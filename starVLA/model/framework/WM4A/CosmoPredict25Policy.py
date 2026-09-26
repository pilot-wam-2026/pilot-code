# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5 Policy Framework.

This follows the Cosmos-Predict2.5 policy recipe: actions are injected into
the Cosmos latent video sequence as an extra latent frame, and the Cosmos DiT
is trained with rectified flow / flow matching to predict latent velocity.

Scope:
  - blank/proprio/current observation image/video + language -> condition latents
  - action chunk -> one injected latent frame
  - optional future image latent frames
  - RF target velocity is noise - clean_latent

It keeps the official Cosmos Policy RF/FM training target and denoising loop,
while adapting the frame layout to the fields available in starVLA batches.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict25PolicyDefaultConfig:
    """Cosmos-Predict2.5 latent-action policy defaults."""

    name: str = "CosmoPredict25Policy"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "nvidia/Cosmos-Predict2.5-2B",
        "revision": "diffusers/base/post-trained",
        "height": 704,
        "width": 1280,
        "conditional_frame_timestep": 0.0,
    })

    # Kept for shared config compatibility.
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "nvidia/Cosmos-Predict2.5-2B",
        "vl_hidden_dim": 2048,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
    })

    policy: dict = field(default_factory=lambda: {
        "train_time_distribution": "logitnormal",
        "train_time_weight": "uniform",
        "shift": 5.0,
        "action_loss_weight": 16.0,
        "future_image_loss_weight": 1.0,
        "num_inference_steps": 5,
        "action_inference_mode": "euler",
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict25Policy")
class CosmoPredict25_Policy(baseframework):
    """Cosmos-Predict2.5 with latent-frame action injection."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict25PolicyDefaultConfig, config)
        self.backbone = get_world_model(config=self.config)
        self._disable_backbone_feature_hooks()

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = self.config.framework.action_model.action_dim

        policy_cfg = self.config.framework.get("policy", {})
        self.train_time_distribution = policy_cfg.get("train_time_distribution", "logitnormal")
        self.train_time_weight = policy_cfg.get("train_time_weight", "uniform")
        self.shift = float(policy_cfg.get("shift", 5.0))
        self.action_loss_weight = float(policy_cfg.get("action_loss_weight", 16.0))
        self.future_image_loss_weight = float(policy_cfg.get("future_image_loss_weight", 1.0))
        self.num_inference_steps = int(policy_cfg.get("num_inference_steps", 5))
        self.action_inference_mode = str(policy_cfg.get("action_inference_mode", "euler"))
        self._wm_debug_count = 0

    def _disable_backbone_feature_hooks(self) -> None:
        for hook in getattr(self.backbone, "_hooks", []):
            hook.remove()
        if hasattr(self.backbone, "_hooks"):
            self.backbone._hooks.clear()
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()

    def _begin_wm_debug_trace(self) -> bool:
        if os.getenv("STARVLA_WM_DEBUG", "0").lower() not in {"1", "true", "yes", "on"}:
            return False
        max_traces = int(os.getenv("STARVLA_WM_DEBUG_MAX", "6"))
        if self._wm_debug_count >= max_traces:
            return False
        self._wm_debug_count += 1
        return True

    @staticmethod
    def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
        tensor = tensor.detach().float()
        return (
            f"{name}:shape={tuple(tensor.shape)},"
            f"mean={tensor.mean().item():.4g},std={tensor.std(unbiased=False).item():.4g},"
            f"min={tensor.min().item():.4g},max={tensor.max().item():.4g},"
            f"nan={torch.isnan(tensor).any().item()},inf={torch.isinf(tensor).any().item()}"
        )

    def _log_wm_debug(self, enabled: bool, label: str, **tensors) -> None:
        if not enabled:
            return
        parts = [f"[WM_DEBUG=step_trace] {type(self).__name__} {label}"]
        for name, tensor in tensors.items():
            if tensor is not None:
                parts.append(self._tensor_stats(name, tensor))
        print(" | ".join(parts), flush=True)

    def _select_action_chunk(self, actions) -> torch.Tensor:
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        return actions[:, -(self.future_action_window_size + 1):, :]

    @staticmethod
    def _latent_slot_count(latents: torch.Tensor) -> int:
        _, channels, _, height, width = latents.shape
        return channels * height * width

    def _tensor_to_latent_frame(self, values: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, height, width = latents.shape
        flat_values = values.reshape(batch_size, -1).to(device=latents.device, dtype=latents.dtype)
        slot_count = self._latent_slot_count(latents)
        repeat_factor = math.ceil(slot_count / flat_values.shape[1])
        tiled_values = flat_values.repeat(1, repeat_factor)[:, :slot_count]
        return tiled_values.reshape(batch_size, channels, height, width)

    def _actions_to_latent_frame(self, actions: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self._tensor_to_latent_frame(actions, latents)

    def _state_to_latent_frame(self, states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return self._tensor_to_latent_frame(states, latents)

    def _latent_frame_to_actions(self, action_frame: torch.Tensor, action_shape) -> torch.Tensor:
        batch_size = action_frame.shape[0]
        flat_dim = int(np.prod(action_shape[1:]))
        flat_frame = action_frame.reshape(batch_size, -1)
        slot_count = flat_frame.shape[1]
        action_indices = torch.arange(slot_count, device=flat_frame.device) % flat_dim

        action_sum = flat_frame.new_zeros((batch_size, flat_dim))
        action_count = flat_frame.new_zeros((batch_size, flat_dim))
        action_sum.scatter_add_(1, action_indices.expand(batch_size, -1), flat_frame)
        action_count.scatter_add_(1, action_indices.expand(batch_size, -1), torch.ones_like(flat_frame))
        return (action_sum / action_count.clamp_min(1.0)).reshape(batch_size, *action_shape[1:])

    def _project_action_frame_to_action_manifold(self, action_frame: torch.Tensor, latents: torch.Tensor, action_shape) -> torch.Tensor:
        actions = self._latent_frame_to_actions(action_frame.float(), action_shape)
        return self._actions_to_latent_frame(actions, latents).to(dtype=action_frame.dtype)

    def _sample_train_times(self, batch_size: int, device: torch.device) -> torch.Tensor:
        distribution = str(self.train_time_distribution).lower()
        if distribution == "logitnormal":
            return torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
        if distribution == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        raise ValueError(f"Unsupported train_time_distribution={self.train_time_distribution!r}")

    def _shift_time(self, time: torch.Tensor, shift: Optional[float] = None) -> torch.Tensor:
        shift = self.shift if shift is None else float(shift)
        return shift * time / (1.0 + (shift - 1.0) * time)

    def _time_to_timestep(self, time: torch.Tensor, shift: Optional[float] = None) -> torch.Tensor:
        return self._shift_time(time, shift=shift)

    @staticmethod
    def _sigma_to_timestep(sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def _flow_inference_schedule(self, num_steps: int, device: torch.device, shift: Optional[float] = None):
        base_times = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
        sigmas = self._shift_time(base_times, shift=shift)
        return sigmas

    @staticmethod
    def _stack_optional_tensor(values, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if values is None:
            return None
        return torch.tensor(np.array(values), device=device, dtype=dtype)

    @staticmethod
    def _as_image_list(images) -> List:
        return list(images) if isinstance(images, (list, tuple)) else [images]

    @staticmethod
    def _blank_like(image):
        if isinstance(image, Image.Image):
            return Image.new(image.mode, image.size)
        image_array = np.asarray(image)
        return np.zeros_like(image_array)

    @staticmethod
    def _duplicate_frame(image, count: int) -> List:
        return [image] * count

    def _split_policy_views(self, images: List, blank_image):
        # LeRobot loader returns primary views first and wrist views after them.
        primary = images[0] if images else blank_image
        wrist = images[1] if len(images) > 1 else blank_image
        secondary = images[2] if len(images) > 2 else blank_image
        return wrist, primary, secondary

    def _build_raw_policy_sequences(
        self,
        batch_images: List,
        future_images: Optional[List] = None,
        include_state: bool = False,
        include_future: bool = True,
    ) -> tuple[List[List], dict]:
        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        raw_sequences = []
        slot_info = None

        for batch_idx, images in enumerate(batch_images):
            current_images = self._as_image_list(images)
            future_image_list = (
                self._as_image_list(future_images[batch_idx])
                if future_images is not None
                else current_images
            )
            if not include_future:
                future_image_list = []

            blank_image = self._blank_like(current_images[0])
            _, current_primary, _ = self._split_policy_views(current_images, blank_image)
            if future_image_list:
                _, future_primary, _ = self._split_policy_views(future_image_list, blank_image)
            else:
                future_primary = blank_image

            # Match the official RoboCasa Cosmos Policy latent role layout:
            # [blank, proprio, wrist, primary, secondary, action,
            #  future_proprio, future_wrist, future_primary, future_secondary, value].
            # When starVLA only provides a primary camera, the missing views keep
            # their official temporal slots as blank placeholders.
            sequence = [blank_image]
            slot_images = [
                blank_image,     # current proprio placeholder
                blank_image,     # current wrist placeholder
                current_primary,
                blank_image,     # current secondary placeholder
                blank_image,     # action placeholder
                blank_image,     # future proprio placeholder
                blank_image,     # future wrist placeholder
                future_primary,
                blank_image,     # future secondary placeholder
                blank_image,     # value placeholder
            ]
            for slot_image in slot_images:
                sequence.extend(self._duplicate_frame(slot_image, temporal_factor))

            blank_idx = 0
            state_idx = 1
            current_wrist_idx = 2
            current_primary_idx = 3
            current_secondary_idx = 4
            action_idx = 5
            future_state_idx = 6
            future_wrist_idx = 7
            future_primary_idx = 8
            future_secondary_idx = 9
            value_idx = 10
            condition_indices = [
                blank_idx,
                state_idx,
                current_wrist_idx,
                current_primary_idx,
                current_secondary_idx,
            ]
            future_indices = [future_primary_idx]
            target_indices = [
                action_idx,
                future_state_idx,
                future_wrist_idx,
                future_primary_idx,
                future_secondary_idx,
                value_idx,
            ]

            sample_slot_info = {
                "condition_indices": condition_indices,
                "state_idx": state_idx,
                "current_wrist_idx": current_wrist_idx,
                "current_image_indices": [current_primary_idx],
                "current_secondary_idx": current_secondary_idx,
                "action_idx": action_idx,
                "future_state_idx": future_state_idx,
                "future_wrist_idx": future_wrist_idx,
                "future_indices": future_indices,
                "future_secondary_idx": future_secondary_idx,
                "value_idx": value_idx,
                "target_indices": target_indices,
            }
            if slot_info is None:
                slot_info = sample_slot_info
            elif sample_slot_info != slot_info:
                raise ValueError(
                    "CosmoPredict25Policy requires the same policy frame layout for every sample in a batch."
                )
            raw_sequences.append(sequence)

        return raw_sequences, slot_info

    def _make_policy_inputs(
        self,
        wm_inputs: dict,
        clean_latents: torch.Tensor,
        condition_indices: List[int],
        target_indices: List[int],
        sample_frames: List[torch.Tensor],
        timestep: torch.Tensor,
    ):
        batch_size, _, t_lat, h_lat, w_lat = clean_latents.shape
        dtype = clean_latents.dtype
        num_target_frames = len(sample_frames)

        hidden_states = clean_latents.clone()
        for frame_idx, target_idx in enumerate(target_indices):
            hidden_states[:, :, target_idx] = sample_frames[frame_idx].to(dtype=dtype)

        condition_mask = clean_latents.new_zeros((batch_size, 1, t_lat, h_lat, w_lat))
        condition_mask[:, :, condition_indices] = 1.0

        timestep_tensor = clean_latents.new_zeros((batch_size, 1, t_lat, 1, 1))
        target_timestep = timestep.view(batch_size, 1, 1, 1, 1).to(device=clean_latents.device, dtype=dtype)
        timestep_tensor[:, :, target_indices] = target_timestep.expand(batch_size, 1, num_target_frames, 1, 1)

        return {
            "hidden_states": hidden_states.to(dtype),
            "timestep": timestep_tensor,
            "encoder_hidden_states": wm_inputs["encoder_hidden_states"],
            "condition_mask": condition_mask.to(dtype),
            "padding_mask": wm_inputs.get("padding_mask", None),
        }

    def _predict_velocity(
        self,
        wm_inputs: dict,
        clean_latents: torch.Tensor,
        condition_indices: List[int],
        target_indices: List[int],
        sample_frames: List[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        policy_inputs = self._make_policy_inputs(
            wm_inputs=wm_inputs,
            clean_latents=clean_latents,
            condition_indices=condition_indices,
            target_indices=target_indices,
            sample_frames=sample_frames,
            timestep=timestep,
        )
        return self._run_transformer(policy_inputs, target_indices=target_indices)

    def _run_transformer(self, policy_inputs: dict, target_indices: List[int]) -> torch.Tensor:
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.backbone.transformer(
                hidden_states=policy_inputs["hidden_states"],
                timestep=policy_inputs["timestep"],
                encoder_hidden_states=policy_inputs["encoder_hidden_states"],
                condition_mask=policy_inputs.get("condition_mask", None),
                padding_mask=policy_inputs.get("padding_mask", None),
                return_dict=False,
            )

        pred = output[0] if isinstance(output, tuple) else output
        pred_frames = pred[:, :, target_indices].float()
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        return pred_frames

    def _encode_future_image_frames(self, future_images: List) -> tuple[torch.Tensor, List[int]]:
        future_latents, frame_counts, counts_are_latent_frames = self.backbone._encode_images(future_images)
        if not counts_are_latent_frames:
            frame_counts = [
                (frame_count - 1) // self.backbone.vae_scale_factor_temporal + 1
                for frame_count in frame_counts
            ]
        return future_latents.float(), frame_counts

    def _decode_latent_frames(self, latent_frames: torch.Tensor, frame_counts: Optional[List[int]] = None) -> List[np.ndarray]:
        batch_size, channels, num_frames, height, width = latent_frames.shape
        latent_video = (
            latent_frames.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * num_frames, channels, 1, height, width)
            .to(device=latent_frames.device, dtype=self.backbone.vae.dtype)
        )

        latents_mean = self.backbone.latents_mean.to(device=latent_video.device, dtype=latent_video.dtype)
        latents_std = self.backbone.latents_std.to(device=latent_video.device, dtype=latent_video.dtype)
        latent_video = latent_video * latents_std + latents_mean

        with torch.no_grad():
            decoded = self.backbone.vae.decode(latent_video).sample

        decoded = decoded[:, :, 0].float().clamp(-1, 1)
        decoded = ((decoded + 1.0) * 127.5).clamp(0, 255).byte()
        decoded = decoded.permute(0, 2, 3, 1)
        decoded = decoded.reshape(batch_size, num_frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])
        decoded = decoded.detach().cpu().numpy()

        images = []
        if frame_counts is None:
            frame_counts = [num_frames] * batch_size
        for sample, frame_count in zip(decoded, frame_counts):
            frame_count = max(1, min(int(frame_count), sample.shape[0]))
            sample = sample[:frame_count]
            if frame_count == 1:
                images.append(sample[0])
            else:
                images.append(np.concatenate([frame for frame in sample], axis=1))
        return images

    def _decode_policy_future_images(
        self,
        latent_sequence: torch.Tensor,
        future_indices: List[int],
    ) -> List[np.ndarray]:
        latent_video = latent_sequence.to(device=latent_sequence.device, dtype=self.backbone.vae.dtype)
        latents_mean = self.backbone.latents_mean.to(device=latent_video.device, dtype=latent_video.dtype)
        latents_std = self.backbone.latents_std.to(device=latent_video.device, dtype=latent_video.dtype)
        latent_video = latent_video * latents_std + latents_mean

        with torch.no_grad():
            decoded = self.backbone.vae.decode(latent_video).sample

        decoded = decoded.float().clamp(-1, 1)
        decoded = ((decoded + 1.0) * 127.5).clamp(0, 255).byte()
        decoded = decoded.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        raw_indices = [
            0 if latent_idx == 0 else (int(latent_idx) - 1) * temporal_factor + 1
            for latent_idx in future_indices
        ]

        images = []
        for sample in decoded:
            frames = [sample[min(raw_idx, sample.shape[0] - 1)] for raw_idx in raw_indices]
            if len(frames) == 1:
                images.append(frames[0])
            else:
                images.append(np.concatenate(frames, axis=1))
        return images

    @staticmethod
    def _image_frame_counts(images) -> List[int]:
        frame_counts = [len(sample) if isinstance(sample, (list, tuple)) else 1 for sample in images]
        return [max(1, int(frame_count)) for frame_count in frame_counts]

    @staticmethod
    def _latent_frame_mask(frame_counts: List[int], num_frames: int, like: torch.Tensor) -> torch.Tensor:
        mask = like.new_zeros((len(frame_counts), 1, num_frames, 1, 1), dtype=torch.float32)
        for batch_idx, frame_count in enumerate(frame_counts):
            mask[batch_idx, :, : max(1, min(int(frame_count), num_frames))] = 1.0
        return mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        future_images = [example["future_image"] for example in examples] if "future_image" in examples[0] else None
        instructions = [example["lang"] for example in examples]
        actions = self._select_action_chunk([example["action"] for example in examples])
        states = [example["state"] for example in examples] if "state" in examples[0] else None

        raw_sequences, slot_info = self._build_raw_policy_sequences(
            batch_images=batch_images,
            future_images=future_images,
            include_state=states is not None,
            include_future=True,
        )
        wm_inputs = self.backbone.build_inputs(images=raw_sequences, instructions=instructions)
        latents = wm_inputs["hidden_states"]
        clean_latents = latents.clone()
        actions = actions.to(device=latents.device, dtype=torch.float32)
        states = self._stack_optional_tensor(states, latents.device, torch.float32)
        if states is not None and slot_info["state_idx"] is not None:
            clean_latents[:, :, slot_info["state_idx"]] = self._state_to_latent_frame(states, clean_latents)
        target_action_frame = self._actions_to_latent_frame(actions, clean_latents).float()
        clean_latents[:, :, slot_info["action_idx"]] = target_action_frame.to(clean_latents.dtype)
        target_indices = slot_info["target_indices"]
        target_frames = [clean_latents[:, :, target_idx].to(latents.dtype) for target_idx in target_indices]

        batch_size = target_action_frame.shape[0]
        train_time = self._sample_train_times(batch_size, latents.device)
        flow_time = self._shift_time(train_time)
        timestep = self._sigma_to_timestep(flow_time)
        time_view = flow_time.view(batch_size, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)

        noises = [torch.randn_like(target_frame).to(latents.dtype) for target_frame in target_frames]
        sample_frames = [
            noise * time_view + target_frame * (1.0 - time_view)
            for target_frame, noise in zip(target_frames, noises)
        ]
        target_velocities = [
            (noise.float() - target_frame.float())
            for target_frame, noise in zip(target_frames, noises)
        ]

        pred_frames = self._predict_velocity(
            wm_inputs=wm_inputs,
            clean_latents=clean_latents,
            condition_indices=slot_info["condition_indices"],
            target_indices=target_indices,
            sample_frames=sample_frames,
            timestep=timestep,
        )
        pred_action_frame = pred_frames[:, :, 0]

        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        action_loss = F.mse_loss(pred_action_frame, target_velocities[0], reduction="none")
        if action_mask is not None:
            action_mask = torch.tensor(np.array(action_mask), device=latents.device, dtype=torch.float32)
            action_mask = action_mask[:, -(self.future_action_window_size + 1):, :]
            latent_mask = self._actions_to_latent_frame(action_mask, latents).float().clamp(0.0, 1.0)
            action_loss_scalar = (action_loss * latent_mask).sum() / latent_mask.sum().clamp_min(1.0)
        else:
            action_loss_scalar = action_loss.mean()

        future_positions = [target_indices.index(idx) for idx in slot_info["future_indices"]]
        pred_future_frame = pred_frames[:, :, future_positions]
        future_target = torch.stack([target_velocities[pos] for pos in future_positions], dim=2).float()
        future_diff = future_target - pred_future_frame
        future_image_loss = (future_diff**2).mean()

        # Official Cosmos Policy multiplies selected frame losses and then
        # averages over the full latent time axis, including conditioned frames
        # whose loss is zero because they are replaced by GT velocity.
        total_loss = (
            action_loss_scalar * self.action_loss_weight
            + future_image_loss * self.future_image_loss_weight * len(future_positions)
        ) / clean_latents.shape[2]

        action_diff = target_velocities[0] - pred_action_frame
        output = {
            "action_loss": total_loss,
            "demo_sample_action_mse_loss": (action_diff**2).mean().detach(),
            "demo_sample_action_l1_loss": torch.abs(action_diff).mean().detach(),
            "demo_sample_future_image_mse_loss": future_image_loss.detach(),
            "demo_sample_future_image_l1_loss": torch.abs(future_diff).mean().detach(),
        }
        output["future_image_loss"] = future_image_loss.detach()

        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        states = [example["state"] for example in examples] if "state" in examples[0] else None
        raw_sequences, slot_info = self._build_raw_policy_sequences(
            batch_images=batch_images,
            future_images=None,
            include_state=states is not None,
            include_future=True,
        )
        wm_inputs = self.backbone.build_inputs(images=raw_sequences, instructions=instructions)
        latents = wm_inputs["hidden_states"]
        clean_latents = latents.clone()
        states = self._stack_optional_tensor(states, latents.device, torch.float32)
        if states is not None and slot_info["state_idx"] is not None:
            clean_latents[:, :, slot_info["state_idx"]] = self._state_to_latent_frame(states, clean_latents)
        batch_size, channels, _, height, width = latents.shape
        action_shape = (batch_size, self.future_action_window_size + 1, self.action_dim)
        future_frame_counts = [len(slot_info["future_indices"])] * batch_size
        target_indices = slot_info["target_indices"]
        action_target_indices = [slot_info["action_idx"]]

        num_inference_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        num_inference_steps = max(num_inference_steps, 1)
        shift = float(kwargs.get("shift", self.shift))
        debug_trace = self._begin_wm_debug_trace()
        self._log_wm_debug(
            debug_trace,
            (
                f"start requested_num_inference_steps={num_inference_steps} "
                f"parameterization=rectified_flow shift={shift:g}"
            ),
            obs_latents=latents,
        )

        target_frames = [
            torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
            for _ in target_indices
        ]

        sigmas = self._flow_inference_schedule(num_inference_steps, latents.device, shift=shift)
        if self.action_inference_mode == "one_step_x0":
            timestep = torch.ones((batch_size,), device=latents.device, dtype=torch.float32)
            pred_frames = self._predict_velocity(
                wm_inputs=wm_inputs,
                clean_latents=clean_latents,
                condition_indices=slot_info["condition_indices"],
                target_indices=target_indices,
                sample_frames=target_frames,
                timestep=timestep,
            )
            self._log_wm_debug(
                debug_trace,
                "one_step_x0",
                sample_action_before=target_frames[0],
                pred_action_velocity=pred_frames[:, :, 0],
                pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
            )
            target_frames = [
                target_frames[frame_idx] - pred_frames[:, :, frame_idx].to(dtype=latents.dtype)
                for frame_idx in range(len(target_frames))
            ]
        else:
            for step_idx in range(num_inference_steps):
                timestep = self._sigma_to_timestep(sigmas[step_idx]).expand(batch_size)
                next_sigma = sigmas[step_idx + 1]
                sigma = sigmas[step_idx]
                pred_frames = self._predict_velocity(
                    wm_inputs=wm_inputs,
                    clean_latents=clean_latents,
                    condition_indices=slot_info["condition_indices"],
                    target_indices=target_indices,
                    sample_frames=target_frames,
                    timestep=timestep,
                )
                self._log_wm_debug(
                    debug_trace,
                    (
                        f"step={step_idx} sigma={sigma.item():.4g}->{next_sigma.item():.4g} "
                        f"timestep={timestep[0].item():.4g}"
                    ),
                    sample_action_before=target_frames[0],
                    pred_action_velocity=pred_frames[:, :, 0],
                    pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
                )
                dt = (next_sigma - sigma).to(device=latents.device, dtype=latents.dtype)
                target_frames = [
                    target_frames[frame_idx] + dt * pred_frames[:, :, frame_idx].to(dtype=latents.dtype)
                    for frame_idx in range(len(target_frames))
                ]
                self._log_wm_debug(
                    debug_trace,
                    f"step={step_idx} after_update",
                    sample_action_after=target_frames[0],
                )

        pred_action_frame = target_frames[0].float()
        pred_actions = self._latent_frame_to_actions(pred_action_frame, action_shape)
        self._log_wm_debug(
            debug_trace,
            "final_action",
            pred_action_frame=pred_action_frame,
            pred_actions=pred_actions,
        )

        generated_latents = clean_latents.clone()
        for frame_idx, target_idx in enumerate(target_indices):
            generated_latents[:, :, target_idx] = target_frames[frame_idx].to(dtype=generated_latents.dtype)
        pred_future_frames = generated_latents[:, :, slot_info["future_indices"]].float()
        self._log_wm_debug(debug_trace, "final_future", pred_future_frames=pred_future_frames)

        decode_latents = generated_latents.clone()
        replace_indices = [
            slot_info["state_idx"],
            slot_info["current_wrist_idx"],
            *slot_info["current_image_indices"],
            slot_info["current_secondary_idx"],
            slot_info["action_idx"],
            slot_info["future_state_idx"],
            slot_info["future_wrist_idx"],
            slot_info["future_secondary_idx"],
            slot_info["value_idx"],
        ]
        for replace_idx in replace_indices:
            if replace_idx is not None:
                decode_latents[:, :, replace_idx] = latents[:, :, replace_idx]
        pred_future_images = self._decode_policy_future_images(
            decode_latents.float(),
            future_indices=slot_info["future_indices"],
        )

        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "pred_future_images": pred_future_images,
        }
