# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2 Policy Framework.

Minimal Cosmos-Policy-style implementation on top of the existing
Cosmos-Predict2 diffusers backbone. The action chunk is injected as one
extra latent frame, and the Cosmos DiT is trained to denoise that frame.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict2PolicyDefaultConfig:
    """Cosmos-Predict2 latent-action policy defaults."""

    name: str = "CosmoPredict2Policy"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "nvidia/Cosmos-Predict2-2B-Video2World",
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
        "min_timestep": 0.02,
        "max_timestep": 1.0,
        "noise_offset": 0.02,
        "action_loss_weight": 1.0,
        "future_image_loss_weight": 1.0,
        "prediction_type": "x0",
        # Keep the default sampler at one denoise step. The hand-written
        # multi-step path below is not the official Cosmos Policy sampler and
        # can drift badly for future-image decoding.
        "num_inference_steps": 1,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict2Policy")
class CosmoPredict2_Policy(baseframework):
    """Cosmos-Predict2 with latent-frame action injection."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict2PolicyDefaultConfig, config)
        self.backbone = get_world_model(config=self.config)
        self._disable_backbone_feature_hooks()

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = self.config.framework.action_model.action_dim

        policy_cfg = self.config.framework.get("policy", {})
        self.min_timestep = float(policy_cfg.get("min_timestep", 0.02))
        self.max_timestep = float(policy_cfg.get("max_timestep", 1.0))
        self.noise_offset = float(policy_cfg.get("noise_offset", 0.02))
        self.action_loss_weight = float(policy_cfg.get("action_loss_weight", 1.0))
        self.future_image_loss_weight = float(policy_cfg.get("future_image_loss_weight", 1.0))
        self.prediction_type = policy_cfg.get("prediction_type", "x0")
        self.num_inference_steps = int(policy_cfg.get("num_inference_steps", 1))
        self._warned_experimental_multistep = False
        self._wm_debug_count = 0

    def _disable_backbone_feature_hooks(self) -> None:
        """Policy training does not use DiT feature hooks.

        The Cosmos2 backbone registers hooks for Perceiver/GR00T-style hidden
        state extraction. In policy mode we call ``backbone.transformer``
        directly; leaving those hooks attached would append every step's DiT
        activations to ``_intermediate_features`` and keep the autograd graph
        alive, causing a linear GPU memory leak.
        """
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

    def _actions_to_latent_frame(self, actions: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, height, width = latents.shape
        flat_actions = actions.reshape(batch_size, -1).to(device=latents.device, dtype=latents.dtype)
        slot_count = self._latent_slot_count(latents)
        repeat_factor = math.ceil(slot_count / flat_actions.shape[1])
        tiled_actions = flat_actions.repeat(1, repeat_factor)[:, :slot_count]
        return tiled_actions.reshape(batch_size, channels, height, width)

    def _latent_frame_to_actions(self, action_frame: torch.Tensor, action_shape) -> torch.Tensor:
        batch_size = action_frame.shape[0]
        flat_frame = action_frame.reshape(batch_size, -1)
        action_dim = int(np.prod(action_shape[1:]))
        num_chunks = flat_frame.shape[1] // action_dim
        action_chunks = flat_frame[:, : num_chunks * action_dim].reshape(batch_size, num_chunks, action_dim)
        return action_chunks.mean(dim=1).reshape(batch_size, *action_shape[1:])

    def _make_policy_inputs(
        self,
        wm_inputs: dict,
        target_frames: List[torch.Tensor],
        noises: List[torch.Tensor],
        timestep: torch.Tensor,
        add_noise_offset: bool = True,
    ):
        latents = wm_inputs["hidden_states"]
        batch_size, _, t_lat, h_lat, w_lat = latents.shape
        dtype = latents.dtype

        timestep_view = timestep.view(batch_size, 1, 1, 1)
        noised_frames = []
        for target_frame, noise in zip(target_frames, noises):
            noised_frame = (1.0 - timestep_view) * target_frame + timestep_view * noise
            if add_noise_offset and self.noise_offset > 0:
                noised_frame = noised_frame + self.noise_offset * torch.randn_like(noised_frame)
            noised_frames.append(noised_frame.unsqueeze(2))

        num_target_frames = len(noised_frames)
        hidden_states = torch.cat([latents, *noised_frames], dim=2)
        condition_mask = latents.new_zeros((batch_size, 1, t_lat + num_target_frames, h_lat, w_lat))
        obs_condition_mask = wm_inputs.get("condition_mask", None)
        if obs_condition_mask is not None:
            condition_mask[:, :, :t_lat] = obs_condition_mask.to(device=latents.device, dtype=latents.dtype)
        else:
            condition_mask[:, :, :t_lat] = 1.0

        return {
            "hidden_states": hidden_states.to(dtype),
            "timestep": timestep.to(device=latents.device, dtype=torch.float32),
            "encoder_hidden_states": wm_inputs["encoder_hidden_states"],
            "condition_mask": condition_mask.to(dtype),
            "padding_mask": wm_inputs.get("padding_mask", None),
        }

    def _run_transformer(self, policy_inputs: dict, num_target_frames: int) -> torch.Tensor:
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.backbone.transformer(
                hidden_states=policy_inputs["hidden_states"],
                timestep=policy_inputs["timestep"],
                encoder_hidden_states=policy_inputs["encoder_hidden_states"],
                condition_mask=policy_inputs.get("condition_mask", None),
                padding_mask=policy_inputs.get("padding_mask", None),
            )

        pred = output.sample if hasattr(output, "sample") else output
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_frames = pred[:, :, -num_target_frames:].float()
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

    def _encode_future_image_frame(self, future_images: List) -> torch.Tensor:
        future_latents, _ = self._encode_future_image_frames(future_images)
        return future_latents[:, :, 0]

    def _decode_latent_frame(self, latent_frame: torch.Tensor) -> List[np.ndarray]:
        return self._decode_latent_frames(latent_frame.unsqueeze(2))

    def _decode_latent_frames(self, latent_frames: torch.Tensor, frame_counts: Optional[List[int]] = None) -> List[np.ndarray]:
        batch_size, channels, num_frames, height, width = latent_frames.shape
        latent_video = (
            latent_frames.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * num_frames, channels, 1, height, width)
            .to(device=latent_frames.device, dtype=self.backbone.vae.dtype)
        )

        if self.backbone.vae.config.latents_mean is not None:
            z_dim = self.backbone.vae.config.z_dim
            latents_mean = (
                torch.tensor(self.backbone.vae.config.latents_mean)
                .view(1, z_dim, 1, 1, 1)
                .to(device=latent_video.device, dtype=latent_video.dtype)
            )
            latents_std = (
                torch.tensor(self.backbone.vae.config.latents_std)
                .view(1, z_dim, 1, 1, 1)
                .to(device=latent_video.device, dtype=latent_video.dtype)
            )
            sigma_data = self.backbone.scheduler.config.sigma_data
            latent_video = latent_video / sigma_data * latents_std + latents_mean

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

    @staticmethod
    def _image_frame_counts(images) -> List[int]:
        frame_counts = [len(sample) if isinstance(sample, (list, tuple)) else 1 for sample in images]
        return [max(1, int(frame_count)) for frame_count in frame_counts]

    def _num_image_frames(self, images) -> int:
        frame_counts = self._image_frame_counts(images)
        return max(frame_counts) if frame_counts else 1

    @staticmethod
    def _latent_frame_mask(frame_counts: List[int], num_frames: int, like: torch.Tensor) -> torch.Tensor:
        mask = like.new_zeros((len(frame_counts), 1, num_frames, 1, 1), dtype=torch.float32)
        for batch_idx, frame_count in enumerate(frame_counts):
            mask[batch_idx, :, : max(1, min(int(frame_count), num_frames))] = 1.0
        return mask

    def _denoise_step(self, sample: torch.Tensor, pred: torch.Tensor, timestep: torch.Tensor, next_timestep: torch.Tensor):
        timestep = timestep.view(-1, 1, 1, 1).to(device=sample.device, dtype=sample.dtype).clamp_min(1e-6)
        next_timestep = next_timestep.view(-1, 1, 1, 1).to(device=sample.device, dtype=sample.dtype)

        if self.prediction_type == "noise":
            pred_noise = pred.to(dtype=sample.dtype)
            if torch.all(timestep >= 1.0 - 1e-6):
                return sample + (next_timestep - timestep) * pred_noise
            pred_x0 = (sample - timestep * pred_noise) / (1.0 - timestep).clamp_min(1e-6)
        else:
            pred_x0 = pred.to(dtype=sample.dtype)
            pred_noise = (sample - (1.0 - timestep) * pred_x0) / timestep

        return (1.0 - next_timestep) * pred_x0 + next_timestep * pred_noise

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        future_images = [example["future_image"] for example in examples] if "future_image" in examples[0] else None
        instructions = [example["lang"] for example in examples]
        actions = self._select_action_chunk([example["action"] for example in examples])

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        latents = wm_inputs["hidden_states"]
        actions = actions.to(device=latents.device, dtype=torch.float32)
        target_action_frame = self._actions_to_latent_frame(actions, latents).float()
        target_frames = [target_action_frame.to(latents.dtype)]
        target_names = ["action"]
        target_future_frames = None
        future_frame_counts = None
        future_start_index = None

        if future_images is not None and self.future_image_loss_weight > 0:
            target_future_frames, future_frame_counts = self._encode_future_image_frames(future_images)
            future_start_index = len(target_frames)
            for frame_idx in range(target_future_frames.shape[2]):
                target_frames.append(target_future_frames[:, :, frame_idx].to(latents.dtype))
                target_names.append("future_image")

        batch_size = target_action_frame.shape[0]
        timestep = torch.empty(batch_size, device=latents.device).uniform_(self.min_timestep, self.max_timestep)
        noises = [torch.randn_like(target_frame).to(latents.dtype) for target_frame in target_frames]

        policy_inputs = self._make_policy_inputs(
            wm_inputs=wm_inputs,
            target_frames=target_frames,
            noises=noises,
            timestep=timestep,
        )
        pred_frames = self._run_transformer(policy_inputs, num_target_frames=len(target_frames))
        pred_action_frame = pred_frames[:, :, 0]

        action_target = noises[0].float() if self.prediction_type == "noise" else target_action_frame
        action_loss = F.mse_loss(pred_action_frame, action_target, reduction="none")

        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        if action_mask is not None:
            action_mask = torch.tensor(np.array(action_mask), device=latents.device, dtype=torch.float32)
            action_mask = action_mask[:, -(self.future_action_window_size + 1):, :]
            latent_mask = self._actions_to_latent_frame(action_mask, latents).float().clamp(0.0, 1.0)
            action_loss = action_loss * latent_mask
            action_loss = action_loss.sum() / latent_mask.sum().clamp_min(1.0)
        else:
            action_loss = action_loss.mean()

        total_loss = action_loss * self.action_loss_weight
        output = {"action_loss": total_loss, "action_latent_loss": action_loss.detach()}

        if target_future_frames is not None and future_frame_counts is not None and future_start_index is not None:
            future_end_index = future_start_index + target_future_frames.shape[2]
            pred_future_frame = pred_frames[:, :, future_start_index:future_end_index]
            if self.prediction_type == "noise":
                future_target = torch.stack(noises[future_start_index:future_end_index], dim=2).float()
            else:
                future_target = target_future_frames
            future_mask = self._latent_frame_mask(
                future_frame_counts,
                target_future_frames.shape[2],
                pred_future_frame,
            )
            future_loss = F.mse_loss(pred_future_frame, future_target.float(), reduction="none") * future_mask
            future_denom = future_mask.sum().clamp_min(1.0) * pred_future_frame.shape[1] * pred_future_frame.shape[3] * pred_future_frame.shape[4]
            future_image_loss = future_loss.sum() / future_denom
            total_loss = total_loss + future_image_loss * self.future_image_loss_weight
            output["action_loss"] = total_loss
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

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        latents = wm_inputs["hidden_states"]
        batch_size, channels, _, height, width = latents.shape
        action_shape = (batch_size, self.future_action_window_size + 1, self.action_dim)
        future_frame_counts = self._image_frame_counts(batch_images)
        num_future_frames = max(future_frame_counts)

        num_inference_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        num_inference_steps = max(num_inference_steps, 1)
        action_inference_steps = num_inference_steps
        if self.prediction_type == "x0":
            action_inference_steps = 1
        debug_trace = self._begin_wm_debug_trace()
        self._log_wm_debug(
            debug_trace,
            (
                f"start requested_num_inference_steps={num_inference_steps} "
                f"action_inference_steps={action_inference_steps} prediction_type={self.prediction_type}"
            ),
            obs_latents=latents,
        )
        if action_inference_steps != num_inference_steps and not self._warned_experimental_multistep:
            print(
                "Warning: CosmoPredict2Policy prediction_type=x0 is a one-step x0 predictor; "
                "using one action denoise step and ignoring requested multi-step action sampling."
            )
            self._warned_experimental_multistep = True
        elif num_inference_steps > 1 and not self._warned_experimental_multistep:
            print(
                "Warning: CosmoPredict2Policy num_inference_steps > 1 uses an experimental "
                "linear denoise loop, not the official Cosmos Policy sampler."
            )
            self._warned_experimental_multistep = True

        target_frames = [torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)]
        if action_inference_steps == 1:
            target_frames.extend(
                torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
                for _ in range(num_future_frames)
            )
        timesteps = torch.linspace(1.0, 0.0, action_inference_steps + 1, device=latents.device, dtype=torch.float32)

        for step_idx in range(action_inference_steps):
            timestep = timesteps[step_idx].expand(batch_size)
            next_timestep = timesteps[step_idx + 1].expand(batch_size)
            policy_inputs = self._make_policy_inputs(
                wm_inputs=wm_inputs,
                target_frames=target_frames,
                noises=target_frames,
                timestep=timestep,
                add_noise_offset=False,
            )
            pred_frames = self._run_transformer(policy_inputs, num_target_frames=len(target_frames))
            self._log_wm_debug(
                debug_trace,
                f"step={step_idx} t={timesteps[step_idx].item():.4g}->next={timesteps[step_idx + 1].item():.4g}",
                sample_action_before=target_frames[0],
                pred_action=pred_frames[:, :, 0],
                pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
            )
            target_frames = [
                self._denoise_step(target_frames[frame_idx], pred_frames[:, :, frame_idx], timestep, next_timestep)
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

        if action_inference_steps == 1:
            pred_future_frames = torch.stack([frame.float() for frame in target_frames[1:]], dim=2)
        else:
            future_frames = [
                torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
                for _ in range(num_future_frames)
            ]
            visual_frames = [
                torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype),
                *future_frames,
            ]
            timestep = torch.ones(batch_size, device=latents.device, dtype=torch.float32)
            policy_inputs = self._make_policy_inputs(
                wm_inputs=wm_inputs,
                target_frames=visual_frames,
                noises=visual_frames,
                timestep=timestep,
                add_noise_offset=False,
            )
            pred_frames = self._run_transformer(policy_inputs, num_target_frames=len(visual_frames))
            pred_future_frames = pred_frames[:, :, 1:].float()
        self._log_wm_debug(debug_trace, "final_future", pred_future_frames=pred_future_frames)
        pred_future_images = self._decode_latent_frames(pred_future_frames, frame_counts=future_frame_counts)

        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "pred_future_images": pred_future_images,
        }
