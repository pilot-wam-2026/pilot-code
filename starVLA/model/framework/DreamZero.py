# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""DreamZero Wan2.2 policy framework for starVLA.

This is a starVLA-native DreamZero model file. It uses the existing
Wan2.2-TI2V-5B-Diffusers world model in ``starVLA.model.modules.world_model.Wan2``
and trains joint action/future-image latent flow targets, so it does not depend
on the official DreamZero training stack, Hydra, GrootSimPolicy, or PEFT.
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
class DreamZeroDefaultConfig:
    """Wan2.2 DreamZero-style action/video policy defaults."""

    name: str = "DreamZero"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
        "height": 224,
        "width": 224,
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_dim": 32,
        "state_dim": 64,
        "future_action_window_size": 15,
        "action_horizon": 16,
        "past_action_window_size": 0,
    })

    policy: dict = field(default_factory=lambda: {
        "train_time_distribution": "logitnormal",
        "shift": 5.0,
        "action_loss_weight": 1.0,
        "future_image_loss_weight": 1.0,
        "num_inference_steps": 5,
        "timestep_scale": 1000.0,
        "decode_future_images": True,
        "autoregressive": True,
        "latent_clip_value": 30.0,
        "prediction_clip_value": 30.0,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("DreamZero")
class DreamZero(baseframework):
    """Wan2.2 latent action and future-image flow-matching policy."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(DreamZeroDefaultConfig, config)
        self.backbone = get_world_model(config=self.config)
        self._disable_backbone_feature_hooks()

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = self.config.framework.action_model.action_dim

        policy_cfg = self.config.framework.get("policy", {})
        self.train_time_distribution = policy_cfg.get("train_time_distribution", "logitnormal")
        self.shift = float(policy_cfg.get("shift", 5.0))
        self.action_loss_weight = float(policy_cfg.get("action_loss_weight", 1.0))
        self.future_image_loss_weight = float(policy_cfg.get("future_image_loss_weight", 1.0))
        self.num_inference_steps = int(policy_cfg.get("num_inference_steps", 5))
        self.timestep_scale = float(policy_cfg.get("timestep_scale", 1000.0))
        self.decode_future_images = self._as_bool(policy_cfg.get("decode_future_images", True))
        self.autoregressive = self._as_bool(policy_cfg.get("autoregressive", True))
        self.latent_clip_value = float(policy_cfg.get("latent_clip_value", 30.0))
        self.prediction_clip_value = float(policy_cfg.get("prediction_clip_value", 30.0))
        self._wm_debug_count = 0

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

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
        parts = [f"[WM_DEBUG_VERSION=DreamZeroWan2Native] {type(self).__name__} {label}"]
        for name, tensor in tensors.items():
            if tensor is not None:
                parts.append(self._tensor_stats(name, tensor))
        print(" | ".join(parts), flush=True)

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor, clip_value: Optional[float] = None) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_value is not None and clip_value > 0:
            tensor = tensor.clamp(min=-clip_value, max=clip_value)
        return tensor

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

    def _latent_frame_to_actions(self, action_frame: torch.Tensor, action_shape) -> torch.Tensor:
        batch_size = action_frame.shape[0]
        flat_dim = int(np.prod(action_shape[1:]))
        flat_frame = action_frame.reshape(batch_size, -1)
        slot_count = flat_frame.shape[1]
        action_indices = torch.arange(slot_count, device=flat_frame.device) % flat_dim

        action_sum = flat_frame.new_zeros((batch_size, flat_dim))
        action_count = flat_frame.new_zeros((batch_size, flat_dim))
        expanded_indices = action_indices.expand(batch_size, -1)
        action_sum.scatter_add_(1, expanded_indices, flat_frame)
        action_count.scatter_add_(1, expanded_indices, torch.ones_like(flat_frame))
        return (action_sum / action_count.clamp_min(1.0)).reshape(batch_size, *action_shape[1:])

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

    def _flow_inference_schedule(self, num_steps: int, device: torch.device, shift: Optional[float] = None):
        base_times = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
        return self._shift_time(base_times, shift=shift)

    def _make_wan_timestep(
        self,
        hidden_states: torch.Tensor,
        target_indices: List[int],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_latent_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.backbone.transformer.config.patch_size
        frame_token_count = (height // p_h) * (width // p_w)
        temporal_token_count = num_latent_frames // p_t
        seq_len = temporal_token_count * frame_token_count
        rope_max_seq_len = getattr(self.backbone.transformer.config, "rope_max_seq_len", None)
        if rope_max_seq_len is not None and seq_len > rope_max_seq_len:
            raise ValueError(
                f"DreamZero Wan policy seq_len={seq_len} exceeds rope_max_seq_len={rope_max_seq_len}. "
                "Reduce observation frames/resolution or disable optional future-image targets."
            )

        timestep_tokens = torch.zeros(
            batch_size,
            seq_len,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        timestep = timestep.to(device=hidden_states.device, dtype=torch.float32) * self.timestep_scale
        for target_idx in target_indices:
            temporal_idx = int(target_idx) // p_t
            start = temporal_idx * frame_token_count
            end = start + frame_token_count
            timestep_tokens[:, start:end] = timestep[:, None]
        return timestep_tokens

    def _make_policy_inputs(
        self,
        wm_inputs: dict,
        target_frames: List[torch.Tensor],
        timestep: torch.Tensor,
        context_frames: Optional[List[torch.Tensor]] = None,
    ):
        condition_latents = wm_inputs["hidden_states"]
        context_frames = context_frames or []
        hidden_states = torch.cat(
            [
                condition_latents,
                *[frame.unsqueeze(2).to(condition_latents.dtype) for frame in context_frames],
                *[frame.unsqueeze(2).to(condition_latents.dtype) for frame in target_frames],
            ],
            dim=2,
        )
        condition_t = condition_latents.shape[2] + len(context_frames)
        target_indices = list(range(condition_t, condition_t + len(target_frames)))
        timestep_tokens = self._make_wan_timestep(hidden_states, target_indices=target_indices, timestep=timestep)
        return {
            "hidden_states": hidden_states,
            "timestep": timestep_tokens,
            "encoder_hidden_states": wm_inputs["encoder_hidden_states"],
            "target_indices": target_indices,
        }

    def _run_transformer(self, policy_inputs: dict) -> torch.Tensor:
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.backbone.transformer(
                hidden_states=policy_inputs["hidden_states"],
                timestep=policy_inputs["timestep"],
                encoder_hidden_states=policy_inputs["encoder_hidden_states"],
            )
        pred = output.sample if hasattr(output, "sample") else output
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = self._sanitize_tensor(pred, clip_value=self.prediction_clip_value)
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        return pred[:, :, policy_inputs["target_indices"]].float()

    def _predict_velocity(
        self,
        wm_inputs: dict,
        target_frames: List[torch.Tensor],
        timestep: torch.Tensor,
        context_frames: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        policy_inputs = self._make_policy_inputs(
            wm_inputs=wm_inputs,
            target_frames=target_frames,
            timestep=timestep,
            context_frames=context_frames,
        )
        return self._run_transformer(policy_inputs)

    def _predict_velocity_autoregressive(
        self,
        wm_inputs: dict,
        sample_frames: List[torch.Tensor],
        timestep: torch.Tensor,
        teacher_context_frames: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        pred_frames = []
        context_frames = list(teacher_context_frames or [])
        for frame_idx, sample_frame in enumerate(sample_frames):
            pred_frame = self._predict_velocity(
                wm_inputs=wm_inputs,
                target_frames=[sample_frame],
                timestep=timestep,
                context_frames=context_frames,
            )[:, :, 0]
            pred_frames.append(pred_frame)
            context_frames.append(sample_frame.detach())
        return torch.stack(pred_frames, dim=2)

    def _encode_future_image_frame(self, future_images: List, like_latents: torch.Tensor) -> Optional[torch.Tensor]:
        if future_images is None or self.future_image_loss_weight <= 0:
            return None
        future_latents = self.backbone._encode_images_vae(future_images, num_frames=1).float()
        future_frame = future_latents[:, :, 0].to(device=like_latents.device, dtype=like_latents.dtype)
        return self._sanitize_tensor(future_frame, clip_value=self.latent_clip_value)

    def _decode_latent_frames(self, latent_frames: torch.Tensor) -> List[np.ndarray]:
        latent_video = latent_frames.to(device=latent_frames.device, dtype=self.backbone.vae.dtype)

        if getattr(self.backbone.vae.config, "latents_mean", None) is not None:
            latents_mean = (
                torch.tensor(self.backbone.vae.config.latents_mean)
                .view(1, self.backbone.vae.config.z_dim, 1, 1, 1)
                .to(device=latent_video.device, dtype=latent_video.dtype)
            )
            latents_std = (
                torch.tensor(self.backbone.vae.config.latents_std)
                .view(1, self.backbone.vae.config.z_dim, 1, 1, 1)
                .to(device=latent_video.device, dtype=latent_video.dtype)
            )
            latent_video = latent_video * latents_std + latents_mean

        with torch.no_grad():
            decoded = self.backbone.vae.decode(latent_video).sample

        decoded = decoded.float().clamp(-1, 1)
        decoded = ((decoded + 1.0) * 127.5).clamp(0, 255).byte()
        decoded = decoded.permute(0, 2, 3, 4, 1).detach().cpu().numpy()
        images = []
        for sample in decoded:
            frames = [sample[idx] for idx in range(sample.shape[0])]
            images.append(frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1))
        return images

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        future_images = [example["future_image"] for example in examples] if "future_image" in examples[0] else None
        instructions = [example["lang"] for example in examples]
        actions = self._select_action_chunk([example["action"] for example in examples])

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        latents = wm_inputs["hidden_states"]
        latents = self._sanitize_tensor(latents, clip_value=self.latent_clip_value)
        wm_inputs["hidden_states"] = latents
        actions = self._sanitize_tensor(actions.to(device=latents.device, dtype=torch.float32), clip_value=1.0)
        target_action_frame = self._actions_to_latent_frame(actions, latents).float()
        target_action_frame = self._sanitize_tensor(target_action_frame, clip_value=self.latent_clip_value)

        target_frames = [target_action_frame.to(latents.dtype)]
        future_frame = self._encode_future_image_frame(future_images, like_latents=latents)
        if future_frame is not None:
            target_frames.append(future_frame)

        batch_size = latents.shape[0]
        train_time = self._sample_train_times(batch_size, latents.device)
        flow_time = self._shift_time(train_time)
        time_view = flow_time.view(batch_size, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)

        noises = [torch.randn_like(target_frame).to(latents.dtype) for target_frame in target_frames]
        sample_frames = [
            self._sanitize_tensor(noise * time_view + target_frame * (1.0 - time_view), clip_value=self.latent_clip_value)
            for target_frame, noise in zip(target_frames, noises)
        ]
        target_velocities = [
            self._sanitize_tensor(noise.float() - target_frame.float(), clip_value=self.latent_clip_value)
            for target_frame, noise in zip(target_frames, noises)
        ]

        if self.autoregressive:
            pred_frames = self._predict_velocity_autoregressive(
                wm_inputs=wm_inputs,
                sample_frames=sample_frames,
                timestep=flow_time,
            )
        else:
            pred_frames = self._predict_velocity(wm_inputs=wm_inputs, target_frames=sample_frames, timestep=flow_time)
        pred_frames = self._sanitize_tensor(pred_frames, clip_value=self.prediction_clip_value)
        pred_action_frame = pred_frames[:, :, 0]

        action_loss = F.mse_loss(pred_action_frame, target_velocities[0], reduction="none")
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        if action_mask is not None:
            action_mask = torch.tensor(np.array(action_mask), device=latents.device, dtype=torch.float32)
            action_mask = action_mask[:, -(self.future_action_window_size + 1):, :]
            latent_mask = self._actions_to_latent_frame(action_mask, latents).float().clamp(0.0, 1.0)
            action_loss_scalar = (action_loss * latent_mask).sum() / latent_mask.sum().clamp_min(1.0)
        else:
            action_loss_scalar = action_loss.mean()

        total_loss = action_loss_scalar * self.action_loss_weight
        output = {
            "action_loss": total_loss,
            "demo_sample_action_mse_loss": action_loss_scalar.detach(),
        }

        if future_frame is not None:
            pred_future_frame = pred_frames[:, :, 1]
            future_loss = F.mse_loss(pred_future_frame, target_velocities[1], reduction="mean")
            total_loss = total_loss + future_loss * self.future_image_loss_weight
            output["action_loss"] = total_loss
            output["future_image_loss"] = future_loss.detach()

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
        latents = self._sanitize_tensor(latents, clip_value=self.latent_clip_value)
        wm_inputs["hidden_states"] = latents
        batch_size, channels, _, height, width = latents.shape
        action_shape = (batch_size, self.future_action_window_size + 1, self.action_dim)

        num_inference_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        num_inference_steps = max(num_inference_steps, 1)
        shift = float(kwargs.get("shift", self.shift))
        debug_trace = self._begin_wm_debug_trace()
        self._log_wm_debug(
            debug_trace,
            f"start requested_num_inference_steps={num_inference_steps} parameterization=rectified_flow shift={shift:g}",
            obs_latents=latents,
        )

        decode_future_images = self._as_bool(kwargs.get("decode_future_images", self.decode_future_images))
        action_frame = torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
        future_frame = (
            torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
            if decode_future_images
            else None
        )

        sigmas = self._flow_inference_schedule(num_inference_steps, latents.device, shift=shift)
        for step_idx in range(num_inference_steps):
            timestep = sigmas[step_idx].expand(batch_size)
            sigma = sigmas[step_idx]
            next_sigma = sigmas[step_idx + 1]
            pred_action = self._predict_velocity(
                wm_inputs=wm_inputs,
                target_frames=[action_frame],
                timestep=timestep,
            )[:, :, 0]
            pred_action = self._sanitize_tensor(pred_action, clip_value=self.prediction_clip_value)

            pred_future = None
            if decode_future_images and future_frame is not None:
                pred_future = self._predict_velocity(
                    wm_inputs=wm_inputs,
                    target_frames=[future_frame],
                    timestep=timestep,
                    context_frames=[action_frame],
                )[:, :, 0]
                pred_future = self._sanitize_tensor(pred_future, clip_value=self.prediction_clip_value)

            self._log_wm_debug(
                debug_trace,
                f"step={step_idx} sigma={sigma.item():.4g}->{next_sigma.item():.4g}",
                sample_action_before=action_frame,
                pred_action_velocity=pred_action,
                pred_future_velocity=pred_future,
            )
            dt = (next_sigma - sigma).to(device=latents.device, dtype=latents.dtype)
            action_frame = self._sanitize_tensor(
                action_frame + dt * pred_action.to(dtype=latents.dtype),
                clip_value=self.latent_clip_value,
            )
            if pred_future is not None and future_frame is not None:
                future_frame = self._sanitize_tensor(
                    future_frame + dt * pred_future.to(dtype=latents.dtype),
                    clip_value=self.latent_clip_value,
                )
        pred_action_frame = action_frame.float()
        pred_actions = self._latent_frame_to_actions(pred_action_frame, action_shape)
        self._log_wm_debug(debug_trace, "final_action", pred_action_frame=pred_action_frame, pred_actions=pred_actions)

        output = {"normalized_actions": pred_actions.detach().cpu().numpy()}
        if decode_future_images and future_frame is not None:
            output["pred_future_images"] = self._decode_latent_frames(future_frame.float().unsqueeze(2))
        return output
