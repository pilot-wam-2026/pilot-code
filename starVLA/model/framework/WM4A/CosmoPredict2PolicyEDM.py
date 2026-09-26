# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2 EDM Policy Framework.

This is an EDM/SDE variant of ``CosmoPredict2Policy`` following the sampling
semantics used by NVLabs Cosmos Policy:

  train:      x_sigma = x0 + sigma * epsilon
  denoise:    model(c_in * x_sigma, c_noise) -> raw output, then EDM x0 head
  inference: randn * sigma_max -> Karras sigma solver -> clean denoise

It intentionally lives in a separate file so the existing policy route remains
available for A/B runs.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.WM4A.CosmoPredict2Policy import CosmoPredict2_Policy
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict2PolicyEDMDefaultConfig:
    """Cosmos-Predict2 latent-action policy defaults with EDM sigma-space."""

    name: str = "CosmoPredict2PolicyEDM"

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
        "sigma_data": 1.0,
        "sigma_min": 0.0002,
        "sigma_max": 80.0,
        "sigma_p_mean": -1.2,
        "sigma_p_std": 1.2,
        "hybrid_sigma_distribution": True,
        "hybrid_sigma_lognormal_prob": 0.7,
        "uniform_lower": 1.0,
        "uniform_upper": 85.0,
        "rho": 7.0,
        "action_loss_weight": 1.0,
        "future_image_loss_weight": 1.0,
        "loss_weighting": "edm",
        "max_loss_weight": 100.0,
        "num_inference_steps": 1,
        # Cosmos Policy Predict2 uses a truncated inference schedule for action
        # prediction; keep training sigma_min separate from the sampling floor.
        "inference_sigma_min": 4.0,
        "inference_sigma_max": 80.0,
        "sampler": "2ab",
        "preconditioning": True,
        "conditional_sigma": 0.002,
        # Diffusers' Cosmos transformer does not expose the official
        # PolicyVideo2World wrapper, so we provide the same c_noise values here.
        "timestep_mapping": "c_noise",
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict2PolicyEDM")
class CosmoPredict2_Policy_EDM(CosmoPredict2_Policy):
    """Cosmos-Predict2 policy trained and sampled in EDM sigma-space."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(CosmoPredict2PolicyEDMDefaultConfig, config)
        self.backbone = get_world_model(config=self.config)
        self._disable_backbone_feature_hooks()

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = self.config.framework.action_model.action_dim

        policy_cfg = self.config.framework.get("policy", {})
        self.sigma_data = float(policy_cfg.get("sigma_data", 1.0))
        self.sigma_min = float(policy_cfg.get("sigma_min", 0.002))
        self.sigma_max = float(policy_cfg.get("sigma_max", 80.0))
        self.sigma_p_mean = float(policy_cfg.get("sigma_p_mean", -1.2))
        self.sigma_p_std = float(policy_cfg.get("sigma_p_std", 1.2))
        self.hybrid_sigma_distribution = self._as_bool(policy_cfg.get("hybrid_sigma_distribution", True))
        self.hybrid_sigma_lognormal_prob = float(policy_cfg.get("hybrid_sigma_lognormal_prob", 0.7))
        self.uniform_lower = float(policy_cfg.get("uniform_lower", 1.0))
        self.uniform_upper = float(policy_cfg.get("uniform_upper", 85.0))
        self.rho = float(policy_cfg.get("rho", 7.0))
        self.action_loss_weight = float(policy_cfg.get("action_loss_weight", 1.0))
        self.future_image_loss_weight = float(policy_cfg.get("future_image_loss_weight", 1.0))
        self.loss_weighting = policy_cfg.get("loss_weighting", "edm")
        self.max_loss_weight = float(policy_cfg.get("max_loss_weight", 100.0))
        self.num_inference_steps = int(policy_cfg.get("num_inference_steps", 1))
        self.inference_sigma_min = float(policy_cfg.get("inference_sigma_min", 4.0))
        self.inference_sigma_max = float(policy_cfg.get("inference_sigma_max", self.sigma_max))
        self.sampler = policy_cfg.get("sampler", "2ab").lower()
        if self.sampler not in {"euler", "heun", "2ab"}:
            raise ValueError(f"Unsupported sampler: {self.sampler}. Expected one of: euler, heun, 2ab.")
        self.preconditioning = self._as_bool(policy_cfg.get("preconditioning", True))
        self.conditional_sigma = float(policy_cfg.get("conditional_sigma", self.sigma_min))
        self.timestep_mapping = policy_cfg.get("timestep_mapping", "c_noise")
        self._wm_debug_count = 0

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

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

    def _project_action_frame_to_action_manifold(
        self,
        action_frame: torch.Tensor,
        action_shape,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        actions = self._latent_frame_to_actions(action_frame.float(), action_shape)
        return self._actions_to_latent_frame(actions, latents).to(dtype=action_frame.dtype)

    def _sample_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        log_sigma = torch.randn(batch_size, device=device, dtype=torch.float32)
        sigmas = torch.exp(log_sigma * self.sigma_p_std + self.sigma_p_mean)
        if self.hybrid_sigma_distribution:
            use_lognormal = torch.rand(batch_size, device=device) < self.hybrid_sigma_lognormal_prob
            uniform_sigmas = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(
                self.uniform_lower,
                self.uniform_upper,
            )
            sigmas = torch.where(use_lognormal, sigmas, uniform_sigmas)
        return sigmas.clamp(self.sigma_min, self.sigma_max)

    def _preconditioning(self, sigmas: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigmas.ndim == 2 and like.ndim == 5:
            sigma_shape = (sigmas.shape[0], 1, sigmas.shape[1], 1, 1)
        else:
            sigma_shape = (sigmas.shape[0],) + (1,) * (like.ndim - 1)
        sigma = sigmas.to(device=like.device, dtype=torch.float32).view(sigma_shape)
        sigma_data = torch.tensor(self.sigma_data, device=like.device, dtype=torch.float32)
        denom = sigma.square() + sigma_data.square()
        c_skip = sigma_data.square() / denom
        c_out = sigma * sigma_data / denom.sqrt()
        c_in = denom.rsqrt()
        return (
            c_skip.to(dtype=like.dtype),
            c_out.to(dtype=like.dtype),
            c_in.to(dtype=like.dtype),
        )

    def _sigma_to_timestep(self, sigmas: torch.Tensor) -> torch.Tensor:
        sigmas = sigmas.to(dtype=torch.float32)
        if self.timestep_mapping == "c_noise":
            return 0.25 * sigmas.clamp_min(1e-6).log()
        if self.timestep_mapping == "sigma":
            return sigmas
        if self.timestep_mapping == "log_sigma":
            return sigmas.clamp_min(1e-6).log()
        if self.timestep_mapping == "sigma_to_t":
            return sigmas / (sigmas + self.sigma_data)
        raise ValueError(f"Unsupported timestep_mapping: {self.timestep_mapping}")

    def _loss_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        if self.loss_weighting == "none":
            weights = torch.ones_like(sigmas)
        elif self.loss_weighting == "edm":
            weights = (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data).clamp_min(1e-6) ** 2
        else:
            raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")

        if self.max_loss_weight > 0:
            weights = weights.clamp_max(self.max_loss_weight)
        return weights

    def _make_edm_policy_inputs(
        self,
        wm_inputs: dict,
        sample_frames: List[torch.Tensor],
        sigmas: torch.Tensor,
        sample_sigmas: Optional[List[torch.Tensor]] = None,
        sample_condition_mask: Optional[List[bool]] = None,
    ) -> dict:
        latents = wm_inputs["hidden_states"]
        batch_size, _, t_lat, h_lat, w_lat = latents.shape
        dtype = latents.dtype
        if sample_sigmas is None:
            sample_sigmas = [sigmas for _ in sample_frames]
        if len(sample_sigmas) != len(sample_frames):
            raise ValueError(f"Expected {len(sample_frames)} sample sigmas, got {len(sample_sigmas)}.")

        scaled_sample_frames = []
        for frame, frame_sigmas in zip(sample_frames, sample_sigmas):
            if self.preconditioning:
                _, _, c_in = self._preconditioning(frame_sigmas, frame)
                frame = frame * c_in
            scaled_sample_frames.append(frame.unsqueeze(2))

        hidden_states = torch.cat([latents, *scaled_sample_frames], dim=2)
        condition_mask = latents.new_zeros((batch_size, 1, t_lat + len(sample_frames), h_lat, w_lat))
        obs_condition_mask = wm_inputs.get("condition_mask", None)
        if obs_condition_mask is not None:
            condition_mask[:, :, :t_lat] = obs_condition_mask.to(device=latents.device, dtype=latents.dtype)
        else:
            condition_mask[:, :, :t_lat] = 1.0

        obs_timestep = wm_inputs.get("timestep", None)
        if obs_timestep is not None:
            obs_timestep = obs_timestep.to(device=latents.device, dtype=dtype)
            if obs_timestep.ndim == 1:
                obs_timestep = obs_timestep.view(batch_size, 1, 1, 1, 1).expand(batch_size, 1, t_lat, 1, 1)
        else:
            obs_sigmas = latents.new_full((batch_size,), self.conditional_sigma, dtype=torch.float32)
            obs_timestep = self._sigma_to_timestep(obs_sigmas).view(batch_size, 1, 1, 1, 1)
            obs_timestep = obs_timestep.to(device=latents.device, dtype=dtype)
            obs_timestep = obs_timestep.expand(batch_size, 1, t_lat, 1, 1)

        target_timesteps = []
        for frame_sigmas in sample_sigmas:
            target_timestep = self._sigma_to_timestep(frame_sigmas).view(batch_size, 1, 1, 1, 1)
            target_timesteps.append(target_timestep.to(device=latents.device, dtype=dtype))
        target_timestep = torch.cat(target_timesteps, dim=2)
        timestep_tensor = torch.cat([obs_timestep, target_timestep], dim=2)

        if sample_condition_mask is not None:
            if len(sample_condition_mask) != len(sample_frames):
                raise ValueError(f"Expected {len(sample_frames)} sample condition masks, got {len(sample_condition_mask)}.")
            for frame_idx, is_condition in enumerate(sample_condition_mask):
                if is_condition:
                    condition_mask[:, :, t_lat + frame_idx] = 1.0

        return {
            "hidden_states": hidden_states.to(dtype),
            "timestep": timestep_tensor,
            "encoder_hidden_states": wm_inputs["encoder_hidden_states"],
            "condition_mask": condition_mask.to(dtype),
            "padding_mask": wm_inputs.get("padding_mask", None),
        }

    def _denoise_x0(
        self,
        wm_inputs: dict,
        sample_frames: List[torch.Tensor],
        sigmas: torch.Tensor,
        sample_sigmas: Optional[List[torch.Tensor]] = None,
        sample_condition_mask: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        if sample_sigmas is None:
            sample_sigmas = [sigmas for _ in sample_frames]
        policy_inputs = self._make_edm_policy_inputs(
            wm_inputs=wm_inputs,
            sample_frames=sample_frames,
            sigmas=sigmas,
            sample_sigmas=sample_sigmas,
            sample_condition_mask=sample_condition_mask,
        )
        raw_pred_frames = self._run_transformer(policy_inputs, num_target_frames=len(sample_frames))
        if not self.preconditioning:
            return raw_pred_frames

        sample_stack = torch.stack([frame.float() for frame in sample_frames], dim=2)
        sigma_stack = torch.stack([frame_sigmas.to(device=sample_stack.device) for frame_sigmas in sample_sigmas], dim=1)
        c_skip, c_out, _ = self._preconditioning(sigma_stack, sample_stack)
        return c_skip.float() * sample_stack + c_out.float() * raw_pred_frames.float()

    def _karras_sigmas(
        self,
        num_steps: int,
        device: torch.device,
        sigma_max: Optional[float] = None,
        sigma_min: Optional[float] = None,
    ) -> torch.Tensor:
        sigma_max = self.sigma_max if sigma_max is None else float(sigma_max)
        sigma_min = self.sigma_min if sigma_min is None else float(sigma_min)
        if sigma_max < sigma_min:
            raise ValueError(f"sigma_max must be >= sigma_min, got {sigma_max:g} < {sigma_min:g}.")
        if num_steps <= 1:
            return torch.tensor([sigma_max], device=device, dtype=torch.float32)

        ramp = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
        min_inv_rho = sigma_min ** (1.0 / self.rho)
        max_inv_rho = sigma_max ** (1.0 / self.rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

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
        sigmas = self._sample_sigmas(batch_size, latents.device)
        sigma_view = sigmas.view(batch_size, 1, 1, 1)
        noises = [torch.randn_like(target_frame).to(latents.dtype) for target_frame in target_frames]
        sample_frames = [
            target_frame + sigma_view.to(target_frame.dtype) * noise
            for target_frame, noise in zip(target_frames, noises)
        ]

        pred_frames = self._denoise_x0(wm_inputs=wm_inputs, sample_frames=sample_frames, sigmas=sigmas)
        pred_action_frame = pred_frames[:, :, 0]

        loss_weight = self._loss_weights(sigmas).view(batch_size, 1, 1, 1)
        action_loss = F.mse_loss(pred_action_frame, target_action_frame, reduction="none") * loss_weight

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
            future_mask = self._latent_frame_mask(
                future_frame_counts,
                target_future_frames.shape[2],
                pred_future_frame,
            )
            future_loss = (
                F.mse_loss(pred_future_frame, target_future_frames.float(), reduction="none")
                * loss_weight.unsqueeze(2)
                * future_mask
            )
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

        num_inference_steps = max(int(kwargs.get("num_inference_steps", self.num_inference_steps)), 1)
        inference_sigma_min = float(kwargs.get("inference_sigma_min", self.inference_sigma_min))
        inference_sigma_max = float(kwargs.get("inference_sigma_max", self.inference_sigma_max))
        solver_steps = num_inference_steps - 1 if num_inference_steps > 1 else 1
        sigmas = self._karras_sigmas(
            solver_steps,
            latents.device,
            sigma_max=inference_sigma_max,
            sigma_min=inference_sigma_min,
        )
        debug_trace = self._begin_wm_debug_trace()
        self._log_wm_debug(
            debug_trace,
            (
                f"start num_inference_steps={num_inference_steps} sampler={self.sampler} "
                f"train_sigma_min={self.sigma_min:g} train_sigma_max={self.sigma_max:g} "
                f"inference_sigma_min={inference_sigma_min:g} "
                f"inference_sigma_max={inference_sigma_max:g} "
                f"preconditioning={self.preconditioning} timestep_mapping={self.timestep_mapping}"
            ),
            obs_latents=latents,
            sigmas=sigmas,
        )
        sample_frames = [
            torch.randn(batch_size, channels, height, width, device=latents.device, dtype=latents.dtype)
            * sigmas[0].to(dtype=latents.dtype)
            for _ in range(1 + num_future_frames)
        ]

        if num_inference_steps == 1:
            sigma_batch = sigmas[0].expand(batch_size)
            pred_frames = self._denoise_x0(wm_inputs=wm_inputs, sample_frames=sample_frames, sigmas=sigma_batch)
            self._log_wm_debug(
                debug_trace,
                f"single_step sigma={sigmas[0].item():.4g}",
                sample_action=sample_frames[0],
                pred_action=pred_frames[:, :, 0],
                pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
            )
        else:
            previous_derivatives = None
            for step_idx in range(len(sigmas) - 1):
                sigma = sigmas[step_idx]
                next_sigma = sigmas[step_idx + 1]
                dt = (next_sigma - sigma).to(device=latents.device, dtype=latents.dtype)
                sigma_batch = sigma.expand(batch_size)
                pred_frames = self._denoise_x0(wm_inputs=wm_inputs, sample_frames=sample_frames, sigmas=sigma_batch)
                derivatives = []
                euler_frames = []
                for frame_idx, sample in enumerate(sample_frames):
                    pred_x0 = pred_frames[:, :, frame_idx].to(dtype=sample.dtype)
                    derivative = (sample - pred_x0) / sigma.clamp_min(1e-6).to(dtype=sample.dtype)
                    derivatives.append(derivative)
                    euler_frames.append(sample + dt.to(dtype=sample.dtype) * derivative)

                self._log_wm_debug(
                    debug_trace,
                    f"step={step_idx} sigma={sigma.item():.4g}->next={next_sigma.item():.4g}",
                    sample_action_before=sample_frames[0],
                    sample_future_before=sample_frames[1] if len(sample_frames) > 1 else None,
                    pred_action=pred_frames[:, :, 0],
                    pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
                    derivative_action=derivatives[0],
                    euler_action=euler_frames[0],
                )

                if self.sampler == "2ab" and previous_derivatives is not None:
                    sample_frames = [
                        sample
                        + dt.to(dtype=sample.dtype)
                        * (1.5 * derivative - 0.5 * previous_derivative.to(dtype=derivative.dtype))
                        for sample, derivative, previous_derivative in zip(
                            sample_frames,
                            derivatives,
                            previous_derivatives,
                        )
                    ]
                elif self.sampler == "heun" and next_sigma > self.sigma_min:
                    next_sigma_batch = next_sigma.expand(batch_size)
                    next_pred_frames = self._denoise_x0(
                        wm_inputs=wm_inputs,
                        sample_frames=euler_frames,
                        sigmas=next_sigma_batch,
                    )
                    sample_frames = [
                        sample
                        + dt.to(dtype=sample.dtype)
                        * (
                            derivative
                            + (euler - next_pred_frames[:, :, frame_idx].to(dtype=sample.dtype))
                            / next_sigma.clamp_min(1e-6).to(dtype=sample.dtype)
                        )
                        * 0.5
                        for frame_idx, (sample, euler, derivative) in enumerate(
                            zip(sample_frames, euler_frames, derivatives)
                        )
                    ]
                else:
                    sample_frames = euler_frames
                previous_derivatives = [derivative.detach() for derivative in derivatives]

                sample_frames[0] = self._project_action_frame_to_action_manifold(
                    sample_frames[0],
                    action_shape,
                    latents,
                )
                self._log_wm_debug(
                    debug_trace,
                    f"step={step_idx} after_update",
                    sample_action_after=sample_frames[0],
                    sample_future_after=sample_frames[1] if len(sample_frames) > 1 else None,
                )

            clean_sigma_batch = sigmas[-1].expand(batch_size)
            pred_frames = self._denoise_x0(wm_inputs=wm_inputs, sample_frames=sample_frames, sigmas=clean_sigma_batch)
            self._log_wm_debug(
                debug_trace,
                f"clean_denoise sigma={sigmas[-1].item():.4g}",
                sample_action=sample_frames[0],
                sample_future=sample_frames[1] if len(sample_frames) > 1 else None,
                pred_action=pred_frames[:, :, 0],
                pred_future=pred_frames[:, :, 1:] if pred_frames.shape[2] > 1 else None,
            )

        pred_action_frame = pred_frames[:, :, 0].float()
        pred_future_frames = pred_frames[:, :, 1:].float()
        pred_actions = self._latent_frame_to_actions(pred_action_frame, action_shape)
        self._log_wm_debug(
            debug_trace,
            "final_action",
            pred_action_frame=pred_action_frame,
            pred_actions=pred_actions,
        )

        self._log_wm_debug(debug_trace, "final_future", pred_future_frames=pred_future_frames)
        pred_future_images = self._decode_latent_frames(pred_future_frames, frame_counts=future_frame_counts)

        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "pred_future_images": pred_future_images,
        }
