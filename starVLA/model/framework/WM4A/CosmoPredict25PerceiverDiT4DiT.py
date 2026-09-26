# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5-Perceiver with DiT4DiT-style video dynamics fusion.

This keeps starVLA's Perceiver action head, but aligns the Cosmos path with
DiT4DiT: the action head consumes the hidden state captured from the first
Cosmos denoising call, and the optional future-video loss supervises the same
condition/future latent layout with rectified-flow targets.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.WM4A.CosmoPredict25PerceiverFutureImage import (
    CosmoPredict25_Perceiver_FutureImage,
    CosmoPredict25PerceiverFutureImageDefaultConfig,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiT4DiTPerceiverHead import get_action_model as get_dit4dit_action_model
from starVLA.model.modules.action_model.PerceiverHead import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict25PerceiverDiT4DiTDefaultConfig(CosmoPredict25PerceiverFutureImageDefaultConfig):
    """Cosmos-Predict2.5-Perceiver defaults for DiT4DiT-style fusion."""

    name: str = "CosmoPredict25PerceiverDiT4DiT"

    dit4dit_video: dict = field(default_factory=lambda: {
        "enabled": True,
        "training": "joint",  # joint | action | video
        "future_loss_type": "flow_matching",
        "future_loss_weight": 1.0,
        "future_num_inference_steps": 1,
        "conditional_frame_timestep": 0.0001,
        "tri_timestep": {
            "video_time_distribution": "uniform",
            "feature_timestep": 1.0,
            "action_time_distribution": "beta_action_head",
        },
        "return_pred_future_video": False,
    })

    dit4dit_action_head: dict = field(default_factory=lambda: {
        "enabled": False,
    })


@FRAMEWORK_REGISTRY.register("CosmoPredict25PerceiverDiT4DiT")
class CosmoPredict25_Perceiver_DiT4DiT(CosmoPredict25_Perceiver_FutureImage):
    """Cosmos first-denoising hidden + starVLA Perceiver action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(CosmoPredict25PerceiverDiT4DiTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        action_head_cfg = dict(self.config.framework.get("dit4dit_action_head", {}))
        if bool(action_head_cfg.get("enabled", False)):
            self.action_model: FlowmatchingActionHead = get_dit4dit_action_model(config=self.config)
        else:
            self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.state_dim = self.config.framework.action_model.get("state_dim", None)

    def _video_cfg(self) -> dict:
        return dict(self.config.framework.get("dit4dit_video", {}))

    @staticmethod
    def _flatten_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 3:
            return hidden
        if hidden.dim() == 5:
            batch, channels, frames, height, width = hidden.shape
            return hidden.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
        raise ValueError(f"Unsupported Cosmos hidden shape: {tuple(hidden.shape)}")

    def _preprocess_video_sequence(self, sequences: List[List], height: int, width: int) -> torch.Tensor:
        videos = []
        for sequence in sequences:
            video = self.backbone.video_processor.preprocess_video(sequence, height=height, width=width)
            videos.append(video.squeeze(0))
        dtype = self.backbone.vae.dtype
        device = next(self.backbone.vae.parameters()).device
        return torch.stack(videos, dim=0).to(device=device, dtype=dtype)

    @staticmethod
    def _is_auto_size(value) -> bool:
        return isinstance(value, str) and value.lower() in {"auto", "infer"}

    def _resolve_video_size(self, cfg: dict, condition_sequences: List[List], future_sequences: Optional[List[List]] = None):
        height_cfg = cfg.get("height", getattr(self.backbone, "_height", 704))
        width_cfg = cfg.get("width", getattr(self.backbone, "_width", 1280))

        height = None if self._is_auto_size(height_cfg) else int(height_cfg)
        width = None if self._is_auto_size(width_cfg) else int(width_cfg)
        if height is not None and width is not None:
            return height, width

        images = []
        for sequence in condition_sequences:
            images.extend(sequence)
        if future_sequences is not None:
            for sequence in future_sequences:
                images.extend(sequence)
        if not images:
            raise ValueError("Cannot infer video size from empty image sequences.")

        if height is None:
            height = max(int(image.height) for image in images)
        if width is None:
            width = max(int(image.width) for image in images)
        return height, width

    def _split_condition_future(self, examples: List[dict]) -> tuple[List[List], Optional[List[List]], List[str]]:
        condition_sequences = []
        future_sequences = []
        has_future = False

        for example in examples:
            image_sequence = self._as_sequence(to_pil_preserve(example["image"]))
            if not image_sequence:
                raise ValueError("Each example must contain at least one image.")

            condition_sequences.append([image_sequence[0]])
            if "future_image" in example:
                future = self._as_sequence(to_pil_preserve(example["future_image"]))
            elif len(image_sequence) > 1:
                future = image_sequence[1:]
            else:
                future = []

            if future:
                has_future = True
            future_sequences.append(future)

        if not has_future:
            return condition_sequences, None, [example["lang"] for example in examples]

        max_future = max(len(future) for future in future_sequences)
        padded_future = []
        for cond, future in zip(condition_sequences, future_sequences):
            if not future:
                future = [cond[-1]]
            if len(future) < max_future:
                future = future + [future[-1]] * (max_future - len(future))
            padded_future.append(future)

        return condition_sequences, padded_future, [example["lang"] for example in examples]

    def _prepare_dit4dit_latents(
        self,
        condition_video: torch.Tensor,
        num_frames_out: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ):
        batch_size = condition_video.shape[0]
        num_channels = int(self.backbone.transformer.config.in_channels) - 1
        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        spatial_factor = int(getattr(self.backbone, "vae_scale_factor_spatial", 16))
        t_lat = (int(num_frames_out) - 1) // temporal_factor + 1
        h_lat = height // spatial_factor
        w_lat = width // spatial_factor

        if condition_video.shape[2] < num_frames_out:
            pad = condition_video.new_zeros(
                batch_size,
                condition_video.shape[1],
                num_frames_out - condition_video.shape[2],
                condition_video.shape[3],
                condition_video.shape[4],
            )
            padded_video = torch.cat([condition_video, pad], dim=2)
        else:
            padded_video = condition_video[:, :, :num_frames_out]

        with torch.no_grad():
            cond_latents = self.backbone.vae.encode(padded_video).latent_dist.sample()

        latents_mean = self.backbone.latents_mean.to(device=cond_latents.device, dtype=cond_latents.dtype)
        latents_std = self.backbone.latents_std.to(device=cond_latents.device, dtype=cond_latents.dtype)
        cond_latents = ((cond_latents - latents_mean) / latents_std).to(dtype)

        target_shape = (batch_size, num_channels, t_lat, h_lat, w_lat)
        if cond_latents.shape != target_shape:
            adjusted = cond_latents.new_zeros(target_shape)
            t_copy = min(cond_latents.shape[2], t_lat)
            adjusted[:, :, :t_copy] = cond_latents[:, :, :t_copy]
            cond_latents = adjusted

        latents = torch.randn(target_shape, device=condition_video.device, dtype=dtype)
        cond_count = min(t_lat, (condition_video.shape[2] - 1) // temporal_factor + 1)
        cond_indicator = latents.new_zeros(batch_size, 1, t_lat, 1, 1)
        cond_indicator[:, :, :cond_count] = 1.0
        cond_mask = cond_indicator.expand(batch_size, 1, t_lat, h_lat, w_lat)
        return latents, cond_latents, cond_mask, cond_indicator, cond_count

    def _encode_video_latents_mean(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.backbone.vae.encode(video.to(dtype=self.backbone.vae.dtype))
            latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
        latents = latents.float()
        mean = self.backbone.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.backbone.latents_std.to(device=latents.device, dtype=latents.dtype)
        if mean.ndim == 5 and mean.shape[2] >= latents.shape[2]:
            mean = mean[:, :, :latents.shape[2]]
        if std.ndim == 5 and std.shape[2] >= latents.shape[2]:
            std = std[:, :, :latents.shape[2]]
        return (latents - mean) / std

    def _decode_latents_to_video(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.backbone.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.backbone.latents_std.to(device=latents.device, dtype=latents.dtype)
        if mean.ndim == 5 and mean.shape[2] >= latents.shape[2]:
            mean = mean[:, :, :latents.shape[2]]
        if std.ndim == 5 and std.shape[2] >= latents.shape[2]:
            std = std[:, :, :latents.shape[2]]
        decoded = self.backbone.vae.decode((latents * std + mean).to(dtype=self.backbone.vae.dtype), return_dict=False)[0]
        if decoded.ndim == 5 and decoded.shape[1] == 3:
            decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()
        elif decoded.ndim == 4 and decoded.shape[1] == 3:
            decoded = decoded.unsqueeze(1)
        decoded = decoded.float()
        if decoded.min() < -0.5:
            decoded = (decoded + 1.0) / 2.0
        return decoded.clamp(0.0, 1.0)

    def _tri_timestep_cfg(self, cfg: Optional[dict] = None) -> dict:
        cfg = self._video_cfg() if cfg is None else cfg
        tri_cfg = dict(cfg.get("tri_timestep", {}))
        tri_cfg.setdefault("video_time_distribution", cfg.get("time_distribution", "uniform"))
        tri_cfg.setdefault("feature_timestep", cfg.get("feature_timestep", 1.0))
        tri_cfg.setdefault("action_time_distribution", "beta_action_head")
        return tri_cfg

    def _sample_video_flow_time(self, batch_size: int, device: torch.device, cfg: dict) -> torch.Tensor:
        tri_cfg = self._tri_timestep_cfg(cfg)
        distribution = str(tri_cfg.get("video_time_distribution", "uniform")).lower()
        if distribution in {"logit_normal", "logitnormal"}:
            time = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
        elif distribution == "uniform":
            time = torch.rand(batch_size, device=device, dtype=torch.float32)
        elif distribution in {"fixed", "constant"}:
            time = torch.full(
                (batch_size,),
                fill_value=float(tri_cfg.get("video_timestep", 0.5)),
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported dit4dit_video.tri_timestep.video_time_distribution={distribution!r}")

        high_sigma_ratio = tri_cfg.get("high_sigma_ratio", cfg.get("high_sigma_ratio", None))
        if high_sigma_ratio is not None and float(high_sigma_ratio) > 0:
            high_sigma_min = float(tri_cfg.get("high_sigma_min", cfg.get("high_sigma_min", 0.98)))
            high_mask = torch.rand(batch_size, device=device) < float(high_sigma_ratio)
            high_time = torch.rand(batch_size, device=device, dtype=torch.float32) * (1.0 - high_sigma_min) + high_sigma_min
            time = torch.where(high_mask, high_time, time)
        return time

    def _dit4dit_cosmos_forward(
        self,
        condition_sequences: List[List],
        instructions: List[str],
        future_sequences: Optional[List[List]] = None,
        return_pred_future_video: bool = False,
    ):
        cfg = self._video_cfg()
        height, width = self._resolve_video_size(cfg, condition_sequences, future_sequences)
        condition_video = self._preprocess_video_sequence(condition_sequences, height, width)

        future_video = None
        num_frames_out = condition_video.shape[2]
        if future_sequences is not None:
            future_video = self._preprocess_video_sequence(future_sequences, height, width)
            num_frames_out += future_video.shape[2]
        num_frames_out = max(num_frames_out, 1 + int(getattr(self.backbone, "vae_scale_factor_temporal", 4)))

        dtype = self.backbone.transformer.dtype
        prompt_embeds = self.backbone._encode_text(instructions).to(device=condition_video.device, dtype=dtype)
        latents, cond_latents, cond_mask, cond_indicator, cond_count = self._prepare_dit4dit_latents(
            condition_video=condition_video,
            num_frames_out=num_frames_out,
            height=height,
            width=width,
            dtype=dtype,
        )

        num_steps = max(1, int(cfg.get("future_num_inference_steps", 1)))
        self.backbone.scheduler.set_timesteps(num_steps, device=condition_video.device)
        timesteps = self.backbone.scheduler.timesteps
        sigma = getattr(self.backbone.scheduler, "sigmas", None)
        tri_cfg = self._tri_timestep_cfg(cfg)
        feature_timestep = float(tri_cfg.get("feature_timestep", 1.0))
        feature_timestep_tensor = torch.full((1,), feature_timestep, device=condition_video.device, dtype=dtype)

        cond_timestep = torch.ones_like(cond_indicator, dtype=dtype) * float(
            cfg.get("conditional_frame_timestep", getattr(self.backbone, "_conditional_frame_timestep", 0.0001))
        )
        in_timestep = (
            cond_indicator.to(dtype) * cond_timestep
            + (1.0 - cond_indicator.to(dtype)) * feature_timestep_tensor
        )
        padding_mask = latents.new_zeros((1, 1, height, width), dtype=dtype)
        cond_mask = cond_mask.to(dtype)

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()

        in_latents = cond_mask * cond_latents + (1.0 - cond_mask) * latents
        model_out = self.backbone.transformer(
            hidden_states=in_latents,
            timestep=in_timestep,
            encoder_hidden_states=prompt_embeds,
            condition_mask=cond_mask,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        if not getattr(self.backbone, "_intermediate_features", None):
            hidden = model_out
        else:
            hidden = self.backbone._intermediate_features[-1]
        hidden = self._flatten_hidden(hidden)

        future_loss = None
        pred_future_video = None
        if future_video is not None and str(cfg.get("future_loss_type", "flow_matching")).lower() in {
            "flow_matching",
            "latent_flow_matching",
            "rectified_flow",
            "rf",
        }:
            full_video = torch.cat([condition_video, future_video], dim=2)
            min_full_frames = 1 + int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
            if full_video.shape[2] < min_full_frames:
                pad = full_video[:, :, -1:].repeat(1, 1, min_full_frames - full_video.shape[2], 1, 1)
                full_video = torch.cat([full_video, pad], dim=2)

            gt_latents = self._encode_video_latents_mean(full_video)
            pred_future_len = latents.shape[2] - cond_count
            gt_future = gt_latents[:, :, cond_count:cond_count + pred_future_len].to(
                device=latents.device, dtype=torch.float32
            )

            if pred_future_len > 0 and gt_future.numel() > 0:
                t_sup = min(pred_future_len, gt_future.shape[2])
                x0_future = gt_future[:, :, :t_sup]
                flow_time = self._sample_video_flow_time(latents.shape[0], latents.device, cfg).view(-1, 1, 1, 1, 1)
                noise_future = torch.randn_like(x0_future)
                xt_future = (1.0 - flow_time) * x0_future + flow_time * noise_future

                xt_full = torch.randn_like(latents.float())
                xt_full[:, :, cond_count:cond_count + t_sup] = xt_future
                t_full = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
                t_full[:, :, cond_count:cond_count + t_sup] = flow_time

                v_pred = self.backbone.transformer(
                    hidden_states=cond_mask * cond_latents + (1.0 - cond_mask) * xt_full.to(dtype),
                    timestep=cond_indicator.to(dtype) * cond_timestep + (1.0 - cond_indicator.to(dtype)) * t_full.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                v_target = (noise_future - x0_future).to(device=v_pred.device, dtype=v_pred.dtype)
                future_loss = F.mse_loss(v_pred[:, :, cond_count:cond_count + t_sup].float(), v_target.float())
            else:
                future_loss = latents.new_tensor(0.0)

        if return_pred_future_video:
            scheduler_cfg = getattr(self.backbone.scheduler, "config", None)
            prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower()
            step_model_out = model_out
            if prediction_type == "flow_prediction":
                gt_velocity = (latents - cond_latents).to(dtype=dtype) * cond_mask
                step_model_out = gt_velocity + model_out * (1.0 - cond_mask)

            denoised_latents = self.backbone.scheduler.step(step_model_out, timesteps[0], latents, return_dict=False)[0]
            for step_idx, timestep in enumerate(timesteps[1:], start=1):
                sigma_t = (
                    torch.tensor(sigma[step_idx].item(), device=condition_video.device, dtype=dtype).view(1)
                    if sigma is not None
                    else torch.ones(1, device=condition_video.device, dtype=dtype)
                )
                in_latents = cond_mask * cond_latents + (1.0 - cond_mask) * denoised_latents
                in_timestep = cond_indicator.to(dtype) * cond_timestep + (1.0 - cond_indicator.to(dtype)) * sigma_t
                model_out_i = self.backbone.transformer(
                    hidden_states=in_latents,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                if prediction_type == "flow_prediction":
                    gt_velocity = (denoised_latents - cond_latents).to(dtype=dtype) * cond_mask
                    model_out_i = gt_velocity + model_out_i * (1.0 - cond_mask)
                denoised_latents = self.backbone.scheduler.step(
                    model_out_i,
                    timestep,
                    denoised_latents,
                    return_dict=False,
                )[0]
            pred_future_video = self._decode_latents_to_video(denoised_latents)

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        return hidden, future_loss, pred_future_video

    def _sample_dit4dit_future_video(
        self,
        condition_sequences: List[List],
        instructions: List[str],
        generation_cfg: dict,
    ) -> torch.Tensor:
        video_cfg = self._video_cfg()
        size_cfg = {
            "height": generation_cfg.get("height", video_cfg.get("height", getattr(self.backbone, "_height", 704))),
            "width": generation_cfg.get("width", video_cfg.get("width", getattr(self.backbone, "_width", 1280))),
        }
        height, width = self._resolve_video_size(size_cfg, condition_sequences)
        condition_video = self._preprocess_video_sequence(condition_sequences, height, width)

        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        num_frames_out = int(generation_cfg.get("num_frames", 1 + temporal_factor))
        num_frames_out = max(num_frames_out, condition_video.shape[2], 1 + temporal_factor)

        dtype = self.backbone.transformer.dtype
        prompt_embeds = self.backbone._encode_text(instructions).to(device=condition_video.device, dtype=dtype)
        latents, cond_latents, cond_mask, cond_indicator, _ = self._prepare_dit4dit_latents(
            condition_video=condition_video,
            num_frames_out=num_frames_out,
            height=height,
            width=width,
            dtype=dtype,
        )

        num_steps = int(generation_cfg.get("num_inference_steps", video_cfg.get("future_num_inference_steps", 1)))
        num_steps = max(1, num_steps)
        self.backbone.scheduler.set_timesteps(num_steps, device=condition_video.device)
        timesteps = self.backbone.scheduler.timesteps
        sigmas = getattr(self.backbone.scheduler, "sigmas", None)

        cond_timestep = torch.ones_like(cond_indicator, dtype=dtype) * float(
            generation_cfg.get(
                "conditional_frame_timestep",
                video_cfg.get(
                    "conditional_frame_timestep",
                    getattr(self.backbone, "_conditional_frame_timestep", 0.0001),
                ),
            )
        )
        padding_mask = latents.new_zeros((1, 1, height, width), dtype=dtype)
        cond_mask = cond_mask.to(dtype)

        scheduler_cfg = getattr(self.backbone.scheduler, "config", None)
        prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower()

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()

        sample_latents = latents
        for step_idx, timestep in enumerate(timesteps):
            sigma_t = (
                torch.tensor(sigmas[step_idx].item(), device=condition_video.device, dtype=dtype).view(1)
                if sigmas is not None
                else torch.ones(1, device=condition_video.device, dtype=dtype)
            )
            in_latents = cond_mask * cond_latents + (1.0 - cond_mask) * sample_latents
            in_timestep = cond_indicator.to(dtype) * cond_timestep + (1.0 - cond_indicator.to(dtype)) * sigma_t
            model_out = self.backbone.transformer(
                hidden_states=in_latents,
                timestep=in_timestep,
                encoder_hidden_states=prompt_embeds,
                condition_mask=cond_mask,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            if prediction_type == "flow_prediction":
                gt_velocity = (sample_latents - cond_latents).to(dtype=dtype) * cond_mask
                model_out = gt_velocity + model_out * (1.0 - cond_mask)
            sample_latents = self.backbone.scheduler.step(model_out, timestep, sample_latents, return_dict=False)[0]

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        return self._decode_latents_to_video(sample_latents)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        condition_sequences, future_sequences, instructions = self._split_condition_future(examples)
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            condition_sequences = [resize_images(seq, target_size=train_obs_image_size) for seq in condition_sequences]
            if future_sequences is not None:
                future_sequences = [resize_images(seq, target_size=train_obs_image_size) for seq in future_sequences]

        video_cfg = self._video_cfg()
        training_mode = str(video_cfg.get("training", "joint")).lower()
        use_video_loss = bool(video_cfg.get("enabled", True)) and training_mode in {"joint", "video"}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden, future_video_loss, _ = self._dit4dit_cosmos_forward(
                condition_sequences=condition_sequences,
                instructions=instructions,
                future_sequences=future_sequences if use_video_loss else None,
                return_pred_future_video=False,
            )

        if training_mode == "video":
            if future_video_loss is None:
                raise ValueError("dit4dit_video.training='video' requires future frames in image or future_image.")
            return {"action_loss": future_video_loss, "future_video_loss": future_video_loss.detach()}

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "trainer")
                else self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state = self._align_state_dim(state)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(np.array(action_mask), device=last_hidden.device, dtype=last_hidden.dtype)
                action_mask_target = action_mask_tensor[:, -(self.future_action_window_size + 1):, :]
                action_mask_repeated = action_mask_target.repeat(repeated_diffusion_steps, 1, 1)

            embodiment_tag_repeated = None
            if embodiment_tag is not None:
                embodiment_tag = torch.tensor(np.array(embodiment_tag), device=last_hidden.device, dtype=torch.int64).view(-1)
                embodiment_tag_repeated = embodiment_tag.repeat(repeated_diffusion_steps)

            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                embodiment_tag_repeated,
                action_mask=action_mask_repeated,
            )

        output = {"action_loss": action_loss}
        if future_video_loss is not None:
            future_video_loss_scaled = future_video_loss * float(video_cfg.get("future_loss_weight", 1.0))
            output["perceiver_action_loss"] = action_loss.detach()
            output["future_video_loss"] = future_video_loss.detach()
            output["future_video_loss_scaled"] = future_video_loss_scaled.detach()
            output["action_loss"] = action_loss + future_video_loss_scaled
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        condition_sequences, _, instructions = self._split_condition_future(examples)
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            condition_sequences = [resize_images(seq, target_size=train_obs_image_size) for seq in condition_sequences]

        generation_cfg = self._get_future_image_generation_cfg(kwargs)
        return_pred_future_video = bool(generation_cfg.get("enabled", True))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden, _, _ = self._dit4dit_cosmos_forward(
                condition_sequences=condition_sequences,
                instructions=instructions,
                future_sequences=None,
                return_pred_future_video=False,
            )

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        state = self._align_state_dim(state)
        embodiment_tag = (
            torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64)
            if embodiment_tag is not None
            else None
        )
        if embodiment_tag is not None:
            embodiment_tag = embodiment_tag.view(-1)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state, embodiment_tag)

        output = {"normalized_actions": pred_actions.detach().cpu().numpy()}
        if return_pred_future_video:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_video = self._sample_dit4dit_future_video(
                    condition_sequences=condition_sequences,
                    instructions=instructions,
                    generation_cfg=generation_cfg,
                )
            if bool(generation_cfg.get("return_full_video", False)):
                output["pred_future_images"] = [sample.detach().cpu() for sample in pred_video]
            else:
                frame_idx = int(generation_cfg.get("future_frame_index", -1))
                output["pred_future_images"] = pred_video[:, frame_idx].detach().cpu()
        return output
