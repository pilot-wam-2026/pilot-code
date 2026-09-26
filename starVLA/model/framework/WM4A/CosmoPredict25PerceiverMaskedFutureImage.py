# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Cosmos-Predict2.5 Perceiver with masked future-image denoising.

This variant keeps the existing action path and Cosmos future-image objective,
but builds a latent-grid change mask from the current/future RGB pair. The mask
can focus the loss on changed regions and optionally add noise only inside those
regions, turning future-image training into an inpainting-style objective.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.WM4A.CosmoPredict25PerceiverFutureImage import (
    CosmoPredict25_Perceiver_FutureImage,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("CosmoPredict25PerceiverMaskedFutureImage")
class CosmoPredict25_Perceiver_MaskedFutureImage(CosmoPredict25_Perceiver_FutureImage):
    """Future-image training that denoises mostly/only changed regions."""

    @staticmethod
    def _pil_rgb_to_tensor(image, size: tuple[int, int] | None = None) -> torch.Tensor:
        image = to_pil_preserve(image).convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    def _build_latent_change_masks(
        self,
        batch_images: List,
        future_images: List,
        latent_height: int,
        latent_width: int,
        training_cfg: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary denoise mask and soft loss weights at latent resolution."""
        top_ratio = float(training_cfg.get("change_mask_top_ratio", 0.25))
        top_ratio = min(max(top_ratio, 1e-4), 1.0)
        static_weight = float(training_cfg.get("change_mask_static_weight", 0.05))
        static_weight = min(max(static_weight, 0.0), 1.0)
        soft_loss = self._as_bool(training_cfg.get("change_mask_soft_loss", True))

        binary_masks = []
        loss_weights = []
        for images, future_image in zip(batch_images, future_images):
            current_sequence = self._as_sequence(images)
            future_sequence = self._as_sequence(future_image)
            current_frame = current_sequence[-1]
            target_frame = future_sequence[0]

            current = self._pil_rgb_to_tensor(current_frame)
            target = self._pil_rgb_to_tensor(target_frame, size=to_pil_preserve(current_frame).size)
            diff = (target - current).abs().mean(dim=1, keepdim=True)
            score = F.interpolate(diff, size=(latent_height, latent_width), mode="area")
            flat = score.flatten(1)
            max_score = flat.amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
            normalized = (score / max_score).clamp(0.0, 1.0)

            k = max(1, int(flat.shape[1] * top_ratio))
            threshold = flat.topk(k, dim=1).values[:, -1].view(-1, 1, 1, 1)
            binary = (score >= threshold).float()

            weight_source = normalized if soft_loss else binary
            loss_weight = static_weight + (1.0 - static_weight) * weight_source
            binary_masks.append(binary)
            loss_weights.append(loss_weight)

        binary_mask = torch.cat(binary_masks, dim=0).to(device=device, dtype=torch.float32)
        loss_weight = torch.cat(loss_weights, dim=0).to(device=device, dtype=torch.float32)
        return binary_mask, loss_weight

    def _cosmos25_future_image_loss(self, examples: List[dict]) -> Optional[torch.Tensor]:
        if "future_image" not in examples[0]:
            return None

        training_cfg = self._future_training_cfg()
        if not self._as_bool(training_cfg.get("enabled", True)):
            return None
        if not self._as_bool(training_cfg.get("change_mask_enabled", True)):
            return super()._cosmos25_future_image_loss(examples)

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        future_images = [to_pil_preserve(example["future_image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            future_images = resize_images(future_images, target_size=train_obs_image_size)

        raw_sequences, condition_counts, sample_counts = self._build_future_video_sequences(
            batch_images=batch_images,
            future_images=future_images,
        )

        prompt_embeds = self.backbone._encode_text(instructions)
        clean_latents, _, _ = self.backbone._encode_images(raw_sequences)
        dtype = self.backbone.transformer.dtype
        clean_latents = clean_latents.to(dtype)
        batch_size, channels, num_latents, height, width = clean_latents.shape
        device = clean_latents.device

        condition_mask = clean_latents.new_zeros((batch_size, 1, num_latents, height, width))
        target_mask = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1), dtype=torch.float32)
        cond_indicator = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1))
        for batch_idx, (condition_count, sample_count) in enumerate(zip(condition_counts, sample_counts)):
            condition_count = max(1, min(int(condition_count), num_latents))
            sample_count = min(max(condition_count + 1, int(sample_count)), num_latents)
            condition_mask[batch_idx, :, :condition_count] = 1.0
            cond_indicator[batch_idx, :, :condition_count] = 1.0
            target_mask[batch_idx, :, condition_count:sample_count] = 1.0

        change_binary, change_loss_weight = self._build_latent_change_masks(
            batch_images=batch_images,
            future_images=future_images,
            latent_height=height,
            latent_width=width,
            training_cfg=training_cfg,
            device=device,
        )
        change_binary = change_binary.unsqueeze(2).to(dtype=clean_latents.dtype)
        change_loss_weight = change_loss_weight.unsqueeze(2).to(dtype=torch.float32)

        train_time = self._sample_train_times(
            batch_size=batch_size,
            device=device,
            distribution=training_cfg.get("train_time_distribution", "logitnormal"),
        )
        flow_time = self._shift_time(train_time, shift=float(training_cfg.get("shift", 5.0)))
        target_time = flow_time.view(batch_size, 1, 1, 1, 1).to(device=device, dtype=clean_latents.dtype)

        noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - target_time) + noise * target_time
        target_spatial_mask = target_mask.expand(batch_size, 1, num_latents, height, width)

        use_masked_denoising = self._as_bool(training_cfg.get("masked_denoising", True))
        use_static_conditioning = use_masked_denoising and self._as_bool(
            training_cfg.get("condition_static_regions", True)
        )

        if use_masked_denoising:
            denoise_mask = target_spatial_mask.to(dtype=clean_latents.dtype) * change_binary
        else:
            denoise_mask = target_spatial_mask.to(dtype=clean_latents.dtype)
        hidden_states = torch.where(denoise_mask.to(dtype=torch.bool), noisy_latents, clean_latents)
        target_velocity = noise.float() - clean_latents.float()

        if use_static_conditioning:
            static_target_mask = target_spatial_mask.to(dtype=clean_latents.dtype) * (1.0 - change_binary)
            condition_mask = torch.maximum(condition_mask, static_target_mask)

        cond_timestep = float(getattr(self.backbone, "_conditional_frame_timestep", 0.1))
        timestep = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1))
        timestep = timestep + cond_indicator.to(dtype=clean_latents.dtype) * cond_timestep
        timestep = torch.where(target_mask.to(dtype=torch.bool), target_time.expand_as(timestep), timestep)

        padding_height = int(height * getattr(self.backbone, "vae_scale_factor_spatial", 16))
        padding_width = int(width * getattr(self.backbone, "vae_scale_factor_spatial", 16))
        padding_mask = clean_latents.new_zeros((1, 1, padding_height, padding_width), dtype=dtype)

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.backbone.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                condition_mask=condition_mask.to(dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred = pred.sample if hasattr(pred, "sample") else pred
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()

        temporal_loss_mask = target_spatial_mask.expand(batch_size, channels, num_latents, height, width)
        if use_static_conditioning:
            change_loss_weight = change_loss_weight * change_binary.float()
        spatial_loss_weight = change_loss_weight.expand(batch_size, 1, num_latents, height, width)
        spatial_loss_weight = spatial_loss_weight.expand(batch_size, channels, num_latents, height, width)
        loss_mask = temporal_loss_mask * spatial_loss_weight
        future_loss = F.mse_loss(pred.float(), target_velocity, reduction="none")
        return (future_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
