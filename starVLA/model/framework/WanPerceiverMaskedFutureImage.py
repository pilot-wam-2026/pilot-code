# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Wan2.2-TI2V Perceiver with masked future-image denoising."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.WM4A.CosmoPredict25PerceiverFutureImage import (
    CosmoPredict25PerceiverFutureImageDefaultConfig,
)
from starVLA.model.framework.WM4A.CosmoPredict25PerceiverMaskedFutureImage import (
    CosmoPredict25_Perceiver_MaskedFutureImage,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.PerceiverHead_original import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanPerceiverMaskedFutureImageDefaultConfig(CosmoPredict25PerceiverFutureImageDefaultConfig):
    """Wan2.2-TI2V Perceiver + masked future-image defaults."""

    name: str = "WanPerceiverMaskedFutureImage"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
        "height": 224,
        "width": 224,
        "freeze_text_encoder": True,
        "freeze_vae": True,
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
    })

    future_image_training: dict = field(default_factory=lambda: {
        "enabled": True,
        "loss_weight": 1.0,
        "train_time_distribution": "logitnormal",
        "shift": 5.0,
        "timestep_scale": 1000.0,
        "single_frame_condition_prob": 0.0,
        "change_mask_enabled": True,
        "change_mask_top_ratio": 0.25,
        "change_mask_static_weight": 0.05,
        "change_mask_soft_loss": True,
        "masked_denoising": False,
        "condition_static_regions": False,
    })

    future_image_generation: dict = field(default_factory=lambda: {
        "enabled": True,
        "num_frames": 9,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "output_type": "pil",
        "height": 224,
        "width": "auto",
        "max_sequence_length": 512,
        "conditioning_mode": "image",
        "max_samples": 1,
        "return_full_video": False,
        "future_frame_index": -1,
        "negative_prompt": None,
    })


@FRAMEWORK_REGISTRY.register("WanPerceiverMaskedFutureImage")
class Wan_Perceiver_MaskedFutureImage(CosmoPredict25_Perceiver_MaskedFutureImage):
    """Perceiver action head plus Wan2.2 per-token masked flow matching."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(WanPerceiverMaskedFutureImageDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)
        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.state_dim = self.config.framework.action_model.get("state_dim", None)

    def _make_wan_timestep(
        self,
        hidden_states: torch.Tensor,
        denoise_mask: torch.Tensor,
        flow_time: torch.Tensor,
        timestep_scale: float,
    ) -> torch.Tensor:
        """Convert the latent denoise mask into Wan's per-patch timestep tokens."""
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.backbone.transformer.config.patch_size
        if num_frames % p_t or height % p_h or width % p_w:
            raise ValueError(
                "Wan latent shape must be divisible by transformer.patch_size; "
                f"got {(num_frames, height, width)} and {(p_t, p_h, p_w)}."
            )

        patch_mask = F.max_pool3d(
            denoise_mask.float(),
            kernel_size=(p_t, p_h, p_w),
            stride=(p_t, p_h, p_w),
        )
        patch_mask = patch_mask[:, 0].flatten(1)
        seq_len = patch_mask.shape[1]
        rope_max_seq_len = getattr(self.backbone.transformer.config, "rope_max_seq_len", None)
        if rope_max_seq_len is not None and seq_len > int(rope_max_seq_len):
            raise ValueError(
                f"Wan future-image seq_len={seq_len} exceeds rope_max_seq_len={rope_max_seq_len}. "
                "Reduce observation frames or image resolution."
            )

        scaled_time = flow_time.to(device=hidden_states.device, dtype=torch.float32) * float(timestep_scale)
        return patch_mask * scaled_time[:, None]

    def _cosmos25_future_image_loss(self, examples: List[dict]) -> Optional[torch.Tensor]:
        """Compute masked Wan flow loss (method name retained for parent forward compatibility)."""
        if "future_image" not in examples[0]:
            return None

        training_cfg = self._future_training_cfg()
        if not self._as_bool(training_cfg.get("enabled", True)):
            return None

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
        clean_latents = self.backbone._encode_images_vae(raw_sequences)
        dtype = self.backbone.transformer.dtype
        clean_latents = clean_latents.to(dtype=dtype)
        batch_size, channels, num_latents, height, width = clean_latents.shape
        device = clean_latents.device

        target_mask = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1), dtype=torch.float32)
        for batch_idx, (condition_count, sample_count) in enumerate(zip(condition_counts, sample_counts)):
            condition_count = max(1, min(int(condition_count), num_latents))
            sample_count = min(max(condition_count + 1, int(sample_count)), num_latents)
            target_mask[batch_idx, :, condition_count:sample_count] = 1.0
        target_spatial_mask = target_mask.expand(batch_size, 1, num_latents, height, width)

        use_change_mask = self._as_bool(training_cfg.get("change_mask_enabled", True))
        if use_change_mask:
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
        else:
            change_binary = clean_latents.new_ones((batch_size, 1, 1, height, width))
            change_loss_weight = clean_latents.new_ones(
                (batch_size, 1, 1, height, width), dtype=torch.float32
            )

        train_time = self._sample_train_times(
            batch_size=batch_size,
            device=device,
            distribution=training_cfg.get("train_time_distribution", "logitnormal"),
        )
        flow_time = self._shift_time(train_time, shift=float(training_cfg.get("shift", 5.0)))
        time_view = flow_time.view(batch_size, 1, 1, 1, 1).to(dtype=clean_latents.dtype)

        noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - time_view) + noise * time_view
        use_masked_denoising = self._as_bool(training_cfg.get("masked_denoising", False))
        use_static_conditioning = use_masked_denoising and self._as_bool(
            training_cfg.get("condition_static_regions", False)
        )
        denoise_mask = target_spatial_mask.to(dtype=clean_latents.dtype)
        if use_masked_denoising:
            denoise_mask = denoise_mask * change_binary
        hidden_states = torch.where(denoise_mask.bool(), noisy_latents, clean_latents)
        target_velocity = noise.float() - clean_latents.float()

        timestep_mask = denoise_mask if use_static_conditioning else target_spatial_mask
        timestep = self._make_wan_timestep(
            hidden_states=hidden_states,
            denoise_mask=timestep_mask,
            flow_time=flow_time,
            timestep_scale=float(training_cfg.get("timestep_scale", 1000.0)),
        )

        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.backbone.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                return_dict=False,
            )
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred = pred.sample if hasattr(pred, "sample") else pred
        if hasattr(self.backbone, "_intermediate_features"):
            self.backbone._intermediate_features.clear()

        if use_static_conditioning:
            change_loss_weight = change_loss_weight * change_binary.float()
        spatial_loss_weight = change_loss_weight.expand(batch_size, 1, num_latents, height, width)
        loss_mask = target_spatial_mask * spatial_loss_weight
        loss_mask = loss_mask.expand(batch_size, channels, num_latents, height, width)
        future_loss = F.mse_loss(pred.float(), target_velocity, reduction="none")
        return (future_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def _build_generation_inputs(self, batch_images: List, instructions: List[str], cfg: dict) -> dict:
        """Build arguments accepted by Diffusers WanImageToVideoPipeline."""
        mode = str(cfg.pop("conditioning_mode", "image")).lower()
        if mode not in {"auto", "image", "video"}:
            raise ValueError(
                "future_image_generation.conditioning_mode must be one of "
                f"'auto', 'image', or 'video', got {mode!r}."
            )

        height = cfg.pop("height", getattr(self.backbone, "height", 480))
        width = cfg.pop("width", getattr(self.backbone, "width", 832))
        resolved_height, resolved_width = self._resolve_generation_size_values(batch_images, height, width)
        generation_inputs = {
            "image": [self._as_sequence(images)[-1] for images in batch_images],
            "prompt": instructions,
            "height": resolved_height,
            "width": resolved_width,
            "num_frames": cfg.pop("num_frames", 9),
            "num_inference_steps": cfg.pop("num_inference_steps", 50),
            "guidance_scale": cfg.pop("guidance_scale", 5.0),
            "output_type": cfg.pop("output_type", "pil"),
            "return_dict": True,
            "max_sequence_length": cfg.pop("max_sequence_length", 512),
        }
        negative_prompt = cfg.pop("negative_prompt", None)
        if negative_prompt is not None:
            generation_inputs["negative_prompt"] = negative_prompt

        # Cosmos-only knobs may still be present in a shared YAML.
        cfg.pop("conditional_frame_timestep", None)
        cfg.pop("num_latent_conditional_frames", None)
        generation_inputs.update({key: value for key, value in cfg.items() if value is not None})
        return generation_inputs
