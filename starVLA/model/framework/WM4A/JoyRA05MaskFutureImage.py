# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Wan2.2-TI2V Perceiver with masked future-image denoising."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
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
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class JoyRA05MaskFutureImageDefaultConfig(CosmoPredict25PerceiverFutureImageDefaultConfig):
    """Wan2.2-TI2V Perceiver + masked future-image defaults."""

    name: str = "JoyRA05MaskFutureImage"

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
        "num_frames": 81,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "output_type": "pil",
        "height": 224,
        "width": 224,
        "max_sequence_length": 512,
        "conditioning_mode": "image",
        "max_samples": 1,
        "return_full_video": False,
        "future_frame_index": -1,
        "negative_prompt": None,
    })


@FRAMEWORK_REGISTRY.register("JoyRA05MaskFutureImage")
class JoyRA05_MaskedFutureImage(CosmoPredict25_Perceiver_MaskedFutureImage):
    """Perceiver action head plus Wan2.2 per-token masked flow matching."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(JoyRA05MaskFutureImageDefaultConfig, config)

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # 保持与历史 checkpoint 键名一致：主模块名使用 wan_interface.*
        self.wan_interface = get_world_model(config=self.config)

        qwen_hidden = self.qwen_vl_interface.model.config.hidden_size
        wm_hidden = self.wan_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = qwen_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = qwen_hidden

        wm_cfg = self.config.framework.get("world_model", {}) if self.config is not None else {}
        self._pool_wm_sequence = bool(wm_cfg.get("pool_wm_sequence", False))
        self.wm_projector = nn.Linear(wm_hidden, qwen_hidden)
        self.wm_post_norm = nn.LayerNorm(qwen_hidden, eps=1e-6)
        gain = float(wm_cfg.get("wm_projector_init_gain", 0.02))
        nn.init.xavier_uniform_(self.wm_projector.weight, gain=gain)
        nn.init.zeros_(self.wm_projector.bias)

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.state_dim = self.config.framework.action_model.get("state_dim", None)

        qwenvl_cfg = self.config.framework.get("qwenvl", {}) if self.config is not None else {}
        if self._as_bool(wm_cfg.get("freeze_text_encoder", True)) and hasattr(self.wan_interface, "text_encoder"):
            for p in self.wan_interface.text_encoder.parameters():
                p.requires_grad = False
        if self._as_bool(wm_cfg.get("freeze_vae", True)) and hasattr(self.wan_interface, "vae"):
            for p in self.wan_interface.vae.parameters():
                p.requires_grad = False
        if self._as_bool(qwenvl_cfg.get("is_frozen", True)):
            for p in self.qwen_vl_interface.parameters():
                p.requires_grad = False


    def _fuse_vlm_with_wm(self, last_vlm_hidden: torch.Tensor, last_wm_hidden: torch.Tensor) -> torch.Tensor:
        if self._pool_wm_sequence:
            last_wm_hidden = last_wm_hidden.mean(dim=1, keepdim=True)
        wm_input = last_wm_hidden.to(dtype=self.wm_projector.weight.dtype)
        wm_aligned = self.wm_post_norm(self.wm_projector(wm_input)).to(dtype=last_vlm_hidden.dtype)
        return torch.cat([last_vlm_hidden, wm_aligned], dim=1)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        training_cfg = self._future_training_cfg()
        examples, single_frame_ratio = self._maybe_apply_single_frame_condition(examples, training_cfg)

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        wm_inputs = self.wan_interface.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.wan_interface(**wm_inputs, output_hidden_states=True, return_dict=True)
            last_wm_hidden = wm_outputs.hidden_states[-1]
            qwen_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_vlm_hidden = qwen_outputs.hidden_states[-1]
        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)

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
        future_image_loss = self._cosmos25_future_image_loss(examples)
        if future_image_loss is None:
            if single_frame_ratio > 0.0:
                output["single_frame_condition_ratio"] = single_frame_ratio
            return output

        loss_weight = float(self._future_training_cfg().get("loss_weight", 1.0))
        output["perceiver_action_loss"] = action_loss.detach()
        output["future_image_loss"] = future_image_loss.detach()
        if single_frame_ratio > 0.0:
            output["single_frame_condition_ratio"] = single_frame_ratio
        output["action_loss"] = action_loss + future_image_loss * loss_weight
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        wm_inputs = self.wan_interface.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.wan_interface(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_wm_hidden = wm_outputs.hidden_states[-1]

            qwen_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_vlm_hidden = qwen_outputs.hidden_states[-1]

        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)

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

        generation_cfg = self._get_future_image_generation_cfg(kwargs)
        if not bool(generation_cfg.pop("enabled", True)):
            return output

        return_full_video = bool(generation_cfg.pop("return_full_video", False))
        future_frame_index = int(generation_cfg.pop("future_frame_index", -1))
        max_samples = int(generation_cfg.pop("max_samples", 1))
        generation_inputs = self._build_generation_inputs(batch_images, instructions, generation_cfg)
        output["pred_future_images"] = self._generate_future_images(
            generation_inputs,
            batch_size=len(examples),
            max_samples=max_samples,
            future_frame_index=future_frame_index,
            return_full_video=return_full_video,
        )

        return output

    def _make_wan_timestep(
        self,
        hidden_states: torch.Tensor,
        denoise_mask: torch.Tensor,
        flow_time: torch.Tensor,
        timestep_scale: float,
    ) -> torch.Tensor:
        """Convert the latent denoise mask into Wan's per-patch timestep tokens."""
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.wan_interface.transformer.config.patch_size
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
        rope_max_seq_len = getattr(self.wan_interface.transformer.config, "rope_max_seq_len", None)
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
        prompt_embeds = self.wan_interface._encode_text(instructions)
        clean_latents = self.wan_interface._encode_images_vae(raw_sequences)
        dtype = self.wan_interface.transformer.dtype
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

        if hasattr(self.wan_interface, "_intermediate_features"):
            self.wan_interface._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.wan_interface.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                return_dict=False,
            )
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred = pred.sample if hasattr(pred, "sample") else pred
        if hasattr(self.wan_interface, "_intermediate_features"):
            self.wan_interface._intermediate_features.clear()

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

        height = cfg.pop("height", getattr(self.wan_interface, "height", 480))
        width = cfg.pop("width", getattr(self.wan_interface, "width", 832))
        resolved_height, resolved_width = self._resolve_generation_size_values(batch_images, height, width)
        generation_inputs = {
            "image": [self._as_sequence(images)[-1] for images in batch_images],
            "prompt": instructions,
            "height": resolved_height,
            "width": resolved_width,
            "num_frames": cfg.pop("num_frames", 81),
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
