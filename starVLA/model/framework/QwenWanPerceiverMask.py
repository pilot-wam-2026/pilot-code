# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenWanPerceiver with masked future-image denoising.

This variant keeps the existing action path and adds a Wan2 future-image
objective with a latent-grid change mask built from the current/future RGB
pair.  The mask focuses the loss on changed regions and optionally adds noise
only inside those regions, turning future-image training into an
inpainting-style objective.

Key adaptation from CosmoPredict25PerceiverMaskedFutureImage:
  - Wan2 uses expand_timesteps (per-token timestep) instead of Cosmos's
    condition_mask / padding_mask.  Condition-frame tokens receive
    timestep=0 (clean); target-frame tokens receive flow_time.
  - Wan2 VAE has z_dim=48 and temporal_downsample_factor=4.

================================================================================
(1) datasets.py 修改说明
================================================================================

为了支持本模型对 future_image 的需求，对 datasets.py 做了两处最小化修改：

A. 新增 _get_future_images 方法（LeRobotMixtureDataset 类）

   逻辑：
     - 取第一个 action key 的 delta_indices 最大值 +1 作为 future_offset
       （即当前 action chunk 最后一步的下一帧）
     - 遍历所有 video key，用 dataset.get_video(trajectory_id, video_key,
       step + future_offset) 读取未来视频帧
     - 取该视频的最后一帧，转为 PIL Image，返回每路相机的未来帧列表

B. 修改 __getitem__ 中的两处 return

   在 return 之前新增条件化获取和返回 future_image 的逻辑：
     future_images = None
     if self.data_cfg is not None and self.data_cfg.get("include_future_image", False):
         future_images = self._get_future_images(dataset, trajectory_id, step, data)

   在两处 return 的 dict 中条件化添加：
     if future_images is not None:
         result["future_image"] = future_images

   核心原则：只有当 data_cfg.include_future_image=True 时才会获取并返回
   future_image，其他模型的配置中没有该字段（默认 False），完全不受影响。

================================================================================
(2) QwenWanPerceiverMask 与 QwenWanPerceiver 的区别
================================================================================

QwenWan_Perceiver_Mask 继承自 QwenWan_Perceiver，保留了原有的 action 预测
路径不变，新增了 masked future-image denoising loss。

新增方法：
  - _build_future_video_sequences():
      构建视频序列 [current_frame] + [future_frame]*4（5帧 → Wan2 VAE 编码
      → 2个 latent frame：1个条件 + 1个目标）

  - _build_latent_change_masks():
      RGB 差异图下采样到 latent 分辨率，top-k 选择变化最大的 25% 区域，
      生成 binary mask + soft loss weight

  - _wan_future_image_loss():
      核心方法，计算 masked future-image denoising loss，流程：
        1. 从 example 中取 future_image，若不存在则返回 None（不训练）
        2. 构建 5 帧视频序列，经 Wan2 VAE 编码得到 clean_latents [B,48,2,H,W]
        3. 构建变化 mask：当前帧 vs 未来帧的 RGB 差异 → 下采样到 latent
           分辨率 → top-k 选变化区域
        4. 采样 flow matching 时间步 t，加噪：x_t = (1-t)*x_0 + t*noise
        5. Masked denoising：只对目标帧中变化区域加噪，静态区域保持干净
        6. Per-token timestep（关键适配）：Wan2 不支持 Cosmos 的
           condition_mask，改用 expand_timesteps——条件帧 token timestep=0
           （干净），目标帧 token timestep=flow_time
        7. Static region conditioning：未变化的目标区域也标记为条件帧
           （timestep=0）
        8. 计算 MSE loss，用 change mask 加权

  - _future_training_cfg():
      读取 config.framework.future_image_training 配置

  - _sample_train_times():
      采样 flow matching 的时间步（支持 logitnormal / uniform 分布）

  - _shift_time():
      对时间步施加 shift 偏移（默认 shift=5.0）

  - 辅助方法：_as_sequence(), _pil_rgb_to_tensor(), _as_bool()

重写的 forward 方法：

  原始 forward 只返回 {"action_loss": action_loss}，新版本：
    output = super().forward(examples=examples, **kwargs)  # 先走原 action 路径
    action_loss = output["action_loss"]
    future_image_loss = self._wan_future_image_loss(examples)
    # 总 loss = action_loss + future_image_loss * loss_weight
    output["action_loss"] = action_loss + future_image_loss * loss_weight
================================================================================
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.QwenWanPerceiver import QwenWan_Perceiver
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenWanPerceiverMask")
class QwenWan_Perceiver_Mask(QwenWan_Perceiver):
    """QwenWanPerceiver with masked future-image denoising."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        qwenvl_cfg = self.config.framework.get("qwenvl", {}) if self.config is not None else {}
        is_frozen = bool(qwenvl_cfg.get("is_frozen", False))

        for p in self.wan_interface.text_encoder.parameters():
            p.requires_grad = False
        for p in self.wan_interface.vae.parameters():
            p.requires_grad = False

        if is_frozen:
            for p in self.qwen_vl_interface.parameters():
                p.requires_grad = False

    @staticmethod
    def _as_sequence(images):
        return list(images) if isinstance(images, (list, tuple)) else [images]

    @staticmethod
    def _pil_rgb_to_tensor(image, size=None) -> torch.Tensor:
        from PIL import Image as _Image
        image = to_pil_preserve(image).convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _future_training_cfg(self) -> dict:
        return dict(self.config.framework.get("future_image_training", {}))

    def _sample_train_times(self, batch_size: int, device: torch.device, distribution: str) -> torch.Tensor:
        distribution = str(distribution).lower()
        if distribution == "logitnormal":
            return torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
        if distribution == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        raise ValueError(f"Unsupported future_image_training.train_time_distribution={distribution!r}")

    @staticmethod
    def _shift_time(time: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * time / (1.0 + (shift - 1.0) * time)

    def _latent_frame_count(self, raw_frame_count: int) -> int:
        temporal_factor = int(getattr(self.wan_interface, "vae_scale_factor_temporal", 4))
        return (int(raw_frame_count) - 1) // temporal_factor + 1

    def _maybe_apply_single_frame_condition(self, examples: List[dict], training_cfg: dict) -> tuple[List[dict], float]:
        prob = float(training_cfg.get("single_frame_condition_prob", 0.0))
        if prob <= 0.0 or not examples:
            return examples, 0.0

        prob = min(max(prob, 0.0), 1.0)
        sampled_examples = []
        single_frame_count = 0
        for example in examples:
            image_sequence = self._as_sequence(example.get("image"))
            if len(image_sequence) > 1 and torch.rand((), device="cpu").item() < prob:
                example = dict(example)
                example["image"] = [image_sequence[-1]]
                single_frame_count += 1
            sampled_examples.append(example)

        return sampled_examples, single_frame_count / max(1, len(examples))

    def _build_future_video_sequences(self, batch_images: List, future_images: List):
        """Build video sequences for Wan2 VAE encoding.

        Keep all observed frames as condition frames, then append the future frame
        repeated by temporal_factor. This mirrors
        CosmoPredict25PerceiverFutureImage._build_future_video_sequences, with
        only the video model swapped from Cosmos to Wan.

        Returns:
            raw_sequences: List of List[PIL Image]
            condition_counts: List[int] — latent condition frame count
            sample_counts: List[int] — latent total frame count
        """
        temporal_factor = int(getattr(self.wan_interface, "vae_scale_factor_temporal", 4))
        raw_sequences = []
        condition_counts = []
        sample_counts = []

        for images, future_image in zip(batch_images, future_images):
            current_sequence = self._as_sequence(images)
            future_sequence = self._as_sequence(future_image)
            if not current_sequence or not future_sequence:
                raise ValueError("future image training needs both current image and future_image.")

            future_frame = future_sequence[0]
            raw_sequence = current_sequence + [future_frame] * temporal_factor
            raw_sequences.append(raw_sequence)
            condition_counts.append(self._latent_frame_count(len(current_sequence)))
            sample_counts.append(self._latent_frame_count(len(raw_sequence)))

        return raw_sequences, condition_counts, sample_counts

    def _build_latent_change_masks(
        self,
        batch_images: List,
        future_images: List,
        latent_height: int,
        latent_width: int,
        training_cfg: dict,
        device: torch.device,
    ) -> tuple:
        """Return binary denoise mask and soft loss weights at latent resolution.

        Adapted from CosmoPredict25PerceiverMaskedFutureImage._build_latent_change_masks.
        """
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

    def _wan_future_image_loss(self, examples: List[dict]) -> Optional[torch.Tensor]:
        """Compute masked future-image denoising loss using Wan2 DiT.

        Adaptation from CosmoPredict25PerceiverMaskedFutureImage for Wan2:
          - Uses Wan2 _encode_images_vae instead of Cosmos _encode_images
          - Uses per-token expand_timesteps instead of condition_mask / padding_mask
          - Condition-frame tokens get timestep=0 (clean), target-frame tokens
            get timestep=flow_time
        """
        if "future_image" not in examples[0]:
            return None

        training_cfg = self._future_training_cfg()
        if not self._as_bool(training_cfg.get("enabled", True)):
            return None

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        future_images = [to_pil_preserve(example["future_image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if not train_obs_image_size:
            train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            future_images = resize_images(future_images, target_size=train_obs_image_size)

        # Build video sequences: [current_frame, future_frame * temporal_factor]
        raw_sequences, condition_counts, sample_counts = self._build_future_video_sequences(
            batch_images=batch_images,
            future_images=future_images,
        )

        # Encode through Wan2 VAE — raw_sequences already contains head-only frames
        # (batch_images/future_images were already filtered via _wm_images_head_only earlier,
        #  so we should NOT apply _wm_images_head_only again on the video sequences)
        with torch.no_grad():
            clean_latents = self.wan_interface._encode_images_vae(raw_sequences)

        dtype = self.wan_interface.transformer.dtype
        clean_latents = clean_latents.to(dtype)
        batch_size, channels, num_latents, height, width = clean_latents.shape
        device = clean_latents.device

        # Build per-latent-frame masks
        # condition_mask: which latent frames are conditions (timestep=0)
        # target_mask: which latent frames are targets (to be denoised)
        condition_mask = clean_latents.new_zeros((batch_size, 1, num_latents, height, width))
        target_mask = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1), dtype=torch.float32)
        for batch_idx, (condition_count, sample_count) in enumerate(zip(condition_counts, sample_counts)):
            condition_count = max(1, min(int(condition_count), num_latents))
            sample_count = min(max(condition_count + 1, int(sample_count)), num_latents)
            condition_mask[batch_idx, :, :condition_count] = 1.0
            target_mask[batch_idx, :, condition_count:sample_count] = 1.0

        # Build change masks at latent spatial resolution
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

        # Sample training time
        train_time = self._sample_train_times(
            batch_size=batch_size,
            device=device,
            distribution=training_cfg.get("train_time_distribution", "logitnormal"),
        )
        flow_time = self._shift_time(train_time, shift=float(training_cfg.get("shift", 5.0)))

        # Flow-matching interpolation: x_t = (1-t)*x_0 + t*noise
        noise = torch.randn_like(clean_latents)
        target_time = flow_time.view(batch_size, 1, 1, 1, 1).to(device=device, dtype=clean_latents.dtype)
        noisy_latents = clean_latents * (1.0 - target_time) + noise * target_time

        # Masked denoising: only noise changed regions in target frames
        use_masked_denoising = self._as_bool(training_cfg.get("masked_denoising", True))
        use_static_conditioning = use_masked_denoising and self._as_bool(
            training_cfg.get("condition_static_regions", True)
        )

        if use_masked_denoising:
            denoise_mask = target_mask.to(dtype=clean_latents.dtype) * change_binary
        else:
            denoise_mask = target_mask.to(dtype=clean_latents.dtype)

        # hidden_states: condition frames stay clean, target frames are noised
        # (within target frames, only changed regions are noised if masked_denoising)
        hidden_states = torch.where(denoise_mask.to(dtype=torch.bool), noisy_latents, clean_latents)
        target_velocity = noise.float() - clean_latents.float()

        # Static region conditioning: treat unchanged target regions as conditions
        if use_static_conditioning:
            static_target_mask = target_mask.to(dtype=clean_latents.dtype) * (1.0 - change_binary)
            condition_mask = torch.maximum(condition_mask, static_target_mask)

        # Build per-token timestep for Wan2 expand_timesteps mode
        # Condition tokens → timestep=0, target tokens → timestep=flow_time
        p_t, p_h, p_w = self.wan_interface.transformer.config.patch_size
        tokens_per_frame = (height // p_h) * (width // p_w)

        # flow_time is [B], expand to per-token: [B, num_latents * tokens_per_frame]
        cond_timestep_val = 0.0  # clean condition
        target_timestep_val = flow_time.to(dtype=torch.float32)  # [B]

        timestep_per_token = torch.zeros(
            batch_size, num_latents * tokens_per_frame,
            device=device, dtype=torch.float32,
        )
        for batch_idx in range(batch_size):
            for latent_idx in range(num_latents):
                start = latent_idx * tokens_per_frame
                end = (latent_idx + 1) * tokens_per_frame
                if target_mask[batch_idx, 0, latent_idx, 0, 0] > 0.5:
                    timestep_per_token[batch_idx, start:end] = target_timestep_val[batch_idx]
                else:
                    timestep_per_token[batch_idx, start:end] = cond_timestep_val

        # Encode text
        with torch.no_grad():
            prompt_embeds = self.wan_interface._encode_text(instructions)

        # DiT forward pass
        self.wan_interface._intermediate_features.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.wan_interface.transformer(
                hidden_states=hidden_states,
                timestep=timestep_per_token,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
            )
        pred = pred.sample if hasattr(pred, "sample") else pred
        if isinstance(pred, tuple):
            pred = pred[0]
        self.wan_interface._intermediate_features.clear()

        # Compute masked loss
        temporal_loss_mask = target_mask.expand(batch_size, channels, num_latents, height, width)
        if use_static_conditioning:
            change_loss_weight = change_loss_weight * change_binary.float()
        spatial_loss_weight = change_loss_weight.expand(batch_size, 1, num_latents, height, width)
        spatial_loss_weight = spatial_loss_weight.expand(batch_size, channels, num_latents, height, width)
        loss_mask = temporal_loss_mask * spatial_loss_weight
        future_loss = F.mse_loss(pred.float(), target_velocity, reduction="none")
        
        return (future_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def forward(self, examples: List[dict] = None, **kwargs):
        """Forward pass: action loss + masked future-image loss."""
        output = super().forward(examples=examples, **kwargs)
        action_loss = output["action_loss"]

        future_image_loss = self._wan_future_image_loss(examples)
        if future_image_loss is None:
            return output

        loss_weight = float(self._future_training_cfg().get("loss_weight", 1.0))
        output["perceiver_action_loss"] = action_loss.detach()
        output["future_image_loss"] = future_image_loss.detach()
        output["action_loss"] = action_loss + future_image_loss * loss_weight
        return output

    # ------------------------------------------------------------------
    # Future-image generation (eval / inference)
    # ------------------------------------------------------------------

    def _get_future_image_generation_cfg(self, kwargs: dict) -> dict:
        """Merge config defaults with runtime kwargs for future-image generation."""
        cfg = dict(self.config.framework.get("future_image_generation", {}))
        override = kwargs.get("future_image_generation", None)
        if override:
            cfg.update(dict(override))

        if "decode_future_images" in kwargs:
            cfg["enabled"] = kwargs["decode_future_images"]
        if "num_ddim_steps" in kwargs and "num_inference_steps" not in kwargs:
            cfg["num_inference_steps"] = kwargs["num_ddim_steps"]

        for key in (
            "enabled",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "output_type",
            "height",
            "width",
            "max_sequence_length",
            "max_samples",
            "return_full_video",
            "future_frame_index",
            "negative_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
        ):
            if key in kwargs:
                cfg[key] = kwargs[key]
        return cfg

    def _build_generation_inputs(self, batch_images: List, instructions: List[str], cfg: dict) -> dict:
        """Build generation inputs for Wan2 TI2V pipeline.
        """
        height = cfg.pop("height", getattr(self.wan_interface, "height", 704))
        width = cfg.pop("width", getattr(self.wan_interface, "width", 1280))
        heights = self._resolve_generation_size_values(batch_images, height, width)

        # Wan2 TI2V: pass conditioning image directly; generate() handles
        # VAE encoding + expand_timesteps internally
        head_only_images = self._wm_images_head_only(batch_images)
        condition_images = [self._as_sequence(imgs)[-1] for imgs in head_only_images]

        generation_inputs = {
            "prompt": instructions,
            "image": condition_images,
            "height": heights[0],
            "width": heights[1],
            "num_frames": cfg.pop("num_frames", 93),
            "num_inference_steps": cfg.pop("num_inference_steps", 36),
            "guidance_scale": cfg.pop("guidance_scale", 7.0),
            "output_type": cfg.pop("output_type", "pil"),
            "return_dict": True,
            "max_sequence_length": cfg.pop("max_sequence_length", 512),
        }

        negative_prompt = cfg.pop("negative_prompt", None)
        if negative_prompt is not None:
            generation_inputs["negative_prompt"] = negative_prompt

        passthrough_keys = (
            "generator",
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "num_videos_per_prompt",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        )
        for key in passthrough_keys:
            if key in cfg and cfg[key] is not None:
                generation_inputs[key] = cfg.pop(key)

        generation_inputs.update({key: value for key, value in cfg.items() if value is not None})
        return generation_inputs

    @staticmethod
    def _is_auto_size(value) -> bool:
        return isinstance(value, str) and value.lower() == "auto"

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return int(((max(1, value) + multiple - 1) // multiple) * multiple)

    @staticmethod
    def _image_size(image) -> tuple[int, int]:
        size = getattr(image, "size", None)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return int(size[0]), int(size[1])
        if torch.is_tensor(image):
            shape = tuple(image.shape)
            if len(shape) >= 3 and shape[0] in (1, 3, 4):
                return int(shape[-1]), int(shape[-2])
            if len(shape) >= 2:
                return int(shape[1]), int(shape[0])
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"Cannot infer image size from shape {array.shape}")
        return int(array.shape[1]), int(array.shape[0])

    def _resolve_generation_size_values(self, batch_images: List, height, width) -> tuple:
        if not self._is_auto_size(height) and not self._is_auto_size(width):
            return int(height), int(width)

        resolved_heights = []
        resolved_widths = []
        spatial_multiple = int(getattr(self.wan_interface, "vae_scale_factor_spatial", 16))
        patch_size = getattr(getattr(self.wan_interface.transformer, "config", None), "patch_size", None)
        if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 3:
            spatial_multiple *= max(int(patch_size[-2]), int(patch_size[-1]), 1)
        elif isinstance(patch_size, int):
            spatial_multiple *= max(int(patch_size), 1)
        for images in batch_images:
            image_sequence = self._as_sequence(images)
            frame_sizes = [self._image_size(image) for image in image_sequence]
            if not frame_sizes:
                raise ValueError("Cannot infer auto generation size from empty image sequence.")

            sample_height = max(frame_height for _, frame_height in frame_sizes) if self._is_auto_size(height) else int(height)
            sample_height = self._round_up_to_multiple(sample_height, spatial_multiple)
            if self._is_auto_size(width):
                max_aspect = max(frame_width / max(1, frame_height) for frame_width, frame_height in frame_sizes)
                sample_width = self._round_up_to_multiple(round(sample_height * max_aspect), spatial_multiple)
            else:
                sample_width = self._round_up_to_multiple(int(width), spatial_multiple)
            resolved_heights.append(int(sample_height))
            resolved_widths.append(int(sample_width))

        height_value = resolved_heights if len(set(resolved_heights)) > 1 else resolved_heights[0]
        width_value = resolved_widths if len(set(resolved_widths)) > 1 else resolved_widths[0]
        return height_value, width_value

    @staticmethod
    def _extract_videos(generation_output):
        if hasattr(generation_output, "frames"):
            return generation_output.frames
        if hasattr(generation_output, "videos"):
            return generation_output.videos
        if hasattr(generation_output, "images"):
            return generation_output.images
        if isinstance(generation_output, tuple):
            return generation_output[0]
        return generation_output

    def _select_future_images(self, generation_output, future_frame_index: int, return_full_video: bool):
        from PIL import Image as _Image
        videos = self._extract_videos(generation_output)
        if return_full_video:
            return videos

        if isinstance(videos, (list, tuple)) and videos and isinstance(videos[0], _Image.Image):
            return [videos[future_frame_index]]

        future_images = []
        for sample in videos:
            if torch.is_tensor(sample):
                if sample.ndim >= 4:
                    future_images.append(sample[future_frame_index])
                else:
                    future_images.append(sample)
            elif isinstance(sample, np.ndarray):
                if sample.ndim >= 4:
                    future_images.append(sample[future_frame_index])
                else:
                    future_images.append(sample)
            elif isinstance(sample, (list, tuple)):
                future_images.append(sample[future_frame_index])
            else:
                future_images.append(sample)
        return future_images

    @staticmethod
    def _slice_generation_value(key: str, value, index: int, batch_size: int):
        if value is None:
            return None
        if key in {"height", "width"} and isinstance(value, (list, tuple)) and len(value) == batch_size:
            return value[index]
        if key in {"image", "video", "condition_image"} and isinstance(value, (list, tuple)) and len(value) == batch_size:
            return value[index]
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            return value[index:index + 1]
        if isinstance(value, np.ndarray) and value.shape[0] == batch_size:
            return value[index:index + 1]
        if isinstance(value, list) and len(value) == batch_size:
            return [value[index]]
        if isinstance(value, tuple) and len(value) == batch_size:
            return (value[index],)
        return value

    def _generate_future_images(
        self,
        generation_inputs: dict,
        batch_size: int,
        max_samples: int,
        future_frame_index: int,
        return_full_video: bool,
    ):
        if max_samples is None or max_samples < 0:
            sample_count = batch_size
        else:
            sample_count = min(batch_size, max(0, int(max_samples)))

        pred_future_images = []
        for sample_idx in range(sample_count):
            single_inputs = {
                key: self._slice_generation_value(key, value, sample_idx, batch_size)
                for key, value in generation_inputs.items()
            }
            generation_output = self.wan_interface.generate(**single_inputs)
            sample_images = self._select_future_images(
                generation_output,
                future_frame_index=future_frame_index,
                return_full_video=return_full_video,
            )
            if isinstance(sample_images, (list, tuple)):
                pred_future_images.extend(sample_images)
            else:
                pred_future_images.append(sample_images)
        return pred_future_images

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], is_run_parallel_eval_submission: bool = False, **kwargs) -> dict:
        """Predict actions and optionally generate future images."""
        output = super().predict_action(examples=examples, **kwargs)

        generation_cfg = self._get_future_image_generation_cfg(kwargs)
        if not bool(generation_cfg.pop("enabled", True)):
            return output

        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if not train_obs_image_size:
            train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

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

        # When called from websocket server, convert PIL Images to numpy arrays
        # for msgpack serialization; otherwise keep PIL Image format
        if is_run_parallel_eval_submission and "pred_future_images" in output:
            output["pred_future_images"] = [
                np.array(img) if isinstance(img, PILImage.Image) else img
                for img in output["pred_future_images"]
            ]

        return output