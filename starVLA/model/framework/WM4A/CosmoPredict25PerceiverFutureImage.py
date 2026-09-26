# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5-Perceiver with future-image prediction.

This framework keeps the existing Cosmos hidden-state -> Perceiver action
path, and additionally runs the Cosmos2.5 generation/decoder path during
inference to return the image predicted after the action chunk horizon.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.WM4A.CosmoPredict25Perceiver import (
    CosmoPredict25_Perceiver,
    CosmoPredict25PerceiverDefaultConfig,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.PerceiverHead_original import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict25PerceiverFutureImageDefaultConfig(CosmoPredict25PerceiverDefaultConfig):
    """Cosmos-Predict2.5-Perceiver + future-image defaults."""

    name: str = "CosmoPredict25PerceiverFutureImage"

    future_image_generation: dict = field(default_factory=lambda: {
        "enabled": True,
        "num_frames": 93,
        "num_inference_steps": 36,
        "guidance_scale": 7.0,
        "output_type": "pil",
        "height": 704,
        "width": 1280,
        "max_sequence_length": 512,
        "conditional_frame_timestep": 0.1,
        "num_latent_conditional_frames": 2,
        "conditioning_mode": "auto",
        "max_samples": 1,
        "return_full_video": False,
        "future_frame_index": -1,
        "negative_prompt": None,
    })

    future_image_training: dict = field(default_factory=lambda: {
        "enabled": True,
        "loss_weight": 1.0,
        "train_time_distribution": "logitnormal",
        "shift": 5.0,
    })


@FRAMEWORK_REGISTRY.register("CosmoPredict25PerceiverFutureImage")
class CosmoPredict25_Perceiver_FutureImage(CosmoPredict25_Perceiver):
    """Cosmos-Predict2.5 hidden-state action head plus generated future images."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        print('################# CosmoPredict25_Perceiver_FutureImage-1')
        baseframework.__init__(self)
        self.config = merge_framework_config(CosmoPredict25PerceiverFutureImageDefaultConfig, config)

        # 1. 初始化CosmosPredictr2.5类作为backbone
        # WM4A/starVLA/model/modules/world_model/CosmoPredict25.py
        self.backbone = get_world_model(config=self.config)
        # print(type(self.backbone))              # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface'>

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden
        # print(wm_hidden)                        # 2048

        # 2. 初始化FlowmatchingActionHead作为action-head
        # WM4A/starVLA/model/modules/action_model/PerceiverHead.py
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        # print(type(self.action_model))          # <class 'starVLA.model.modules.action_model.PerceiverHead.FlowmatchingActionHead'>

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.state_dim = self.config.framework.action_model.get("state_dim", None)
        # print(self.future_action_window_size)   # 15
        # print(self.past_action_window_size)     # 0
        # print(self.chunk_len)                   # 16
        # print(self.state_dim)                   # 64
        print('################# CosmoPredict25_Perceiver_FutureImage-2')

    @staticmethod
    def _as_sequence(images):
        return list(images) if isinstance(images, (list, tuple)) else [images]

    def _align_state_dim(self, state: torch.Tensor) -> torch.Tensor:
        if state is None or self.state_dim is None:
            return state
        target_dim = int(self.state_dim)
        current_dim = state.shape[-1]
        if current_dim == target_dim:
            return state
        if current_dim > target_dim:
            return state[..., :target_dim]

        pad_shape = (*state.shape[:-1], target_dim - current_dim)
        padding = state.new_zeros(pad_shape)
        return torch.cat([state, padding], dim=-1)

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
        if hasattr(self, 'backbone'):
            temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        else:
            temporal_factor = int(getattr(self.wan_interface, "vae_scale_factor_temporal", 4))
        return (int(raw_frame_count) - 1) // temporal_factor + 1

    def _build_future_video_sequences(self, batch_images: List, future_images: List) -> tuple[List[List], List[int], List[int]]:
        # print('################# CosmoPredict25_Perceiver_FutureImage._build_future_video_sequences-1')
        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        raw_sequences = []
        condition_counts = []
        sample_counts = []
        # print(temporal_factor)              # 4
        # print(len(batch_images))            # 8
        # print(len(future_images))           # 8

        for images, future_image in zip(batch_images, future_images):
            current_sequence = self._as_sequence(images)
            future_sequence = self._as_sequence(future_image)
            # print(len(current_sequence))    # 1
            # print(len(future_sequence))     # 1
            if not current_sequence or not future_sequence:
                raise ValueError("future image training needs both current image and future_image.")

            future_frame = future_sequence[0]
            # print(type(future_frame))       # <class 'PIL.Image.Image'>
            raw_sequence = current_sequence + [future_frame] * temporal_factor
            raw_sequences.append(raw_sequence)

            condition_count = self._latent_frame_count(len(current_sequence))
            sample_count = self._latent_frame_count(len(raw_sequence))
            # print(condition_count)          # 1
            # print(sample_count)             # 2
            condition_counts.append(condition_count)
            sample_counts.append(sample_count)

        # print('################# CosmoPredict25_Perceiver_FutureImage._build_future_video_sequences-2')
        return raw_sequences, condition_counts, sample_counts

    # 训练时
    # future_image支路，self.backbone.transformer()
    def _cosmos25_future_image_loss(self, examples: List[dict]) -> Optional[torch.Tensor]:
        # print('################# CosmoPredict25_Perceiver_FutureImage._cosmos25_future_image_loss-1')
        if "future_image" not in examples[0]:
            return None
        training_cfg = self._future_training_cfg()
        if not self._as_bool(training_cfg.get("enabled", True)):
            return None

        # 1. 将输入数据【当前观测】和【未来观测】预处理为可以被CosmosPredict接受的形状
        # batch_size=8时，batch_images、future_images和instructions都是长度为8的列表，分别包含当前观测图像、未来观测图像和文本prompt；
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        future_images = [to_pil_preserve(example["future_image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        # print(len(examples))                # 8
        # print(type(batch_images[0]))        # <class 'list'>
        # print(type(future_images[0]))       # <class 'list'>
        # print(type(instructions[0]))        # <class 'str'>

        # 我们将每个样本对应的当前和未来图像组装为raw_sequence=concat[obs,future_obs,future_obs,future_obs,future_obs]
        # 而raw_sequences也是由8个这样的raw_sequence组成的列表，
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        # print(train_obs_image_size)         # None
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            future_images = resize_images(future_images, target_size=train_obs_image_size)
        raw_sequences, condition_counts, sample_counts = self._build_future_video_sequences(
            batch_images=batch_images,
            future_images=future_images,
        )
        # print(len(raw_sequences))           # 8
        # print(len(raw_sequences[0]))        # 5
        # print(raw_sequences[0][0])          # <PIL.Image.Image image mode=RGB size=224x224 at 0x7FF6B20D60E0>
        # print(condition_counts)             # [1, 1, 1, 1, 1, 1, 1, 1]
        # print(sample_counts)                # [2, 2, 2, 2, 2, 2, 2, 2]

        # 2. 调用CosmosPredict的文本和视觉tokenizer，
        # 将每个样本的prompt编码为prompt_emb(512,100352)，将raw_sequence编码为clean_latents(16,2,28,28)，
        # 从mask上来看，1个image和4个future_image分别被编码为了两个(16,28,28)，分别记为obs_tokens和future_obs_tokens，二者拼起来就是clean_latents
        prompt_embeds = self.backbone._encode_text(instructions)
        clean_latents, _, _ = self.backbone._encode_images(raw_sequences)
        dtype = self.backbone.transformer.dtype
        clean_latents = clean_latents.to(dtype)
        batch_size, channels, num_latents, height, width = clean_latents.shape
        device = clean_latents.device
        # print(prompt_embeds.shape)          # torch.Size([8, 512, 100352])
        # print(clean_latents.shape)          # torch.Size([8, 16, 2, 28, 28])
        # print(dtype)                        # torch.bfloat16
        condition_mask = clean_latents.new_zeros((batch_size, 1, num_latents, height, width))
        target_mask = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1), dtype=torch.float32)
        cond_indicator = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1))
        # print(condition_mask.shape)         # torch.Size([8, 1, 2, 28, 28])
        # print(target_mask.shape)            # torch.Size([8, 1, 2, 1, 1])
        # print(cond_indicator.shape)         # torch.Size([8, 1, 2, 1, 1])
        for batch_idx, (condition_count, sample_count) in enumerate(zip(condition_counts, sample_counts)):
            condition_count = max(1, min(int(condition_count), num_latents))
            sample_count = min(max(condition_count + 1, int(sample_count)), num_latents)
            condition_mask[batch_idx, :, :condition_count] = 1.0
            cond_indicator[batch_idx, :, :condition_count] = 1.0
            target_mask[batch_idx, :, condition_count:sample_count] = 1.0

        # 3. 在clean_latents中，对obs_tokens部分保持不变、对future_obs_tokens部分加噪
        train_time = self._sample_train_times(
            batch_size=batch_size,
            device=device,
            distribution=training_cfg.get("train_time_distribution", "logitnormal"),
        )
        # print(train_time.shape)             # torch.Size([8])
        flow_time = self._shift_time(train_time, shift=float(training_cfg.get("shift", 5.0)))
        # print(flow_time.shape)              # torch.Size([8])
        target_time = flow_time.view(batch_size, 1, 1, 1, 1).to(device=device, dtype=clean_latents.dtype)
        # print(target_time.shape)            # torch.Size([8, 1, 1, 1, 1])
        # print(clean_latents.shape)          # torch.Size([8, 16, 2, 28, 28])
        noise = torch.randn_like(clean_latents)
        # print(noise.shape)                  # torch.Size([8, 16, 2, 28, 28])
        # print(target_time.shape)            # torch.Size([8, 1, 1, 1, 1])
        noisy_latents = clean_latents * (1.0 - target_time) + noise * target_time
        # print(noisy_latents.shape)          # torch.Size([8, 16, 2, 28, 28])
        # print(target_mask.shape)            # torch.Size([8, 1, 2, 1, 1])
        hidden_states = torch.where(target_mask.to(dtype=torch.bool), noisy_latents, clean_latents)
        # print(hidden_states.shape)          # torch.Size([8, 16, 2, 28, 28])

        # 4. 对future_obs_tokens部分，计算其去噪目标target_velocity
        target_velocity = noise.float() - clean_latents.float()
        # print(target_velocity.shape)        # torch.Size([8, 16, 2, 28, 28])

        # 5. 计算timestep和padding_mask
        cond_timestep = float(getattr(self.backbone, "_conditional_frame_timestep", 0.1))
        # print(cond_timestep)                # 0.1
        timestep = clean_latents.new_zeros((batch_size, 1, num_latents, 1, 1))
        # print(timestep.shape)               # torch.Size([8, 1, 2, 1, 1])
        timestep = timestep + cond_indicator.to(dtype=clean_latents.dtype) * cond_timestep
        # print(timestep.shape)               # torch.Size([8, 1, 2, 1, 1])
        timestep = torch.where(target_mask.to(dtype=torch.bool), target_time.expand_as(timestep), timestep)
        # print(timestep.shape)               # torch.Size([8, 1, 2, 1, 1])

        padding_height = int(height * getattr(self.backbone, "vae_scale_factor_spatial", 16))
        padding_width = int(width * getattr(self.backbone, "vae_scale_factor_spatial", 16))
        padding_mask = clean_latents.new_zeros((1, 1, padding_height, padding_width), dtype=dtype)
        # print(padding_height)               # 224
        # print(padding_width)                # 224
        # print(padding_mask.shape)           # torch.Size([1, 1, 224, 224])

        # 7. CosmosPredict前向传播
        # 以prompt_emb(8,512,100352)和clean_latents(8,16,2,28,28)作为输入，
        # 其中clean_latents包含【当前观测的tokens1】和【未来观测加噪后的tokens2】这两部分，CosmosPredict采用了一个经典的DiT架构，其中，
        # self-attn使得tokens2能够看到tokens1中真实的当前观测中是视觉信息、cross-attn使得tokens2能够理解prompt_emb中的人类指令信息，
        # 这两部分信息一起作为tokens2部分的降噪条件，得益于loss_mask，最终参与计算future_image_loss的只有tokens2部分，
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
        # print(type(self.backbone))          # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface'>
        # print(pred.shape)                   # torch.Size([8, 16, 2, 28, 28])

        # 8. 计算future-image-loss
        # print(target_mask.shape)            # torch.Size([8, 1, 2, 1, 1])
        loss_mask = target_mask.expand(batch_size, channels, num_latents, height, width)
        # print(loss_mask.shape)              # torch.Size([8, 16, 2, 28, 28])
        future_loss = F.mse_loss(pred.float(), target_velocity, reduction="none")
        # print(future_loss.shape)            # torch.Size([8, 16, 2, 28, 28])
        future_image_loss = (future_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        # print(future_image_loss)            # tensor(0.3386, device='cuda:0', grad_fn=<DivBackward0>)
        # print('################# CosmoPredict25_Perceiver_FutureImage._cosmos25_future_image_loss-2')
        return future_image_loss

    # 训练时
    # 生成future_image和action
    # 最终返回的是total_loss=action_loss+future_image_loss*loss_weight
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        # print('################# CosmoPredict25_Perceiver_FutureImage.forward-1')
        # 1. 调用CosmosPredict+ActionHead生成action_loss，这个函数定义在父类中
        output = super().forward(examples=examples, **kwargs)
        action_loss = output["action_loss"]

        # 2. 调用CosmosPredict生成future_image_loss
        future_image_loss = self._cosmos25_future_image_loss(examples)
        if future_image_loss is None:
            return output

        # 3. 最终返回的是total_loss=action_loss+future_image_loss*loss_weight
        loss_weight = float(self._future_training_cfg().get("loss_weight", 1.0))
        # print(loss_weight)      # 1.0
        output["perceiver_action_loss"] = action_loss.detach()
        output["future_image_loss"] = future_image_loss.detach()
        output["action_loss"] = action_loss + future_image_loss * loss_weight
        # print('################# CosmoPredict25_Perceiver_FutureImage.forward-2')
        return output

    def _get_future_image_generation_cfg(self, kwargs: dict) -> dict:
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
            "conditional_frame_timestep",
            "num_latent_conditional_frames",
            "conditioning_mode",
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
        mode = str(cfg.pop("conditioning_mode", "auto")).lower()
        _wm = self.backbone if hasattr(self, "backbone") else self.wan_interface
        height = cfg.pop("height", getattr(_wm, "_height", 704))
        width = cfg.pop("width", getattr(_wm, "_width", 1280))
        heights = self._resolve_generation_size_values(batch_images, height, width)
        generation_inputs = {
            "prompt": instructions,
            "height": heights[0],
            "width": heights[1],
            "num_frames": cfg.pop("num_frames", 93),
            "num_inference_steps": cfg.pop("num_inference_steps", 36),
            "guidance_scale": cfg.pop("guidance_scale", 7.0),
            "output_type": cfg.pop("output_type", "pil"),
            "return_dict": True,
            "max_sequence_length": cfg.pop("max_sequence_length", 512),
            "conditional_frame_timestep": cfg.pop("conditional_frame_timestep", 0.1),
            "num_latent_conditional_frames": cfg.pop("num_latent_conditional_frames", 2),
        }

        negative_prompt = cfg.pop("negative_prompt", None)
        if negative_prompt is not None:
            generation_inputs["negative_prompt"] = negative_prompt

        if mode == "image":
            generation_inputs["image"] = [self._as_sequence(images)[-1] for images in batch_images]
        elif mode == "video":
            generation_inputs["video"] = [self._as_sequence(images) for images in batch_images]
        elif mode == "auto":
            if any(len(self._as_sequence(images)) > 1 for images in batch_images):
                generation_inputs["video"] = [self._as_sequence(images) for images in batch_images]
            else:
                generation_inputs["image"] = [self._as_sequence(images)[0] for images in batch_images]
        else:
            raise ValueError(
                "future_image_generation.conditioning_mode must be one of "
                f"'auto', 'image', or 'video', got {mode!r}."
            )

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
        _wm = self.backbone if hasattr(self, "backbone") else self.wan_interface
        spatial_multiple = int(getattr(_wm, "vae_scale_factor_spatial", 16))
        for images in batch_images:
            image_sequence = self._as_sequence(images)
            frame_sizes = [self._image_size(image) for image in image_sequence]
            if not frame_sizes:
                raise ValueError("Cannot infer auto generation size from empty image sequence.")

            sample_height = max(frame_height for _, frame_height in frame_sizes) if self._is_auto_size(height) else int(height)
            if self._is_auto_size(width):
                max_aspect = max(frame_width / max(1, frame_height) for frame_width, frame_height in frame_sizes)
                sample_width = self._round_up_to_multiple(round(sample_height * max_aspect), spatial_multiple)
            else:
                sample_width = int(width)
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
        videos = self._extract_videos(generation_output)
        if return_full_video:
            return videos

        if isinstance(videos, (list, tuple)) and videos and isinstance(videos[0], Image.Image):
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
        if key in {"image", "video"} and isinstance(value, (list, tuple)) and len(value) == batch_size:
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

    # 推理时
    # future_image支路，self.backbone.generate()
    def _generate_future_images(self, generation_inputs: dict, batch_size: int, max_samples: int, future_frame_index: int, return_full_video: bool):
        # print('################# CosmoPredict25_Perceiver_FutureImage._generate_future_images-1')
        if max_samples is None or max_samples < 0:
            sample_count = batch_size
        else:
            sample_count = min(batch_size, max(0, int(max_samples)))
        # print(sample_count)         # 1

        # print(generation_inputs)
        """
        {
            'prompt': [
                'Subtask: unlocked_waist: pick the bell pepper from the plate and place it in the cardboard box.', 
                'Subtask: unlocked_waist: pick the tomato from the tray and place it in the cardboard box.', 
                'Subtask: unlocked_waist: pick up the milk, place it into the microwave and close the microwave.', 
                'Subtask: unlocked_waist: pick the bell pepper from the cutting board and place it in the pot.', 
                'Subtask: unlocked_waist: pick up the potato, place it into the microwave and close the microwave.', 
                'Subtask: unlocked_waist: pick the tomato from the placemat and place it in the tiered shelf.', 
                'Subtask: unlocked_waist: pick the can from the cutting board and place it in the basket.', 
                'Subtask: unlocked_waist: pick up the wine, place it into the cabinet and close the cabinet.'
            ], 
            'height': 224, 
            'width': 224, 
            'num_frames': 93, 
            'num_inference_steps': 20, 
            'guidance_scale': 7.0, 
            'output_type': 'pil', 
            'return_dict': True, 
            'max_sequence_length': 512, 
            'conditional_frame_timestep': 0.1, 
            'num_latent_conditional_frames': 2, 
            'image': [
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487370>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D74872B0>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487280>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D74872E0>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D74877F0>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487850>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487F70>, 
                <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487580>
            ]
        }
        """
        pred_future_images = []
        for sample_idx in range(sample_count):

            # 1. CosmosPredict：构建输入
            # print(sample_idx)       # 0
            single_inputs = {
                key: self._slice_generation_value(key, value, sample_idx, batch_size)
                for key, value in generation_inputs.items()
            }
            # print(single_inputs)
            """
            {
                'prompt': ['Subtask: unlocked_waist: pick the bell pepper from the plate and place it in the cardboard box.'], 
                'height': 224, 
                'width': 224, 
                'num_frames': 93, 
                'num_inference_steps': 20, 
                'guidance_scale': 7.0, 
                'output_type': 'pil', 
                'return_dict': True, 
                'max_sequence_length': 512, 
                'conditional_frame_timestep': 0.1, 
                'num_latent_conditional_frames': 2, 
                'image': <PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7487370>
            }
            """

            # 2. CosmosPredict：前向推理
            generation_output = self.backbone.generate(**single_inputs) if hasattr(self, "backbone") else self.wan_interface.generate(**single_inputs)
            # print(type(self.backbone))      # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface'>
            # print(type(generation_output))  # <class 'diffusers.pipelines.cosmos.pipeline_output.CosmosPipelineOutput'>
            # print(future_frame_index)       # -1
            # print(return_full_video)        # False
            sample_images = self._select_future_images(
                generation_output,
                future_frame_index=future_frame_index,
                return_full_video=return_full_video,
            )
            # print(sample_images)            # [<PIL.Image.Image image mode=RGB size=224x224 at 0x7F96D7423220>]
            if isinstance(sample_images, (list, tuple)):
                pred_future_images.extend(sample_images)
            else:
                pred_future_images.append(sample_images)
        # print('################# CosmoPredict25_Perceiver_FutureImage._generate_future_images-2')
        return pred_future_images

    # 推理时
    # action支路，self.backbone.build_inputs()、self.backbone.forward()、self.action_model.predict_action()
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], is_run_parallel_eval_submission: bool = False, **kwargs) -> dict:
        # print('################# CosmoPredict25_Perceiver_FutureImage.predict_action-1')
        if type(examples) is not list:
            examples = [examples]

        # 1. CosmosPredict：构建输入
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        # print(len(examples))            # 8

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        # print(train_obs_image_size)     # None
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        # print(type(self.backbone))      # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface'>
        # print(wm_inputs.keys())         # dict_keys(['hidden_states', 'timestep', 'encoder_hidden_states', 'condition_mask', 'padding_mask', '_is_wm_input'])
        # print(wm_inputs['hidden_states'].shape)             # torch.Size([8, 16, 1, 28, 28])
        # print(wm_inputs['timestep'].shape)                  # torch.Size([8, 1, 1, 1, 1])
        # print(wm_inputs['encoder_hidden_states'].shape)     # torch.Size([8, 512, 100352])
        # print(wm_inputs['condition_mask'].shape)            # torch.Size([8, 1, 1, 28, 28])
        # print(wm_inputs['padding_mask'].shape)              # torch.Size([1, 1, 224, 224])
        # print(wm_inputs['_is_wm_input'])                    # True

        # 2. CosmosPredict：前向推理
        # 这里的CosmosPredict起到了一个编码器的作用，将输入的当前观测和prompt编码为vl-tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]
            # print(type(wm_outputs))         # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface.forward.<locals>._WMOutput'>
            # print(last_hidden.shape)        # torch.Size([8, 196, 2048])

        # 3. ActionHead：构建输入
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        # print(state.shape)                  # torch.Size([8, 1, 64])
        state = self._align_state_dim(state)
        # print(state.shape)                  # torch.Size([8, 1, 64])

        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        embodiment_tag = (
            torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64)
            if embodiment_tag is not None
            else None
        )
        # print(embodiment_tag.shape)         # torch.Size([8])
        if embodiment_tag is not None:
            embodiment_tag = embodiment_tag.view(-1)
        # print(embodiment_tag.shape)         # torch.Size([8])

        # 4. ActionHead：前向推理
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state, embodiment_tag)
            # print(type(self.action_model))  # <class 'starVLA.model.modules.action_model.PerceiverHead.FlowmatchingActionHead'>
            # print(last_hidden.shape)        # torch.Size([8, 196, 2048])
            # print(state.shape)              # torch.Size([8, 1, 64])
            # print(embodiment_tag.shape)     # torch.Size([8])
            # print(pred_actions.shape)       # torch.Size([8, 16, 32])
        output = {"normalized_actions": pred_actions.detach().cpu().numpy()}

        # 5. 若enabled为True，只生成action即可返回；若enabled为False，则需要
        generation_cfg = self._get_future_image_generation_cfg(kwargs)
        is_enabled = generation_cfg.pop("enabled", True)
        # print(is_enabled)                   # True
        if not bool(is_enabled):
            # print('is_enabled is False')
            # print('################# CosmoPredict25_Perceiver_FutureImage.predict_action-2')
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
        # When called from websocket server, convert PIL Images to numpy arrays
        # for msgpack serialization; otherwise keep PIL Image format
        if is_run_parallel_eval_submission and "pred_future_images" in output:
            output["pred_future_images"] = [
                np.array(img) if isinstance(img, Image.Image) else img
                for img in output["pred_future_images"]
            ]
        # print('################# CosmoPredict25_Perceiver_FutureImage.predict_action-2')
        return output
