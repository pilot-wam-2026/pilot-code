# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5-Perceiver with DINOv3 future-latent prediction.

This framework keeps the existing Cosmos hidden-state -> Perceiver action
path, and additionally uses DINOv3 encoder + kaiming-initialized predictor
to compute future-latent loss during training.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.PerceiverHead import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

from starVLA.model.framework.WAM_VJEPA.CosmoPredict25Perceiver import (
    CosmoPredict25_Perceiver,
    CosmoPredict25PerceiverDefaultConfig,
)



@dataclass
class CosmoPredict25PerceiverDINODefaultConfig(CosmoPredict25PerceiverDefaultConfig):
    """Cosmos-Predict2.5-Perceiver + DINOv3 defaults."""

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


@FRAMEWORK_REGISTRY.register("CosmoPredict25PerceiverDINO")
class CosmoPredict25_Perceiver_DINO(CosmoPredict25_Perceiver):

    # DINOv3 ViT-H+/16 常量
    DINO_EMBED_DIM = 1280
    DINO_PATCH_SIZE = 16
    DINO_IMG_SIZE = 224
    DINO_PREDICTOR_EMBED_DIM = 1024

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(CosmoPredict25PerceiverDINODefaultConfig, config)

        # 1. 加载预训练的CosmoPredictr2.5类作为self.backbone
        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        # 2. 初始化FlowmatchingActionHead作为self.action_model
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.state_dim = self.config.framework.action_model.get("state_dim", None)

        # 3. 加载DINOv3编码器
        dino_cfg = self.config.framework.get("dinov3")
        dino_repo_dir = dino_cfg.get("repo_dir")
        dino_pretrained = dino_cfg.get("pretrained")
        # 直接从dinov3代码构建模型，绕过torch.hub.load()。
        # 原因：torch.hub.load()内部dinov3_vith16plus会强制从远程URL下载.pth权重，
        # 而本地权重是safetensors格式且网络不可达，导致URLError。
        import sys as _sys
        import os as _os
        # 将dinov3 repo根目录加入sys.path，使 `from dinov3.hub.backbones import ...` 可用
        # 注意：dinov3仓库结构为 repo_root/dinov3/hub/backbones.py，
        # 需要将repo_root加入sys.path，Python才能找到内部的dinov3包（有__init__.py）。
        # 若加入repo_root的父目录，Python会找到没有__init__.py的repo_root目录作为命名空间包，导致dinov3.hub找不到。
        _dino_repo_root = _os.path.abspath(dino_repo_dir)
        if _dino_repo_root not in _sys.path:
            _sys.path.insert(0, _dino_repo_root)
        # mock torchmetrics，dinov3的hubconf.py顶层import链依赖它
        if "torchmetrics" not in _sys.modules:
            import types as _types
            _tm = _types.ModuleType("torchmetrics")
            _tm.Metric = type("Metric", (), {})
            _sys.modules["torchmetrics"] = _tm
            _sys.modules["torchmetrics.utilities"] = _types.ModuleType("torchmetrics.utilities")
            _sys.modules["torchmetrics.utilities.distributed"] = _types.ModuleType("torchmetrics.utilities.distributed")
            _sys.modules["torchmetrics.utilities.distributed.gather"] = type("gather", (), {})
            _sys.modules["torchmetrics.utilities.prints"] = _types.ModuleType("torchmetrics.utilities.prints")
        from dinov3.hub.backbones import dinov3_vith16plus as _dinov3_vith16plus
        # pretrained=False：只构建模型结构，不下载权重
        self.dino_encoder = _dinov3_vith16plus(pretrained=False)
        # 手动加载本地safetensors权重
        _weight_file = _os.path.join(dino_pretrained, "model.safetensors")
        if not _os.path.isfile(_weight_file):
            raise FileNotFoundError(f"DINOv3 weight file not found: {_weight_file}")
        from safetensors.torch import load_file as _load_safetensors
        _state_dict = _load_safetensors(_weight_file)
        self.dino_encoder.load_state_dict(_state_dict, strict=False)
        self.dino_encoder = self.dino_encoder.to(torch.bfloat16).cuda()
        # 冻结dino_encoder全部参数
        for p in self.dino_encoder.parameters():
            p.requires_grad = False

        # 4. 初始化DINO Predictor（同结构kaiming随机初始化）
        # 延迟导入VisionTransformerPredictorAC，避免模块级导入时因src不在sys.path而报错
        # （VJEPA2AC版本通过torch.hub.load触发导入，此处仿照其思路在__init__内导入）
        from starVLA.facebookresearch_vjepa2_main.src.models.ac_predictor import VisionTransformerPredictorAC
        # num_frames=4, tubelet_size=2 → grid_depth=2 → attn_mask足够大
        # forward_for_WAM_VJEPA2AC中实际序列长度=cond_tokens(65)+196=261，
        # attn_mask大小=grid_depth*(add_tokens+H*W)=2*(2+196)=396>261 ✓
        # 注意：若num_frames太小(如默认1)，grid_depth=0会导致attn_mask=[0,0]崩溃
        self.dino_predictor = VisionTransformerPredictorAC(
            img_size=self.DINO_IMG_SIZE,
            patch_size=self.DINO_PATCH_SIZE,
            in_chans=3,
            num_frames=4,
            tubelet_size=2,
            embed_dim=self.DINO_EMBED_DIM,
            predictor_embed_dim=self.DINO_PREDICTOR_EMBED_DIM,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            action_embed_dim=self.state_dim,
        )
        # kaiming随机初始化
        def _kaiming_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.dino_predictor.apply(_kaiming_init)
        self.dino_predictor = self.dino_predictor.to(torch.bfloat16).cuda()

        # 冻结predictor内部不再使用的encoder——它们在forward_for_WAM_VJEPA2AC中从未被调用，
        # 但作为nn.Linear子模块仍被注册为可训练参数，DeepSpeed ZeRO-2会为它们分配
        # fp32 master weight和优化器状态，由于它们永远不参与前向计算，梯度始终为None，
        # 在optimizer.step()或fp32 master同步时可能产生异常值，最终污染backbone权重
        for p in self.dino_predictor.action_encoder.parameters():
            p.requires_grad = False
        for p in self.dino_predictor.state_encoder.parameters():
            p.requires_grad = False
        for p in self.dino_predictor.extrinsics_encoder.parameters():
            p.requires_grad = False

        for name, p in self.dino_encoder.named_parameters():
            assert p.dtype == torch.bfloat16, f"dino_encoder.{name} dtype={p.dtype}, expected bfloat16"
        for name, p in self.dino_predictor.named_parameters():
            assert p.dtype == torch.bfloat16, f"dino_predictor.{name} dtype={p.dtype}, expected bfloat16"

        # 初始化DINOv3的图像预处理transform（ImageNet标准）
        self._dino_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.DINO_IMG_SIZE, self.DINO_IMG_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.state_projector = torch.nn.Linear(
            self.state_dim,
            self.DINO_PREDICTOR_EMBED_DIM,
            bias=True
        ).to(torch.bfloat16)
        self.action_projector = torch.nn.Linear(
            self.action_model.model.config.output_dim,
            self.DINO_PREDICTOR_EMBED_DIM,
            bias=True
        ).to(torch.bfloat16)

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

    def _future_image_training_cfg(self) -> dict:
        return dict(self.config.framework.get("future_image_training", {}))

    def _future_latent_training_cfg(self) -> dict:
        return dict(self.config.framework.get("future_latent_training", {}))

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
        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
        return (int(raw_frame_count) - 1) // temporal_factor + 1

    def _build_future_video_sequences(self, batch_images: List, future_images: List) -> tuple[List[List], List[int], List[int]]:
        temporal_factor = int(getattr(self.backbone, "vae_scale_factor_temporal", 4))
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

            condition_count = self._latent_frame_count(len(current_sequence))
            sample_count = self._latent_frame_count(len(raw_sequence))
            condition_counts.append(condition_count)
            sample_counts.append(sample_count)

        return raw_sequences, condition_counts, sample_counts

    # 训练时
    # future_image支路，self.backbone.transformer()
    def _cosmos25_future_image_loss(self, examples: List[dict]) -> Optional[torch.Tensor]:
        if "future_image" not in examples[0]:
            return None
        training_cfg = self._future_image_training_cfg()
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

        train_time = self._sample_train_times(
            batch_size=batch_size,
            device=device,
            distribution=training_cfg.get("train_time_distribution", "logitnormal"),
        )
        flow_time = self._shift_time(train_time, shift=float(training_cfg.get("shift", 5.0)))
        target_time = flow_time.view(batch_size, 1, 1, 1, 1).to(device=device, dtype=clean_latents.dtype)
        noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - target_time) + noise * target_time
        hidden_states = torch.where(target_mask.to(dtype=torch.bool), noisy_latents, clean_latents)

        target_velocity = noise.float() - clean_latents.float()

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

        loss_mask = target_mask.expand(batch_size, channels, num_latents, height, width)
        future_loss = F.mse_loss(pred.float(), target_velocity, reduction="none")
        future_image_loss = (future_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return future_image_loss

    # 训练时
    # future_latent支路，self.dino_encoder + self.dino_predictor
    def _dino_future_latent_loss(self, examples: List[dict], target_emb: torch.Tensor) -> Optional[torch.Tensor]:
        if "future_image" not in examples[0]:
            return None

        # 2. 从examples中取出当前图像和未来图像，并使用DINOv3的ImageNet标准transform做预处理
        # to_pil_preserve可能返回PIL.Image或list[PIL.Image]，需要统一取单帧
        batch_current_images = [self._as_sequence(to_pil_preserve(example["image"]))[0] for example in examples]
        batch_future_images = [self._as_sequence(to_pil_preserve(example["future_image"]))[0] for example in examples]

        # DINOv3是2D图像编码器，需要分别处理当前图像和未来图像
        current_tensors = torch.stack([
            self._dino_transform(img) for img in batch_current_images
        ]).to(self.dino_encoder.patch_embed.proj.weight.device)

        future_tensors = torch.stack([
            self._dino_transform(img) for img in batch_future_images
        ]).to(self.dino_encoder.patch_embed.proj.weight.device)

        # 3. DINOv3编码，分别获取当前图像和未来图像的patch tokens
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            current_tokens = self.dino_encoder.forward_features(current_tensors)["x_norm_patchtokens"]
            future_tokens = self.dino_encoder.forward_features(future_tensors)["x_norm_patchtokens"]
        # current_tokens: [B, 196, 1280], future_tokens: [B, 196, 1280]

        # 4. 从examples中取出current_state
        current_state = [example["state"] for example in examples] if "state" in examples[0] else None
        current_state = (torch.from_numpy(np.array(current_state)).to(current_tokens.device, dtype=current_tokens.dtype))
        current_state = self._align_state_dim(current_state)

        # 5. dino_predictor推理时使用fp32
        with torch.autocast("cuda", dtype=torch.float32):
            # 通过projector映射维度
            current_state_emb = self.state_projector(current_state)
            target_emb_proj = self.action_projector(target_emb)

            # 6. 调用dino_predictor预测future_tokens
            future_tokens_pred = self.dino_predictor.forward_for_WAM_VJEPA2AC(
                x=current_tokens,
                target_emb=target_emb_proj,
                state_emb=current_state_emb,
            )

            # 7. 计算future_latent_loss
            future_latent_loss = F.smooth_l1_loss(future_tokens_pred, future_tokens)

        return future_latent_loss

    # 训练时
    # 生成future_image、future_latent和action
    # 最终返回的是total_loss=action_loss+future_image_loss*loss_weight+future_latent_loss*loss_weight
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        output = {}
        future_image_cfg = self._future_image_training_cfg()
        future_latent_cfg = self._future_latent_training_cfg()

        # 1. action支路（必定执行）
        perceiver_action_loss, target_emb = super().forward(examples=examples, **kwargs)
        output["perceiver_action_loss"] = perceiver_action_loss.detach()
        total_loss = perceiver_action_loss

        # 2. future_image支路（根据enabled配置决定是否执行）
        if self._as_bool(future_image_cfg.get("enabled")):
            future_image_loss = self._cosmos25_future_image_loss(examples)
            if future_image_loss is not None:
                future_image_loss_weight = float(future_image_cfg.get("loss_weight"))
                output["future_image_loss"] = future_image_loss.detach()
                total_loss = total_loss + future_image_loss * future_image_loss_weight

        # 3. future_latent支路（根据enabled配置决定是否执行）
        if self._as_bool(future_latent_cfg.get("enabled")):
            future_latent_loss = self._dino_future_latent_loss(examples, target_emb)
            if future_latent_loss is not None:
                future_latent_loss_weight = float(future_latent_cfg.get("loss_weight"))
                output["future_latent_loss"] = future_latent_loss.detach()
                total_loss = total_loss + future_latent_loss * future_latent_loss_weight

        output["action_loss"] = total_loss

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
        height = cfg.pop("height", getattr(self.backbone, "_height", 704))
        width = cfg.pop("width", getattr(self.backbone, "_width", 1280))
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
        spatial_multiple = int(getattr(self.backbone, "vae_scale_factor_spatial", 16))
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

            generation_output = self.backbone.generate(**single_inputs)
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

    # 推理时
    # action支路，self.backbone.build_inputs()、self.backbone.forward()、self.action_model.predict_action()
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        # 1. CosmosPredict：构建输入
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        # 2. CosmosPredict：前向推理
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]

        # 3. ActionHead：构建输入
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        state = self._align_state_dim(state)

        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        embodiment_tag = (
            torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64)
            if embodiment_tag is not None
            else None
        )
        if embodiment_tag is not None:
            embodiment_tag = embodiment_tag.view(-1)

        # 4. ActionHead：前向推理
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state, embodiment_tag)
        output = {"normalized_actions": pred_actions.detach().cpu().numpy()}

        return output