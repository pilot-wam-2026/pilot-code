# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5 world-model interface.

Predict2.5 keeps a Cosmos DiT + Wan VAE backbone, but differs from
Cosmos-Predict2 in its diffusers pipeline and Qwen2.5-VL text encoder.
This wrapper mirrors the starVLA world-model API used by CosmoPredict2.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from .latent_contract import configure_coordinates

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _DisabledCosmosSafetyChecker:
    """No-op safety checker for backbone-only training."""

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video, *args, **kwargs):
        return video

# 训练时
# future_image支路，self.backbone.transformer()
# 训练时
# action支路，self.backbone.build_inputs、self.backbone.forward()、self.action_model.forward()
# 推理时
# future_image支路，self.backbone.generate()
# 推理时
# action支路，self.backbone.build_inputs()、self.backbone.forward()、self.action_model.predict_action()
class _CosmoPredict25_Interface(nn.Module):
    """World model wrapper for Cosmos-Predict2.5 diffusers checkpoints."""

    def __init__(self, config: Optional[dict] = None, **kwargs):

        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get("base_wm", config.framework.get("qwenvl", {}).get("base_vlm", "nvidia/Cosmos-Predict2.5-2B"),)
        revision = wm_cfg.get("revision", "diffusers/base/post-trained")
        attn_implementation = wm_cfg.get("attn_implementation", None)
        enable_safety_checker = bool(wm_cfg.get("enable_safety_checker", False))
        self.config = config





        if self._looks_like_native_checkpoint(model_name):
            raise ValueError(
                "Detected NVIDIA Cosmos-Predict2.5 native checkpoint layout under "
                f"{model_name!r} (for example tokenizer.pth/base/post-trained/*.pt). "
                "The current starVLA wrapper uses the diffusers Cosmos2_5_PredictBasePipeline, "
                "which requires a diffusers-format checkpoint with model_index.json and "
                "transformer/vae/text_encoder/scheduler subfolders. Use the HF diffusers "
                "revision (default: diffusers/base/post-trained), or convert/download the "
                "checkpoint in diffusers format before using this wrapper."
            )

        try:
            from diffusers import Cosmos2_5_PredictBasePipeline
        except ImportError as exc:
            raise ImportError(
                "Cosmos-Predict2.5 requires diffusers with "
                "Cosmos2_5_PredictBasePipeline support. Install diffusers>=0.37.0 "
                "in the training environment, for example: "
                "pip install -U 'diffusers>=0.37.0'."
            ) from exc

        load_kwargs = {"torch_dtype": torch.bfloat16}
        if revision and not os.path.isdir(model_name):
            load_kwargs["revision"] = revision
        if not enable_safety_checker:
            from diffusers.pipelines.cosmos import pipeline_cosmos2_5_predict
            pipeline_cosmos2_5_predict.CosmosSafetyChecker = _DisabledCosmosSafetyChecker

        logger.info(
            f"Loading Cosmos-Predict2.5 from {model_name} "
            f"(revision={revision}, attn_implementation={attn_implementation})"
        )
        self.pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model_name, **load_kwargs)
        self.pipe.set_progress_bar_config(disable=True)


        self.transformer = self.pipe.transformer
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.video_processor = self.pipe.video_processor







        self.vae_scale_factor_spatial = self.pipe.vae_scale_factor_spatial
        self.vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
        self.latent_normalization = wm_cfg.get("latent_normalization", "canonical")
        mean, divisor, self.native_latent_convention = configure_coordinates(
            self.pipe, self.latent_normalization
        )
        self.register_buffer("latents_mean", mean, persistent=False)
        self.register_buffer("latents_std", divisor, persistent=False)





        # 冻结self.vae和self.text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self._configure_attention_backend(attn_implementation)
        transformer_config = self.transformer.config
        self._hidden_size = getattr(transformer_config, "inner_dim", None)
        if self._hidden_size is None:
            self._hidden_size = transformer_config.num_attention_heads * transformer_config.attention_head_dim

        class _FakeConfig:
            pass
        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # 启动hook机制
        self._intermediate_features = []
        self._hooks = []
        self._extract_layers = wm_cfg.get("extract_layers", [-1])
        self._height = wm_cfg.get("height", 704)
        self._width = wm_cfg.get("width", 1280)
        self._max_sequence_length = wm_cfg.get("max_sequence_length", 512)
        self._conditional_frame_timestep = wm_cfg.get("conditional_frame_timestep", 0.1)
        self._register_hooks()


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

    def _resolve_video_size(self, images) -> tuple[int, int]:
        height = self._height
        width = self._width
        if not self._is_auto_size(height) and not self._is_auto_size(width):
            return int(height), int(width)

        frame_sizes = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]
            frame_sizes.extend(self._image_size(image) for image in sample_imgs)
        if not frame_sizes:
            raise ValueError("Cannot infer auto video size from empty image batch.")

        if self._is_auto_size(height):
            height = max(frame_height for _, frame_height in frame_sizes)
        height = int(height)

        if self._is_auto_size(width):
            max_aspect = max(frame_width / max(1, frame_height) for frame_width, frame_height in frame_sizes)
            width = round(height * max_aspect)
            width = self._round_up_to_multiple(width, int(getattr(self, "vae_scale_factor_spatial", 16)))
        width = int(width)

        return height, width

    def _configure_attention_backend(self, attn_implementation: Optional[str]) -> None:
        if not attn_implementation:
            return

        backend_map = {
            "flash_attention_2": "flash",
            "flash-attention-2": "flash",
            "flash": "flash",
            "sdpa": "native",
            "native": "native",
            "native_flash": "_native_flash",
            "_native_flash": "_native_flash",
            "xformers": "xformers",
        }
        backend = backend_map.get(str(attn_implementation).lower(), str(attn_implementation))
        if not hasattr(self.transformer, "set_attention_backend"):
            logger.warning(
                "Cosmos transformer does not expose set_attention_backend(); "
                f"cannot enable attention backend {backend!r}."
            )
            return

        try:
            self.transformer.set_attention_backend(backend)
            logger.info(
                f"Enabled Cosmos transformer attention backend {backend!r} "
                f"from attn_implementation={attn_implementation!r}."
            )
        except Exception as exc:
            logger.warning(
                f"Failed to enable Cosmos transformer attention backend {backend!r} "
                f"from attn_implementation={attn_implementation!r}; using default backend. "
                f"Error: {exc}"
            )

    @staticmethod
    def _looks_like_native_checkpoint(model_name: str) -> bool:
        if not os.path.isdir(model_name):
            return False
        has_diffusers_index = os.path.exists(os.path.join(model_name, "model_index.json"))
        has_native_tokenizer = os.path.exists(os.path.join(model_name, "tokenizer.pth"))
        has_native_base = os.path.isdir(os.path.join(model_name, "base"))
        return not has_diffusers_index and (has_native_tokenizer or has_native_base)

    @property
    def model(self):
        """Compatibility shim for framework code that reads model.config.hidden_size."""
        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.transformer_blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                hook = self.transformer.transformer_blocks[actual_idx].register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _encode_text(self, instructions):
        device = next(self.text_encoder.parameters()).device
        dtype = self.transformer.dtype
        with torch.no_grad():
            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=instructions,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=self._max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds

    def _encode_images(self, images, num_frames=None):
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = self._resolve_video_size(images)

        preprocessed = []
        cond_frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = self.video_processor.preprocess_video(
                sample_imgs,
                height=height,
                width=width,
            )
            video_tensor = video_tensor.to(device=device, dtype=dtype)
            preprocessed.append(video_tensor)
            cond_frame_counts.append(video_tensor.shape[2])

        target_frames = max(cond_frame_counts) if num_frames is None else num_frames
        batch_videos = []
        for i, video_tensor in enumerate(preprocessed):
            n_frames = video_tensor.shape[2]
            if n_frames > target_frames:
                video_tensor = video_tensor[:, :, -target_frames:]
                cond_frame_counts[i] = target_frames
            elif n_frames < target_frames:
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n_frames, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))

        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()

        latents_mean = self.latents_mean.to(device=device, dtype=latents.dtype)
        latents_std = self.latents_std.to(device=device, dtype=latents.dtype)
        latents = (latents - latents_mean) / latents_std

        return latents, cond_frame_counts, False

    # 生成action时，将当前obs和prompt打包为inputs
    def build_inputs(self, images, instructions, **kwargs):

        assert len(images) == len(instructions)

        # 1. 编码当前obs和prompt
        prompt_embeds = self._encode_text(instructions)


        latents, cond_frame_counts, counts_are_latent_frames = self._encode_images(images)





        batch_size = latents.shape[0]
        device = latents.device
        dtype = self.transformer.dtype
        _, _, t_lat, h_lat, w_lat = latents.shape

        # 2. 生成mask
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        cond_indicator = latents.new_zeros(batch_size, 1, t_lat, 1, 1)
        for i, n_cond in enumerate(cond_frame_counts):
            if counts_are_latent_frames:
                n_cond_latent = n_cond
            else:
                n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0
            cond_indicator[i, :, :n_cond_latent] = 1.0



        cond_timestep = torch.ones_like(cond_indicator) * self._conditional_frame_timestep


        timestep = cond_indicator * cond_timestep


        padding_height = int(h_lat * self.vae_scale_factor_spatial)
        padding_width = int(w_lat * self.vae_scale_factor_spatial)
        padding_mask = latents.new_zeros((1, 1, padding_height, padding_width), dtype=dtype)



        return {
            "hidden_states": latents.to(dtype),
            "timestep": timestep.to(dtype),
            "encoder_hidden_states": prompt_embeds.to(device=device, dtype=dtype),
            "condition_mask": condition_mask.to(dtype),
            "padding_mask": padding_mask,
            "_is_wm_input": True,
        }

    # 生成action时，接受obs和prompt作为inputs，并将其编码为vl-tokens
    def forward(self, **kwargs):

        kwargs.pop("_is_wm_input", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("output_attentions", None)

        self._intermediate_features.clear()

        # 3. 调用self.backbone.transformer()
        # 利用hook机制，将CosmosPredict前向推理的中间变量存储在列表self._intermediate_features中，
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                condition_mask=kwargs.get("condition_mask", None),
                padding_mask=kwargs.get("padding_mask", None),
                return_dict=False,
            )




        # 若self._intermediate_features非空，则将self._intermediate_features中的表征作为vl-tokens
        extracted = []

        for feat in self._intermediate_features:

            if feat.dim() == 5:
                batch, channels, frames, height, width = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)

            extracted.append(feat)

        # 若self._intermediate_features为空，则将最后一层输出的表征作为vl-tokens
        if not extracted:
            out = dit_output[0] if isinstance(dit_output, tuple) else dit_output
            if out.dim() == 5:
                batch, channels, frames, height, width = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
            extracted.append(out)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss


        return _WMOutput(hidden_states_tuple=tuple(extracted))

    # 生成future-image时，接受当前obs和prompt以及加噪的未来obs，生成降噪后的未来obs
    def generate(self, **kwargs):
        return self.pipe(**kwargs)

    # 生成future-image时，接受当前obs和prompt，生成预测的未来obs
    # def transform(self, **kwargs):
    #     return self.transformer(**kwargs)
