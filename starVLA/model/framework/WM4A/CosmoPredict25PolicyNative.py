# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Native Cosmos-Predict2.5 Policy backend.

This wrapper intentionally does not use the diffusers Cosmos pipeline.  It
loads NVIDIA's native Cosmos Policy implementation and adapts starVLA batches
to the official ``CosmosPolicyVideo2WorldModelRectifiedFlow`` data_batch
format.  The policy model, scheduler, tokenizer/VAE, latent injection, and
loss computation are therefore owned by the official code.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
import types
import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class CosmoPredict25PolicyNativeDefaultConfig:
    name: str = "CosmoPredict25PolicyNative"

    world_model: dict = field(default_factory=lambda: {
        # Native checkpoint path, not a diffusers folder.
        "base_wm": "",
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "",
        "vl_hidden_dim": 2048,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 15,
        "action_horizon": 16,
        "past_action_window_size": 0,
    })

    policy: dict = field(default_factory=lambda: {
        "native_repo_path": "",
        "config_file": "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
        "experiment": "cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task_no_s3",
        "text_embeddings_path": "",
        "text_embeddings_kind": "reason1",
        "use_online_text_encoder": True,
        "text_encoder_ckpt_path": "",
        "height": 224,
        "width": 224,
        "shift": 5.0,
        "num_inference_steps": 5,
        # starVLA currently trains demos, so only supervise action + future
        # primary image instead of blank unused future/value slots.
        "mask_policy_loss": True,
        "action_loss_multiplier": 16,
        "load_ema_to_reg": True,
        "instantiate_ema": False,
        "skip_native_package_init": True,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict25PolicyNative")
class CosmoPredict25_Policy_Native(baseframework):
    """Adapter around NVIDIA's native Cosmos-Predict2.5 policy model."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict25PolicyNativeDefaultConfig, config)
        self.policy_cfg = self.config.framework.get("policy", {})

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_dim = self.config.framework.action_model.action_dim
        self.state_dim = self.config.framework.action_model.get("state_dim", self.action_dim)

        self.height = int(self.policy_cfg.get("height", 224))
        self.width = int(self.policy_cfg.get("width", 224))
        self.shift = float(self.policy_cfg.get("shift", 5.0))
        self.num_inference_steps = int(self.policy_cfg.get("num_inference_steps", 5))

        self._add_native_repo_to_path()
        self.native_model, self.native_config = self._load_native_model()
        self._configure_native_policy_loss()
        self._freeze_native_non_policy_modules()
        self._use_online_text_encoder = bool(self.policy_cfg.get("use_online_text_encoder", True))
        self._text_embedding_cache = {} if self._use_online_text_encoder else self._load_text_embedding_cache()
        self._iteration = 0

    def _add_native_repo_to_path(self) -> None:
        repo_path = self.policy_cfg.get("native_repo_path") or os.getenv("COSMOS_PREDICT25_REPO", "")
        if repo_path:
            repo_path = os.path.abspath(os.path.expanduser(repo_path))
            if bool(self.policy_cfg.get("skip_native_package_init", True)):
                self._install_cosmos_predict2_package_shim(repo_path)
                self._install_optional_dependency_shims()
            packages_root = os.path.join(repo_path, "packages")
            package_paths = []
            if os.path.isdir(packages_root):
                package_paths = [
                    os.path.join(packages_root, name)
                    for name in sorted(os.listdir(packages_root))
                    if os.path.isdir(os.path.join(packages_root, name))
                ]
            extra_paths = [repo_path, *package_paths]
            for extra_path in reversed(extra_paths):
                if os.path.isdir(extra_path) and extra_path not in sys.path:
                    sys.path.insert(0, extra_path)

    @staticmethod
    def _install_cosmos_predict2_package_shim(repo_path: str) -> None:
        """Expose cosmos_predict2 submodules without running its top-level check."""
        package_root = os.path.join(repo_path, "cosmos_predict2")
        if not os.path.isdir(package_root):
            return
        existing = sys.modules.get("cosmos_predict2")
        if existing is not None and getattr(existing, "__file__", None):
            return

        module = types.ModuleType("cosmos_predict2")
        module.__path__ = [package_root]
        module.__package__ = "cosmos_predict2"
        module.__file__ = os.path.join(package_root, "__init__.py")
        module.__version__ = "1.5.0"

        about_path = os.path.join(package_root, "__about__.py")
        if os.path.exists(about_path):
            namespace = {}
            with open(about_path, "r", encoding="utf-8") as f:
                exec(compile(f.read(), about_path, "exec"), namespace)
            module.__version__ = namespace.get("__version__", module.__version__)

        sys.modules["cosmos_predict2"] = module

    @staticmethod
    def _install_optional_dependency_shims() -> None:
        """Let local-only official config imports pass without S3 packages."""
        if "boto3" not in sys.modules:
            boto3_module = types.ModuleType("boto3")

            def _missing_boto3(*args, **kwargs):
                raise RuntimeError("boto3 is not installed; S3 access is disabled in CosmoPredict25PolicyNative.")

            boto3_module.client = _missing_boto3
            boto3_module.Session = _missing_boto3
            sys.modules["boto3"] = boto3_module

        if "botocore" not in sys.modules:
            botocore_module = types.ModuleType("botocore")
            sys.modules["botocore"] = botocore_module

        if "botocore.config" not in sys.modules:
            config_module = types.ModuleType("botocore.config")

            class Config:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            config_module.Config = Config
            sys.modules["botocore.config"] = config_module

        if "botocore.exceptions" not in sys.modules:
            exceptions_module = types.ModuleType("botocore.exceptions")

            class ClientError(Exception):
                pass

            exceptions_module.ClientError = ClientError
            sys.modules["botocore.exceptions"] = exceptions_module

        if "multistorageclient" not in sys.modules:
            msc_module = types.ModuleType("multistorageclient")

            class StorageClient:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "multistorageclient is not installed; MSC/S3 access is disabled in "
                        "CosmoPredict25PolicyNative."
                    )

            class StorageClientConfig:
                @staticmethod
                def read_msc_config(*args, **kwargs):
                    raise RuntimeError(
                        "multistorageclient is not installed; MSC/S3 access is disabled in "
                        "CosmoPredict25PolicyNative."
                    )

                @staticmethod
                def from_dict(*args, **kwargs):
                    return None

            msc_module.StorageClient = StorageClient
            msc_module.StorageClientConfig = StorageClientConfig
            sys.modules["multistorageclient"] = msc_module

        if "multistorageclient.types" not in sys.modules:
            msc_types_module = types.ModuleType("multistorageclient.types")

            class Range:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            msc_types_module.Range = Range
            sys.modules["multistorageclient.types"] = msc_types_module

        if "transformer_engine" not in sys.modules:
            te_module = types.ModuleType("transformer_engine")
            te_pytorch_module = types.ModuleType("transformer_engine.pytorch")
            te_attention_module = types.ModuleType("transformer_engine.pytorch.attention")
            te_rope_module = types.ModuleType("transformer_engine.pytorch.attention.rope")
            te_optimizers_module = types.ModuleType("transformer_engine.pytorch.optimizers")

            class RMSNorm(torch.nn.Module):
                def __init__(self, hidden_size, eps=1e-5, **kwargs):
                    super().__init__()
                    self.eps = eps
                    self.weight = torch.nn.Parameter(torch.ones(hidden_size))

                def reset_parameters(self):
                    torch.nn.init.ones_(self.weight)

                def forward(self, x):
                    out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
                    return out.to(dtype=x.dtype) * self.weight

            def _rotate_half(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)

            def apply_rotary_pos_emb(x, rope_emb, tensor_format="bshd", fused=True):
                del tensor_format, fused
                if rope_emb is None:
                    return x
                cos = rope_emb.cos().to(device=x.device, dtype=x.dtype)
                sin = rope_emb.sin().to(device=x.device, dtype=x.dtype)
                while cos.dim() < x.dim():
                    cos = cos.unsqueeze(0)
                    sin = sin.unsqueeze(0)
                return (x * cos) + (_rotate_half(x) * sin)

            class DotProductAttention(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()

                def set_context_parallel_group(self, *args, **kwargs):
                    return None

                def forward(self, q, k, v, *args, **kwargs):
                    del args, kwargs
                    q_bhsd = q.permute(0, 2, 1, 3)
                    k_bhsd = k.permute(0, 2, 1, 3)
                    v_bhsd = v.permute(0, 2, 1, 3)
                    out = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
                    out = out.permute(0, 2, 1, 3).contiguous()
                    return out.flatten(start_dim=2)

            def _missing_multi_tensor_applier(*args, **kwargs):
                raise RuntimeError(
                    "transformer_engine is not installed; official FusedAdam is unavailable. "
                    "Use the starVLA optimizer path instead."
                )

            te_pytorch_module.RMSNorm = RMSNorm
            te_attention_module.DotProductAttention = DotProductAttention
            te_attention_module.apply_rotary_pos_emb = apply_rotary_pos_emb
            te_rope_module.apply_rotary_pos_emb = apply_rotary_pos_emb
            te_optimizers_module.multi_tensor_applier = _missing_multi_tensor_applier
            te_pytorch_module.attention = te_attention_module
            te_pytorch_module.optimizers = te_optimizers_module
            te_module.pytorch = te_pytorch_module
            sys.modules["transformer_engine"] = te_module
            sys.modules["transformer_engine.pytorch"] = te_pytorch_module
            sys.modules["transformer_engine.pytorch.attention"] = te_attention_module
            sys.modules["transformer_engine.pytorch.attention.rope"] = te_rope_module
            sys.modules["transformer_engine.pytorch.optimizers"] = te_optimizers_module

        if "transformer_engine_torch" not in sys.modules:
            tex_module = types.ModuleType("transformer_engine_torch")

            def _missing_transformer_engine_torch(*args, **kwargs):
                raise RuntimeError(
                    "transformer_engine_torch is not installed; official FusedAdam is unavailable."
                )

            tex_module.multi_tensor_adam = _missing_transformer_engine_torch
            tex_module.multi_tensor_adam_capturable = _missing_transformer_engine_torch
            tex_module.multi_tensor_adam_capturable_master = _missing_transformer_engine_torch
            sys.modules["transformer_engine_torch"] = tex_module

        try:
            from transformers import cache_utils as transformers_cache_utils

            if not hasattr(transformers_cache_utils, "SlidingWindowCache"):
                class SlidingWindowCache(transformers_cache_utils.DynamicCache):
                    pass

                transformers_cache_utils.SlidingWindowCache = SlidingWindowCache
        except Exception:
            pass

        if "megatron" not in sys.modules:
            sys.modules["megatron"] = types.ModuleType("megatron")
        if "megatron.core" not in sys.modules:
            core_module = types.ModuleType("megatron.core")
            sys.modules["megatron.core"] = core_module
        else:
            core_module = sys.modules["megatron.core"]
        sys.modules["megatron"].core = core_module

        if not hasattr(core_module, "ModelParallelConfig"):
            try:
                import attrs as _attrs

                @_attrs.define(slots=False)
                class ModelParallelConfig:
                    tensor_model_parallel_size: int = 1
                    pipeline_model_parallel_size: int = 1
                    context_parallel_size: int = 1
                    expert_model_parallel_size: int = 1
                    sequence_parallel: bool = False

            except Exception:
                @dataclass
                class ModelParallelConfig:
                    tensor_model_parallel_size: int = 1
                    pipeline_model_parallel_size: int = 1
                    context_parallel_size: int = 1
                    expert_model_parallel_size: int = 1
                    sequence_parallel: bool = False

            core_module.ModelParallelConfig = ModelParallelConfig

        if "megatron.core.parallel_state" not in sys.modules:
            parallel_state_module = types.ModuleType("megatron.core.parallel_state")

            class _SingleProcessGroup:
                def size(self):
                    return 1

                def rank(self):
                    return 0

            _group = _SingleProcessGroup()
            parallel_state_module.sequence_parallel = False
            parallel_state_module.cp_size_t = 1
            parallel_state_module.is_initialized = lambda: False
            parallel_state_module.initialize_model_parallel = lambda *args, **kwargs: None
            parallel_state_module.destroy_model_parallel = lambda *args, **kwargs: None
            parallel_state_module.get_data_parallel_world_size = lambda *args, **kwargs: 1
            parallel_state_module.get_data_parallel_rank = lambda *args, **kwargs: 0
            parallel_state_module.get_context_parallel_world_size = lambda *args, **kwargs: 1
            parallel_state_module.get_context_parallel_rank = lambda *args, **kwargs: 0
            parallel_state_module.get_context_parallel_group = lambda *args, **kwargs: None
            parallel_state_module.get_tensor_model_parallel_world_size = lambda *args, **kwargs: 1
            parallel_state_module.get_tensor_model_parallel_rank = lambda *args, **kwargs: 0
            parallel_state_module.get_pipeline_model_parallel_world_size = lambda *args, **kwargs: 1
            parallel_state_module.get_pipeline_model_parallel_rank = lambda *args, **kwargs: 0
            parallel_state_module.get_expert_model_parallel_world_size = lambda *args, **kwargs: 1
            parallel_state_module.get_model_parallel_group = lambda *args, **kwargs: _group
            parallel_state_module.get_data_parallel_group = lambda *args, **kwargs: None
            sys.modules["megatron.core.parallel_state"] = parallel_state_module
            core_module.parallel_state = parallel_state_module
        else:
            core_module.parallel_state = sys.modules["megatron.core.parallel_state"]

    def _load_native_model(self):
        try:
            from cosmos_predict2._src.imaginaire.lazy_config import instantiate
            from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
        except ImportError as exc:
            raise ImportError(
                "Failed to import the minimal NVIDIA cosmos-predict2.5 config/model helpers. "
                "This can mean framework.policy.native_repo_path is wrong, or one of the native repo "
                "dependencies is missing from the current Python environment. "
                f"native_repo_path={self.policy_cfg.get('native_repo_path')!r}, "
                f"COSMOS_PREDICT25_REPO={os.getenv('COSMOS_PREDICT25_REPO')!r}. "
                f"Original ImportError: {exc!r}\n{traceback.format_exc()}"
            ) from exc

        checkpoint_path = (
            self.policy_cfg.get("checkpoint_path")
            or self.config.framework.world_model.get("base_wm")
            or self.config.framework.qwenvl.get("base_vlm")
        )
        if not checkpoint_path:
            raise ValueError(
                "CosmoPredict25PolicyNative needs a native Cosmos checkpoint path in "
                "framework.world_model.base_wm or framework.policy.checkpoint_path."
            )

        use_online_text_encoder = bool(self.policy_cfg.get("use_online_text_encoder", True))
        if use_online_text_encoder:
            transformers_version = importlib.metadata.version("transformers")
            if transformers_version != "4.51.3":
                raise ValueError(
                    "Online Reason1 must run with transformers==4.51.3 to match the official "
                    f"cosmos-predict2.5 lockfile, but the current environment has transformers=={transformers_version}. "
                    "Please switch/pin the training environment before online Reason1 training."
                )
            text_encoder_ckpt_path = self.policy_cfg.get("text_encoder_ckpt_path", "")
            if not text_encoder_ckpt_path:
                raise ValueError(
                    "Online Reason1 requires framework.policy.text_encoder_ckpt_path to point to a local "
                    "Reason1 checkpoint. The official default is an internal S3 path, which this wrapper "
                    "does not use."
                )
            if not os.path.exists(os.path.abspath(os.path.expanduser(text_encoder_ckpt_path))):
                raise FileNotFoundError(
                    "framework.policy.text_encoder_ckpt_path does not exist: "
                    f"{text_encoder_ckpt_path!r}"
                )

        experiment_opts = list(self.policy_cfg.get("experiment_opts", []))
        experiment_opts.extend([
            "model.config.fsdp_shard_size=1",
            "model.config.ema.enabled=false",
            "model.config.net.atten_backend=torch",
            f"model.config.action_loss_multiplier={int(self.policy_cfg.get('action_loss_multiplier', 16))}",
        ])
        if bool(self.policy_cfg.get("mask_policy_loss", True)):
            experiment_opts.append("model.config.mask_value_prediction_loss_for_policy_prediction=true")

        config_module = get_config_module(self.policy_cfg.get("config_file"))
        native_config = importlib.import_module(config_module).make_config()
        native_config = override(
            native_config,
            ["--", f"experiment={self.policy_cfg.get('experiment')}", *experiment_opts],
        )
        if use_online_text_encoder:
            text_encoder_ckpt_path = self.policy_cfg.get("text_encoder_ckpt_path", "")
            from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoderConfig

            text_encoder_config = TextEncoderConfig(
                compute_online=True,
                ckpt_path=os.path.abspath(os.path.expanduser(text_encoder_ckpt_path)),
            )
            native_config.model.config.text_encoder_class = "reason1p1_7B"
            native_config.model.config.text_encoder_config = text_encoder_config

        model = instantiate(native_config.model)
        if hasattr(model, "on_train_start"):
            model.on_train_start()
        if torch.cuda.is_available():
            model = model.cuda()
        self._load_local_checkpoint(model, checkpoint_path)
        return model, native_config

    @staticmethod
    def _candidate_checkpoint_files(checkpoint_path: str) -> List[str]:
        if os.path.isfile(checkpoint_path):
            return [checkpoint_path]
        candidates = [
            os.path.join(checkpoint_path, "model.pt"),
            os.path.join(checkpoint_path, "model.pth"),
            os.path.join(checkpoint_path, "pytorch_model.pt"),
            os.path.join(checkpoint_path, "pytorch_model.bin"),
        ]
        base_dir = os.path.join(checkpoint_path, "base", "pre-trained")
        candidates.extend([
            os.path.join(base_dir, "model.pt"),
            os.path.join(base_dir, "model.pth"),
            os.path.join(base_dir, "pytorch_model.pt"),
        ])
        return [path for path in candidates if os.path.isfile(path)]

    def _load_local_checkpoint(self, model, checkpoint_path: str) -> None:
        checkpoint_files = self._candidate_checkpoint_files(os.path.expanduser(checkpoint_path))
        if not checkpoint_files:
            raise FileNotFoundError(
                "CosmoPredict25PolicyNative now uses a minimal local torch.load checkpoint path "
                "and does not import NVIDIA model_loader/easy_io/boto3. Please point "
                "framework.policy.checkpoint_path or framework.world_model.base_wm to a local .pt/.pth file. "
                f"No supported checkpoint file found under: {checkpoint_path}"
            )

        state = torch.load(checkpoint_files[0], map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        load_info = model.load_state_dict(state, strict=False)
        if load_info is not None and (load_info.missing_keys or load_info.unexpected_keys):
            print(
                "[CosmoPredict25PolicyNative] checkpoint loaded with "
                f"{len(load_info.missing_keys)} missing keys and "
                f"{len(load_info.unexpected_keys)} unexpected keys. "
                f"First missing={load_info.missing_keys[:5]}, "
                f"first unexpected={load_info.unexpected_keys[:5]}",
                flush=True,
            )

    def _configure_native_policy_loss(self) -> None:
        if bool(self.policy_cfg.get("mask_policy_loss", True)):
            self.native_model.config.mask_value_prediction_loss_for_policy_prediction = True
        self.native_model.config.action_loss_multiplier = int(self.policy_cfg.get("action_loss_multiplier", 16))

    def _freeze_native_non_policy_modules(self) -> None:
        for module_name in ("tokenizer", "text_encoder"):
            module = getattr(self.native_model, module_name, None)
            if module is not None and hasattr(module, "requires_grad_"):
                module.requires_grad_(False)

    def _load_text_embedding_cache(self) -> Dict[str, torch.Tensor]:
        path = self.policy_cfg.get("text_embeddings_path", "")
        if not path:
            return {}
        path = os.path.abspath(os.path.expanduser(path))
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _as_image_list(images) -> List:
        return list(images) if isinstance(images, (list, tuple)) else [images]

    @staticmethod
    def _blank_like(image):
        if isinstance(image, Image.Image):
            return Image.new(image.mode, image.size)
        image_array = np.asarray(image)
        return np.zeros_like(image_array)

    @staticmethod
    def _to_uint8_array(image) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if torch.is_tensor(image):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if array.max() <= 1.0 and array.min() >= 0.0:
                array = array * 255.0
            elif array.min() >= -1.0 and array.max() <= 1.0:
                array = (array + 1.0) * 127.5
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return array[..., :3]

    def _resize_uint8(self, image) -> np.ndarray:
        array = self._to_uint8_array(image)
        pil = Image.fromarray(array)
        pil = pil.resize((self.width, self.height), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    def _split_policy_views(self, images: List, blank_image):
        primary = images[0] if images else blank_image
        wrist = images[1] if len(images) > 1 else blank_image
        secondary = images[2] if len(images) > 2 else blank_image
        return wrist, primary, secondary

    def _build_video_frames(self, example: dict, include_future: bool) -> List[np.ndarray]:
        temporal_factor = 4
        current_images = self._as_image_list(example["image"])
        future_images = self._as_image_list(example.get("future_image", current_images)) if include_future else []
        blank_image = self._blank_like(current_images[0])

        wrist, primary, secondary = self._split_policy_views(current_images, blank_image)
        if future_images:
            future_wrist, future_primary, future_secondary = self._split_policy_views(future_images, blank_image)
        else:
            future_wrist, future_primary, future_secondary = blank_image, blank_image, blank_image

        slot_images = [
            blank_image,       # 0: special blank, not repeated
            blank_image,       # 1: current proprio placeholder
            wrist,             # 2: current wrist
            primary,           # 3: current primary
            secondary,         # 4: current secondary
            blank_image,       # 5: action placeholder
            blank_image,       # 6: future proprio placeholder
            future_wrist,      # 7: future wrist
            future_primary,    # 8: future primary
            future_secondary,  # 9: future secondary
            blank_image,       # 10: value placeholder
        ]

        frames = [self._resize_uint8(slot_images[0])]
        for slot_image in slot_images[1:]:
            resized = self._resize_uint8(slot_image)
            frames.extend([resized] * temporal_factor)
        return frames

    def _stack_video(self, examples: List[dict], include_future: bool) -> torch.Tensor:
        videos = []
        for example in examples:
            frames = np.stack(self._build_video_frames(example, include_future=include_future), axis=0)
            videos.append(torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous())
        return torch.stack(videos, dim=0).to(dtype=torch.uint8)

    def _select_action_chunk(self, actions) -> torch.Tensor:
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        return actions[:, -(self.future_action_window_size + 1):, :]

    def _stack_state(self, examples: List[dict], key: str, device: torch.device) -> torch.Tensor:
        if key in examples[0]:
            return torch.tensor(np.array([example[key] for example in examples]), device=device, dtype=torch.float32)
        return torch.zeros((len(examples), self.state_dim), device=device, dtype=torch.float32)

    def _stack_text_embeddings(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        if self._use_online_text_encoder:
            raise RuntimeError("_stack_text_embeddings should not be called when use_online_text_encoder=true.")
        if "t5_text_embeddings" in examples[0]:
            embeddings = [example["t5_text_embeddings"] for example in examples]
        else:
            embeddings = []
            missing = []
            for example in examples:
                instruction = example.get("lang") or example.get("instruction") or example.get("ai_caption")
                if instruction in self._text_embedding_cache:
                    embeddings.append(self._text_embedding_cache[instruction])
                else:
                    missing.append(instruction)
            if missing:
                raise KeyError(
                    "CosmoPredict25PolicyNative needs official text embeddings. "
                    "Provide example['t5_text_embeddings'] or set framework.policy.text_embeddings_path. "
                    f"Missing instruction example: {missing[0]!r}"
                )

        tensors = []
        for embedding in embeddings:
            tensor = embedding if torch.is_tensor(embedding) else torch.as_tensor(embedding)
            tensor = tensor.squeeze(0).to(device=device, dtype=torch.bfloat16)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _make_native_batch(self, examples: List[dict], include_future: bool = True) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = len(examples)

        actions = self._select_action_chunk([example["action"] for example in examples]).to(device=device)
        video = self._stack_video(examples, include_future=include_future)
        proprio = self._stack_state(examples, "state", device=device)
        future_proprio = self._stack_state(examples, "future_state", device=device)

        captions = [
            example.get("ai_caption") or example.get("lang") or example.get("instruction") or ""
            for example in examples
        ]
        if self._use_online_text_encoder:
            text_embeddings = None
            text_mask = None
        else:
            text_embeddings = self._stack_text_embeddings(examples, device=device)
            text_mask = torch.ones(
                (batch_size, text_embeddings.shape[1]),
                device=device,
                dtype=torch.int64,
            )

        def full_idx(value: int) -> torch.Tensor:
            return torch.full((batch_size,), value, device=device, dtype=torch.int64)

        zeros = torch.zeros((batch_size,), device=device, dtype=torch.int64)
        data_batch = {
            "video": video,
            "actions": actions,
            "proprio": proprio,
            "future_proprio": future_proprio,
            "ai_caption": captions,
            "num_conditional_frames": torch.tensor(5, device=device, dtype=torch.int64),
            "current_proprio_latent_idx": full_idx(1) if "state" in examples[0] else full_idx(-1),
            "action_latent_idx": full_idx(5),
            "future_proprio_latent_idx": full_idx(-1),
            "future_wrist_image_latent_idx": full_idx(-1),
            "future_wrist_image2_latent_idx": full_idx(-1),
            "future_image_latent_idx": full_idx(8) if include_future else full_idx(-1),
            "future_image2_latent_idx": full_idx(-1),
            "value_latent_idx": full_idx(10),
            "rollout_data_mask": zeros.clone(),
            "world_model_sample_mask": zeros.clone(),
            "value_function_sample_mask": zeros.clone(),
            "value_function_return": torch.zeros((batch_size,), device=device, dtype=torch.float32),
        }
        if not self._use_online_text_encoder:
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = text_mask
        return data_batch

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        data_batch = self._make_native_batch(examples, include_future=True)
        output_batch, loss = self.native_model.training_step(data_batch, self._iteration)
        self._iteration += 1

        metrics = {"action_loss": loss}
        for key in (
            "demo_sample_action_mse_loss",
            "demo_sample_action_l1_loss",
            "demo_sample_future_image_mse_loss",
            "demo_sample_future_image_l1_loss",
            "velocity_mse_loss",
            "edm_loss",
        ):
            value = output_batch.get(key)
            if torch.is_tensor(value):
                metrics[key] = value.detach()
        return metrics

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
            extract_action_chunk_from_latent_sequence,
        )

        data_batch = self._make_native_batch(examples, include_future=True)
        batch_size = len(examples)
        num_steps = int(kwargs.get("num_inference_steps", self.num_inference_steps))
        generated_latent = self.native_model.generate_samples_from_batch(
            data_batch,
            n_sample=batch_size,
            num_steps=num_steps,
            seed=int(kwargs.get("seed", 0)),
            is_negative_prompt=False,
            use_variance_scale=False,
            return_orig_clean_latent_frames=False,
            shift=float(kwargs.get("shift", self.shift)),
            guidance=0,
        )
        action_indices = torch.full(
            (batch_size,),
            5,
            dtype=torch.int64,
            device=generated_latent.device,
        )
        actions = extract_action_chunk_from_latent_sequence(
            generated_latent,
            action_shape=(self.future_action_window_size + 1, self.action_dim),
            action_indices=action_indices,
        )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
