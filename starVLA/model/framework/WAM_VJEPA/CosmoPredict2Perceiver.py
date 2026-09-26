# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
CosmoPredict2-Perceiver Framework.

Uses Cosmos-Predict2 as the world-model backbone and the Perceiver-style
flow-matching action head for continuous action prediction.
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.PerceiverHead import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class CosmoPredict2PerceiverDefaultConfig:
    """CosmoPredict2-Perceiver default parameters."""

    name: str = "CosmoPredict2Perceiver"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    })

    # Kept for compatibility with shared utilities and action-head configs.
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "vl_hidden_dim": 2048,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "max_num_embodiments": 32,
        "use_category_specific": False,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict2Perceiver")
class CosmoPredict2_Perceiver(baseframework):
    """World-model-for-action framework using Cosmos-Predict2 + Perceiver head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict2PerceiverDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    # 训练时
    # action支路，self.backbone.build_inputs、self.backbone.forward()、self.action_model.forward()
    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        # print('################# CosmoPredict2_Perceiver.forward-1')

        # 1. CosmosPredict：构建输入
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        # print(len(examples))                                # 8

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        # print(type(self.backbone))                          # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface'>
        # print(wm_inputs.keys())                             # dict_keys(['hidden_states', 'timestep', 'encoder_hidden_states', 'condition_mask', 'padding_mask', '_is_wm_input'])
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
            # print(type(wm_outputs))                         # <class 'starVLA.model.modules.world_model.CosmoPredict25._CosmoPredict25_Interface.forward.<locals>._WMOutput'>
            # print(last_hidden.shape)                        # torch.Size([8, 196, 2048])

        # 3. ActionHead：构建输入
        with torch.autocast("cuda", dtype=torch.float32):
            
            # 3.1 初始化action
            actions = [example["action"] for example in examples]
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            # print(actions.shape)                            # torch.Size([8, 16, 32])
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]
            # print(self.future_action_window_size)           # 15
            # print(actions_target.shape)                     # torch.Size([8, 16, 32])

            # 3.2 重复降噪
            # 将action和vl-tokens复制多次，重复多次降噪
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "trainer")
                else self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            # print(actions_target_repeated.shape)            # torch.Size([32, 16, 32])
            # print(last_hidden_repeated.shape)               # torch.Size([32, 196, 2048])

            # 3.3 准备state、action_mask和embodiment_tag
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            if state is not None:
                # print('state is not None')
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                # print(state.shape)                          # torch.Size([8, 1, 64])
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
                # print(state_repeated.shape)                 # torch.Size([32, 1, 64])
            else:
                state_repeated = None

            action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
            if action_mask is not None:
                # print('action_mask is not None')
                action_mask_tensor = torch.tensor(np.array(action_mask), device=last_hidden.device, dtype=last_hidden.dtype)
                # print(action_mask_tensor.shape)             # torch.Size([8, 16, 32])
                action_mask_target = action_mask_tensor[:, -(self.future_action_window_size + 1):, :]
                # print(action_mask_target.shape)             # torch.Size([8, 16, 32])
                action_mask_repeated = action_mask_target.repeat(repeated_diffusion_steps, 1, 1)
                # print(action_mask_repeated.shape)           # torch.Size([32, 16, 32])
            else:
                action_mask_repeated = None
            
            embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
            if embodiment_tag is not None:
                # print('embodiment_tag is not None')
                embodiment_tag = torch.tensor(np.array(embodiment_tag), device=last_hidden.device, dtype=torch.int64)
                # print(embodiment_tag.shape)                 # torch.Size([8])
                embodiment_tag = embodiment_tag.view(-1)
                # print(embodiment_tag.shape)                 # torch.Size([8])
                embodiment_tag_repeated = embodiment_tag.repeat(repeated_diffusion_steps)
                # print(embodiment_tag_repeated.shape)        # torch.Size([32])
            else:
                embodiment_tag_repeated = None

            action_loss, target_emb = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                embodiment_tag_repeated,
                action_mask=action_mask_repeated,
                use_vjepa2ac=True,
                repeated_diffusion_steps=repeated_diffusion_steps,
            )
            # print(type(self.action_model))          # <class 'starVLA.model.modules.action_model.PerceiverHead.FlowmatchingActionHead'>
            # print(last_hidden_repeated.shape)       # torch.Size([32, 196, 2048])
            # print(actions_target_repeated.shape)    # torch.Size([32, 16, 32])
            # print(state_repeated.shape)             # torch.Size([32, 1, 64])
            # print(embodiment_tag_repeated.shape)    # torch.Size([32])
            # print(action_mask_repeated.shape)       # torch.Size([32, 16, 32])
            # print(action_loss)                      # tensor(1.5625, device='cuda:0', grad_fn=<DivBackward0>)
            # print(target_emb.shape)                 # torch.Size([8, 64, 1024])

        # print('################# CosmoPredict2_Perceiver.forward-2')
        return action_loss, target_emb

    # @torch.inference_mode()
    # def predict_action(self, examples: List[dict], **kwargs) -> dict:
    #     if type(examples) is not list:
    #         examples = [examples]

    #     batch_images = [to_pil_preserve(example["image"]) for example in examples]
    #     instructions = [example["lang"] for example in examples]
    #     state = [example["state"] for example in examples] if "state" in examples[0] else None
    #     embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None

    #     train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
    #     if train_obs_image_size:
    #         batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    #     wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         wm_outputs = self.backbone(
    #             **wm_inputs,
    #             output_hidden_states=True,
    #             return_dict=True,
    #         )
    #         last_hidden = wm_outputs.hidden_states[-1]

    #     state = (
    #         torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
    #         if state is not None
    #         else None
    #     )
    #     embodiment_tag = (
    #         torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64)
    #         if embodiment_tag is not None
    #         else None
    #     )
    #     if embodiment_tag is not None:
    #         embodiment_tag = embodiment_tag.view(-1)

    #     with torch.autocast("cuda", dtype=torch.float32):
    #         pred_actions = self.action_model.predict_action(last_hidden, state, embodiment_tag)

    #     normalized_actions = pred_actions.detach().cpu().numpy()
    #     return {"normalized_actions": normalized_actions}
