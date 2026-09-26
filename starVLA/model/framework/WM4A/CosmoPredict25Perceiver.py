# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2.5-Perceiver Framework.

This keeps Predict2.5 separate from the existing Cosmos-Predict2 framework
while reusing the same Perceiver action-head training/inference logic.
"""

from dataclasses import dataclass, field
from typing import Optional

from starVLA.model.framework.WM4A.CosmoPredict2Perceiver import CosmoPredict2_Perceiver
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.PerceiverHead_original import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class CosmoPredict25PerceiverDefaultConfig:
    """Cosmos-Predict2.5-Perceiver default parameters."""

    name: str = "CosmoPredict25Perceiver"

    world_model: dict = field(default_factory=lambda: {
        "base_wm": "nvidia/Cosmos-Predict2.5-2B",
        "revision": "diffusers/base/post-trained",
        "extract_layers": [-1],
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "nvidia/Cosmos-Predict2.5-2B",
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


@FRAMEWORK_REGISTRY.register("CosmoPredict25Perceiver")
class CosmoPredict25_Perceiver(CosmoPredict2_Perceiver):
    """World-model-for-action framework using Cosmos-Predict2.5 + Perceiver."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(CosmoPredict25PerceiverDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
