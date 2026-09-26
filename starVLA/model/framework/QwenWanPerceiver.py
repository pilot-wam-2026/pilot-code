# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model import get_world_model

from starVLA.model.modules.action_model.PerceiverHead_original import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

# from starVLA.model.modules.action_model.flow_matching_action_head import (
#     FlowmatchingActionHead,
#     FlowmatchingActionHeadConfig,
# )


@FRAMEWORK_REGISTRY.register("QwenWanPerceiver")
class QwenWan_Perceiver(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.wan_interface = get_world_model(config=self.config)

        wm_cfg = config.framework.get("world_model", {})
        # 将 Wan 多 token 压成 1 个全局条件，避免 action DiT 的 KV 比 QwenPerceiver 长数倍、稀释 cross-attn
        self._pool_wm_sequence = bool(wm_cfg.get("pool_wm_sequence", False))
        wm_hidden = self.wan_interface.model.config.hidden_size
        cross_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = cross_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_dim)
        self.wm_post_norm = torch.nn.LayerNorm(cross_dim, eps=1e-6)
        gain = float(wm_cfg.get("wm_projector_init_gain", 0.02))
        torch.nn.init.xavier_uniform_(self.wm_projector.weight, gain=gain)
        torch.nn.init.zeros_(self.wm_projector.bias)
        logger.info(
            f"QwenWanPerceiver: pool_wm_sequence={self._pool_wm_sequence}, wm_projector_init_gain={gain}"
        )

        self.use_time_aware_action_head = True

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # Whether to freeze VLM for VLA data (no gradient backprop to VLM)
        self.freeze_vlm_for_vla = (
            config.trainer.get("freeze_vlm_for_vla", False) 
            if config and hasattr(config, "trainer") else False
        )
        print("freeze_vlm_for_vla:  ",self.freeze_vlm_for_vla)

    @staticmethod
    def _wm_images_head_only(batch_images: List) -> List:
        """World model 只使用每个样本的第一路相机（例如 [头, 左手, 右手] 中的头部）。"""
        wm_images = []
        for imgs in batch_images:
            if isinstance(imgs, (list, tuple)):
                wm_images.append([imgs[0]] if len(imgs) > 0 else [])
            else:
                wm_images.append([imgs])
        return wm_images

    def _fuse_vlm_with_wm(self, last_vlm_hidden: torch.Tensor, last_wm_hidden: torch.Tensor) -> torch.Tensor:
        """Wan 特征与 Qwen 最后一层 hidden 对齐维后沿序列维拼接。"""
        if self._pool_wm_sequence:
            last_wm_hidden = last_wm_hidden.mean(dim=1, keepdim=True)
        # Wan 在 bf16 autocast 下常为 BF16，Linear/LayerNorm 参数默认 FP32，需对齐后再乘权重
        proj_dtype = self.wm_projector.weight.dtype
        x = last_wm_hidden.to(dtype=proj_dtype)
        wm_aligned = self.wm_post_norm(self.wm_projector(x)).to(dtype=last_vlm_hidden.dtype)
        return torch.cat([last_vlm_hidden, wm_aligned], dim=1)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None  # [B, 1]

        # Extract action_mask if available
        if "action_mask" in examples[0]:
            action_mask = [example["action_mask"] for example in examples]  # [B, len, action_dim]
            # Check if all action_mask values are None, if so set to None
            if all(mask is None for mask in action_mask):
                action_mask = None
        else:
            action_mask = None

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Step 2–3: Wan 仅作冻结特征提取：no_grad 不保留反传激活；bf16 autocast 降低 Wan 前向峰值显存
        # with torch.no_grad():
        #     with torch.autocast("cuda", dtype=torch.bfloat16):
        #         wm_inputs = self.wan_interface.build_inputs(
        #             images=self._wm_images_head_only(batch_images), instructions=instructions
        #         )
        #         wm_outputs = self.wan_interface(
        #             **wm_inputs,
        #             output_hidden_states=True,
        #             return_dict=True,
        #         )
        #         last_wm_hidden = wm_outputs.hidden_states[-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_inputs = self.wan_interface.build_inputs(
                images=self._wm_images_head_only(batch_images), instructions=instructions
            )
            wm_outputs = self.wan_interface(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_wm_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_vlm_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H] [ 2, 221, 3072 ]
            
            # Detach hidden states to prevent gradient backprop to VLM for VLA data
            if self.freeze_vlm_for_vla:
                last_vlm_hidden = last_vlm_hidden.detach()

        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)
        # last_hidden = last_vlm_hidden

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # Process action_mask if available
            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    np.array(action_mask), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]

                action_mask_target = action_mask_tensor[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)
                action_mask_repeated = action_mask_target.repeat(repeated_diffusion_steps, 1, 1)

            if embodiment_tag is not None:
                embodiment_tag = torch.tensor(np.array(embodiment_tag), device=last_hidden.device, dtype=torch.int64)
                embodiment_tag_repeated = embodiment_tag.repeat(repeated_diffusion_steps) # embodiment_tag is index
            else:
                embodiment_tag_repeated = None

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated, embodiment_tag_repeated, action_mask=action_mask_repeated)  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        embodiment_tag = [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None  # [B, 1]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL + Wan（与 forward 一致，否则推理与训练条件分布不一致）
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                wm_inputs = self.wan_interface.build_inputs(
                    images=self._wm_images_head_only(batch_images), instructions=instructions
                )
                wm_outputs = self.wan_interface(
                    **wm_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_wm_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_vlm_hidden = qwenvl_outputs.hidden_states[-1]

        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        embodiment_tag = torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64) if embodiment_tag is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            if embodiment_tag is not None:
                pred_actions = self.action_model.predict_action(last_hidden, state, embodiment_tag)  # (B, chunk_len, action_dim)
            else:
                pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model: QwenWan_Perceiver = QwenWan_Perceiver(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")
