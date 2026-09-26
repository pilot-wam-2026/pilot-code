# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by combining QwenFast + QwenWanPerceiver.
"""
QwenFastWanPerceiver Framework

VLM 内部通过 next-token prediction 学习 fast action（QwenFast），
同时用纯 VL 条件（图像 + 指令，不含 action token）融合 Wan 特征后，
交给 Perceiver flow-matching head 预测连续动作（QwenWanPerceiver）。

因果 LM 下，action token 之前的 hidden state 与仅 VL 输入前向等价，
因此训练时可单次前向（含 teacher-forced action）并截取 VL 段特征。
"""
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.PerceiverHead_original import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("QwenFastWanPerceiver")
class QwenFastWanPerceiver(baseframework):
    """
    Qwen-VL + Wan + Perceiver，VLM 内部额外学习 fast action token 预测。

    - Fast action：VLM 对离散 action token 做自回归监督（训练）/ generate（推理）
    - Continuous action：仅用 VL hidden（不含 action token 段）+ Wan 特征 → Perceiver head
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.wan_interface = get_world_model(config=self.config)

        wm_cfg = config.framework.get("world_model", {})
        self._pool_wm_sequence = bool(wm_cfg.get("pool_wm_sequence", False))
        wm_hidden = self.wan_interface.model.config.hidden_size
        cross_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = cross_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_dim)
        self.wm_post_norm = torch.nn.LayerNorm(cross_dim, eps=1e-6)
        gain = float(wm_cfg.get("wm_projector_init_gain", 0.02))
        torch.nn.init.xavier_uniform_(self.wm_projector.weight, gain=gain)
        torch.nn.init.zeros_(self.wm_projector.bias)

        self.fast_action_model = Fast_Action_Tokenizer()
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.fast_action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.fast_action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

        self.fast_loss_weight = float(config.framework.get("fast_loss_weight", 1.0))

        self.freeze_vlm_for_vla = (
            config.trainer.get("freeze_vlm_for_vla", False)
            if config and hasattr(config, "trainer")
            else False
        )

        logger.info(
            f"QwenFastWanPerceiver: pool_wm_sequence={self._pool_wm_sequence}, "
            f"fast_loss_weight={self.fast_loss_weight}"
        )

    @staticmethod
    def _wm_images_head_only(batch_images: List) -> List:
        wm_images = []
        for imgs in batch_images:
            if isinstance(imgs, (list, tuple)):
                wm_images.append([imgs[0]] if len(imgs) > 0 else [])
            else:
                wm_images.append([imgs])
        return wm_images

    def _fuse_vlm_with_wm(self, last_vlm_hidden: torch.Tensor, last_wm_hidden: torch.Tensor) -> torch.Tensor:
        if self._pool_wm_sequence:
            last_wm_hidden = last_wm_hidden.mean(dim=1, keepdim=True)
        proj_dtype = self.wm_projector.weight.dtype
        x = last_wm_hidden.to(dtype=proj_dtype)
        wm_aligned = self.wm_post_norm(self.wm_projector(x)).to(dtype=last_vlm_hidden.dtype)
        return torch.cat([last_vlm_hidden, wm_aligned], dim=1)

    def _first_action_token_index(self, input_ids: torch.Tensor) -> int:
        """返回 batch 内第一个 action token 的位置（各样本 VL 段长度应一致）。"""
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        seq = input_ids[0]
        is_action = (seq >= act_min) & (seq <= act_max)
        nonzero = torch.nonzero(is_action, as_tuple=False)
        return nonzero[0].item() if nonzero.numel() > 0 else input_ids.size(1)

    def _extract_vl_only_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        从含 fast action 的完整序列 hidden 中截取纯 VL 段。
        因果 LM 下，action token 之前的表示与仅 VL 输入前向一致。
        """
        end_idx = self._first_action_token_index(input_ids)
        return hidden_states[:, :end_idx, :]

    def _encode_wan_hidden(self, batch_images: List, instructions: List[str]) -> torch.Tensor:
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
                return wm_outputs.hidden_states[-1]

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        return "".join([f"<robot_action_{token}>" for token in tokens])

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = (
            [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        )

        if "action_mask" in examples[0]:
            action_mask = [example["action_mask"] for example in examples]
            if all(mask is None for mask in action_mask):
                action_mask = None
        else:
            action_mask = None

        batch_fast_tokens = self.fast_action_model.encoder_action2fastoken(actions)
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )
        last_wm_hidden = self._encode_wan_hidden(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_vlm_hidden = self._extract_vl_only_hidden(
                qwenvl_outputs.hidden_states[-1], qwen_inputs["input_ids"]
            )
            if self.freeze_vlm_for_vla:
                last_vlm_hidden = last_vlm_hidden.detach()

            vlm_fast_loss = qwenvl_outputs.loss
            if vlm_fast_loss is None or torch.isnan(vlm_fast_loss):
                vlm_fast_loss = torch.tensor(0.0, device=last_vlm_hidden.device)

        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions_tensor[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    np.array(action_mask), device=last_hidden.device, dtype=last_hidden.dtype
                )
                action_mask_target = action_mask_tensor[:, -(self.future_action_window_size + 1) :, :]
                action_mask_repeated = action_mask_target.repeat(repeated_diffusion_steps, 1, 1)

            if embodiment_tag is not None:
                embodiment_tag_tensor = torch.tensor(
                    np.array(embodiment_tag), device=last_hidden.device, dtype=torch.int64
                )
                embodiment_tag_repeated = embodiment_tag_tensor.repeat(repeated_diffusion_steps)
            else:
                embodiment_tag_repeated = None

            diffusion_action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                embodiment_tag_repeated,
                action_mask=action_mask_repeated,
            )

        action_loss = diffusion_action_loss + self.fast_loss_weight * vlm_fast_loss
        return {
            "action_loss": action_loss,
            "vlm_fast_loss": vlm_fast_loss,
            "diffusion_action_loss": diffusion_action_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        embodiment_tag = (
            [example["embodiment_tag"] for example in examples] if "embodiment_tag" in examples[0] else None
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        last_wm_hidden = self._encode_wan_hidden(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_vlm_hidden = qwenvl_outputs.hidden_states[-1]

        last_hidden = self._fuse_vlm_with_wm(last_vlm_hidden, last_wm_hidden)

        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        embodiment_tag_tensor = (
            torch.from_numpy(np.array(embodiment_tag)).to(last_hidden.device, dtype=torch.int64)
            if embodiment_tag is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            if embodiment_tag_tensor is not None:
                pred_actions = self.action_model.predict_action(last_hidden, state_tensor, embodiment_tag_tensor)
            else:
                pred_actions = self.action_model.predict_action(last_hidden, state_tensor)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_fast_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict:
        """可选：单独通过 VLM generate 预测 fast 离散 action。"""
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(**qwen_inputs, max_length=2048)

        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        normalized_actions = self.fast_action_model.fast_tokenizer.decode(batch_fast_action_token_idx)
        return {"normalized_actions": normalized_actions}

    def _extract_action_token_ids(self, generated_ids: torch.LongTensor) -> List[List[int]]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            results.append(generated_ids[b, idx].tolist())
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            batch_fast_token_ids.append([t - act_min for t in seq])
        return batch_fast_token_ids


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml")
    model: QwenFastWanPerceiver = QwenFastWanPerceiver(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room.",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Put all the toys in the child's room.",
    }

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")
    print(f"VLM Fast Loss: {forward_output['vlm_fast_loss'].item()}")
    print(f"Diffusion Loss: {forward_output['diffusion_action_loss'].item()}")

    predict_output = model.predict_action(examples=[sample])
    print(f"Predicted Actions: {predict_output['normalized_actions']}")
