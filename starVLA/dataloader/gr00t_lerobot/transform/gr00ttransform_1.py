# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
from starVLA.dataloader.gr00t_lerobot.transform.base import InvertibleModalityTransform
# ============================================================
# Unified schema definition
# ============================================================
UNIFIED_STATE_SCHEMA: List[Tuple[str, int]] = [
    ("left_arm", 6),
    ("right_arm", 6),
    ("left_gripper", 1),
    ("right_gripper", 1),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
    ("head", 2),
]
UNIFIED_ACTION_SCHEMA: List[Tuple[str, int]] = [
    ("left_arm", 6),
    ("right_arm", 6),
    ("left_gripper", 1),
    ("right_gripper", 1),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
    ("head", 2),
]
# alias：支持不同数据集字段名映射到统一 canonical block
DEFAULT_SUFFIX_TO_CANONICAL: Dict[str, str] = {
    "left_arm": "left_arm",
    "right_arm": "right_arm",
    "left_gripper": "left_gripper",
    "right_gripper": "right_gripper",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "waist": "waist",
    "head": "head",
    "left_pose": "left_arm",
    "right_pose": "right_arm",
    "original_state_head_position": "head",
    "original_state_waist_position": "waist",
    "original_action_head_position": "head",
    "original_action_waist_position": "waist",
}
# 反向：canonical -> 所有可接受的 suffix 候选
CANONICAL_TO_SUFFIX_CANDIDATES: Dict[str, List[str]] = {}
for suffix, canonical in DEFAULT_SUFFIX_TO_CANONICAL.items():
    CANONICAL_TO_SUFFIX_CANDIDATES.setdefault(canonical, []).append(suffix)
def collate(features: List[dict]) -> dict:
    batch = {}
    keys = features[0].keys()
    for key in keys:
        values = [elem[key] for elem in features]
        if key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            batch[key] = torch.cat(values)
        else:
            # numpy / torch 都兼容
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = torch.from_numpy(np.stack(values))
    return batch
class DefaultDataCollator(DataCollatorMixin):
    def __init__(self):
        super().__init__()
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features)
class GR00TTransform(InvertibleModalityTransform):
    apply_to: list[str] = Field(default_factory=list)
    training: bool = Field(default=True)
    embodiment_tag_mapping: dict[str, int] = Field(default=EMBODIMENT_TAG_MAPPING)
    language_dropout_prob: float = Field(default=0.0)
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int
    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None
    # 可选：如果某些数据集字段名不标准，可以额外传 alias 覆盖
    # 例如 {"state.ee_pos": "left_arm"}
    extra_suffix_to_canonical: Dict[str, str] = Field(default_factory=dict)
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag
    def get_embodiment_tag(self) -> int:
        assert self.embodiment_tag is not None, "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]
    def check_keys_and_batch_size(self, data):
        # BUG
        video_ndim = data["video"].ndim
        if video_ndim == 5:   # [T, V, H, W, C]
            return False, 1
        elif video_ndim == 6: # [B, T, V, H, W, C]
            return True, data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")
    # ============================================================
    # helpers
    # ============================================================
    def _build_schema_slices(self, schema: List[Tuple[str, int]]):
        slices = {}
        start = 0
        for name, dim in schema:
            slices[name] = slice(start, start + dim)
            start += dim
        return slices, start
    def _to_tensor_2d(self, x, horizon: int, name: str) -> torch.Tensor:
        """
        把输入转成 [T, D] 的 torch.Tensor。
        支持 list / np.ndarray / torch.Tensor。
        """
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if t.ndim == 1:
            # [T] -> [T, 1]
            if t.shape[0] != horizon:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, expected first dim == horizon={horizon}"
                )
            t = t.unsqueeze(-1)
        elif t.ndim == 2:
            if t.shape[0] != horizon:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, expected first dim == horizon={horizon}"
                )
        else:
            raise ValueError(
                f"{name} must be 1D or 2D ([T] or [T, D]), but got shape {tuple(t.shape)}"
            )
        return t.float()
    def _merged_suffix_to_canonical(self) -> Dict[str, str]:
        merged = dict(DEFAULT_SUFFIX_TO_CANONICAL)
        merged.update(self.extra_suffix_to_canonical)
        return merged
    def _merged_canonical_to_candidates(self) -> Dict[str, List[str]]:
        suffix_to_canonical = self._merged_suffix_to_canonical()
        canonical_to_candidates: Dict[str, List[str]] = {}
        for suffix, canonical in suffix_to_canonical.items():
            canonical_to_candidates.setdefault(canonical, []).append(suffix)
        return canonical_to_candidates
    def _get_block_from_data(
        self,
        data: dict,
        prefix: str,          # "state" or "action"
        canonical_name: str,  # e.g. "left_arm"
        horizon: int,
    ) -> Optional[torch.Tensor]:
        """
        从 data 里找一个 block。
        支持：
          1) data["state.left_arm"]
          2) data["state"]["left_arm"]
          3) alias, 例如 left_pose -> left_arm
        """
        canonical_to_candidates = self._merged_canonical_to_candidates()
        candidate_suffixes = canonical_to_candidates.get(canonical_name, [canonical_name])
        # 先查扁平 key：data["state.left_arm"]
        for suffix in candidate_suffixes:
            flat_key = f"{prefix}.{suffix}"
            if flat_key in data:
                return self._to_tensor_2d(data[flat_key], horizon=horizon, name=flat_key)
        # 再查嵌套 key：data["state"]["left_arm"]
        if prefix in data and isinstance(data[prefix], dict):
            subdict = data[prefix]
            for suffix in candidate_suffixes:
                if suffix in subdict:
                    return self._to_tensor_2d(subdict[suffix], horizon=horizon, name=f"{prefix}.{suffix}")
        return None
    def _assemble_from_blocks(
        self,
        data: dict,
        prefix: str,                 # "state" or "action"
        schema: List[Tuple[str, int]],
        horizon: int,
        max_dim: int,
    ):
        schema_slices, schema_total_dim = self._build_schema_slices(schema)
        if max_dim < schema_total_dim:
            raise ValueError(
                f"{prefix}: max_dim={max_dim} is smaller than schema_total_dim={schema_total_dim}"
            )
        out = torch.zeros((horizon, max_dim), dtype=torch.float32)
        mask = torch.zeros((horizon, max_dim), dtype=torch.bool)
        for canonical_name, dst_dim in schema:
            dst_slice = schema_slices[canonical_name]
            block = self._get_block_from_data(
                data=data,
                prefix=prefix,
                canonical_name=canonical_name,
                horizon=horizon,
            )
            if block is None:
                continue
            src_dim = block.shape[1]
            copy_dim = min(src_dim, dst_dim)
            out[:, dst_slice.start: dst_slice.start + copy_dim] = block[:, :copy_dim]
            mask[:, dst_slice.start: dst_slice.start + copy_dim] = True
            if canonical_name == "waist" and src_dim == 2 and dst_dim == 3:
                continue
            # 如果原始维度和 schema 不一致，给出提示
            if src_dim != dst_dim:
                print(
                    f"[GR00TTransform][{prefix}] block '{canonical_name}' dim mismatch: "
                    f"input={src_dim}, schema={dst_dim}, copy_dim={copy_dim}"
                )
        return out, mask
    # ============================================================
    # prepare state / action
    # ============================================================
    def _prepare_state(self, data: dict):
        """
        不再依赖 data["state"] 是预先 concat 好的矩阵。
        直接从：
            data["state.left_arm"], data["state.right_arm"], ...
        或：
            data["state"]["left_arm"], ...
        中组装。
        """
        state, state_mask = self._assemble_from_blocks(
            data=data,
            prefix="state",
            schema=UNIFIED_STATE_SCHEMA,
            horizon=self.state_horizon,
            max_dim=self.max_state_dim,
        )
        return state, state_mask, self.state_horizon
    def _prepare_action(self, data: dict):
        """
        不再依赖 data["action"] 是预先 concat 好的矩阵。
        """
        action, action_mask = self._assemble_from_blocks(
            data=data,
            prefix="action",
            schema=UNIFIED_ACTION_SCHEMA,
            horizon=self.action_horizon,
            max_dim=self.max_action_dim,
        )
        return action, action_mask, self.action_horizon
    # ============================================================
    # main apply
    # ============================================================
    def apply_single(self, data: dict) -> dict:
        transformed_data = data.copy()
        vlm_outputs = {}
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state.numpy()
        transformed_data["state_mask"] = state_mask.numpy()
        if self.training:
            transformed_data["segmentation_target"] = np.zeros((2,), dtype=np.float32)
            transformed_data["segmentation_target_mask"] = np.zeros((1,), dtype=bool)
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            action, action_mask, _ = self._prepare_action(data)
            transformed_data["action"] = action.numpy()
            transformed_data["action_mask"] = action_mask.numpy()
        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v
        transformed_data["embodiment_id"] = self.get_embodiment_tag()
        if self.training:
            assert transformed_data["action"].shape == transformed_data["action_mask"].shape, (
                transformed_data["action"].shape,
                transformed_data["action_mask"].shape,
            )
        return transformed_data
    def apply_batch(self, data: dict, batch_size: int) -> dict:
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed)
    def apply(self, data: dict) -> dict:
        return self.apply_single(data)
    def unapply(self, data: dict) -> dict:
        return data
    def __call__(self, data: dict) -> dict:
        return self.apply(data)
