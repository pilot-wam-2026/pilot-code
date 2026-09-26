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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import os
import hashlib
import json, torch
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag, EMBODIMENT_TAG_MAPPING
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform

from functools import partial
from typing import Tuple, List
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t_old.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_SUBTASKS_FILENAME = "meta/subtasks.jsonl"
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


# def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
#     """Calculate the dataset statistics of all columns for a list of parquet files."""
#     # Dataset statistics
#     all_low_dim_data_list = []
#     # Collect all the data
#     # parquet_paths = parquet_paths[:3]
#     for parquet_path in tqdm(
#         sorted(list(parquet_paths)),
#         desc="Collecting all parquet files...",
#     ):
#         # Load the parquet file
#         parquet_data = pd.read_parquet(parquet_path)
#         parquet_data = parquet_data
#         all_low_dim_data_list.append(parquet_data)

#     all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
#     # Compute dataset statistics
#     dataset_statistics = {}
#     for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
#         print(le_modality)
#         if "task_info" in le_modality:
#             continue
#         print(f"Computing statistics for {le_modality}...")
#         # 检查数据是否为空或无效
#         try:
#             np_data = np.vstack(
#                 [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
#             )
#         except Exception as e:
#             print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
#             continue

#         dataset_statistics[le_modality] = {
#             "mean": np.mean(np_data, axis=0).tolist(),
#             "std": np.std(np_data, axis=0).tolist(),
#             "min": np.min(np_data, axis=0).tolist(),
#             "max": np.max(np_data, axis=0).tolist(),
#             "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
#             "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
#         }
#     return dataset_statistics


def _read_parquet_batch(paths: list[Path]) -> pd.DataFrame:
    """Read a batch of parquet files and concatenate. Used for parallel loading."""
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()


def _merge_rank_stats(rank_stats_list: list[dict]) -> dict:
    """Merge per-rank stats (n, sum, sum_sq, min, max, q01, q99) into one dataset_statistics dict.
    Key order follows the first rank's keys (same as pandas column order), then any extra keys from other ranks sorted.
    """
    if not rank_stats_list:
        return {}
    # Preserve order: first rank's keys (matches single-rank / pandas column order), then sorted union of rest
    order = list(rank_stats_list[0].keys())
    seen = set(order)
    for r in rank_stats_list[1:]:
        for k in r:
            if k not in seen:
                seen.add(k)
                order.append(k)
    out = {}
    for mod in order:
        # Only merge ranks that have this modality
        ranks_with_mod = [r for r in rank_stats_list if mod in r]
        if not ranks_with_mod:
            continue
        n_total = sum(r[mod]["n"] for r in ranks_with_mod)
        if n_total == 0:
            continue
        sum_ = sum(r[mod]["sum"] for r in ranks_with_mod)
        sum_sq = sum(r[mod]["sum_sq"] for r in ranks_with_mod)
        mins = np.array([r[mod]["min"] for r in ranks_with_mod])
        maxs = np.array([r[mod]["max"] for r in ranks_with_mod])
        q01s = np.array([r[mod]["q01"] for r in ranks_with_mod])
        q99s = np.array([r[mod]["q99"] for r in ranks_with_mod])
        ns = np.array([r[mod]["n"] for r in ranks_with_mod], dtype=np.float64)

        mean = sum_ / n_total
        var = np.maximum((sum_sq / n_total) - (mean**2), 0.0)
        std = np.sqrt(var)
        q01 = np.average(q01s, axis=0, weights=ns).tolist()
        q99 = np.average(q99s, axis=0, weights=ns).tolist()

        out[mod] = {
            "mean": (mean if isinstance(mean, np.ndarray) else np.array(mean)).tolist(),
            "std": (std if isinstance(std, np.ndarray) else np.array(std)).tolist(),
            "min": np.min(mins, axis=0).tolist(),
            "max": np.max(maxs, axis=0).tolist(),
            "q01": q01,
            "q99": q99,
        }
    return out


def _stack_column_to_array(series: pd.Series) -> np.ndarray:
    """Stack series of array-like to (N, dim) float64."""
    n = len(series)
    it = iter(series)
    first = np.asarray(next(it), dtype=np.float64)
    d = first.size
    out = np.empty((n, d), dtype=np.float64)
    out[0] = first
    for i in range(1, n):
        out[i] = np.asarray(next(it), dtype=np.float64)
    return out


def _compute_local_stats(
    sorted_paths: list[Path],
    batch_size: int,
    n_workers: int,
    parallel_modalities: int,
    max_paths: int | None,
    tqdm_desc: str = "Loading & streaming stats...",
) -> dict:
    """Compute per-modality (n, sum, sum_sq, min, max, q01, q99) on the given paths. Used by single-rank and per-rank in multi-rank."""
    if not sorted_paths:
        return {}
    paths = sorted_paths[:max_paths] if max_paths is not None else sorted_paths
    sample_df = pd.read_parquet(paths[0])
    modalities = [c for c in sample_df.columns if "task_info" not in c]
    del sample_df

    running = {}
    modality_arrays = defaultdict(list)
    n_batches = (len(paths) + batch_size - 1) // batch_size

    def _read_batch(batch_paths: list, executor: ThreadPoolExecutor | None) -> pd.DataFrame:
        if n_workers <= 1 or executor is None:
            return _read_parquet_batch(batch_paths)
        dfs = list(executor.map(pd.read_parquet, batch_paths))
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

    executor_ctx = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for batch_start in tqdm(range(0, len(paths), batch_size), desc=tqdm_desc, total=n_batches):
            batch_paths = paths[batch_start:batch_start + batch_size]
            batch_df = _read_batch(batch_paths, executor_ctx)

            for le_modality in modalities:
                if le_modality not in batch_df.columns:
                    continue
                try:
                    np_data = _stack_column_to_array(batch_df[le_modality])
                except Exception as e:
                    print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
                    continue
                n = np_data.shape[0]
                s = np_data.sum(axis=0)
                s_sq = (np_data.astype(np.float64)**2).sum(axis=0)
                mn, mx = np_data.min(axis=0), np_data.max(axis=0)
                if le_modality not in running:
                    running[le_modality] = {"n": n, "sum": s, "sum_sq": s_sq, "min": mn.copy(), "max": mx.copy()}
                else:
                    r = running[le_modality]
                    r["n"] += n
                    r["sum"] += s
                    r["sum_sq"] += s_sq
                    r["min"] = np.minimum(r["min"], mn)
                    r["max"] = np.maximum(r["max"], mx)
                modality_arrays[le_modality].append(np_data)
            del batch_df
    finally:
        if executor_ctx is not None:
            executor_ctx.shutdown(wait=True)

    # Build local stats: n, sum, sum_sq, min, max, q01, q99 (for merge)
    local = {}
    for le_modality, r in running.items():
        if r["n"] == 0:
            continue
        arrs = modality_arrays.get(le_modality, [])
        if arrs:
            full = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
            q01 = np.quantile(full, 0.01, axis=0)
            q99 = np.quantile(full, 0.99, axis=0)
        else:
            q01 = r["min"].copy()
            q99 = r["max"].copy()
        local[le_modality] = {
            "n": r["n"],
            "sum": r["sum"],
            "sum_sq": r["sum_sq"],
            "min": r["min"],
            "max": r["max"],
            "q01": q01,
            "q99": q99,
        }
    return local


def calculate_dataset_statistics(
    parquet_paths: list[Path],
    batch_size: int = 3000,
    n_workers: int = 16,
    parallel_modalities: int = 0,
    stats_output_path: Path | str | None = None,
) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Optimizations:
    - Single pass, stream batch-by-batch; merge mean/std/min/max incrementally.
    - One shared ThreadPoolExecutor per run; parallel parquet read per batch (n_workers).
    - Multi-rank: when torch.distributed is initialized and stats_output_path is set,
      paths are split across ranks, each rank computes on its subset, then rank 0 merges
      and writes; all ranks load the result. Speeds up proportionally to world_size.
    """
    sorted_paths = sorted(list(parquet_paths))

    if not sorted_paths:
        return {}

    dist = torch.distributed
    use_multi_rank = (dist.is_initialized() and dist.get_world_size() > 1 and stats_output_path is not None)
    stats_output_path = Path(stats_output_path) if stats_output_path else None

    if use_multi_rank:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Interleaved split for better load balance when file sizes vary
        my_paths = [p for i, p in enumerate(sorted_paths) if i % world_size == rank]
        if not my_paths:
            local_stats = {}
        else:
            local_stats = _compute_local_stats(
                my_paths,
                batch_size=batch_size,
                n_workers=n_workers,
                parallel_modalities=0,
                max_paths=None,
                tqdm_desc=f"Rank {rank} stats",
            )
        # Serialize for rank 0 to gather (n, sum, sum_sq, min, max, q01, q99 are numpy/list)
        tmp_dir = stats_output_path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rank_pkl = tmp_dir / f".stats_rank_{rank}.pkl"
        with open(rank_pkl, "wb") as f:
            pickle.dump(local_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

        dist.barrier()

        if rank == 0:
            all_stats = []
            for r in range(world_size):
                pkl_path = tmp_dir / f".stats_rank_{r}.pkl"
                with open(pkl_path, "rb") as f:
                    all_stats.append(pickle.load(f))
                try:
                    pkl_path.unlink()
                except OSError:
                    pass
            dataset_statistics = _merge_rank_stats(all_stats)
            with open(stats_output_path, "w", encoding="utf-8") as f:
                json.dump(dataset_statistics, f, indent=4, ensure_ascii=False)

        dist.barrier()

        with open(stats_output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Single process or no stats_output_path
    local = _compute_local_stats(
        sorted_paths,
        batch_size=batch_size,
        n_workers=n_workers,
        parallel_modalities=parallel_modalities,
        max_paths=None,
    )
    return _merge_rank_stats([local])


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version =  self.data_cfg.get("lerobot_version", "v2.0") #self._indict_lerobot_version(**kwargs)

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
            self.tag_index = EMBODIMENT_TAG_MAPPING[self.tag]
        else:
            self.tag = embodiment_tag
            self.tag_index = EMBODIMENT_TAG_MAPPING[EmbodimentTag.NEW_EMBODIMENT.value]


        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self._subtasks = self._get_subtasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")


        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> np.ndarray:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return:
                np.array([
                    [0, 0], [0, 1], [0, 2],
                    [1, 0], [1, 1],
                    [2, 0], [2, 1], [2, 2], [2, 3],
                ], dtype=np.int64)
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    @property
    def subtasks(self) -> dict:
        """The subtasks for the dataset."""
        return self._subtasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            print(f"Loading stats from {stats_path}")
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)

        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}",stats_path)
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # assert False
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            parquet_files_filtered = []
            #  parquet_files[0].name = "episode_033675.parquet" is broken file
            for pf in parquet_files:
                if "episode_033675.parquet" in pf.name:
                    continue
                parquet_files_filtered.append(pf)

            stats_path.parent.mkdir(parents=True, exist_ok=True)
            # Multi-rank: all ranks call; paths are split by rank, rank 0 merges and writes.
            # Single-process: compute then write here.
            if torch.distributed.is_initialized():
                le_statistics = calculate_dataset_statistics(
                    parquet_files_filtered,
                    stats_output_path=stats_path,
                )
            else:
                le_statistics = calculate_dataset_statistics(parquet_files_filtered)
                tmp_path = stats_path.with_suffix(stats_path.suffix + f".tmp.{os.getpid()}")
                with open(tmp_path, "w") as f:
                    json.dump(le_statistics, f, indent=4)
                os.replace(tmp_path, stats_path)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    # TODO auto map key? just map to file_path and file_from_index
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/observation.images.wrist/from_timestamp": episode["videos/observation.images.wrist/from_timestamp"],
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # 这里应该可以直接读取到 save index 信息
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> np.ndarray:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            np.ndarray: Shape [N, 2], where each row is [trajectory_id, base_index].
        """
        # Create a hash key based on configuration to ensure cache validity
        config_key = self._get_steps_config_key()

        # Create a unique filename based on config_key
        steps_filename = f"steps_{config_key}.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename

        # Try to load cached steps first
        try:
            if steps_path.exists():
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("config_key") != config_key:
                    raise KeyError("cached steps config_key mismatch")
                cached_steps = cached_data["steps"]
                if isinstance(cached_steps, np.ndarray):
                    return cached_steps.astype(np.int64, copy=False)
                return np.asarray(cached_steps, dtype=np.int64)
        except (EOFError, FileNotFoundError, pickle.PickleError, KeyError) as e:
            print(f"Failed to load cached steps: {e}")
            print("Computing steps from scratch...")

        # Compute steps using single process
        all_steps = self._get_all_steps_single_process()

        # Cache the computed steps with unique filename
        try:
            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }

            # Ensure the meta directory exists
            steps_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_steps_path = steps_path.with_suffix(f"{steps_path.suffix}.tmp.{os.getpid()}")
            with open(tmp_steps_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_steps_path, steps_path)
            print(f"Cached steps saved to {steps_path}")
        except Exception as e:
            print(f"Failed to cache steps: {e}")

        return all_steps

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        future_video_delta_indices = None
        if self.data_cfg is not None:
            future_video_delta_indices = self.data_cfg.get("future_video_delta_indices", None)
        config_dict = {
            "steps_filter_version": 2,
            "positive_offset_policy": "pad_or_clamp",
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "action_delta_indices": {
                key: self.delta_indices[key].tolist()
                for key in self.modality_keys.get("action", [])
            },
            "future_video_delta_indices": (
                list(future_video_delta_indices)
                if future_video_delta_indices is not None
                else None
            ),
        }
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #

    def _min_valid_index_offset(self) -> int:
        """Return the minimum relative offset that must be valid before sampling.
        Positive offsets are intentionally not treated as hard bounds: action/state
        chunks already pad at the end, and video retrieval clamps to the last frame.
        Keeping those samples lets the final action chunk of each trajectory train.
        """
        min_offset = 0
        for action_key in self.modality_keys.get("action", []):
            action_offsets = self.delta_indices[action_key]
            if len(action_offsets) > 0:
                min_offset = min(min_offset, int(np.min(action_offsets)))

        future_video_delta_indices = None
        if self.data_cfg is not None:
            future_video_delta_indices = self.data_cfg.get("future_video_delta_indices", None)
        if future_video_delta_indices is not None:
            future_offsets = [int(offset) for offset in future_video_delta_indices]
            if future_offsets:
                min_offset = min(min_offset, min(future_offsets))

        return min_offset


    def _get_all_steps_single_process(self) -> np.ndarray:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        skipped_steps = 0
        min_required_offset = self._min_valid_index_offset()

        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)

                trajectory_skipped = False

                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if not trajectory_skipped:
                processed_trajectories += 1

            for base_index in range(trajectory_length):
                if base_index + min_required_offset < 0:
                    skipped_steps += 1
                    continue
                all_steps.append((trajectory_id, base_index))

        # Print summary statistics
        print(f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories")
        if skipped_steps:
            print(
                f"Skipped {skipped_steps} front boundary steps requiring "
                f"min offset {min_required_offset} without padding"
            )
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")

        if not all_steps:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(all_steps, dtype=np.int64)

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])

        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy()
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _process_task_text(self, task_text: str) -> str:
        """Process task text to extract English part after @ and validate.
        
        Args:
            task_text: Original task text, may contain Chinese and English separated by @
            
        Returns:
            Processed task text (English part only), or None if invalid
        """
        if not isinstance(task_text, str):
            return None

        # Extract English part after @
        if "@" in task_text:
            parts = task_text.split("@", 1)
            if len(parts) > 1:
                english_part = parts[1].strip()
            else:
                english_part = task_text.strip()
        else:
            # If no @, use the whole text
            english_part = task_text.strip()

        # Filter out invalid instructions
        invalid_keywords = ["null", "unqualified", "qualified"]
        if english_part.lower() in invalid_keywords:
            return None

        # Return empty string if result is empty
        if not english_part:
            return None

        return english_part

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            # Process task text: extract English part and filter invalid ones
            df["task"] = df["task"].apply(self._process_task_text)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # 把索引变成一列，列名通常为 'index'
            df = df.rename(columns={'index': 'task'})  # 把 'index' 列重命名为 'task'
            df = df[['task_index', 'task']]  # 调整列顺序
            return df

    def _get_subtasks(self) -> dict:
        """Get the subtasks for the dataset."""
        subtasks_path = self.dataset_path / LE_ROBOT_SUBTASKS_FILENAME
        # check if the file exists, if not, return None
        if not subtasks_path.exists():
            # check if subtask in episodes.json
            episodes_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(episodes_path, "r") as f:
                episodes = [json.loads(line) for line in f]
            if "subtask" in episodes[0]:
                print(f"Subtask not found in {subtasks_path}, extracting from {episodes_path}")
                self._extract_subtasks(episodes_path, subtasks_path)
            else:
                return None
        with open(subtasks_path, "r") as f:
            subtasks = [json.loads(line) for line in f]
        df = pd.DataFrame(subtasks)
        df["task"] = df["task"].apply(self._process_task_text)
        return df.set_index("subtask_index")

    def _extract_subtasks(self, input_path, output_path):
        """Extract all subtasks from episodes.jsonl and write them to output_path in subtask_index order."""
        all_subtasks = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {idx + 1} 行解析失败，已跳过 - {e}")
                    continue

                subtask_list = episode.get('subtask', [])
                for st in subtask_list:
                    subtask_index = st.get('subtask_index')
                    task = st.get('task', '')
                    all_subtasks.append({"subtask_index": subtask_index, "task": task})

        # 按 subtask_index 排序（应对可能的乱序）
        all_subtasks.sort(key=lambda x: x['subtask_index'])

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_subtasks:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"已从 {input_path} 提取 {len(all_subtasks)} 条 subtask，写入 {output_path}")


    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        trajectory_id, base_index = int(trajectory_id), int(base_index)
        data = self.get_step_data(trajectory_id, base_index)

        # Process all video keys dynamically
        images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]

            # Apply image cropping if enabled and the video key is base_view
            # Note: crop_obs_camera functionality has been removed

            image = Image.fromarray(image).resize((224, 224))
            images.append(image)

        # Get language and action data
        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1)

        return dict(action=action, image=images, language=language)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # TODO @contributor The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                return pd.read_parquet(parquet_path)
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else: #TODO check detail later
            chunk_index = self.get_episode_chunk(trajectory_id)

            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)


            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()

            # fix timestamp from epis index to file index
            from_timestamp = self.trajectory_ids_to_metadata[trajectory_id]["videos/observation.images.wrist/from_timestamp"]
            episode_data["timestamp"] = episode_data["timestamp"] + from_timestamp

            return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend, # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)): #
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]] # TODO check v2.0
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        # Get language instructions and filter out None values (invalid instructions)
        if original_key == "subtask_index":
            language_list = self.subtasks.loc[task_indices]["task"].tolist()
        else:
            language_list = self.tasks.loc[task_indices]["task"].tolist()
        # Filter out None values (invalid instructions that were filtered in _process_task_text)
        language_list = [lang for lang in language_list if lang is not None]

        return language_list

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys()
                )
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed

                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats

def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []

    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()

            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True

    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None
        self._retry_log_counts = defaultdict(int)

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        print(f"Dataset sampling weights: {self._dataset_sampling_weights}")
        print(f"balance_dataset_weights: {self.balance_dataset_weights}")

        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        print(f"After balance_dataset_weights: {self._dataset_sampling_weights}")

        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum
        print(f'After normalize weights: {self._dataset_sampling_weights}')

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        # trajectory_index = rng.choice(
        #     len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        # )
        # trajectory_id = dataset.trajectory_ids[trajectory_index]

        # # Sample step
        # base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        # return dataset, trajectory_id, base_index
        single_step_index = rng.choice(len(dataset.all_steps))
        trajectory_id, base_index = dataset.all_steps[single_step_index]
        trajectory_id, base_index = int(trajectory_id), int(base_index)
        return dataset, trajectory_id, base_index

    def filter_zero_action(self, dataset, raw_data, step):
        # raw_data keys are always "action.xxx" format; collect and concatenate directly
        raw_action_list = [raw_data[action_key] for action_key in dataset.modality_keys["action"]]
        if isinstance(raw_action_list[0], torch.Tensor):
            raw_action = (raw_action_list[0] if len(raw_action_list) == 1 else torch.cat(raw_action_list, dim=1))
        else:
            raw_action = (raw_action_list[0] if len(raw_action_list) == 1 else np.concatenate(raw_action_list, axis=1))
        # Only check non-padded frames: delta actions at trajectory boundaries are legitimately zero-padded,
        # so filtering on padded frames causes valid end-of-trajectory samples to be incorrectly discarded.
        #
        # Use curr_traj_data length directly — get_step_data already loaded it, so this is O(1)
        # rather than the O(N) np.where scan inside get_trajectory_index.
        trajectory_length = len(dataset.curr_traj_data)
        first_action_key = dataset.modality_keys["action"][0]
        action_delta_indices = dataset.delta_indices[first_action_key]
        valid_frame_mask = ((step + action_delta_indices >= 0) & (step + action_delta_indices < trajectory_length))
        if valid_frame_mask.any():
            if isinstance(raw_action, torch.Tensor):
                valid_mask_t = torch.from_numpy(valid_frame_mask)
                has_zero_action = bool(torch.any(torch.all(raw_action[valid_mask_t] == 0, dim=-1)))
            else:
                has_zero_action = bool(np.any(np.all(raw_action[valid_frame_mask] == 0, axis=-1)))
            if has_zero_action:
                raise ValueError(f"Found raw action with all dimensions equal to zero in non-padded action chunk "
                                 f"for tag {dataset.tag_index}")

    def _log_retry_failure(self, exc: Exception, index: int, attempt: int, max_retries: int) -> None:
        """Throttle noisy retry logs while keeping failure visibility."""
        error_key = str(exc)
        self._retry_log_counts[error_key] += 1
        seen = self._retry_log_counts[error_key]

        should_log = seen <= 3 or seen % 100 == 0 or attempt == max_retries - 1
        if should_log:
            print(
                f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {exc} "
                f"(seen {seen} times)"
            )
            if seen == 3:
                print("Suppressing repeated retry logs for this error; will report every 100 occurrences.")

    def _build_language_instruction(self, dataset, data: dict, trajectory_length: int) -> str:
        """Build hierarchical language like: Task: ... Subtask: ... Speed: ..."""
        task = None
        subtask = None
        speed = int(round(trajectory_length / 500) * 500)

        language_texts = []
        for language_key in dataset.modality_keys["language"]:
            language_values = data.get(language_key)
            if language_values is None or len(language_values) == 0:
                continue

            language_text = str(language_values[0]).strip()
            if not language_text:
                continue

            language_texts.append(language_text)

        if len(language_texts) == 1:
            subtask = language_texts[0]
        elif len(language_texts) >= 2:
            task = language_texts[0]
            subtask = language_texts[1]

        language_parts = []
        if task:
            language_parts.append(("Task", task))
        if subtask:
            language_parts.append(("Subtask", subtask))

        # language_parts.append(("Speed", str(speed)))
        # language_parts.append(("Mistake", "false"))
        # language_parts.append(("Control Mode", "ee"))

        return " ".join(
            f"{label}: {text.rstrip('.')}."
            for label, text in language_parts
        )

    @staticmethod
    def _concat_view_images(images, view_size=(224, 224)):
        """Concatenate camera views into one panorama frame."""
        if not images:
            return []

        images = list(images)
        resized = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            resized.append(image.convert("RGB").resize(view_size))

        panorama = Image.new("RGB", (sum(image.width for image in resized), max(image.height for image in resized)))
        x_offset = 0
        for image in resized:
            panorama.paste(image, (x_offset, 0))
            x_offset += image.width
        return panorama

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True: # @DUG
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                self.filter_zero_action(dataset, raw_data, step)
                data = dataset.transforms(raw_data)

                # Process video: check if ConcatTransform has been applied
                prim_images = []
                wrist_views = []

                if "video" in data and not any(k.startswith("video.") for k in data.keys()):
                    # Videos have been concatenated by ConcatTransform
                    # data["video"] shape: [T, V, H, W, C] where V is number of views
                    video_data = data["video"]
                    num_views = video_data.shape[1]  # Number of video views

                    for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
                        # Extract the view at view_idx
                        image = video_data[0, view_idx, :, :, :]  # Get first frame [H, W, C]
                        image = Image.fromarray(image)

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)
                else:
                    # Videos are still individual keys (no ConcatTransform applied)
                    for video_key in dataset.modality_keys["video"]:
                        image = data[video_key][0]  # Get first frame
                        image = Image.fromarray(image).resize((224, 224))

                        if "wrist" not in video_key and "hand" not in video_key:
                            prim_images.append(image)
                        else:
                            wrist_views.append(image)

                all_images = prim_images + wrist_views
                if len(all_images) > 1:
                    all_images = [self._concat_view_images(all_images)]

                # Future visual supervision. By default this is one image after
                # the action chunk; DiT4DiT-style training can request a full
                # future sequence through data_cfg.future_video_delta_indices.
                future_images = []
                try:
                    first_action_key = dataset.modality_keys["action"][0]
                    future_video_delta_indices = None
                    if self.data_cfg is not None:
                        future_video_delta_indices = self.data_cfg.get("future_video_delta_indices", None)
                    if future_video_delta_indices is None:
                        future_offsets = [int(np.max(dataset.delta_indices[first_action_key])) + 1]
                    else:
                        future_offsets = [int(offset) for offset in future_video_delta_indices]
                    future_view_mode = "all"
                    if self.data_cfg is not None:
                        future_view_mode = str(
                            self.data_cfg.get(
                                "future_video_view_mode",
                                "concat"
                                if len(dataset.modality_keys["video"]) > 1
                                else ("primary" if future_video_delta_indices is not None else "all"),
                            )
                        ).lower()
                    future_prim_images = []
                    future_wrist_views = []

                    if future_view_mode == "concat":
                        for future_offset in future_offsets:
                            future_view_images = []
                            for video_key in dataset.modality_keys["video"]:
                                future_video = dataset.get_video(trajectory_id, video_key, step + future_offset)
                                future_view_images.append(Image.fromarray(future_video[0]))
                            future_prim_images.append(self._concat_view_images(future_view_images))
                    elif "video" in data and not any(k.startswith("video.") for k in data.keys()):
                        for view_idx, video_key in enumerate(dataset.modality_keys["video"]):
                            if future_view_mode == "primary" and ("wrist" in video_key or "hand" in video_key):
                                continue
                            if future_view_mode == "primary" and future_prim_images:
                                continue
                            for future_offset in future_offsets:
                                future_video = dataset.get_video(trajectory_id, video_key, step + future_offset)
                                future_image = Image.fromarray(future_video[0])
                                if "wrist" not in video_key and "hand" not in video_key:
                                    future_prim_images.append(future_image)
                                else:
                                    future_wrist_views.append(future_image)
                    else:
                        for video_key in dataset.modality_keys["video"]:
                            if future_view_mode == "primary" and ("wrist" in video_key or "hand" in video_key):
                                continue
                            if future_view_mode == "primary" and future_prim_images:
                                continue
                            for future_offset in future_offsets:
                                future_video = dataset.get_video(trajectory_id, video_key, step + future_offset)
                                future_image = Image.fromarray(future_video[0]).resize((224, 224))
                                if "wrist" not in video_key and "hand" not in video_key:
                                    future_prim_images.append(future_image)
                                else:
                                    future_wrist_views.append(future_image)
                    future_images = future_prim_images + future_wrist_views
                except Exception:
                    if future_video_delta_indices is not None:
                        raise
                    future_images = all_images

                # Get language data. Format as: Task: ... Subtask: ... Speed: ... Mistake: ... Control Mode: ...
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                trajectory_length = dataset.trajectory_lengths[trajectory_index]
                language = self._build_language_instruction(dataset, data, trajectory_length)
                # print(dataset.modality_keys["language"], language)

                # Get action data: check if ConcatTransform has been applied
                # if "action" in data and not any(k.startswith("action.") for k in data.keys()):
                # Actions have been concatenated
                action = data["action"].astype(np.float16)
                # else:
                #     # Individual action keys
                #     action = []
                #     for action_key in dataset.modality_keys["action"]:
                #         action.append(data[action_key])
                #     action = np.concatenate(action, axis=1).astype(np.float16)

                # if "action_mask" in data and not any(k.startswith("action.") for k in data.keys()):
                # Actions have been concatenated
                action_mask = data["action_mask"].astype(np.float16)
                # else:
                #     # Individual action keys
                #     action_mask = []
                #     for action_key in dataset.modality_keys["action_mask"]:
                #         action_mask.append(data[action_key])
                #     action_mask = np.concatenate(action_mask, axis=1).astype(np.float16)
                # Get state data if needed
                state = None
                if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
                    if "state" in data:
                        # Prefer the transformed state tensor. GR00TTransform keeps
                        # raw state.* keys around, but data["state"] is the padded
                        # unified state consumed by action heads.
                        state = data["state"].astype(np.float16)
                    else:
                        # Individual state keys
                        state = []
                        for state_key in dataset.modality_keys["state"]:
                            state.append(data[state_key])
                        state = np.concatenate(state, axis=1).astype(np.float16)

                    return dict(
                        action=action,
                        image=all_images,
                        future_image=future_images,
                        lang=language,
                        state=state,
                        action_mask=action_mask,
                        embodiment_tag=dataset.tag_index,
                    )

                return dict(
                    action=action,
                    image=all_images,
                    future_image=future_images,
                    lang=language,
                    action_mask=action_mask,
                    embodiment_tag=dataset.tag_index,
                )

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    self._log_retry_failure(e, index, attempt, max_retries)
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys - use union of all keys from all datasets
        all_modality_keys = set()
        for task_stats in per_task_stats:
            all_modality_keys.update(task_stats.keys())

        # For each modality key, only include datasets that have this key
        for modality in all_modality_keys:
            # Find which datasets have this modality
            datasets_with_modality = []
            weights_for_modality = []

            for task_idx, task_stats in enumerate(per_task_stats):
                if modality in task_stats:
                    datasets_with_modality.append(task_idx)
                    weights_for_modality.append(normalized_weights[task_idx])

            # Skip if no dataset has this modality (shouldn't happen)
            if not datasets_with_modality:
                continue

            # Re-normalize weights for this modality
            weights_for_modality = np.array(weights_for_modality)
            weights_for_modality = weights_for_modality / weights_for_modality.sum()

            # Number of dimensions (from first dataset that has this modality)
            num_dims = len(per_task_stats[datasets_with_modality[0]][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for i, task_idx in enumerate(datasets_with_modality):
                w_i = weights_for_modality[i]
                task_stats = per_task_stats[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return

        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue

            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }

            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data

            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
