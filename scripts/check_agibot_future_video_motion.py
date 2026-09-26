#!/usr/bin/env python3
"""Check whether AgiBot GT future video frames are static.

This script reads LeRobot samples through the same starVLA dataset code used by
cosmos25_perceiver_pretrain.sh, then compares frames at:

    t, t + future_video_delta_indices[0], ..., t + future_video_delta_indices[-1]

It reports pixel-difference metrics per camera view and writes contact sheets so
you can visually confirm whether the GT future frames actually move.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import av
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


DEFAULT_CONFIG_YAML = REPO_ROOT / "examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml"
DEFAULT_DATA_ROOT_DIR = "/path/to/workspace/"
DEFAULT_DATA_MIX = "sq_agi-beta_robocasa"
DEFAULT_FUTURE_OFFSETS = [2, 4, 6, 8, 10, 12, 14, 16]
DEFAULT_ROBOT_TYPES = ["agibot_beta", "agibot_genie1"]


def parse_int_list(value: str) -> list[int]:
    text = value.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        return [int(item) for item in parsed]
    return [int(item) for item in text.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", default=str(DEFAULT_CONFIG_YAML))
    parser.add_argument("--data-root-dir", default=DEFAULT_DATA_ROOT_DIR)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument(
        "--compare-video-backends",
        default="",
        help="Comma-separated extra backends to read with the same timestamps, e.g. decord,torchvision_av.",
    )
    parser.add_argument("--future-video-delta-indices", default=json.dumps(DEFAULT_FUTURE_OFFSETS))
    parser.add_argument(
        "--robot-types",
        default=",".join(DEFAULT_ROBOT_TYPES),
        help="Comma-separated robot types to inspect from the mixture.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/agibot_future_video_debug")
    parser.add_argument(
        "--changed-pixel-threshold",
        type=float,
        default=3.0,
        help="A pixel is treated as changed if any RGB channel differs by more than this value.",
    )
    parser.add_argument(
        "--frozen-ratio-threshold",
        type=float,
        default=0.001,
        help="Frame pairs below this changed-pixel ratio are flagged as frozen.",
    )
    parser.add_argument("--max-contact-sheets", type=int, default=64)
    parser.add_argument("--skip-video-timestamp-map", action="store_true")
    parser.add_argument(
        "--show-all-frames",
        action="store_true",
        help="Save all sampled frames in contact sheets. Default saves only first and last frames for easier inspection.",
    )
    return parser.parse_args()


def image_to_uint8_rgb(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image or THWC video with T=1, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA image, got shape {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def resize_for_metrics(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] == b.shape[:2]:
        return a, b
    image = Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.Resampling.BILINEAR)
    return a, np.asarray(image, dtype=np.uint8)


def frame_diff_metrics(a: np.ndarray, b: np.ndarray, changed_threshold: float) -> dict[str, float]:
    a, b = resize_for_metrics(a, b)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    changed = np.any(diff > changed_threshold, axis=-1)
    return {
        "mae": float(np.mean(diff)),
        "rmse": float(math.sqrt(mse)),
        "psnr": psnr,
        "max_abs": float(np.max(diff)),
        "changed_pixel_ratio": float(np.mean(changed)),
    }


def concat_raw_action(raw_data: dict[str, Any], action_keys: list[str]) -> np.ndarray | None:
    chunks = []
    for key in action_keys:
        if key not in raw_data:
            continue
        value = np.asarray(raw_data[key])
        if value.ndim == 1:
            value = value[:, None]
        chunks.append(value)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=1).astype(np.float32)


def get_video_frame_timestamps(video_path: Path) -> np.ndarray:
    container = av.open(video_path.as_posix(), options={"threads": "1"})
    stream = container.streams.video[0]
    timestamps = []
    try:
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            timestamps.append(float(frame.pts * stream.time_base))
    finally:
        container.close()
    if not timestamps:
        raise ValueError(f"No decodable frame timestamps in {video_path}")
    return np.asarray(timestamps, dtype=np.float64)


def get_video_frames_by_indices_pyav(video_path: Path, indices: np.ndarray) -> list[np.ndarray]:
    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        requested = requested.reshape(-1)
    if len(requested) == 0:
        return []

    order = np.argsort(requested)
    sorted_indices = requested[order]
    frames_sorted: list[np.ndarray | None] = [None] * len(sorted_indices)
    container = av.open(video_path.as_posix(), options={"threads": "1"})
    try:
        want = 0
        for frame_index, frame in enumerate(container.decode(video=0)):
            while want < len(sorted_indices) and sorted_indices[want] < frame_index:
                want += 1
            if want >= len(sorted_indices):
                break
            if frame_index == sorted_indices[want]:
                frame_array = frame.to_ndarray(format="rgb24")
                while want < len(sorted_indices) and sorted_indices[want] == frame_index:
                    frames_sorted[want] = frame_array
                    want += 1
                if want >= len(sorted_indices):
                    break
    finally:
        container.close()

    if any(frame is None for frame in frames_sorted):
        missing = [int(sorted_indices[idx]) for idx, frame in enumerate(frames_sorted) if frame is None]
        raise ValueError(f"Missing video frame indices {missing[:8]} from {video_path}")

    restored: list[np.ndarray | None] = [None] * len(requested)
    for sorted_pos, original_pos in enumerate(order):
        restored[int(original_pos)] = frames_sorted[sorted_pos]
    return [image_to_uint8_rgb(frame) for frame in restored if frame is not None]


def nearest_video_frame_indices(video_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(video_timestamps, target_timestamps)
    positions = np.clip(positions, 0, len(video_timestamps) - 1)
    prev_positions = np.clip(positions - 1, 0, len(video_timestamps) - 1)
    use_prev = np.abs(video_timestamps[prev_positions] - target_timestamps) <= np.abs(
        video_timestamps[positions] - target_timestamps
    )
    return np.where(use_prev, prev_positions, positions).astype(np.int64)


def proportional_video_indices(step_indices: list[int], trajectory_length: int, num_video_frames: int) -> np.ndarray:
    if trajectory_length <= 1 or num_video_frames <= 1:
        return np.zeros(len(step_indices), dtype=np.int64)
    ratio = np.asarray(step_indices, dtype=np.float64) / float(trajectory_length - 1)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.rint(ratio * float(num_video_frames - 1)).astype(np.int64)


def save_contact_sheet(
    frames: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    cell_width: int = 224,
    label_height: int = 24,
) -> None:
    if not frames:
        return
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = [
        Image.fromarray(frame).convert("RGB").resize((cell_width, int(cell_width * frame.shape[0] / frame.shape[1])), resample)
        for frame in frames
    ]
    cell_height = max(image.height for image in resized)
    sheet = Image.new("RGB", (cell_width * len(resized), cell_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(resized):
        x = idx * cell_width
        y = label_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, 4), labels[idx], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_comparison_contact_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    labels: list[str],
    output_path: Path,
    cell_width: int = 224,
    label_height: int = 24,
    row_label_width: int = 92,
) -> None:
    rows = [(name, frames) for name, frames in rows if frames]
    if not rows:
        return
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized_rows = []
    max_cell_height = 1
    for row_name, frames in rows:
        resized = [
            Image.fromarray(frame)
            .convert("RGB")
            .resize((cell_width, int(cell_width * frame.shape[0] / frame.shape[1])), resample)
            for frame in frames
        ]
        max_cell_height = max(max_cell_height, *(image.height for image in resized))
        resized_rows.append((row_name, resized))

    row_height = max_cell_height + label_height
    sheet = Image.new("RGB", (row_label_width + cell_width * len(labels), row_height * len(resized_rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, (row_name, images) in enumerate(resized_rows):
        y0 = row_idx * row_height
        draw.text((4, y0 + label_height + 4), row_name, fill=(0, 0, 0))
        for col_idx, image in enumerate(images):
            x = row_label_width + col_idx * cell_width
            if row_idx == 0:
                draw.text((x + 4, y0 + 4), labels[col_idx], fill=(0, 0, 0))
            sheet.paste(image, (x, y0 + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def select_visual_frames(frames: list[np.ndarray], labels: list[str], show_all_frames: bool) -> tuple[list[np.ndarray], list[str]]:
    if show_all_frames or len(frames) <= 2:
        return frames, labels
    return [frames[0], frames[-1]], [labels[0], labels[-1]]


def build_data_cfg(args: argparse.Namespace, future_offsets: list[int]):
    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_root_dir = args.data_root_dir
    data_cfg.data_mix = args.data_mix
    data_cfg.include_state = True
    data_cfg.video_backend = args.video_backend
    data_cfg.future_video_delta_indices = future_offsets
    return data_cfg


def choose_steps(dataset, count: int, rng: random.Random) -> list[tuple[int, int]]:
    all_steps = [(int(traj), int(step)) for traj, step in np.asarray(dataset.all_steps)]
    if len(all_steps) <= count:
        return all_steps
    return rng.sample(all_steps, count)


def inspect_dataset(
    data_cfg,
    data_name: str,
    robot_type: str,
    args: argparse.Namespace,
    future_offsets: list[int],
    rng: random.Random,
    csv_writer,
) -> dict[str, Any]:
    dataset = make_LeRobotSingleDataset(
        Path(args.data_root_dir),
        data_name,
        robot_type,
        delete_pause_frame=data_cfg.get("delete_pause_frame", False),
        data_cfg=data_cfg,
    )
    chosen_steps = choose_steps(dataset, args.samples_per_dataset, rng)
    dataset_summary: dict[str, Any] = {
        "data_name": data_name,
        "robot_type": robot_type,
        "dataset_path": str(dataset.dataset_path),
        "num_trajectories": int(len(dataset.trajectory_ids)),
        "num_valid_steps": int(len(dataset.all_steps)),
        "num_checked_steps": int(len(chosen_steps)),
        "views": {},
    }
    view_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    frozen_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[str, int] = defaultdict(int)
    out_dir = Path(args.out_dir)
    saved_sheets = 0
    video_timestamp_cache: dict[Path, np.ndarray] = {}
    video_ratio_frame_cache: dict[tuple[Path, tuple[int, ...]], list[np.ndarray]] = {}

    for sample_idx, (trajectory_id, step) in enumerate(chosen_steps):
        raw_data = dataset.get_step_data(trajectory_id, step)
        trajectory_index = dataset.get_trajectory_index(trajectory_id)
        trajectory_length = int(dataset.trajectory_lengths[trajectory_index])
        action = concat_raw_action(raw_data, dataset.modality_keys.get("action", []))
        action_mean_l2 = float(np.mean(np.linalg.norm(action, axis=1))) if action is not None else float("nan")
        action_max_l2 = float(np.max(np.linalg.norm(action, axis=1))) if action is not None else float("nan")

        for video_key in dataset.modality_keys["video"]:
            frames = []
            offsets = [0] + future_offsets
            frame_indices = [step + offset for offset in offsets]
            timestamps = dataset.curr_traj_data["timestamp"].to_numpy()[frame_indices]
            video_path = dataset.get_video_path(trajectory_id, video_key.replace("video.", ""))
            mapped_video_indices = None
            mapped_video_timestamps = None
            ratio_video_indices = None
            ratio_frames = None
            video_num_frames = None
            video_duration = None
            timestamps_in_video_range = None
            if not args.skip_video_timestamp_map:
                try:
                    if video_path not in video_timestamp_cache:
                        video_timestamp_cache[video_path] = get_video_frame_timestamps(video_path)
                    video_timestamps = video_timestamp_cache[video_path]
                    video_num_frames = int(len(video_timestamps))
                    video_duration = float(video_timestamps[-1] - video_timestamps[0]) if len(video_timestamps) > 1 else 0.0
                    timestamps_in_video_range = bool(
                        np.min(timestamps) >= video_timestamps[0] and np.max(timestamps) <= video_timestamps[-1]
                    )
                    mapped_video_indices = nearest_video_frame_indices(video_timestamps, timestamps)
                    mapped_video_timestamps = video_timestamps[mapped_video_indices]
                    ratio_video_indices = proportional_video_indices(frame_indices, trajectory_length, len(video_timestamps))
                    ratio_key = (video_path, tuple(int(index) for index in ratio_video_indices))
                    if ratio_key not in video_ratio_frame_cache:
                        video_ratio_frame_cache[ratio_key] = get_video_frames_by_indices_pyav(video_path, ratio_video_indices)
                    ratio_frames = video_ratio_frame_cache[ratio_key]
                except Exception as exc:
                    print(
                        f"Warning: video index diagnosis failed for {data_name} "
                        f"traj={trajectory_id} step={step} {video_key}: {exc}",
                        flush=True,
                    )
            labels = []
            for offset in offsets:
                frame = dataset.get_video(trajectory_id, video_key, step + offset)
                frames.append(image_to_uint8_rgb(frame))
                labels.append(f"t+{offset}")
            backend_compare_frames: dict[str, list[np.ndarray]] = {}
            for compare_backend in parse_str_list(args.compare_video_backends):
                try:
                    compare_frames = get_frames_by_timestamps(
                        video_path.as_posix(),
                        timestamps,
                        video_backend=compare_backend,
                        video_backend_kwargs={},
                    )
                    backend_compare_frames[compare_backend] = [image_to_uint8_rgb(frame) for frame in compare_frames]
                except Exception as exc:
                    print(
                        f"Warning: backend compare failed for {compare_backend} on {data_name} "
                        f"traj={trajectory_id} step={step} {video_key}: {exc}",
                        flush=True,
                    )

            metrics_vs_prev = []
            metrics_vs_t0 = []
            for pair_idx in range(1, len(frames)):
                prev_metrics = frame_diff_metrics(frames[pair_idx - 1], frames[pair_idx], args.changed_pixel_threshold)
                t0_metrics = frame_diff_metrics(frames[0], frames[pair_idx], args.changed_pixel_threshold)
                ratio_prev_metrics = None
                ratio_t0_metrics = None
                if ratio_frames is not None and pair_idx < len(ratio_frames):
                    ratio_prev_metrics = frame_diff_metrics(
                        ratio_frames[pair_idx - 1], ratio_frames[pair_idx], args.changed_pixel_threshold
                    )
                    ratio_t0_metrics = frame_diff_metrics(ratio_frames[0], ratio_frames[pair_idx], args.changed_pixel_threshold)
                metrics_vs_prev.append(prev_metrics)
                metrics_vs_t0.append(t0_metrics)
                pair_counts[video_key] += 1
                for name, value in prev_metrics.items():
                    view_stats[video_key][f"prev_{name}"].append(value)
                for name, value in t0_metrics.items():
                    view_stats[video_key][f"t0_{name}"].append(value)
                if prev_metrics["changed_pixel_ratio"] < args.frozen_ratio_threshold:
                    frozen_counts[video_key] += 1

                csv_writer.writerow(
                    {
                        "dataset": data_name,
                        "robot_type": robot_type,
                        "trajectory_id": trajectory_id,
                        "step": step,
                        "video_key": video_key,
                        "offset_from": offsets[pair_idx - 1],
                        "offset_to": offsets[pair_idx],
                        "frame_index_from": frame_indices[pair_idx - 1],
                        "frame_index_to": frame_indices[pair_idx],
                        "trajectory_length": trajectory_length,
                        "timestamp_from": float(timestamps[pair_idx - 1]),
                        "timestamp_to": float(timestamps[pair_idx]),
                        "timestamp_delta": float(timestamps[pair_idx] - timestamps[pair_idx - 1]),
                        "video_path": str(video_path),
                        "video_num_frames": video_num_frames if video_num_frames is not None else "",
                        "video_duration": video_duration if video_duration is not None else "",
                        "timestamps_in_video_range": timestamps_in_video_range if timestamps_in_video_range is not None else "",
                        "video_frame_index_from": int(mapped_video_indices[pair_idx - 1])
                        if mapped_video_indices is not None
                        else "",
                        "video_frame_index_to": int(mapped_video_indices[pair_idx])
                        if mapped_video_indices is not None
                        else "",
                        "video_timestamp_from": float(mapped_video_timestamps[pair_idx - 1])
                        if mapped_video_timestamps is not None
                        else "",
                        "video_timestamp_to": float(mapped_video_timestamps[pair_idx])
                        if mapped_video_timestamps is not None
                        else "",
                        "video_frame_index_delta": int(mapped_video_indices[pair_idx] - mapped_video_indices[pair_idx - 1])
                        if mapped_video_indices is not None
                        else "",
                        "ratio_video_frame_index_from": int(ratio_video_indices[pair_idx - 1])
                        if ratio_video_indices is not None
                        else "",
                        "ratio_video_frame_index_to": int(ratio_video_indices[pair_idx])
                        if ratio_video_indices is not None
                        else "",
                        "ratio_video_frame_index_delta": int(ratio_video_indices[pair_idx] - ratio_video_indices[pair_idx - 1])
                        if ratio_video_indices is not None
                        else "",
                        "prev_mae": prev_metrics["mae"],
                        "prev_changed_pixel_ratio": prev_metrics["changed_pixel_ratio"],
                        "t0_mae": t0_metrics["mae"],
                        "t0_changed_pixel_ratio": t0_metrics["changed_pixel_ratio"],
                        "ratio_prev_mae": ratio_prev_metrics["mae"] if ratio_prev_metrics is not None else "",
                        "ratio_prev_changed_pixel_ratio": ratio_prev_metrics["changed_pixel_ratio"]
                        if ratio_prev_metrics is not None
                        else "",
                        "ratio_t0_mae": ratio_t0_metrics["mae"] if ratio_t0_metrics is not None else "",
                        "ratio_t0_changed_pixel_ratio": ratio_t0_metrics["changed_pixel_ratio"]
                        if ratio_t0_metrics is not None
                        else "",
                        "action_mean_l2": action_mean_l2,
                        "action_max_l2": action_max_l2,
                    }
                )

            if saved_sheets < args.max_contact_sheets:
                safe_name = data_name.replace("/", "__").replace("..", "dotdot")
                safe_view = video_key.replace(".", "_").replace("/", "_")
                visual_frames, visual_labels = select_visual_frames(frames, labels, args.show_all_frames)
                sheet_path = out_dir / "contact_sheets" / safe_name / (
                    f"{sample_idx:04d}_traj{trajectory_id}_step{step}_{safe_view}.jpg"
                )
                save_contact_sheet(visual_frames, visual_labels, sheet_path)
                compare_rows = [(f"loader_{args.video_backend}", visual_frames)]
                for compare_backend, compare_frames in backend_compare_frames.items():
                    compare_visual_frames, _ = select_visual_frames(compare_frames, labels, args.show_all_frames)
                    compare_rows.append((f"backend_{compare_backend}", compare_visual_frames))
                if ratio_frames is not None:
                    ratio_visual_frames, _ = select_visual_frames(ratio_frames, labels, args.show_all_frames)
                    compare_rows.append(("ratio_idx", ratio_visual_frames))
                compare_sheet_path = out_dir / "contact_sheets_compare" / safe_name / (
                    f"{sample_idx:04d}_traj{trajectory_id}_step{step}_{safe_view}.jpg"
                )
                save_comparison_contact_sheet(compare_rows, visual_labels, compare_sheet_path)
                saved_sheets += 1

    for video_key, stats in view_stats.items():
        frozen_pairs = frozen_counts[video_key]
        total_pairs = pair_counts[video_key]
        dataset_summary["views"][video_key] = {
            "pair_count": total_pairs,
            "frozen_pair_count": frozen_pairs,
            "frozen_pair_ratio": float(frozen_pairs / total_pairs) if total_pairs else None,
            "mean_prev_changed_pixel_ratio": float(np.mean(stats["prev_changed_pixel_ratio"]))
            if stats["prev_changed_pixel_ratio"]
            else None,
            "median_prev_changed_pixel_ratio": float(np.median(stats["prev_changed_pixel_ratio"]))
            if stats["prev_changed_pixel_ratio"]
            else None,
            "mean_t0_changed_pixel_ratio": float(np.mean(stats["t0_changed_pixel_ratio"]))
            if stats["t0_changed_pixel_ratio"]
            else None,
            "median_t0_changed_pixel_ratio": float(np.median(stats["t0_changed_pixel_ratio"]))
            if stats["t0_changed_pixel_ratio"]
            else None,
            "mean_prev_mae": float(np.mean(stats["prev_mae"])) if stats["prev_mae"] else None,
            "median_prev_mae": float(np.median(stats["prev_mae"])) if stats["prev_mae"] else None,
            "mean_t0_mae": float(np.mean(stats["t0_mae"])) if stats["t0_mae"] else None,
            "median_t0_mae": float(np.median(stats["t0_mae"])) if stats["t0_mae"] else None,
        }
    return dataset_summary


def main() -> None:
    args = parse_args()
    future_offsets = parse_int_list(args.future_video_delta_indices)
    robot_types = {item.strip() for item in args.robot_types.split(",") if item.strip()}
    if not future_offsets:
        raise ValueError("--future-video-delta-indices must not be empty")
    if args.data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(f"Unknown data mix {args.data_mix!r}. Available keys include: {list(DATASET_NAMED_MIXTURES)[:10]}")

    data_cfg = build_data_cfg(args, future_offsets)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixture_entries = [
        (data_name, weight, robot_type)
        for data_name, weight, robot_type in DATASET_NAMED_MIXTURES[args.data_mix]
        if "all" in robot_types or robot_type in robot_types
    ]
    if not mixture_entries:
        raise ValueError(f"No entries in data_mix={args.data_mix!r} matched robot_types={sorted(robot_types)}")

    summary = {
        "config_yaml": str(args.config_yaml),
        "data_root_dir": args.data_root_dir,
        "data_mix": args.data_mix,
        "video_backend": args.video_backend,
        "future_video_delta_indices": future_offsets,
        "changed_pixel_threshold": args.changed_pixel_threshold,
        "frozen_ratio_threshold": args.frozen_ratio_threshold,
        "datasets": [],
    }

    csv_path = out_dir / "frame_diff_rows.csv"
    with csv_path.open("w", newline="") as csv_file:
        fieldnames = [
            "dataset",
            "robot_type",
            "trajectory_id",
            "step",
            "video_key",
            "offset_from",
            "offset_to",
            "frame_index_from",
            "frame_index_to",
            "trajectory_length",
            "timestamp_from",
            "timestamp_to",
            "timestamp_delta",
            "video_path",
            "video_num_frames",
            "video_duration",
            "timestamps_in_video_range",
            "video_frame_index_from",
            "video_frame_index_to",
            "video_timestamp_from",
            "video_timestamp_to",
            "video_frame_index_delta",
            "ratio_video_frame_index_from",
            "ratio_video_frame_index_to",
            "ratio_video_frame_index_delta",
            "prev_mae",
            "prev_changed_pixel_ratio",
            "t0_mae",
            "t0_changed_pixel_ratio",
            "ratio_prev_mae",
            "ratio_prev_changed_pixel_ratio",
            "ratio_t0_mae",
            "ratio_t0_changed_pixel_ratio",
            "action_mean_l2",
            "action_max_l2",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data_name, _weight, robot_type in mixture_entries:
            print(f"\n=== Checking {robot_type}: {data_name} ===", flush=True)
            dataset_summary = inspect_dataset(data_cfg, data_name, robot_type, args, future_offsets, rng, writer)
            summary["datasets"].append(dataset_summary)
            for view_key, view_summary in dataset_summary["views"].items():
                print(
                    f"{view_key}: frozen_pair_ratio={view_summary['frozen_pair_ratio']:.4f}, "
                    f"mean_prev_changed={view_summary['mean_prev_changed_pixel_ratio']:.4f}, "
                    f"mean_t0_changed={view_summary['mean_t0_changed_pixel_ratio']:.4f}",
                    flush=True,
                )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote per-pair rows: {csv_path}")
    print(f"Wrote contact sheets under: {out_dir / 'contact_sheets'}")
    print(f"Wrote loader-vs-index contact sheets under: {out_dir / 'contact_sheets_compare'}")


if __name__ == "__main__":
    main()
