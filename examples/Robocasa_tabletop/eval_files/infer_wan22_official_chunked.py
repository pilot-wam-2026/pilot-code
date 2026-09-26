#!/usr/bin/env python3
"""Run official Wan2.2 TI2V in short teacher-forced chunks on RoboCasa.

Each chunk conditions on the current GT frame, generates chunk_size + 1 frames,
drops the first generated frame, and appends the remaining chunk_size future
frames. This gives a stitched long rollout while avoiding long open-loop Wan
generation from only the initial frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Robocasa_tabletop.eval_files.infer_cosmos25_future_video import (
    add_label,
    get_full_trajectory_video_frames,
    load_training_dataset,
    parse_sample_indices,
    read_video,
    require_pil,
    resolve_sample_location,
    save_video,
    to_pil_rgb,
)
from examples.Robocasa_tabletop.eval_files.infer_wan22_official import (
    DEFAULT_WAN_CKPT,
    DEFAULT_WAN_REPO,
    ROBOCASA_NEGATIVE_PROMPT,
    VALID_WAN_SIZES,
    build_prompt,
    build_wan_command,
    fit_frame_to_size,
    parse_wan_size,
    patch_wan_imageio_quality,
    patch_wan_negative_prompt_arg,
    resolve_ckpt_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", default=DEFAULT_WAN_REPO)
    parser.add_argument("--ckpt-dir", default=DEFAULT_WAN_CKPT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--out-dir", default="outputs/robocasa_wan22_official_chunked")
    parser.add_argument("--config-yaml", default="./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml")
    parser.add_argument("--data-root-dir", default="/path/to/workspace/datasets")
    parser.add_argument("--data-mix", default="robocasa_teleop_ee")
    parser.add_argument("--dataset-mode", default="train", choices=("train", "val"))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--condition-frames", type=int, default=1, help="Kept for dataset helper compatibility; Wan uses one GT frame per chunk.")
    parser.add_argument("--condition-stride", type=int, default=1)
    parser.add_argument("--size", default="1280*704", choices=VALID_WAN_SIZES)
    parser.add_argument("--wan-image-mode", default="letterbox", choices=("letterbox", "stretch", "raw"))
    parser.add_argument("--chunk-size", type=int, default=16, help="Number of future frames kept from each Wan chunk.")
    parser.add_argument("--max-rollout-frames", type=int, default=None, help="Optional cap including the initial GT condition frame.")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-guide-scale", type=float, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=("unipc", "dpm++"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1, help="Added to --seed for each successive chunk.")
    parser.add_argument("--offload-model", default="True", choices=("True", "False", "true", "false"))
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--prompt", default=None, help="Override dataset instruction.")
    parser.add_argument("--prompt-style", default="robocasa", choices=("robocasa", "raw"))
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--negative-prompt", default=ROBOCASA_NEGATIVE_PROMPT)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.chunk_size % 4 != 0:
        raise ValueError("Wan frame_num must be 4n+1, so --chunk-size must be a multiple of 4.")
    args.frame_num = args.chunk_size + 1


def save_rollout_compare(gt_frames: list, pred_frames: list, save_path: Path, fps: int) -> None:
    Image, _, _ = require_pil()
    target_size = pred_frames[0].size
    frames = []
    for gt, pred in zip(gt_frames, pred_frames):
        gt_panel = add_label(fit_frame_to_size(gt, target_size, mode="letterbox"), "GT")
        pred_panel = add_label(to_pil_rgb(pred).resize(target_size).copy(), "Wan Chunked")
        canvas = Image.new("RGB", (target_size[0] * 2, target_size[1]))
        canvas.paste(gt_panel, (0, 0))
        canvas.paste(pred_panel, (target_size[0], 0))
        frames.append(canvas)
    save_video(frames, save_path, fps=fps)


def run_chunk(args: argparse.Namespace, chunk_dir: Path, condition_frame, prompt: str, chunk_index: int) -> list:
    image_path = chunk_dir / "wan_input.png"
    pred_path = chunk_dir / "pred.mp4"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    wan_input_size = parse_wan_size(args.size)
    wan_input_frame = fit_frame_to_size(condition_frame, wan_input_size, mode=args.wan_image_mode)
    wan_input_frame.save(image_path)

    chunk_args = argparse.Namespace(**vars(args))
    chunk_args.seed = int(args.seed) + chunk_index * int(args.seed_stride)
    chunk_args.frame_num = int(args.chunk_size) + 1
    cmd = build_wan_command(chunk_args, image_path=image_path, prompt=prompt, save_file=pred_path)
    (chunk_dir / "wan_command.json").write_text(json.dumps(cmd, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Chunk {chunk_index:03d}: condition={image_path}")
    print(" ".join(cmd))
    if args.dry_run:
        return []
    subprocess.run(cmd, cwd=args.wan_repo, check=True)
    return [to_pil_rgb(frame) for frame in read_video(pred_path)]


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    patch_wan_imageio_quality(args.wan_repo)
    patch_wan_negative_prompt_arg(args.wan_repo)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    dataset = load_training_dataset(args)
    sample_indices = parse_sample_indices(args)

    for sample_ord, dataset_index in enumerate(sample_indices):
        source_dataset, trajectory_id, base_index = resolve_sample_location(dataset, dataset_index)
        example = dataset[dataset_index]
        all_traj_frames = get_full_trajectory_video_frames(source_dataset, trajectory_id, base_index, args)
        rollout_frames = all_traj_frames[int(base_index):]
        if args.max_rollout_frames is not None:
            rollout_frames = rollout_frames[: max(1, int(args.max_rollout_frames))]
        if len(rollout_frames) < 2:
            raise ValueError(f"Need at least two rollout frames, got {len(rollout_frames)}.")

        raw_instruction = args.prompt or str(example["lang"])
        prompt = build_prompt(args, raw_instruction)
        sample_out_dir = out_dir / f"sample_{sample_ord:03d}_idx_{dataset_index:06d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        stitched_pred = []
        chunks_meta = []
        cursor = 0
        chunk_index = 0
        while cursor < len(rollout_frames) - 1:
            keep = min(args.chunk_size, len(rollout_frames) - 1 - cursor)
            chunk_dir = sample_out_dir / f"chunk_{chunk_index:03d}_gt_{cursor:06d}"
            pred_chunk = run_chunk(args, chunk_dir, rollout_frames[cursor], prompt, chunk_index)
            chunks_meta.append({"chunk_index": chunk_index, "gt_start_frame": cursor, "kept_future_frames": keep, "chunk_dir": str(chunk_dir)})
            if not args.dry_run:
                if len(pred_chunk) < keep + 1:
                    raise ValueError(f"Chunk {chunk_index} returned {len(pred_chunk)} frames, need at least {keep + 1}.")
                if not stitched_pred:
                    stitched_pred.append(fit_frame_to_size(rollout_frames[0], pred_chunk[0].size, mode="letterbox"))
                stitched_pred.extend(pred_chunk[1: keep + 1])
            cursor += keep
            chunk_index += 1

        meta = {
            "dataset_index": dataset_index,
            "trajectory_id": trajectory_id,
            "base_index": base_index,
            "instruction": raw_instruction,
            "wan_prompt": prompt,
            "wan_negative_prompt": args.negative_prompt,
            "chunk_size": args.chunk_size,
            "frame_num_per_chunk": args.frame_num,
            "rollout_frames": len(rollout_frames),
            "chunks": chunks_meta,
        }
        (sample_out_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        save_video(rollout_frames, sample_out_dir / "gt_rollout.mp4", fps=args.fps)
        if not args.dry_run:
            stitched_pred = stitched_pred[: len(rollout_frames)]
            save_video(stitched_pred, sample_out_dir / "stitched_pred.mp4", fps=args.fps)
            save_rollout_compare(rollout_frames, stitched_pred, sample_out_dir / "compare.mp4", fps=args.fps)
            print(f"Saved stitched prediction: {sample_out_dir / 'stitched_pred.mp4'}")
            print(f"Saved comparison video: {sample_out_dir / 'compare.mp4'}")


if __name__ == "__main__":
    main()
