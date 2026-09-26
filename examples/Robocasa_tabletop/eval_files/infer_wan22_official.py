#!/usr/bin/env python3
"""Run official Wan2.2 TI2V code on RoboCasa dataset samples.

This script does not use diffusers.WanPipeline. It reads RoboCasa samples with
the starVLA dataloader, saves input/trajectory previews, then calls the official
Wan2.2 repository's generate.py with --task ti2v-5B and --image.
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
    as_image_list,
    get_condition_video_frames,
    get_full_trajectory_video_frames,
    load_training_dataset,
    parse_sample_indices,
    read_video,
    require_pil,
    resample_to_length,
    resolve_sample_location,
    save_video,
    to_pil_rgb,
)


DEFAULT_WAN_REPO = "/path/to/workspace/code/Wan2.2"
DEFAULT_WAN_CKPT = "/path/to/workspace/share/hf_models/Wan-AI/Wan2.2-TI2V-5B"
VALID_WAN_SIZES = (
    "720*1280",
    "1280*720",
    "480*832",
    "832*480",
    "704*1280",
    "1280*704",
    "1024*704",
    "704*1024",
)
ROBOCASA_PROMPT_TEMPLATE = (
    "A fixed-view RoboCasa simulation trajectory prediction. "
    "The input image is the first reference frame. Generate future frames as a continuation of the same robot rollout. "
    "Keep the background, tabletop workspace, robot, task-relevant objects, object identities, object colors, object shapes, camera pose, "
    "camera scale, image framing, and field of view identical to the first frame. "
    "The scene is a low-texture simulated robotics environment, not a real kitchen video. "
    "The robot end effector is a black five-finger robotic dexterous hand, similar in shape to a human hand but clearly mechanical. "
    "Only the black five-finger robotic dexterous hand, robot arm, and task-relevant movable objects are allowed to move. "
    "Predict how the robot executes the task while all unrelated background objects remain static. "
    "Do not zoom in, do not crop, do not pan, do not rotate the camera, and do not cut to a close-up. "
    "No human, no human hand, no fingers, no skin, no real-world kitchen, no extra arm, no scene change. "
    "Task instruction: {instruction}"
)
ROBOCASA_NEGATIVE_PROMPT = (
    "human, person, human hand, fingers, skin, real-world kitchen, photorealistic kitchen, "
    "close-up, zoom in, camera zoom, camera pan, camera rotation, camera movement, camera cut, "
    "cropped view, changing viewpoint, changing field of view, scene change, new object, "
    "parallel gripper, two-finger gripper, claw gripper, extra robot arm, extra dexterous hand, "
    "duplicated robotic hand, deformed robotic hand, distorted robot, "
    "blur, low quality, text, watermark, subtitles"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", default=DEFAULT_WAN_REPO)
    parser.add_argument("--ckpt-dir", default=DEFAULT_WAN_CKPT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--out-dir", default="outputs/robocasa_wan22_official_ti2v")
    parser.add_argument("--config-yaml", default="./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml")
    parser.add_argument("--data-root-dir", default="/path/to/workspace/datasets")
    parser.add_argument("--data-mix", default="robocasa_teleop_ee")
    parser.add_argument("--dataset-mode", default="train", choices=("train", "val"))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument(
        "--condition-frames",
        type=int,
        default=1,
        help="Number of dataset frames to save for visualization. Official Wan TI2V conditions on the latest frame only.",
    )
    parser.add_argument("--condition-stride", type=int, default=1)
    parser.add_argument(
        "--size",
        default="1280*704",
        choices=VALID_WAN_SIZES,
        help="Official Wan size bucket. For I2V, output aspect ratio follows --image; this is not an exact output size.",
    )
    parser.add_argument(
        "--wan-image-mode",
        default="letterbox",
        choices=("letterbox", "stretch", "raw"),
        help="How to adapt the square RoboCasa frame before passing it to official Wan TI2V.",
    )
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-guide-scale", type=float, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=("unipc", "dpm++"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", default="True", choices=("True", "False", "true", "false"))
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--prompt", default=None, help="Override dataset instruction.")
    parser.add_argument(
        "--prompt-style",
        default="robocasa",
        choices=("robocasa", "raw"),
        help="robocasa wraps the instruction to bias Wan toward simulated robot trajectory prediction.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional format string with {instruction}; overrides --prompt-style robocasa template.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=ROBOCASA_NEGATIVE_PROMPT,
        help="Negative prompt passed to official Wan. Defaults to RoboCasa-specific camera/hand artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_ckpt_dir(ckpt_dir: str) -> str:
    root = Path(ckpt_dir)
    nested = root / "Wan2.2-TI2V-5B"
    if nested.is_dir() and (nested / "Wan2.2_VAE.pth").exists():
        return str(nested)
    return str(root)


def patch_wan_imageio_quality(wan_repo: str) -> None:
    """Patch official Wan save_video for imageio/PyAV versions without quality=."""
    utils_path = Path(wan_repo) / "wan" / "utils" / "utils.py"
    if not utils_path.exists():
        return
    text = utils_path.read_text(encoding="utf-8")
    old = "cache_file, fps=fps, codec='libx264', quality=8)"
    new = "cache_file, fps=fps, codec='libx264')"
    if old in text:
        utils_path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched Wan save_video imageio quality arg: {utils_path}")


def patch_wan_negative_prompt_arg(wan_repo: str) -> None:
    """Expose WanTI2V.generate(..., n_prompt=...) through official generate.py."""
    generate_path = Path(wan_repo) / "generate.py"
    if not generate_path.exists():
        return
    text = generate_path.read_text(encoding="utf-8")
    original = text

    prompt_arg = '''    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
'''
    negative_arg = prompt_arg + '''    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="The negative prompt to avoid unwanted content.")
'''
    if "--negative_prompt" not in text and prompt_arg in text:
        text = text.replace(prompt_arg, negative_arg)

    if "wan_ti2v.generate" in text and "n_prompt=args.negative_prompt" not in text:
        marker = "        video = wan_ti2v.generate(\n"
        start = text.find(marker)
        if start != -1:
            end = text.find("\n        )", start)
            if end != -1:
                end += len("\n        )")
                call = text[start:end]
                old = "            guide_scale=args.sample_guide_scale,\n            seed=args.base_seed,"
                new = "            guide_scale=args.sample_guide_scale,\n            n_prompt=args.negative_prompt or \"\",\n            seed=args.base_seed,"
                if old in call:
                    text = text[:start] + call.replace(old, new) + text[end:]

    if "--negative_prompt" not in text or "n_prompt=args.negative_prompt" not in text:
        raise RuntimeError(f"Failed to patch Wan negative_prompt support in {generate_path}")

    if text != original:
        generate_path.write_text(text, encoding="utf-8")
        print(f"Patched Wan generate.py negative_prompt arg: {generate_path}")


def parse_wan_size(size: str) -> tuple[int, int]:
    width, height = size.lower().split("*")
    return int(width), int(height)


def fit_frame_to_size(frame, target_size: tuple[int, int], mode: str = "letterbox"):
    Image, _, _ = require_pil()
    frame = to_pil_rgb(frame)
    if mode == "raw":
        return frame.copy()
    if mode == "stretch":
        return frame.resize(target_size)
    if mode != "letterbox":
        raise ValueError(f"Unknown frame fit mode: {mode}")

    target_w, target_h = target_size
    scale = min(target_w / frame.width, target_h / frame.height)
    resized = frame.resize((max(1, round(frame.width * scale)), max(1, round(frame.height * scale))))
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    paste_xy = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    canvas.paste(resized, paste_xy)
    return canvas


def resample_and_fit_frames(frames: list, target_len: int, target_size: tuple[int, int], mode: str = "letterbox") -> list:
    if not frames:
        return []
    if len(frames) == 1:
        sampled = [frames[0]] * target_len
    else:
        sampled = [frames[round(i * (len(frames) - 1) / max(target_len - 1, 1))] for i in range(target_len)]
    return [fit_frame_to_size(frame, target_size, mode=mode) for frame in sampled]


def save_compare_video(input_frames, traj_frames, pred_frames, save_path: Path, fps: int = 16) -> None:
    Image, _, _ = require_pil()
    target_size = pred_frames[0].size
    input_video = resample_and_fit_frames(input_frames, len(pred_frames), target_size)
    traj_video = resample_and_fit_frames(traj_frames, len(pred_frames), target_size)

    compare_frames = []
    for input_frame, traj_frame, pred_frame in zip(input_video, traj_video, pred_frames):
        panels = [
            add_label(input_frame.copy(), "Input"),
            add_label(traj_frame.copy(), "Traj"),
            add_label(pred_frame.copy(), "Wan"),
        ]
        canvas = Image.new("RGB", (target_size[0] * len(panels), target_size[1]))
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * target_size[0], 0))
        compare_frames.append(canvas)
    save_video(compare_frames, save_path, fps=fps)


def build_prompt(args: argparse.Namespace, instruction: str) -> str:
    if args.prompt_style == "raw":
        return instruction
    template = args.prompt_template or ROBOCASA_PROMPT_TEMPLATE
    return template.format(instruction=instruction)


def build_wan_command(args: argparse.Namespace, image_path: Path, prompt: str, save_file: Path) -> list[str]:
    cmd = [
        args.python,
        "generate.py",
        "--task",
        "ti2v-5B",
        "--ckpt_dir",
        args.ckpt_dir,
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--size",
        args.size,
        "--frame_num",
        str(args.frame_num),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_solver",
        args.sample_solver,
        "--base_seed",
        str(args.seed),
        "--offload_model",
        args.offload_model,
        "--save_file",
        str(save_file),
    ]
    if args.sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(args.sample_guide_scale)])
    if args.sample_shift is not None:
        cmd.extend(["--sample_shift", str(args.sample_shift)])
    if args.t5_cpu:
        cmd.append("--t5_cpu")
    if args.convert_model_dtype:
        cmd.append("--convert_model_dtype")
    return cmd


def main() -> None:
    args = parse_args()
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
        condition_frames = get_condition_video_frames(source_dataset, trajectory_id, base_index, args)
        traj_frames = get_full_trajectory_video_frames(source_dataset, trajectory_id, base_index, args)
        raw_instruction = args.prompt or str(example["lang"])
        prompt = build_prompt(args, raw_instruction)

        sample_out_dir = out_dir / f"sample_{sample_ord:03d}_idx_{dataset_index:06d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        image_path = sample_out_dir / "wan_input.png"
        input_video_path = sample_out_dir / "input.mp4"
        traj_video_path = sample_out_dir / "traj.mp4"
        pred_path = sample_out_dir / "pred.mp4"
        compare_path = sample_out_dir / "compare.mp4"
        meta_path = sample_out_dir / "sample_meta.json"

        # Official TI2V accepts one image. Use the latest dataset frame as the
        # actual condition; input.mp4 is only kept for visualization.
        wan_input_size = parse_wan_size(args.size)
        wan_input_frame = fit_frame_to_size(condition_frames[-1], wan_input_size, mode=args.wan_image_mode)
        wan_input_frame.save(image_path)
        save_video(condition_frames, input_video_path, fps=16)
        save_video(traj_frames, traj_video_path, fps=16)

        cmd = build_wan_command(args, image_path=image_path, prompt=prompt, save_file=pred_path)
        meta_path.write_text(
            json.dumps(
                {
                    "dataset_index": dataset_index,
                    "trajectory_id": trajectory_id,
                    "base_index": base_index,
                    "instruction": raw_instruction,
                    "wan_prompt": prompt,
                    "wan_negative_prompt": args.negative_prompt,
                    "prompt_style": args.prompt_style,
                    "wan_command": cmd,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"[{sample_ord + 1}/{len(sample_indices)}] dataset index={dataset_index}")
        print(f"Instruction: {raw_instruction}")
        print(f"Wan prompt: {prompt}")
        print(f"Wan negative prompt: {args.negative_prompt}")
        print("Conditioning: official Wan TI2V receives only wan_input.png; input.mp4 is visualization only.")
        print("Wan command:")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=args.wan_repo, check=True)
            pred_frames = [to_pil_rgb(frame) for frame in read_video(pred_path)]
            save_compare_video(condition_frames, traj_frames, pred_frames, compare_path)
            print(f"Saved comparison video: {compare_path}")


if __name__ == "__main__":
    main()
