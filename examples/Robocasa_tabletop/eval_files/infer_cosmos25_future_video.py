#!/usr/bin/env python3
"""Run Cosmos-Predict2.5 future-video inference for RoboCasa images.

This script reads RoboCasa samples like training, then runs the official
diffusers Cosmos2.5 Image2World-style call for video prediction.

Dataset example:
    python examples/Robocasa_tabletop/eval_files/infer_cosmos25_future_video.py \
        --num-samples 8 \
        --out-dir outputs/robocasa_cosmos25_dataset_infer

Manual example:
    python examples/Robocasa_tabletop/eval_files/infer_cosmos25_future_video.py \
        --image current.png \
        --instruction "pick up the cup and put it into the drawer" \
        --gt gt.mp4 \
        --out-dir outputs/robocasa_cosmos25_infer/cup_to_drawer
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_COSMOS25_WEIGHT = (
    "/path/to/workspace/code/starVLA/playground/Pretrained_models/"
    "Cosmos-Predict2.5-2B-Post-Trained"
)
DEFAULT_CONFIG_YAML = "./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml"
DEFAULT_DATA_ROOT_DIR = "/path/to/workspace/datasets"
DEFAULT_DATA_MIX = "robocasa_teleop_ee"
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 704
ROBOCASA_PROMPT_TEMPLATE = (
    "Continue the given simulator image as a future robot manipulation video. "
    "Keep the same camera, viewpoint, framing, lighting, room, surface materials, robot, and objects as the input image. "
    "Only the existing robot hand, robot arm, and task-relevant movable objects should move. "
    "All background surfaces and unrelated objects should remain static and unchanged. "
    "Do not add text, labels, borders, logos, panels, signs, people, human hands, new objects, or new scene elements. "
    "Do not change the camera, zoom, crop, pan, rotate, or redesign the scene. "
    "Task instruction: {instruction}"
)
ROBOCASA_NEGATIVE_PROMPT = (
    "human, person, human hand, fingers, skin, real-world kitchen, photorealistic kitchen, "
    "text, letters, words, writing, handwriting, label, caption, title, logo, watermark, brand mark, sign, signage, "
    "border, picture frame, poster, page, document, panel, user interface, split screen, "
    "black dots, speckles, dotted countertop, spotted countertop, noisy countertop, dirt, stains, crumbs, grid pattern, checker pattern, "
    "close-up, zoom in, camera zoom, camera pan, camera rotation, camera movement, camera cut, "
    "wide-angle view, widened view, cropped view, changing viewpoint, changing field of view, scene change, new object, "
    "parallel gripper, two-finger gripper, claw gripper, extra robot arm, extra dexterous hand, "
    "duplicated robotic hand, deformed robotic hand, distorted robot, "
    "blur, low quality, subtitles"
)
DEFAULT_NEGATIVE_PROMPT = (
    ROBOCASA_NEGATIVE_PROMPT
)


class _DisabledCosmosSafetyChecker:
    """No-op safety checker for local RoboCasa visualization."""

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video, *args, **kwargs):
        return video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_COSMOS25_WEIGHT)
    parser.add_argument("--revision", default=os.environ.get("COSMOS_PREDICT25_REVISION", "diffusers/base/post-trained"))
    parser.add_argument(
        "--image",
        nargs="+",
        default=None,
        help="Optional manual current RGB image path(s). If omitted, samples are read from the RoboCasa training set.",
    )
    parser.add_argument("--instruction", default=None, help="Manual task instruction, or @path/to/instruction.txt.")
    parser.add_argument("--gt", default=None, help="Optional manual GT image/video path for side-by-side comparison.")
    parser.add_argument("--out-dir", default="outputs/robocasa_cosmos25_dataset_future_video")
    parser.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    parser.add_argument("--data-root-dir", default=DEFAULT_DATA_ROOT_DIR)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--dataset-mode", default="train", choices=("train", "val"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated dataset indices. Overrides --num-samples/--start-index.",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=("auto", "image", "video", "text"),
        default="image",
        help="image uses current frame, video uses historical frames, text passes only the prompt.",
    )
    parser.add_argument("--view-index", type=int, default=0, help="RoboCasa camera view used for current image and GT future image.")
    parser.add_argument("--condition-frames", type=int, default=5, help="Number of historical frames for --conditioning-mode video.")
    parser.add_argument("--condition-stride", type=int, default=1, help="Stride in environment steps between historical frames.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--prompt-style",
        default="raw",
        choices=("raw", "empty", "robocasa"),
        help="raw uses only the task instruction; empty uses no text prompt; robocasa wraps the instruction.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional format string with {instruction}; overrides --prompt-style robocasa template.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--conditional-frame-timestep", type=float, default=0.1)
    parser.add_argument("--num-latent-conditional-frames", type=int, default=2)
    parser.add_argument(
        "--compare-panels",
        choices=("gt_pred", "current_gt_pred", "video_traj_pred"),
        default="video_traj_pred",
        help="Panels to write into compare.mp4 when --gt is provided.",
    )
    parser.add_argument("--enable-safety-checker", action="store_true")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Alias for --conditioning-mode text.",
    )
    return parser.parse_args()


def apply_arg_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if args.text_only:
        args.conditioning_mode = "text"
    return args


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required. Please run this script in the starVLA environment.") from exc
    return np


def require_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("pillow is required. Please run this script in the starVLA environment.") from exc
    return Image, ImageDraw, ImageFont


def require_av():
    try:
        import av
    except ImportError as exc:
        raise ImportError("av is required for video IO. Please run this script in the starVLA environment.") from exc
    return av


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for Cosmos inference. Please run this script in the starVLA environment.") from exc
    return torch


def maybe_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def load_generation_pipeline(args: argparse.Namespace, dtype):
    from diffusers import Cosmos2_5_PredictBasePipeline

    load_kwargs = {"torch_dtype": dtype}
    if args.revision and not os.path.isdir(args.model_path):
        load_kwargs["revision"] = args.revision
    return Cosmos2_5_PredictBasePipeline.from_pretrained(args.model_path, **load_kwargs)


def pipeline_capabilities(pipe) -> dict:
    signature = inspect.signature(pipe.__call__)
    params = set(signature.parameters)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    return {
        "class": type(pipe).__name__,
        "accepts_image": "image" in params or accepts_kwargs,
        "accepts_video": "video" in params or accepts_kwargs,
        "accepts_kwargs": accepts_kwargs,
    }


def read_instruction(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8").strip()
    return value


def build_prompt(args: argparse.Namespace, instruction: str) -> str:
    if args.prompt_style == "raw":
        return instruction
    if args.prompt_style == "empty":
        return ""
    template = args.prompt_template or ROBOCASA_PROMPT_TEMPLATE
    return template.format(instruction=instruction)


def parse_sample_indices(args: argparse.Namespace) -> list[int]:
    if args.sample_indices:
        return [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    return list(range(args.start_index, args.start_index + args.num_samples))


def load_training_dataset(args: argparse.Namespace):
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    cfg.datasets.vla_data.data_mix = args.data_mix
    cfg.datasets.vla_data.include_state = True
    cfg.datasets.vla_data.per_device_batch_size = 1
    return get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode=args.dataset_mode)


def resolve_sample_location(mixture_dataset, dataset_index: int) -> tuple:
    """Mirror LeRobotMixtureDataset sampling so we can read temporal frames."""
    dataset, trajectory_id, base_index = mixture_dataset.sample_step(dataset_index)
    return dataset, int(trajectory_id), int(base_index)


def load_image(path: str):
    Image, _, _ = require_pil()
    return Image.open(path).convert("RGB")


def as_image_list(images) -> list:
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def to_pil_rgb(frame):
    Image, _, _ = require_pil()
    np = require_numpy()
    torch = maybe_torch()

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch is not None and torch.is_tensor(frame):
        array = frame.detach().float().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.size and array.min() >= -1.0 and array.max() <= 1.0:
            array = (array + 1.0) * 127.5
        elif array.size and array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


def extract_frames(output) -> list:
    np = require_numpy()
    torch = maybe_torch()

    videos = getattr(output, "frames", None)
    if videos is None:
        videos = getattr(output, "videos", None)
    if videos is None:
        videos = getattr(output, "images", None)
    if videos is None and isinstance(output, tuple):
        videos = output[0]
    if videos is None:
        videos = output

    if torch is not None and torch.is_tensor(videos):
        videos = videos.detach().cpu()
        if videos.ndim == 5:
            videos = videos[0]
        return [to_pil_rgb(frame) for frame in videos]

    if isinstance(videos, np.ndarray):
        if videos.ndim == 5:
            videos = videos[0]
        return [to_pil_rgb(frame) for frame in videos]

    if isinstance(videos, (list, tuple)) and videos:
        first = videos[0]
        if isinstance(first, (list, tuple)):
            return [to_pil_rgb(frame) for frame in first]
        if torch is not None and torch.is_tensor(first) and first.ndim >= 4:
            return [to_pil_rgb(frame) for frame in first]
        if isinstance(first, np.ndarray) and first.ndim >= 4:
            return [to_pil_rgb(frame) for frame in first]
        return [to_pil_rgb(frame) for frame in videos]

    raise ValueError("Could not extract generated frames from Cosmos output.")


def read_video(path: Path) -> list:
    av = require_av()
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"No frames decoded from {path}.")
    return frames


def read_gt(path: str, target_len: int) -> list:
    np = require_numpy()
    gt_path = Path(path)
    if gt_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        frame = load_image(str(gt_path))
        return [frame.copy() for _ in range(target_len)]

    frames = read_video(gt_path)
    if len(frames) == target_len:
        return frames
    if len(frames) == 1:
        return [frames[0].copy() for _ in range(target_len)]

    indices = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[int(i)] for i in indices]


def gt_frames_from_dataset(example: dict, target_len: int, view_index: int) -> list | None:
    future_images = example.get("future_image")
    if future_images is None:
        return None
    future_images = as_image_list(future_images)
    if not future_images:
        return None
    view_index = min(max(int(view_index), 0), len(future_images) - 1)
    frame = to_pil_rgb(future_images[view_index])
    return [frame.copy() for _ in range(target_len)]


def get_condition_video_frames(dataset, trajectory_id: int, base_index: int, args: argparse.Namespace) -> list:
    """Read same-view history frames ending at base_index for Cosmos Video2World."""
    np = require_numpy()

    video_keys = list(dataset.modality_keys["video"])
    if not video_keys:
        raise ValueError("Dataset sample has no video keys.")
    video_key = video_keys[min(max(args.view_index, 0), len(video_keys) - 1)]

    # Ensure curr_traj_data is populated for timestamp-based frame lookup.
    dataset.get_step_data(trajectory_id, base_index)
    condition_frames = max(1, int(args.condition_frames))
    stride = max(1, int(args.condition_stride))
    offsets = np.arange(-(condition_frames - 1) * stride, 1, stride, dtype=np.int64)
    old_delta_indices = dataset.delta_indices[video_key]
    try:
        dataset.delta_indices[video_key] = offsets
        frames = dataset.get_video(trajectory_id, video_key, base_index)
    finally:
        dataset.delta_indices[video_key] = old_delta_indices

    return [to_pil_rgb(frame) for frame in frames]


def get_full_trajectory_video_frames(dataset, trajectory_id: int, base_index: int, args: argparse.Namespace) -> list:
    """Read the full same-view RoboCasa trajectory for visualization only."""
    np = require_numpy()

    video_keys = list(dataset.modality_keys["video"])
    if not video_keys:
        raise ValueError("Dataset sample has no video keys.")
    video_key = video_keys[min(max(args.view_index, 0), len(video_keys) - 1)]

    # Ensure curr_traj_data is populated for timestamp-based frame lookup.
    dataset.get_step_data(trajectory_id, base_index)
    trajectory_index = dataset.get_trajectory_index(trajectory_id)
    trajectory_length = int(dataset.trajectory_lengths[trajectory_index])
    offsets = np.arange(trajectory_length, dtype=np.int64) - int(base_index)

    old_delta_indices = dataset.delta_indices[video_key]
    try:
        dataset.delta_indices[video_key] = offsets
        frames = dataset.get_video(trajectory_id, video_key, base_index)
    finally:
        dataset.delta_indices[video_key] = old_delta_indices

    return [to_pil_rgb(frame) for frame in frames]


def hold_last_to_length(frames: list, target_len: int, target_size) -> list:
    if not frames:
        return []
    frames = [to_pil_rgb(frame).resize(target_size).copy() for frame in frames]
    if len(frames) >= target_len:
        return frames[:target_len]
    return frames + [frames[-1].copy() for _ in range(target_len - len(frames))]


def resample_to_length(frames: list, target_len: int, target_size) -> list:
    np = require_numpy()
    if not frames:
        return []
    frames = [to_pil_rgb(frame).resize(target_size).copy() for frame in frames]
    if len(frames) == target_len:
        return frames
    if len(frames) == 1:
        return [frames[0].copy() for _ in range(target_len)]
    indices = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[int(index)] for index in indices]


def add_label(frame, label: str):
    _, ImageDraw, ImageFont = require_pil()
    frame = frame.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)
    pad = 8
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return frame


def make_compare_frames(
    pred_frames: list,
    gt_frames: list | None,
    current_image,
    compare_panels: str,
    condition_frames: list | None = None,
    trajectory_frames: list | None = None,
) -> list:
    if gt_frames is None and not (compare_panels == "video_traj_pred" and trajectory_frames):
        return pred_frames

    Image, _, _ = require_pil()
    compare_frames = []
    condition_video = []
    trajectory_video = []
    if compare_panels == "video_traj_pred":
        target_size = pred_frames[0].size
        condition_video = hold_last_to_length(condition_frames or [current_image], len(pred_frames), target_size)
        trajectory_video = resample_to_length(trajectory_frames or gt_frames, len(pred_frames), target_size)

    if gt_frames is None:
        gt_frames = [pred_frames[-1] for _ in pred_frames]

    for frame_idx, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
        gt = gt.resize(pred.size)
        panels = []
        if compare_panels == "video_traj_pred":
            cond_panel = condition_video[frame_idx] if condition_video else current_image.resize(pred.size)
            traj_panel = trajectory_video[frame_idx] if trajectory_video else gt
            panels.extend([add_label(cond_panel.copy(), "Input"), add_label(traj_panel.copy(), "Traj")])
        elif compare_panels == "current_gt_pred":
            panels.append(add_label(current_image.resize(pred.size).copy(), "Current"))
        if compare_panels == "video_traj_pred":
            panels.append(add_label(pred.copy(), "Cosmos"))
        else:
            panels.extend([add_label(gt.copy(), "GT"), add_label(pred.copy(), "Cosmos")])

        canvas = Image.new("RGB", (pred.width * len(panels), pred.height))
        for panel_idx, panel in enumerate(panels):
            canvas.paste(panel, (pred.width * panel_idx, 0))
        compare_frames.append(canvas)
    return compare_frames


def save_video(frames: Iterable, path: Path, fps: int) -> None:
    frames = [frame.convert("RGB") for frame in frames]
    if not frames:
        raise ValueError(f"No frames to save for {path}.")

    width, height = frames[0].size
    encoded_width = width + (width % 2)
    encoded_height = height + (height % 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_video_with_codec(frames, path, fps, "h264", encoded_width, encoded_height)
    except Exception:
        save_video_with_codec(frames, path, fps, "mpeg4", encoded_width, encoded_height)


def save_video_with_codec(frames: list, path: Path, fps: int, codec: str, width: int, height: int) -> None:
    av = require_av()
    Image, _, _ = require_pil()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            if frame.size != (width, height):
                canvas = Image.new("RGB", (width, height))
                canvas.paste(frame, (0, 0))
                frame = canvas
            video_frame = av.VideoFrame.from_image(frame)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_generation_kwargs(args: argparse.Namespace, instruction: str, images: list, generator) -> dict:
    generation_kwargs = {
        "image": None,
        "video": None,
        "prompt": instruction,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "output_type": "pil",
        "return_dict": True,
        "max_sequence_length": args.max_sequence_length,
        "conditional_frame_timestep": args.conditional_frame_timestep,
        "num_latent_conditional_frames": args.num_latent_conditional_frames,
        "generator": generator,
    }

    if args.conditioning_mode == "image":
        view_index = min(max(args.view_index, 0), len(images) - 1)
        generation_kwargs["image"] = images[view_index]
    elif args.conditioning_mode == "video":
        generation_kwargs["video"] = images
    elif args.conditioning_mode == "text":
        pass
    else:
        if len(images) == 1:
            generation_kwargs["image"] = images[0]
        else:
            # Dataset samples store multiple camera views, not temporal frames.
            # Use the selected view as the current frame unless the caller
            # explicitly asks for --conditioning-mode video.
            view_index = min(max(args.view_index, 0), len(images) - 1)
            generation_kwargs["image"] = images[view_index]
    return generation_kwargs


def save_inference_outputs(
    args: argparse.Namespace,
    sample_out_dir: Path,
    pred_frames: list,
    gt_frames: list | None,
    current_image,
    metadata: dict,
    condition_frames: list | None = None,
    trajectory_frames: list | None = None,
) -> None:
    compare_frames = make_compare_frames(
        pred_frames,
        gt_frames,
        current_image,
        args.compare_panels,
        condition_frames=condition_frames,
        trajectory_frames=trajectory_frames,
    )

    pred_path = sample_out_dir / "pred.mp4"
    input_path = sample_out_dir / "input.mp4"
    traj_path = sample_out_dir / "traj.mp4"
    compare_path = sample_out_dir / "compare.mp4"
    last_path = sample_out_dir / "last_frame_compare.png"
    meta_path = sample_out_dir / "sample_meta.json"

    save_video(pred_frames, pred_path, args.fps)
    if condition_frames:
        save_video(condition_frames, input_path, args.fps)
    if trajectory_frames:
        save_video(trajectory_frames, traj_path, args.fps)
    save_video(compare_frames, compare_path, args.fps)
    compare_frames[-1].save(last_path)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved prediction video: {pred_path}")
    if condition_frames:
        print(f"Saved input video: {input_path}")
    if trajectory_frames:
        print(f"Saved trajectory video: {traj_path}")
    print(f"Saved comparison video: {compare_path}")
    print(f"Saved last-frame comparison: {last_path}")


def main() -> None:
    args = parse_args()
    args = apply_arg_aliases(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.enable_safety_checker:
        from diffusers.pipelines.cosmos import pipeline_cosmos2_5_predict

        pipeline_cosmos2_5_predict.CosmosSafetyChecker = _DisabledCosmosSafetyChecker

    torch = require_torch()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    pipe = load_generation_pipeline(args, dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    args._pipe_call = pipe.__call__
    print(f"Loaded Cosmos model={args.model_path}")
    print(f"Pipeline capabilities: {pipeline_capabilities(pipe)}")
    print(
        "Generation config: "
        f"height={args.height}, width={args.width}, "
        f"num_frames={args.num_frames}, steps={args.num_inference_steps}, "
        f"guidance_scale={args.guidance_scale}, conditioning_mode={args.conditioning_mode}"
    )

    if args.image is not None:
        if args.instruction is None:
            raise ValueError("--instruction is required when using manual --image input.")
        images = [load_image(path) for path in args.image]
        instruction = read_instruction(args.instruction)
        prompt = build_prompt(args, instruction)
        generator = torch.Generator().manual_seed(args.seed)
        print(f"Instruction: {instruction}")
        print(f"Prompt: {prompt}")
        print(f"Negative prompt: {args.negative_prompt}")
        with torch.inference_mode():
            output = pipe(**build_generation_kwargs(args, prompt, images, generator))
        pred_frames = extract_frames(output)
        gt_frames = read_gt(args.gt, len(pred_frames)) if args.gt else None
        save_inference_outputs(
            args=args,
            sample_out_dir=out_dir,
            pred_frames=pred_frames,
            gt_frames=gt_frames,
            current_image=images[min(max(args.view_index, 0), len(images) - 1)],
            metadata={
                "args": serializable_args(args),
                "instruction": instruction,
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "prompt_style": args.prompt_style,
                "manual_images": args.image,
            },
            condition_frames=images,
        )
        return

    dataset = load_training_dataset(args)
    sample_indices = parse_sample_indices(args)
    (out_dir / "run_args.json").write_text(json.dumps(serializable_args(args), indent=2, ensure_ascii=False), encoding="utf-8")

    for sample_ord, dataset_index in enumerate(sample_indices):
        source_dataset, trajectory_id, base_index = resolve_sample_location(dataset, dataset_index)
        example = dataset[dataset_index]
        images = [to_pil_rgb(image) for image in as_image_list(example["image"])]
        if args.conditioning_mode == "video":
            images = get_condition_video_frames(source_dataset, trajectory_id, base_index, args)
        instruction = str(example["lang"])
        prompt = build_prompt(args, instruction)
        sample_out_dir = out_dir / f"sample_{sample_ord:03d}_idx_{dataset_index:06d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator().manual_seed(args.seed + sample_ord)
        print(f"[{sample_ord + 1}/{len(sample_indices)}] dataset index={dataset_index}")
        print(f"Instruction: {instruction}")
        print(f"Prompt: {prompt}")
        print(f"Negative prompt: {args.negative_prompt}")

        with torch.inference_mode():
            output = pipe(**build_generation_kwargs(args, prompt, images, generator))

        pred_frames = extract_frames(output)
        gt_frames = gt_frames_from_dataset(example, len(pred_frames), args.view_index)
        trajectory_frames = get_full_trajectory_video_frames(source_dataset, trajectory_id, base_index, args)
        save_inference_outputs(
            args=args,
            sample_out_dir=sample_out_dir,
            pred_frames=pred_frames,
            gt_frames=gt_frames,
            current_image=images[min(max(args.view_index, 0), len(images) - 1)],
            metadata={
                "dataset_index": dataset_index,
                "trajectory_id": trajectory_id,
                "base_index": base_index,
                "instruction": instruction,
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "prompt_style": args.prompt_style,
                "num_condition_images": len(images),
                "conditioning_mode": args.conditioning_mode,
                "view_index": args.view_index,
                "condition_frames": args.condition_frames,
                "condition_stride": args.condition_stride,
            },
            condition_frames=images,
            trajectory_frames=trajectory_frames,
        )


if __name__ == "__main__":
    main()
