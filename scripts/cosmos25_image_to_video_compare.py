#!/usr/bin/env python3
"""Generate a Cosmos-Predict2.5 video from current image(s) and compare with GT.

Example:
    python scripts/cosmos25_image_to_video_compare.py \
        --model-path /path/to/Cosmos-Predict2.5-2B-Post-Trained \
        --image current.png \
        --instruction "pick up the mug and place it on the plate" \
        --gt gt.mp4 \
        --out-dir outputs/cosmos25_debug
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


class _DisabledCosmosSafetyChecker:
    """No-op safety checker for local robot data visualization."""

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video, *args, **kwargs):
        return video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF repo id or local diffusers-format Cosmos2.5 checkpoint.")
    parser.add_argument("--revision", default="diffusers/base/post-trained", help="HF revision. Ignored for local paths.")
    parser.add_argument("--image", nargs="+", required=True, help="Current image path(s). One image uses image conditioning; multiple images use video conditioning.")
    parser.add_argument("--instruction", required=True, help="Language instruction, or @path/to/file.txt.")
    parser.add_argument("--gt", default=None, help="Optional GT image/video path for side-by-side comparison.")
    parser.add_argument("--out-dir", default="outputs/cosmos25_image_to_video_compare", help="Directory for generated artifacts.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--conditional-frame-timestep", type=float, default=0.1)
    parser.add_argument("--num-latent-conditional-frames", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda", help="Usually cuda. Use cpu only for very small smoke tests.")
    parser.add_argument("--cpu-offload", action="store_true", help="Use diffusers CPU offload instead of moving the full pipe to CUDA.")
    parser.add_argument("--enable-safety-checker", action="store_true", help="Keep the Cosmos text/video safety checker enabled.")
    return parser.parse_args()


def read_instruction(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8").strip()
    return value


def load_image(path: str) -> Image.Image:
    Image, _, _ = require_pil()
    return Image.open(path).convert("RGB")


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("This script needs torch installed to run Cosmos generation.") from exc
    return torch


def maybe_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def require_av():
    try:
        import av
    except ImportError as exc:
        raise ImportError("This script needs av installed to read/write videos. Install requirements.txt first.") from exc
    return av


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("This script needs numpy installed. Install requirements.txt first.") from exc
    return np


def require_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("This script needs pillow installed. Install requirements.txt first.") from exc
    return Image, ImageDraw, ImageFont


def to_uint8_image(frame) -> Image.Image:
    Image, _, _ = require_pil()
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    np = require_numpy()
    torch = maybe_torch()
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


def normalize_generated_frames(output) -> list[Image.Image]:
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
        return [to_uint8_image(frame) for frame in videos]

    if isinstance(videos, np.ndarray):
        if videos.ndim == 5:
            videos = videos[0]
        return [to_uint8_image(frame) for frame in videos]

    if isinstance(videos, (list, tuple)) and videos:
        first = videos[0]
        if isinstance(first, (list, tuple)):
            return [to_uint8_image(frame) for frame in first]
        if torch is not None and torch.is_tensor(first) and first.ndim >= 4:
            return [to_uint8_image(frame) for frame in first]
        if isinstance(first, np.ndarray) and first.ndim >= 4:
            return [to_uint8_image(frame) for frame in first]
        return [to_uint8_image(frame) for frame in videos]

    raise ValueError("Could not extract frames from Cosmos generation output.")


def read_video(path: Path) -> list[Image.Image]:
    Image, _, _ = require_pil()
    av = require_av()
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"No video frames decoded from {path}.")
    return frames


def read_gt(path: str, target_len: int) -> list[Image.Image]:
    np = require_numpy()
    gt_path = Path(path)
    suffix = gt_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        frame = load_image(str(gt_path))
        return [frame.copy() for _ in range(target_len)]

    frames = read_video(gt_path)
    if len(frames) == target_len:
        return frames
    if len(frames) == 1:
        return [frames[0].copy() for _ in range(target_len)]

    # Resample by nearest index so comparison stays aligned even when FPS/count differs.
    indices = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[int(i)] for i in indices]


def save_video(frames: Iterable[Image.Image], path: Path, fps: int) -> None:
    require_av()
    frames = [frame.convert("RGB") for frame in frames]
    if not frames:
        raise ValueError(f"No frames to save for {path}.")

    width, height = frames[0].size
    encoded_width = width + (width % 2)
    encoded_height = height + (height % 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _save_video_with_codec(frames, path, fps, "h264", encoded_width, encoded_height)
    except Exception:
        _save_video_with_codec(frames, path, fps, "mpeg4", encoded_width, encoded_height)


def _save_video_with_codec(
    frames: list[Image.Image],
    path: Path,
    fps: int,
    codec: str,
    encoded_width: int,
    encoded_height: int,
) -> None:
    av = require_av()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = encoded_width
        stream.height = encoded_height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            if frame.size != (encoded_width, encoded_height):
                canvas = Image.new("RGB", (encoded_width, encoded_height))
                canvas.paste(frame, (0, 0))
                frame = canvas
            video_frame = av.VideoFrame.from_image(frame)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def labelled(frame: Image.Image, label: str) -> Image.Image:
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


def make_compare_frames(pred_frames: list[Image.Image], gt_frames: list[Image.Image] | None) -> list[Image.Image]:
    Image, _, _ = require_pil()
    if gt_frames is None:
        return pred_frames

    compare_frames = []
    for pred, gt in zip(pred_frames, gt_frames):
        gt = gt.resize(pred.size, Image.BILINEAR)
        canvas = Image.new("RGB", (pred.width * 2, pred.height))
        canvas.paste(labelled(gt.copy(), "GT"), (0, 0))
        canvas.paste(labelled(pred.copy(), "Cosmos"), (pred.width, 0))
        compare_frames.append(canvas)
    return compare_frames


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.enable_safety_checker:
        from diffusers.pipelines.cosmos import pipeline_cosmos2_5_predict

        pipeline_cosmos2_5_predict.CosmosSafetyChecker = _DisabledCosmosSafetyChecker

    from diffusers import Cosmos2_5_PredictBasePipeline

    torch = require_torch()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    load_kwargs = {"torch_dtype": dtype}
    if args.revision and not os.path.isdir(args.model_path):
        load_kwargs["revision"] = args.revision

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(args.model_path, **load_kwargs)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    images = [load_image(path) for path in args.image]
    instruction = read_instruction(args.instruction)
    generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu").manual_seed(args.seed)

    generation_kwargs = {
        "prompt": instruction,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "return_dict": True,
        "output_type": "pil",
        "max_sequence_length": args.max_sequence_length,
        "conditional_frame_timestep": args.conditional_frame_timestep,
        "num_latent_conditional_frames": args.num_latent_conditional_frames,
        "generator": generator,
    }
    if args.negative_prompt:
        generation_kwargs["negative_prompt"] = args.negative_prompt
    if len(images) == 1:
        generation_kwargs["image"] = images[0]
    else:
        generation_kwargs["video"] = images

    with torch.inference_mode():
        output = pipe(**generation_kwargs)

    pred_frames = normalize_generated_frames(output)
    gt_frames = read_gt(args.gt, len(pred_frames)) if args.gt else None
    compare_frames = make_compare_frames(pred_frames, gt_frames)

    pred_path = out_dir / "pred.mp4"
    compare_path = out_dir / "compare.mp4"
    last_path = out_dir / "last_frame_compare.png"
    meta_path = out_dir / "run_args.json"

    save_video(pred_frames, pred_path, args.fps)
    save_video(compare_frames, compare_path, args.fps)
    compare_frames[-1].save(last_path)
    meta_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved prediction video: {pred_path}")
    print(f"Saved comparison video: {compare_path}")
    print(f"Saved last-frame comparison: {last_path}")


if __name__ == "__main__":
    main()
