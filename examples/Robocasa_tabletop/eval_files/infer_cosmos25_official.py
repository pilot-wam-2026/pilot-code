#!/usr/bin/env python3
"""Run official nvidia-cosmos/cosmos-predict2.5 on RoboCasa samples.

This wrapper reads RoboCasa samples with the same dataset path as training,
writes official Cosmos-Predict2.5 JSON inference requests, calls the native
`examples/inference.py` entrypoint, then saves input/traj/pred/compare videos.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Robocasa_tabletop.eval_files.infer_cosmos25_future_video import (
    add_label,
    get_condition_video_frames,
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
    ROBOCASA_NEGATIVE_PROMPT,
    build_prompt,
)


DEFAULT_COSMOS_REPO = "/path/to/workspace/code/cosmos-predict2.5"
DEFAULT_COSMOS_CKPT_ROOT = "/path/to/workspace/models/Cosmos-Predict2.5-2B"
DEFAULT_COSMOS_CKPT = (
    "/path/to/workspace/models/Cosmos-Predict2.5-2B/"
    "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
DEFAULT_COSMOS_TOKENIZER_PATH = "/path/to/workspace/models/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-repo", default=DEFAULT_COSMOS_REPO)
    parser.add_argument("--ckpt-path", default=DEFAULT_COSMOS_CKPT_ROOT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--out-dir", default="outputs/robocasa_cosmos25_official")
    parser.add_argument("--config-yaml", default="./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml")
    parser.add_argument("--data-root-dir", default="/path/to/workspace/datasets")
    parser.add_argument("--data-mix", default="robocasa_teleop_ee")
    parser.add_argument("--dataset-mode", default="train", choices=("train", "val"))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--conditioning-mode", choices=("image", "video"), default="image")
    parser.add_argument("--condition-frames", type=int, default=2, help="Cosmos base supports 1 image frame or 2 latent video frames.")
    parser.add_argument("--condition-stride", type=int, default=1)
    parser.add_argument("--model", default="2B/post-trained")
    parser.add_argument("--cosmos-tokenizer-path", default=DEFAULT_COSMOS_TOKENIZER_PATH)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--resolution", default="none", help='Official Cosmos resolution string, e.g. "704,1280"; "none" uses model default.')
    parser.add_argument("--num-output-frames", type=int, default=77)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--disable-guardrails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-diffusion-model", action="store_true")
    parser.add_argument("--offload-text-encoder", action="store_true")
    parser.add_argument("--offload-tokenizer", action="store_true")
    parser.add_argument("--cosmos-entrypoint", choices=("direct", "native"), default="direct")
    parser.add_argument("--prompt", default=None, help="Override dataset instruction.")
    parser.add_argument("--prompt-style", default="robocasa", choices=("robocasa", "raw"))
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--negative-prompt", default=ROBOCASA_NEGATIVE_PROMPT)
    parser.add_argument(
        "--use-compat-sitecustomize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject local inference-only compatibility shims before importing official Cosmos.",
    )
    parser.add_argument(
        "--patch-cosmos-repo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Patch the official Cosmos repo in-place. Off by default; the compat shim is preferred.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_cosmos_ckpt(path: str) -> str:
    root = Path(path)
    if root.is_file():
        return str(root)
    candidate = root / "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
    if candidate.exists():
        return str(candidate)
    matches = sorted(root.glob("base/post-trained/*_ema_bf16.pt"))
    if matches:
        return str(matches[0])
    return path


def cosmos_pythonpath(cosmos_repo: str, env: dict[str, str]) -> str:
    repo = Path(cosmos_repo)
    paths = [
        str(repo),
        str(repo / "packages/cosmos-oss"),
        str(repo / "packages/cosmos-cuda"),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def write_cosmos_compat_sitecustomize(compat_dir: Path) -> Path:
    """Create subprocess-only import shims for optional Cosmos training/storage deps."""
    compat_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = compat_dir / "sitecustomize.py"
    sitecustomize.write_text(
        r'''
"""Inference-only compatibility shims for the official Cosmos-Predict2.5 CLI.

The native entrypoint imports some training/cloud-storage modules at module load
time. RoboCasa evaluation only needs local-file inference, so these shims keep
optional dependencies from blocking import while still failing loudly if a
training-only object is actually used.
"""

import math
import sys
import types
import importlib.machinery

try:
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover
    importlib_metadata = None


def _unavailable(*args, **kwargs):
    raise RuntimeError("This optional Cosmos dependency is not available in the inference shim.")


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if "transformer_engine" not in sys.modules:
    te = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_attention = types.ModuleType("transformer_engine.pytorch.attention")
    te_rope = types.ModuleType("transformer_engine.pytorch.attention.rope")
    te_float8_tensor = types.ModuleType("transformer_engine.pytorch.float8_tensor")
    te.__spec__ = importlib.machinery.ModuleSpec("transformer_engine", loader=None, is_package=True)
    te_pytorch.__spec__ = importlib.machinery.ModuleSpec("transformer_engine.pytorch", loader=None, is_package=True)
    te_attention.__spec__ = importlib.machinery.ModuleSpec(
        "transformer_engine.pytorch.attention",
        loader=None,
        is_package=True,
    )
    te_rope.__spec__ = importlib.machinery.ModuleSpec("transformer_engine.pytorch.attention.rope", loader=None)
    te_float8_tensor.__spec__ = importlib.machinery.ModuleSpec(
        "transformer_engine.pytorch.float8_tensor",
        loader=None,
    )
    te.__path__ = []
    te_pytorch.__path__ = []
    te_attention.__path__ = []

    if nn is not None:
        te_pytorch.RMSNorm = nn.RMSNorm
    else:
        te_pytorch.RMSNorm = _unavailable

    class DotProductAttention(nn.Module if nn is not None else object):
        def __init__(self, *args, **kwargs):
            if nn is not None:
                super().__init__()

        def forward(self, query_layer, key_layer, value_layer, *args, **kwargs):
            if torch is None:
                return _unavailable()
            q = query_layer
            k = key_layer
            v = value_layer
            scale = 1.0 / math.sqrt(q.shape[-1])
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

    def apply_rotary_pos_emb(t, freqs, *args, **kwargs):
        return t

    class Float8Tensor:
        pass

    te_attention.DotProductAttention = DotProductAttention
    te_attention.apply_rotary_pos_emb = apply_rotary_pos_emb
    te_rope.apply_rotary_pos_emb = apply_rotary_pos_emb
    te_float8_tensor.Float8Tensor = Float8Tensor
    te_pytorch.attention = te_attention
    te_pytorch.float8_tensor = te_float8_tensor
    te.pytorch = te_pytorch
    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = te_pytorch
    sys.modules["transformer_engine.pytorch.attention"] = te_attention
    sys.modules["transformer_engine.pytorch.attention.rope"] = te_rope
    sys.modules["transformer_engine.pytorch.float8_tensor"] = te_float8_tensor

    if importlib_metadata is not None:
        _orig_version = importlib_metadata.version
        _orig_distribution = importlib_metadata.distribution

        class _TransformerEngineDistribution:
            version = "0.0.0"
            metadata = {"Name": "transformer-engine", "Version": version}

            def read_text(self, filename):
                return ""

            @property
            def files(self):
                return []

        def _version(name):
            normalized = name.replace("_", "-").lower()
            if normalized in {"transformer-engine", "transformer-engine-torch"}:
                return "0.0.0"
            return _orig_version(name)

        def _distribution(name):
            normalized = name.replace("_", "-").lower()
            if normalized in {"transformer-engine", "transformer-engine-torch"}:
                return _TransformerEngineDistribution()
            return _orig_distribution(name)

        importlib_metadata.version = _version
        importlib_metadata.distribution = _distribution


if "transformer_engine_torch" not in sys.modules:
    tex = types.ModuleType("transformer_engine_torch")
    tex.__spec__ = importlib.machinery.ModuleSpec("transformer_engine_torch", loader=None)
    tex.multi_tensor_adam = _unavailable
    tex.multi_tensor_adam_capturable = _unavailable
    tex.multi_tensor_adam_capturable_master = _unavailable
    sys.modules["transformer_engine_torch"] = tex


if "megatron.core.parallel_state" not in sys.modules:
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    megatron.__spec__ = importlib.machinery.ModuleSpec("megatron", loader=None, is_package=True)
    megatron_core.__spec__ = importlib.machinery.ModuleSpec("megatron.core", loader=None, is_package=True)
    parallel_state.__spec__ = importlib.machinery.ModuleSpec("megatron.core.parallel_state", loader=None)
    megatron.__path__ = []
    megatron_core.__path__ = []

    _MODEL_PARALLEL_INITIALIZED = False
    parallel_state.cp_size_t = 1

    class ModelParallelConfig:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def initialize_model_parallel(*args, **kwargs):
        global _MODEL_PARALLEL_INITIALIZED
        _MODEL_PARALLEL_INITIALIZED = True

    def destroy_model_parallel(*args, **kwargs):
        global _MODEL_PARALLEL_INITIALIZED
        _MODEL_PARALLEL_INITIALIZED = False

    def is_initialized(*args, **kwargs):
        return _MODEL_PARALLEL_INITIALIZED

    def _none(*args, **kwargs):
        return None

    def _one(*args, **kwargs):
        return 1

    def _zero(*args, **kwargs):
        return 0

    parallel_state.initialize_model_parallel = initialize_model_parallel
    parallel_state.destroy_model_parallel = destroy_model_parallel
    parallel_state.is_initialized = is_initialized
    parallel_state.get_context_parallel_group = _none
    parallel_state.get_context_parallel_world_size = _one
    parallel_state.get_context_parallel_rank = _zero
    parallel_state.get_data_parallel_group = _none
    parallel_state.get_data_parallel_world_size = _one
    parallel_state.get_data_parallel_rank = _zero
    parallel_state.get_tensor_model_parallel_group = _none
    parallel_state.get_tensor_model_parallel_world_size = _one
    parallel_state.get_tensor_model_parallel_rank = _zero
    parallel_state.get_pipeline_model_parallel_group = _none
    parallel_state.get_pipeline_model_parallel_world_size = _one
    parallel_state.get_pipeline_model_parallel_rank = _zero
    parallel_state.get_virtual_pipeline_model_parallel_rank = _zero
    parallel_state.is_pipeline_first_stage = lambda *args, **kwargs: True
    parallel_state.is_pipeline_last_stage = lambda *args, **kwargs: True
    parallel_state.model_parallel_is_initialized = is_initialized

    megatron_core.parallel_state = parallel_state
    megatron_core.ModelParallelConfig = ModelParallelConfig
    megatron.core = megatron_core
    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = megatron_core
    sys.modules["megatron.core.parallel_state"] = parallel_state


if "multistorageclient" not in sys.modules:
    msc = types.ModuleType("multistorageclient")

    class StorageClient:
        def __init__(self, *args, **kwargs):
            _unavailable()

    class StorageClientConfig:
        def __init__(self, *args, **kwargs):
            _unavailable()

    msc.StorageClient = StorageClient
    msc.StorageClientConfig = StorageClientConfig
    sys.modules["multistorageclient"] = msc


try:
    import transformers.cache_utils as cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = getattr(
            cache_utils,
            "DynamicCache",
            getattr(cache_utils, "Cache", object),
        )
except Exception:
    pass
'''.lstrip(),
        encoding="utf-8",
    )
    return sitecustomize


def write_cosmos_direct_runner(compat_dir: Path) -> Path:
    """Write a minimal local-file Cosmos runner that bypasses official Inference."""
    compat_dir.mkdir(parents=True, exist_ok=True)
    runner = compat_dir / "cosmos_direct_runner.py"
    runner.write_text(
        r'''
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="2B/post-trained")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--config-file", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--offload-diffusion-model", action="store_true")
    parser.add_argument("--offload-text-encoder", action="store_true")
    parser.add_argument("--offload-tokenizer", action="store_true")
    return parser.parse_args()


def save_video_av(frames: np.ndarray, path: Path, fps: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def resolve_setup(args):
    from cosmos_predict2.config import SetupArguments

    setup = SetupArguments.model_validate(
        {
            "output_dir": str(args.output_dir),
            "model": args.model,
            "checkpoint_path": args.checkpoint_path,
            "experiment": args.experiment,
            "config_file": args.config_file,
            "disable_guardrails": True,
            "offload_diffusion_model": args.offload_diffusion_model,
            "offload_text_encoder": args.offload_text_encoder,
            "offload_tokenizer": args.offload_tokenizer,
        }
    )
    return setup


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    setup = resolve_setup(args)

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    experiment_opts = []
    if setup.model_key.distilled:
        experiment_opts.append("model.config.init_student_with_teacher=False")
    if args.tokenizer_path:
        experiment_opts.append(f"+model.config.tokenizer.vae_pth={args.tokenizer_path}")

    torch.enable_grad(False)
    pipe = Video2WorldInference(
        experiment_name=setup.experiment,
        ckpt_path=setup.checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=setup.config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=setup.offload_diffusion_model,
        offload_text_encoder=setup.offload_text_encoder,
        offload_tokenizer=setup.offload_tokenizer,
    )

    inference_type = str(request.get("inference_type", "video2world"))
    num_input_frames = 1 if inference_type == "image2world" else 2
    video = pipe.generate_vid2world(
        prompt=request["prompt"],
        input_path=request["input_path"],
        guidance=int(request.get("guidance", 7)),
        num_video_frames=int(request.get("num_output_frames", 77)),
        num_latent_conditional_frames=num_input_frames,
        resolution=request.get("resolution", "none"),
        seed=int(request.get("seed", 0)),
        negative_prompt=request.get("negative_prompt", ""),
        num_steps=int(request.get("num_steps", 35)),
    )
    video = ((1.0 + video[0]) / 2.0).clamp(0.0, 1.0)
    frames = (video * 255.0).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    save_path = output_dir / f"{request.get('name', 'pred')}.mp4"
    save_video_av(frames, save_path, fps=16)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )
    return runner


def patch_cosmos_lazy_guardrail_import(cosmos_repo: str) -> None:
    """Avoid importing guardrail/easy_io storage backends when guardrails are disabled."""
    inference_path = Path(cosmos_repo) / "cosmos_predict2/inference.py"
    if not inference_path.exists():
        return
    text = inference_path.read_text(encoding="utf-8")
    original = text
    text = text.replace(
        "from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets\n",
        "",
    )
    marker = "from cosmos_predict2.config import InferenceArguments, SetupArguments, path_to_str\n"
    replacement = marker + "\n\ndef get_guardrail_presets():\n    from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets\n\n    return guardrail_presets\n"
    if "def get_guardrail_presets():" not in text and marker in text:
        text = text.replace(marker, replacement)
    text = text.replace(
        "            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(\n",
        "            guardrail_presets = get_guardrail_presets()\n            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(\n",
    )
    text = text.replace(
        "                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):\n",
        "                guardrail_presets = get_guardrail_presets()\n                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):\n",
    )
    text = text.replace(
        "                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)\n",
        "                guardrail_presets = get_guardrail_presets()\n                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)\n",
    )
    if text != original:
        inference_path.write_text(text, encoding="utf-8")
        print(f"Patched Cosmos lazy guardrail import: {inference_path}")


def save_compare_video(input_frames, traj_frames, pred_frames, save_path: Path, fps: int) -> None:
    Image, _, _ = require_pil()
    target_size = pred_frames[0].size
    input_video = [to_pil_rgb(input_frames[min(i, len(input_frames) - 1)]).resize(target_size) for i in range(len(pred_frames))]
    if len(traj_frames) == 1:
        traj_video = [to_pil_rgb(traj_frames[0]).resize(target_size) for _ in pred_frames]
    else:
        traj_video = [
            to_pil_rgb(traj_frames[round(i * (len(traj_frames) - 1) / max(len(pred_frames) - 1, 1))]).resize(target_size)
            for i in range(len(pred_frames))
        ]

    compare_frames = []
    for input_frame, traj_frame, pred_frame in zip(input_video, traj_video, pred_frames):
        panels = [
            add_label(input_frame.copy(), "Input"),
            add_label(traj_frame.copy(), "Traj"),
            add_label(pred_frame.copy(), "Cosmos"),
        ]
        canvas = Image.new("RGB", (target_size[0] * 3, target_size[1]))
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * target_size[0], 0))
        compare_frames.append(canvas)
    save_video(compare_frames, save_path, fps=fps)


def build_cosmos_command(
    args: argparse.Namespace,
    request_path: Path,
    native_out_dir: Path,
    direct_runner_path: Path | None = None,
) -> list[str]:
    if args.cosmos_entrypoint == "direct":
        if direct_runner_path is None:
            raise ValueError("direct_runner_path is required for --cosmos-entrypoint direct")
        cmd = [
            args.python,
            str(direct_runner_path),
            "--request",
            str(request_path),
            "--output-dir",
            str(native_out_dir),
            "--model",
            args.model,
            "--checkpoint-path",
            args.ckpt_path,
        ]
        if args.cosmos_tokenizer_path:
            cmd.extend(["--tokenizer-path", args.cosmos_tokenizer_path])
    else:
        cmd = [
            args.python,
            "examples/inference.py",
            "-i",
            str(request_path),
            "-o",
            str(native_out_dir),
            "--model",
            args.model,
            "--checkpoint-path",
            args.ckpt_path,
        ]
    if args.experiment:
        cmd.extend(["--experiment", args.experiment])
    if args.cosmos_entrypoint == "native" and args.disable_guardrails:
        cmd.append("--disable-guardrails")
    if args.offload_diffusion_model:
        cmd.append("--offload-diffusion-model")
    if args.offload_text_encoder:
        cmd.append("--offload-text-encoder")
    if args.offload_tokenizer:
        cmd.append("--offload-tokenizer")
    return cmd


def main() -> None:
    args = parse_args()
    args.ckpt_path = resolve_cosmos_ckpt(args.ckpt_path)
    if args.patch_cosmos_repo and args.disable_guardrails:
        patch_cosmos_lazy_guardrail_import(args.cosmos_repo)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compat_dir = out_dir / "_cosmos_compat"
    direct_runner_path = None
    if args.use_compat_sitecustomize:
        sitecustomize_path = write_cosmos_compat_sitecustomize(compat_dir)
        print(f"Using Cosmos compat shim: {sitecustomize_path}")
    if args.cosmos_entrypoint == "direct":
        direct_runner_path = write_cosmos_direct_runner(compat_dir)
        print(f"Using direct Cosmos runner: {direct_runner_path}")
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    dataset = load_training_dataset(args)
    sample_indices = parse_sample_indices(args)

    for sample_ord, dataset_index in enumerate(sample_indices):
        source_dataset, trajectory_id, base_index = resolve_sample_location(dataset, dataset_index)
        example = dataset[dataset_index]

        sample_out_dir = out_dir / f"sample_{sample_ord:03d}_idx_{dataset_index:06d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        native_out_dir = sample_out_dir / "native_outputs"
        native_out_dir.mkdir(parents=True, exist_ok=True)

        raw_instruction = args.prompt or str(example["lang"])
        prompt = build_prompt(args, raw_instruction)
        condition_args = argparse.Namespace(**vars(args))
        if args.conditioning_mode == "image":
            condition_args.condition_frames = 1
        else:
            condition_args.condition_frames = 2
        condition_frames = get_condition_video_frames(source_dataset, trajectory_id, base_index, condition_args)
        traj_frames = get_full_trajectory_video_frames(source_dataset, trajectory_id, base_index, args)

        input_path = sample_out_dir / ("cosmos_input.mp4" if args.conditioning_mode == "video" else "cosmos_input.png")
        if args.conditioning_mode == "video":
            save_video(condition_frames, input_path, fps=args.fps)
            inference_type = "video2world"
        else:
            condition_frames[-1].save(input_path)
            inference_type = "image2world"
        save_video(condition_frames, sample_out_dir / "input.mp4", fps=args.fps)
        save_video(traj_frames, sample_out_dir / "traj.mp4", fps=args.fps)

        request = {
            "inference_type": inference_type,
            "name": "pred",
            "input_path": str(input_path),
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "seed": int(args.seed) + sample_ord,
            "guidance": int(args.guidance),
            "resolution": args.resolution,
            "num_output_frames": int(args.num_output_frames),
            "num_steps": int(args.num_steps),
        }
        request_path = sample_out_dir / "cosmos_request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")

        cmd = build_cosmos_command(
            args,
            request_path=request_path,
            native_out_dir=native_out_dir,
            direct_runner_path=direct_runner_path,
        )
        meta = {
            "dataset_index": dataset_index,
            "trajectory_id": trajectory_id,
            "base_index": base_index,
            "instruction": raw_instruction,
            "cosmos_prompt": prompt,
            "cosmos_negative_prompt": args.negative_prompt,
            "request": request,
            "cosmos_command": cmd,
        }
        (sample_out_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[{sample_ord + 1}/{len(sample_indices)}] dataset index={dataset_index}")
        print(f"Instruction: {raw_instruction}")
        print(f"Cosmos prompt: {prompt}")
        print("Cosmos command:")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = cosmos_pythonpath(args.cosmos_repo, env)
        if args.use_compat_sitecustomize:
            env["PYTHONPATH"] = f"{compat_dir}:{env['PYTHONPATH']}"
        subprocess.run(cmd, cwd=args.cosmos_repo, env=env, check=True)

        native_pred_path = native_out_dir / "pred.mp4"
        if not native_pred_path.exists():
            candidates = sorted(native_out_dir.glob("pred*.mp4"))
            if not candidates:
                raise FileNotFoundError(f"Could not find Cosmos output under {native_out_dir}")
            native_pred_path = candidates[0]
        pred_frames = [to_pil_rgb(frame) for frame in read_video(native_pred_path)]
        save_video(pred_frames, sample_out_dir / "pred.mp4", fps=args.fps)
        save_compare_video(condition_frames, traj_frames, pred_frames, sample_out_dir / "compare.mp4", fps=args.fps)
        print(f"Saved prediction video: {sample_out_dir / 'pred.mp4'}")
        print(f"Saved comparison video: {sample_out_dir / 'compare.mp4'}")


if __name__ == "__main__":
    main()
