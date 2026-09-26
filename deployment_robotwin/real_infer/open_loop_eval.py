import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
import tyro

# ====== 你项目里的 import（按你现有路径保留）======
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

# ✅ 关键：导入你训练用的 VideoResizePad（按你项目实际路径改这里）
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoResizePad


# -----------------------------
# 0) 图像：单张图复用 VideoResizePad（resize longest edge -> pad 到 224）
# -----------------------------
def _img_to_torch_nchw(img) -> torch.Tensor:
    """
    Convert input image to torch float tensor [1,C,H,W] in [0,1].
    Supports: np.ndarray(H,W,C) uint8/float, torch.Tensor, PIL.Image
    """
    if isinstance(img, torch.Tensor):
        x = img
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
            x = x.permute(2, 0, 1)     # HWC -> CHW
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Unexpected torch image shape: {x.shape}")
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.unsqueeze(0)     # [1,C,H,W]

    if isinstance(img, np.ndarray):
        x = torch.from_numpy(img)
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4):     # HWC
            x = x.permute(2, 0, 1)     # -> CHW
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Unexpected numpy image shape: {img.shape}")
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.unsqueeze(0)

    # 避免强依赖 PIL：只有你真的传 PIL.Image 才需要
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            x = torch.from_numpy(np.array(img))
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            x = x.permute(2, 0, 1).float()
            if x.max() > 1.5:
                x = x / 255.0
            return x.unsqueeze(0)
    except Exception:
        pass

    raise TypeError(f"Unsupported image type: {type(img)}")


def _torch_chw_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    """
    x: torch.Tensor [C,H,W] float [0,1]
    return: np.uint8 [H,W,C]
    """
    x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return x


class OpenLoopVideoResizePadWrapper:
    """
    复用训练同款 VideoResizePad 的 torchvision resize+pad 逻辑，
    但输入/输出是“单张图”。
    """

    def __init__(self, resize_pad_transform: VideoResizePad, return_numpy_uint8_hwc: bool = True):
        self.t = resize_pad_transform
        self.return_numpy_uint8_hwc = return_numpy_uint8_hwc

    def __call__(self, img):
        # 单图 -> [1,C,H,W]
        x = _img_to_torch_nchw(img)
        # 复用类内部 torchvision resize+pad
        y = self.t._torchvision_resize_pad(x)     # [1,C,S,S]
        y = y.squeeze(0)     # [C,S,S]
        if self.return_numpy_uint8_hwc:
            return _torch_chw_to_uint8_hwc(y)     # [S,S,C] uint8，最保险（兼容你现有 predict_action）
        return y     # torch [C,S,S]


# -----------------------------
# 1) 你已有的归一化/反归一化逻辑（原封不动）
# -----------------------------
def _minmax_to_minus1_1(x: np.ndarray, stats: dict, eps: float = 1e-6) -> np.ndarray:
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    denom = np.maximum(hi - lo, eps)
    x01 = (x - lo) / denom
    xn = x01 * 2.0 - 1.0
    return xn.astype(np.float32)


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    actions = np.where(
        mask,
        (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


# -----------------------------
# 2) Args（按你 server 的 Args 简化必要项）
# -----------------------------
@dataclass
class Args:
    ckpt_path: str = "/path/to/policy_run/checkpoints/model.pt"
    unnorm_key: str = "agibot_genie1"
    gripper_threshold: float = 0.5

    # ✅ 关键：对齐训练图像预处理
    image_size: int = 224
    image_interpolation: str = "linear"
    image_fill_value: float = 0.0


def _build_policy(args: Args):
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()

    cfg, norm_stats = read_mode_config(Path(args.ckpt_path))
    unnorm_key = args.unnorm_key
    cfg = OmegaConf.create(cfg)

    vla.action_norm_stats = norm_stats[unnorm_key]["action"]
    vla.state_norm_stats = norm_stats[unnorm_key]["state"]
    return vla, cfg


def _infer_single(policy, obs: Dict, args: Args, gripper_th: float) -> Dict:
    prompt = obs.get("prompt", "")

    state = np.asarray(obs["observation/state"], dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
    elif state.ndim != 2:
        raise ValueError(f"Unexpected state shape: {state.shape}")

    state_norm = _minmax_to_minus1_1(state, policy.state_norm_stats)

    # padding 到 64
    cur_dim = state_norm.shape[-1]
    if cur_dim < 64:
        pad_width = 64 - cur_dim
        state_norm = np.pad(
            state_norm,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )

    fake_data = {
        "image": [
            obs["observation/image"],
            obs["observation/wrist_left_image"],
            obs["observation/wrist_right_image"],
        ],
        "lang": prompt,
        "state": state_norm,
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = policy.predict_action(fake_data)
    infer_ms = (time.perf_counter() - t0) * 1000

    na = out["normalized_actions"]     # (1, H, D)
    na2 = na[0, :, :18]     # (H, D) 你原逻辑保留

    raw2 = unnormalize_actions(na2, policy.action_norm_stats)     # (H, D)
    actions = raw2[None, ...]     # (1, H, D)

    return {"actions": actions, "policy_timing": {"infer_ms": infer_ms}}


# -----------------------------
# 3) 从 single_ds.get_step_data 组装 obs / gt_chunk
# -----------------------------
def build_obs_from_step_data(step_data: dict, single_ds, img_proc: Optional[callable] = None) -> dict:
    """
    step_data: single_ds.get_step_data(traj_id, base_index) 的 RAW 输出
    返回：_infer_single 所需的 obs dict
    """
    video_keys = single_ds.modality_keys["video"]
    if len(video_keys) < 3:
        raise ValueError(f"Need 3 cameras, got video_keys={video_keys}")

    head = step_data[video_keys[0]][0]
    left = step_data[video_keys[1]][0]
    right = step_data[video_keys[2]][0]

    # ✅ 对齐训练：resize+pad 到 224（或 args.image_size）
    if img_proc is not None:
        head = img_proc(head)
        left = img_proc(left)
        right = img_proc(right)

    # state concat: (T, D_total) -> 取第 0 帧对应 base step
    state_parts = [step_data[k] for k in single_ds.modality_keys["state"]]
    state = np.concatenate(state_parts, axis=1)
    state0 = state[0]

    # language
    lang_key = single_ds.modality_keys["language"][0]
    lang = step_data[lang_key][0]
    if isinstance(lang, list):
        lang = lang[0]

    obs = {
        "observation/image": head,
        "observation/wrist_left_image": left,
        "observation/wrist_right_image": right,
        "observation/state": state0,
        "prompt": lang,
    }
    return obs


def get_gt_action_chunk(step_data: dict, single_ds) -> np.ndarray:
    action_parts = [step_data[k] for k in single_ds.modality_keys["action"]]
    gt = np.concatenate(action_parts, axis=1)     # (H, D)
    return gt


# -----------------------------
# 4) rollout：拼整段 pred/gt（像 gr00t open-loop eval）
# -----------------------------
def rollout_open_loop(
    single_ds,
    pi0_policy,
    args: Args,
    traj_id: int,
    steps: int = 200,
    action_horizon: int = 16,
    img_proc: Optional[callable] = None,
):
    traj_len = int(single_ds.trajectory_lengths[single_ds.get_trajectory_index(traj_id)])
    actual_steps = min(steps, traj_len)

    pred_list = []
    gt_list = []
    infer_ms_list = []

    for t in range(0, actual_steps, action_horizon):
        step_data = single_ds.get_step_data(traj_id, t)

        obs = build_obs_from_step_data(step_data, single_ds, img_proc=img_proc)
        pred = _infer_single(pi0_policy, obs, args, args.gripper_threshold)

        pred_chunk = pred["actions"][0]     # (H, D)
        gt_chunk = get_gt_action_chunk(step_data, single_ds)     # (H, D)

        remain = actual_steps - t
        pred_chunk = pred_chunk[:remain]
        gt_chunk = gt_chunk[:remain]

        pred_list.append(pred_chunk)
        gt_list.append(gt_chunk)
        infer_ms_list.append(pred["policy_timing"]["infer_ms"])

    pred_action_across_time = np.concatenate(pred_list, axis=0)
    gt_action_across_time = np.concatenate(gt_list, axis=0)

    assert pred_action_across_time.shape == gt_action_across_time.shape, (
        f"pred={pred_action_across_time.shape}, gt={gt_action_across_time.shape}")
    return pred_action_across_time, gt_action_across_time, infer_ms_list


# -----------------------------
# 5) 画图（整段轨迹）
# -----------------------------
def plot_action_curves(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    dims: Optional[List[int]] = None,
    title: str = "",
):
    T, D = gt.shape
    if dims is None:
        dims = list(range(D))

    n = len(dims)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 2.2 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    x = np.arange(T)
    for i, d in enumerate(dims):
        ax = axes[i]
        ax.plot(x, gt[:, d], label="gt")
        ax.plot(x, pred[:, d], label="pred")
        ax.set_ylabel(f"dim {d}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_over_time(gt: np.ndarray, pred: np.ndarray, save_path: str, title: str = ""):
    err = pred - gt
    mse_t = np.mean(err**2, axis=1)
    mae_t = np.mean(np.abs(err), axis=1)

    x = np.arange(len(mse_t))
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x, mse_t, label="MSE per step")
    plt.plot(x, mae_t, label="MAE per step")
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------
# 6) main：把 dataset + policy 接起来
# -----------------------------
def main(args: Args):
    # ===== load policy =====
    pi0_policy, cfg = _build_policy(args)
    print("policy loaded")

    # ===== 构造与训练一致的 VideoResizePad =====
    # 训练时写的是 VideoResizePad(apply_to=[video_key], size=224, interpolation="linear", fill_value=0.0)
    # 这里我们直接复用其内部 torchvision resize+pad 实现
    resize_pad = VideoResizePad(
        apply_to=["video"],     # 占位即可
        size=args.image_size,
        interpolation=args.image_interpolation,
        fill_value=args.image_fill_value,
        backend="torchvision",     # 确保走 torchvision
    )
    img_proc = OpenLoopVideoResizePadWrapper(resize_pad, return_numpy_uint8_hwc=True)

    # ===== 载入训练 yaml 的 dataset 配置 =====
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.task_id = "all"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    single_ds = dataset.datasets[0] if hasattr(dataset, "datasets") else dataset

    # 选一条 traj
    traj_id = int(single_ds.trajectory_ids[0])
    print("Using traj_id:", traj_id)

    # ===== rollout open-loop =====
    pred, gt, infer_ms_list = rollout_open_loop(
        single_ds,
        pi0_policy,
        args,
        traj_id=traj_id,
        steps=200,
        action_horizon=30,
        img_proc=img_proc,     # ✅ 关键：评测也做 resize+pad
    )

    mse = np.mean((pred - gt)**2)
    mae = np.mean(np.abs(pred - gt))
    print(f"Open-loop MSE={mse:.6f}, MAE={mae:.6f}")
    print(f"Infer ms: mean={np.mean(infer_ms_list):.2f}, p95={np.percentile(infer_ms_list, 95):.2f}")

    out_dir = Path("outputs/open_loop_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_action_curves(
        gt,
        pred,
        save_path=str(out_dir / f"traj{traj_id}_action_curves.png"),
     # dims=list(range(min(gt.shape[1], 18))),  # 只看前 18 维就打开
        title=f"traj {traj_id} action curves (T={gt.shape[0]}, D={gt.shape[1]})",
    )
    plot_error_over_time(
        gt,
        pred,
        save_path=str(out_dir / f"traj{traj_id}_error.png"),
        title=f"traj {traj_id} error over time",
    )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
