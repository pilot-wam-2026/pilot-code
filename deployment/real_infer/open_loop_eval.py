import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
import tyro

# ====== 你项目里的 import（按你现有路径保留）======
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


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

    na = out["normalized_actions"]  # (1, 16, D)
    na2 = na[0]                     # (16, D)

    raw2 = unnormalize_actions(na2, policy.action_norm_stats)  # (16, D)
    actions = raw2[None, ...]       # (1, 16, D)

    return {"actions": actions, "policy_timing": {"infer_ms": infer_ms}}


# -----------------------------
# 3) 从 single_ds.get_step_data 组装 obs / gt_chunk
# -----------------------------
def build_obs_from_step_data(step_data: dict, single_ds) -> dict:
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

    # state concat: (T, D_total) -> 取第 0 帧对应 base step
    state_parts = [step_data[k] for k in single_ds.modality_keys["state"]]
    state = np.concatenate(state_parts, axis=1)
    state0 = state[0]

    # language（通常是 annotation.xxx），get_language 返回 list[str]
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
    gt = np.concatenate(action_parts, axis=1)  # (H, D)
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
):
    """
    返回：
      pred_action_across_time: (actual_steps, D)
      gt_action_across_time:   (actual_steps, D)
      infer_ms_list: 每次推理耗时
    """
    traj_len = int(single_ds.trajectory_lengths[single_ds.get_trajectory_index(traj_id)])
    actual_steps = min(steps, traj_len)

    pred_list = []
    gt_list = []
    infer_ms_list = []

    # 每 action_horizon 步推理一次，把 chunk 展开 append
    for t in range(0, actual_steps, action_horizon):
        step_data = single_ds.get_step_data(traj_id, t)

        obs = build_obs_from_step_data(step_data, single_ds)
        pred = _infer_single(pi0_policy, obs, args, args.gripper_threshold)

        pred_chunk = pred["actions"][0]  # (H, D)
        gt_chunk = get_gt_action_chunk(step_data, single_ds)  # (H, D)

        # 可能最后一段不足 action_horizon（如果 traj 尾巴短），这里裁一下
        remain = actual_steps - t
        pred_chunk = pred_chunk[:remain]
        gt_chunk = gt_chunk[:remain]

        pred_list.append(pred_chunk)
        gt_list.append(gt_chunk)
        infer_ms_list.append(pred["policy_timing"]["infer_ms"])

    pred_action_across_time = np.concatenate(pred_list, axis=0)
    gt_action_across_time = np.concatenate(gt_list, axis=0)

    assert pred_action_across_time.shape == gt_action_across_time.shape, (
        f"pred={pred_action_across_time.shape}, gt={gt_action_across_time.shape}"
    )
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
    """
    画每个维度一张子图（很多维会很长，但 debug 很直观）
    """
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
    """
    画每个时间步的 L2/MSE（按维度平均） 和 MAE（按维度平均）
    """
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
    # ===== 2) load policy =====
    pi0_policy, cfg = _build_policy(args)
    print("policy loaded")

    # ===== 1) 载入训练 yaml，复用你的 dataset 配置 =====
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.task_id = "all"  # 或者某个 task_id
    # vla_dataset_cfg.data_mix = "suqian_test"

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # get_vla_dataset 可能返回 mixture（里面有 datasets 列表）
    single_ds = dataset.datasets[0] if hasattr(dataset, "datasets") else dataset

    # 选一条 traj
    traj_id = int(single_ds.trajectory_ids[0])
    print("Using traj_id:", traj_id)

    # ===== 3) rollout open-loop =====
    pred, gt, infer_ms_list = rollout_open_loop(
        single_ds,
        pi0_policy,
        args,
        traj_id=traj_id,
        steps=200,
        action_horizon=30,
    )

    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    print(f"Open-loop MSE={mse:.6f}, MAE={mae:.6f}")
    print(f"Infer ms: mean={np.mean(infer_ms_list):.2f}, p95={np.percentile(infer_ms_list, 95):.2f}")

    out_dir = Path("outputs/open_loop_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 4) 画图 =====
    plot_action_curves(
        gt, pred,
        save_path=str(out_dir / f"traj{traj_id}_action_curves.png"),
        # dims=list(range(min(gt.shape[1], 18))),  # 你想只看前 18 维就开这行
        title=f"traj {traj_id} action curves (T={gt.shape[0]}, D={gt.shape[1]})",
    )
    plot_error_over_time(
        gt, pred,
        save_path=str(out_dir / f"traj{traj_id}_error.png"),
        title=f"traj {traj_id} error over time",
    )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
