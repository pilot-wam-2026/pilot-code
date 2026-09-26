"""
Pi0 WebSocket 推理服务端，兼容 openpi-client 客户端协议。

- 使用 websockets.asyncio.server，客户端用 WebsocketClientPolicy 连接。
- 首次连接发送 metadata（简单字典）。
- 收到 obs 后，调用 Pi0 推理并返回 {"actions": np.ndarray, "policy_timing": {"infer_ms": ...}}。

支持 joint_position 和 camera_position（通过 state 维度判断：16 -> joint，14 -> camera）。
"""

import asyncio
import http
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import tyro
import websockets.asyncio.server as ws_server
import websockets.frames

import torch

# 兼容 protobuf 版本
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# 路径设置
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_openpi_client_src = _project_root / "workspace" / "openpi-client" / "src"
if _openpi_client_src.exists() and str(_openpi_client_src) not in sys.path:
    sys.path.insert(0, str(_openpi_client_src))

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config
from openpi_client import msgpack_numpy


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    ckpt_path: str = "/path/to/policy_run/checkpoints/model.pt"
    data_statistics_path: str = "/path/to/workspace/VLAPipeline_RL_BY_anonymous/output_train/agibot/benchmark_pro_max/ds_stats.json"
    language_tokenizer_path: Optional[str] = "/path/to/workspace/paligemma-3b-pt-224"
    unnorm_key: Optional[str] = "agibot_genie1"
    policy_setup: str = "agibot_genie1"
    obs_camera_name: List[str] = field(default_factory=lambda: ["head_color", "hand_left", "hand_right"])
    action_scale: float = 1.0
    use_filter: bool = False
    use_multi_action_head: str = "raw"
    use_arm_pose_type: str = "euler"
    use_paligemma_original_vocab_size: bool = True
    n_action_steps: Optional[int] = None
    inference_mode: str = "synchronous"
    # 兼容 pi0inference 期望的其他开关
    use_distillation_distribution: bool = False
    use_embodiment_id: bool = False
    use_vit_for_action_head: bool = False
    use_distillation: bool = False
    use_best_of_n: bool = False
    best_of_n: int = 32
    use_step_action_mask_for_q: bool = False

    gripper_threshold: float = 0.5


# def _build_policy(args: Args) -> LerobotPiFastInference:
#     if not args.data_statistics_path:
#         raise ValueError("需要 --data_statistics_path")

#     return LerobotPiFastInference(
#         args=args,
#         saved_model_path=args.ckpt_path,
#         policy_setup=args.policy_setup,
#         action_scale=args.action_scale,
#         use_filter=args.use_filter,
#         image_name_list=args.obs_camera_name,
#         data_statistics_path=args.data_statistics_path,
#         language_tokenlizer_path=args.language_tokenizer_path,
#         unnorm_key=args.unnorm_key,
#         use_multi_action_head=args.use_multi_action_head,
#         use_arm_pose_type=args.use_arm_pose_type,
#         inference_mode=args.inference_mode,
#     )


def _minmax_to_minus1_1(x: np.ndarray, stats: dict, eps: float = 1e-6) -> np.ndarray:
    """
    x: (..., D)
    stats: contains "min" and "max" (D,)
    return: (..., D) in [-1, 1]
    """
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    denom = np.maximum(hi - lo, eps)
    x01 = (x - lo) / denom
    xn = x01 * 2.0 - 1.0
    return xn.astype(np.float32)

def _sincos_state(x: np.ndarray) -> np.ndarray:
    """
    x: (B, D) or (D,)
    return: (B, 2D) in [-1, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]  # (1, D)
    elif x.ndim != 2:
        raise ValueError(f"Unexpected state shape: {x.shape}")

    sin_x = np.sin(x)
    cos_x = np.cos(x)
    return np.concatenate([sin_x, cos_x], axis=-1).astype(np.float32)  # (B, 2D)


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Args:
        normalized_actions: shape (B, chunk, D) (chunk, D)
        action_norm_stats:
    Returns:
        actions
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    
    normalized_actions = np.clip(normalized_actions, -1, 1)
    
    actions = np.where(
        mask,
        (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
        normalized_actions,
    )
    
    return actions


def _build_policy(args: Args):
    vla = baseframework.from_pretrained(args.ckpt_path)
    vla = vla.to("cuda").eval()

    _, norm_stats = read_mode_config(Path(args.ckpt_path))
    unnorm_key = args.unnorm_key

    # ✅ action stats（你原来就有）
    vla.action_norm_stats = norm_stats[unnorm_key]["action"]

    # ✅ state stats（新增：用于输入归一化，训练一致）
    vla.state_norm_stats = norm_stats[unnorm_key]["state"]

    return vla


def _infer_single(policy, obs: Dict, args: Args, gripper_th: float) -> Dict:
    prompt = obs.get("prompt", "")
    print("prompt", prompt)

    # -------- state: raw -> normalized (min_max to [-1,1]) --------
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    print('state', state.shape, state)
    if state.ndim == 1:
        state = state[None, :]   # (1, D)
    elif state.ndim != 2:
        raise ValueError(f"Unexpected state shape: {state.shape}")

    # ✅ 训练一致：min_max 归一化
    state_norm = _minmax_to_minus1_1(state, policy.state_norm_stats)

    fake_data = {
        "image": [
            obs["observation/image"],
            obs["observation/wrist_left_image"],
            obs["observation/wrist_right_image"],
        ],
        "lang": prompt,
        "state": state_norm,                  # ✅ 用归一化后的 state
    }

    print('img', obs["observation/image"].shape)
    raw = state
    sn = state_norm
    print("state raw min/max:", raw.min(), raw.max())
    print("state_norm min/max:", sn.min(), sn.max(), "clip_ratio:", np.mean((sn <= -0.999) | (sn >= 0.999)))

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = policy.predict_action(fake_data)
    infer_ms = (time.perf_counter() - t0) * 1000

    # out 来自 predict_action
    na = out["normalized_actions"]        # (1, 16, 18) numpy

    # 1️⃣ 去掉 batch 维
    na2 = na[0]                           # (16, 18)

    # 2️⃣ 用 numpy 版 unnormalize（CPU）
    raw2 = unnormalize_actions(
        na2,
        policy.action_norm_stats,
    )                                     # (16, 18)

    # 3️⃣ 如有需要再加回 batch 维
    actions = raw2[None, ...]             # (1, 16, 18)
    print(actions[0, 0])

    return {"actions": actions, "policy_timing": {"infer_ms": infer_ms}}

def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request):
    """健康检查端点"""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class Pi0PolicyServer:
    """Pi0 WebSocket 推理服务端"""

    def __init__(self, pi0_policy, args: Args):
        self._policy = pi0_policy
        self._args = args
        self._metadata = {
            "status": "ok",
            "policy": "pi0",
            "obs_cameras": args.obs_camera_name,
            "action_dim": 14,
        }
        self._packer = msgpack_numpy.Packer()

    def serve_forever(self) -> None:
        """阻塞式启动服务"""
        asyncio.run(self._run())

    async def _run(self):
        """异步运行服务"""
        async with ws_server.serve(
            self._handler,
            self._args.host,
            self._args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            print("✓ 服务已启动，等待客户端连接...")
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection):
        """处理单个客户端连接"""
        print(f"[连接] 客户端 {websocket.remote_address} 已连接")

        # 发送 metadata
        await websocket.send(self._packer.pack(self._metadata))

        while True:
            try:
                data = await websocket.recv()
                if isinstance(data, str):
                    await websocket.send(f"Invalid data type: {type(data)}")
                    continue

                obs = msgpack_numpy.unpackb(data)
                result = _infer_single(self._policy, obs, self._args, self._args.gripper_threshold)
                await websocket.send(self._packer.pack(result))

            except websockets.ConnectionClosed:
                print(f"[断开] 客户端 {websocket.remote_address} 已断开")
                break
            except Exception:
                error_msg = traceback.format_exc()
                print(f"[错误] {error_msg}")
                await websocket.send(error_msg)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                raise


def serve(args: Args):
    print("=" * 60)
    print("Pi0 WebSocket 推理服务端 (openpi-client 兼容)")
    print("=" * 60)
    print(f"  监听: ws://{args.host}:{args.port}")
    print(f"  权重: {args.ckpt_path}")
    print(f"  数据统计: {args.data_statistics_path}")
    print(f"  相机键: {args.obs_camera_name}")
    print(f"  unnorm_key: {args.unnorm_key}")
    print(f"  tokenizer: {args.language_tokenizer_path}")
    print("=" * 60)

    print("[1/2] 初始化模型...")
    pi0_policy = _build_policy(args)
    print("✓ 模型就绪")


    # ===== 在这里加 =====
    print("Has predict_action:", hasattr(pi0_policy, "predict_action"))
    print("Has __call__:", callable(pi0_policy))
    print("Action-related methods:",
          [m for m in dir(pi0_policy) if "action" in m.lower()])

    am = pi0_policy.action_model
    print("past_action_window_size:", getattr(pi0_policy, "past_action_window_size", None))
    print("future_action_window_size:", getattr(pi0_policy, "future_action_window_size", None))
    print("action_model:", type(am))
    print("action_model attrs (state):", [x for x in dir(am) if "state" in x.lower()])

    # ===================

    print("[2/2] 启动 WebSocket 服务...")
    server = Pi0PolicyServer(pi0_policy, args)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 收到中断信号，正在关闭服务...")


def main():
    serve(tyro.cli(Args))


if __name__ == "__main__":
    main()