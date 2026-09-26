"""
ALOHA-AGILEX 双臂机器人末端执行器投影可视化脚本（简化版）

功能：
1. 读取 ALOHA-AGILEX URDF，建立运动学链（基座 = footprint，末端 = fl_link6 / fr_link6）
2. 从 LeRobot 数据集读取关节角度（parquet）
3. 用 KDL 做 FK，得到末端在基座坐标系 (footprint) 下的 4x4 位姿矩阵
4. 使用给定的 T_foot_cam（相机在 footprint 坐标系下的外参）转换到相机坐标系
5. 使用相机内参将 3D 末端点投影到每一帧图像上并可视化

相机外参使用固定矩阵（你给的 T_foot_cam）：
T_foot_cam = ^foot T_cam  满足：p_foot = T_foot_cam @ p_cam

代码中会先求逆得到：
T_base_to_camera = T_cam_foot = (T_foot_cam)^(-1)
然后做：
T_cam_ee = T_base_to_camera @ T_base_to_ee
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass , field
import subprocess
import tempfile
import os

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import PyKDL as kdl

"""
"kdl库的FK+IK求解"
python /path/to/workspace/code/robotwin_aloha_ee_projection_kdl2.py \
    --verify_ik

"""

# ^foot T_cam: p_foot = T_foot_cam @ p_cam
T_FOOT_CAM_DEFAULT = np.array(
    [
        [0.6,  0.0,  0.8,   0.2],
        [0.0,  1.0,  0.0,   0.032],
        [-0.8, 0.0,  0.6,   1.35],
        [0.0,  0.0,  0.0,   1.0],
    ],
    dtype=np.float64,
)

# ALOHA 14-dim layout
ALOHA_ACTION_LAYOUT = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}


def slice_action_vector(action: np.ndarray) -> Dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1] != 14:
        raise ValueError(f"ALOHA action/state 最后一维必须是 14，当前 shape={action.shape}")
    return {k: action[..., s:e] for k, (s, e) in ALOHA_ACTION_LAYOUT.items()}

def _as_batch(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[None, :], True
    return x, False


@dataclass
class RobotwinRetargetConfig:
    """模仿 GR1RetargetConfig 的形式，但 Robotwin/Aloha 使用 14-dim state。"""
    urdf_path: str = str(
        Path(__file__).parent / "arx5_description_isaac.urdf"
    )
    camera_intrinsics: Dict[str, float] = field(
        default_factory=lambda: dict(fx=717.28, fy=717.28, cx=320.0, cy=240.0)
    )
    T_foot_cam: np.ndarray = field(default_factory=lambda: T_FOOT_CAM_DEFAULT.copy())

    base_link: str = "footprint"
    left_ee_link: str = "fl_link6"
    right_ee_link: str = "fr_link6"


# ======================= 视频工具函数 =======================

def transcode_video_to_h264(input_path: Path, output_path: Path = None) -> Path:
    """
    将视频转码为 H.264 格式 (兼容 OpenCV)

    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径，如果为 None 则创建临时文件

    返回:
        转码后的视频路径
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        output_path = Path(output_path)

    print(f"正在将视频转码为 H.264 格式: {input_path.name}")

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"转码完成: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"转码失败: {e.stderr.decode()}")
        raise


def check_video_codec(video_path: Path) -> str:
    """检查视频编解码器"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "unknown"

# ======================= 核心类：AlohaRetargeter =======================

class AlohaRetargeter:
    """ALOHA-AGILEX 双臂机器人末端执行器投影器（使用固定 T_foot_cam）"""

    def __init__(
        self,
        urdf_path: Path,
        T_foot_cam: Optional[np.ndarray] = None,
        
    ):
        """
        初始化 ALOHA 投影器

        参数:
            urdf_path: URDF 文件路径
            T_foot_cam: 4x4 外参矩阵 ^foot T_cam（相机坐标系 → footprint）
        """
        print("正在初始化 AlohaRetargeter...")

        cfg = RobotwinRetargetConfig()

        self.T_foot_cam = np.asarray(T_foot_cam if T_foot_cam is not None else cfg.T_foot_cam, dtype=np.float64).reshape(4, 4)
        self.T_base_to_camera = np.linalg.inv(self.T_foot_cam)  # ^cam T_foot

        robot_urdf = URDF.from_xml_file(str(urdf_path))
        tree = kdl_tree_from_urdf_model(robot_urdf)

        self.kin_left_arm = KDLKinematics(robot_urdf, cfg.base_link, cfg.left_ee_link, tree)
        self.kin_right_arm = KDLKinematics(robot_urdf, cfg.base_link, cfg.right_ee_link, tree)

        self.chain_left_arm = tree.getChain(cfg.base_link, cfg.left_ee_link)
        self.chain_right_arm = tree.getChain(cfg.base_link, cfg.right_ee_link)

        self._ik_last_solution = {"left": None, "right": None}
        
    def reset_ik_cache(self, env_idx: int | None = None):
        """
        清空 IK 历史缓存（robotwin 串行评测 env_idx 可忽略）。
        """
        self._ik_last_solution = {"left": None, "right": None}

    # ------------ FK + 末端在相机坐标系下的位姿 ------------

    def process_frame_kinematics_axisangle(
        self,
        action_vector: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        action_vector = np.asarray(action_vector, dtype=np.float64)
        if action_vector.ndim == 3:
            # (B,T,14) -> (B,14) 取最后一步
            action_vector = action_vector[:, -1, :]
        if action_vector.ndim == 1:
            action_vector = action_vector[None, :]
        if action_vector.ndim != 2 or action_vector.shape[-1] != 14:
            raise ValueError(f"action_vector 期望 (14,) 或 (B,14) 或 (B,T,14)，实际 shape={action_vector.shape}")

        action_slices = slice_action_vector(action_vector)
        q_left_all = action_slices["left_arm"]    # (B,6)
        q_right_all = action_slices["right_arm"]  # (B,6)

        B = action_vector.shape[0]
        left_pos = np.zeros((B, 3), dtype=np.float64)
        right_pos = np.zeros((B, 3), dtype=np.float64)
        left_axis = np.zeros((B, 3), dtype=np.float64)
        right_axis = np.zeros((B, 3), dtype=np.float64)

        for b in range(B):
            q_left = q_left_all[b].astype(np.float64).tolist()
            q_right = q_right_all[b].astype(np.float64).tolist()

            T_base_left = np.asarray(self.kin_left_arm.forward(q_left), dtype=np.float64)
            T_base_right = np.asarray(self.kin_right_arm.forward(q_right), dtype=np.float64)

            T_cam_left = self.T_base_to_camera @ T_base_left
            T_cam_right = self.T_base_to_camera @ T_base_right

            left_pos[b] = T_cam_left[:3, 3].reshape(3)
            right_pos[b] = T_cam_right[:3, 3].reshape(3)
            left_axis[b] = R.from_matrix(T_cam_left[:3, :3]).as_rotvec().reshape(3)
            right_axis[b] = R.from_matrix(T_cam_right[:3, :3]).as_rotvec().reshape(3)


        left_q = np.asarray(q_left_all, dtype=np.float64).reshape(B, 6)
        right_q = np.asarray(q_right_all, dtype=np.float64).reshape(B, 6)

        # print(f"----------debug----------")
        # print(f"left_pos:{left_pos}, left_axis:{left_axis}")
        # print(f"right_pos:{right_pos}, right_axis:{right_axis}")
        # print(f"left_q_shape:{left_q.shape}")
        # print(f"left_q:{left_q}")
        # print(f"right_q:{right_q}")

        return (left_pos, left_axis), (right_pos, right_axis), (left_q, right_q)
    
    # ------------ FK/IK 验证（保持原逻辑，可选） ------------
    def verify_ik_solution(self, action_vector: np.ndarray, verbose: bool = True) -> Dict:
        """
        验证 FK -> IK -> FK 的完整流程
        """
        action_slices = slice_action_vector(action_vector)

        q_left_original = list(action_slices["left_arm"])
        q_right_original = list(action_slices["right_arm"])

        T_left_original = np.array(self.kin_left_arm.forward(q_left_original))
        T_right_original = np.array(self.kin_right_arm.forward(q_right_original))

        q_left_ik = self._solve_ik_kdl("left", T_left_original, q_left_original)
        q_right_ik = self._solve_ik_kdl("right", T_right_original, q_right_original)

        T_left_verified = None if q_left_ik is None else np.array(self.kin_left_arm.forward(q_left_ik))
        T_right_verified = None if q_right_ik is None else np.array(self.kin_right_arm.forward(q_right_ik))

        results = {
            "left_arm": self._compute_errors(
                q_left_original, q_left_ik, T_left_original, T_left_verified
            ),
            "right_arm": self._compute_errors(
                q_right_original, q_right_ik, T_right_original, T_right_verified
            ),
        }

        if verbose:
            print("\n=== IK 验证结果 ===")
            for arm_name, result in results.items():
                print(f"\n{arm_name}:")
                if result["ik_success"]:
                    print(f"  关节角度误差 (度): {result['joint_error_deg']:.4f}")
                    print(f"  位置误差 (m): {result['position_error']:.6f}")
                    print(f"  姿态误差 (度): {result['orientation_error_deg']:.4f}")
                else:
                    print("  IK 求解失败")

        return results

    def _solve_ik_kdl(
        self,
        arm: str,
        target_pose: np.ndarray,
        q_init_arm: List[float],
        max_iter: int = 200,
        tol: float = 1e-6,
        damping: float = 1e-3,
    ) -> Optional[List[float]]:
        assert arm in ("left", "right"), f"arm 必须是 'left' 或 'right'，当前为 {arm}"


        chain = self.chain_left_arm if arm == "left" else self.chain_right_arm
        kin = self.kin_left_arm if arm == "left" else self.kin_right_arm
        chain_label = arm

        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        target_frame = self._numpy_to_kdl_frame(target_pose)

        num_joints = chain.getNrOfJoints()
        wide = 2.0 * np.pi
        global_lower = np.full((num_joints,), -wide, dtype=np.float64)
        global_upper = np.full((num_joints,), +wide, dtype=np.float64)

        q_init_arm = np.asarray(q_init_arm, dtype=np.float64).flatten()
        if q_init_arm.shape[0] != num_joints:
            # 防御：长度不对就用 0 初始化
            q_init_arm = np.zeros((num_joints,), dtype=np.float64)

        fallback = self._ik_last_solution.get(chain_label, None)
        if fallback is None:
            fallback = q_init_arm.tolist()

        max_attempts = 5
        eps_schedule = [tol, tol * 2, tol * 5, tol * 10, tol * 20]
        seed_noise = 0.25  # rad

        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        def make_vel_solver(local_damping: float):
            # 优先用 WDLS（更像 Pin 的 damping），没有就退回 pinv
            try:
                vel = kdl.ChainIkSolverVel_wdls(chain)
                vel.setLambda(float(local_damping))
                return vel
            except Exception:
                return kdl.ChainIkSolverVel_pinv(chain)

        for attempt in range(max_attempts):
            local_tol = float(eps_schedule[min(attempt, len(eps_schedule) - 1)])
            local_margin = 0.75 * (attempt + 1)      # rad
            local_damping = float(damping) * (10.0 ** attempt)

            seed = np.asarray(fallback, dtype=np.float64).copy()
            if attempt > 0:
                noise_scale = seed_noise * attempt
                seed = seed + np.random.uniform(-noise_scale, noise_scale, size=seed.shape)

            local_lower = np.maximum(global_lower, seed - local_margin)
            local_upper = np.minimum(global_upper, seed + local_margin)
            bad = local_lower >= local_upper
            if np.any(bad):
                local_lower[bad] = global_lower[bad]
                local_upper[bad] = global_upper[bad]

            q_min = kdl.JntArray(num_joints)
            q_max = kdl.JntArray(num_joints)
            for i in range(num_joints):
                q_min[i] = float(local_lower[i])
                q_max[i] = float(local_upper[i])

            ik_vel_solver = make_vel_solver(local_damping)
            ik_solver = kdl.ChainIkSolverPos_NR_JL(
                chain,
                q_min,
                q_max,
                fk_solver,
                ik_vel_solver,
                maxiter=int(max_iter),
                eps=float(local_tol),
            )

            q_init_kdl = kdl.JntArray(num_joints)
            for i in range(num_joints):
                q_init_kdl[i] = float(np.clip(seed[i], local_lower[i], local_upper[i]))

            q_out = kdl.JntArray(num_joints)
            ret = ik_solver.CartToJnt(q_init_kdl, target_frame, q_out)

            if ret >= 0:
                q_sol = [float(q_out[i]) for i in range(num_joints)]
                # 额外做一次 FK 误差校验，避免“形式成功但误差很大”
                try:
                    T_fk = np.array(kin.forward(q_sol))
                    err = self._compute_errors(q_original=q_sol, q_ik=q_sol, T_original=target_pose, T_ik=T_fk)
                    if err["position_error"] is None:
                        continue
                    # 位置/姿态任意一个达标就接受（KDL 有时姿态收敛慢）
                    if (err["position_error"] < max(1e-4, 10 * local_tol)) or (err["orientation_error_deg"] < 1.0):
                        self._ik_last_solution[chain_label] = q_sol
                        return q_sol
                except Exception:
                    self._ik_last_solution[chain_label] = q_sol
                    return q_sol

        # 全失败：兜底返回（跟 Pin 一样尽量不返回 None）
        if fallback is not None and len(fallback) == num_joints:
            self._ik_last_solution[chain_label] = list(map(float, fallback))
            return list(map(float, fallback))
        return None

    def _numpy_to_kdl_frame(self, T: np.ndarray) -> kdl.Frame:
        """将 numpy 4x4 变换矩阵转换为 PyKDL Frame"""
        rotation = kdl.Rotation(
            T[0, 0], T[0, 1], T[0, 2],
            T[1, 0], T[1, 1], T[1, 2],
            T[2, 0], T[2, 1], T[2, 2],
        )
        position = kdl.Vector(T[0, 3], T[1, 3], T[2, 3])
        return kdl.Frame(rotation, position)

    def _compute_errors(
        self,
        q_original: List[float],
        q_ik: Optional[List[float]],
        T_original: np.ndarray,
        T_ik: Optional[np.ndarray],
    ) -> Dict:
        """计算原始值和 IK 求解值之间的误差"""
        if q_ik is None or T_ik is None:
            return {
                "ik_success": False,
                "joint_error_deg": None,
                "position_error": None,
                "orientation_error_deg": None,
            }

        q_original_arr = np.array(q_original)
        q_ik_arr = np.array(q_ik)
        joint_error = np.linalg.norm(q_original_arr - q_ik_arr)
        joint_error_deg = np.rad2deg(joint_error)

        pos_original = T_original[:3, 3]
        pos_ik = T_ik[:3, 3]
        position_error = np.linalg.norm(pos_original - pos_ik)

        R_original = T_original[:3, :3]
        R_ik = T_ik[:3, :3]
        R_error = R_original.T @ R_ik
        trace = np.trace(R_error)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        orientation_error_deg = np.rad2deg(angle_error)

        return {
            "ik_success": True,
            "joint_error_deg": joint_error_deg,
            "position_error": position_error,
            "orientation_error_deg": orientation_error_deg,
            "q_original": q_original,
            "q_ik": q_ik,
        }

    # ---------------------------------------------------------------------
    # 新增：模仿 Pin 的 verify_ik_solution（从“相机系位姿”转回 base 系做 IK）
    # ---------------------------------------------------------------------
    def inverse_kinematics_from_camera_axisangle(
        self,
        left_hand_pos: np.ndarray,
        left_hand_axisangle: np.ndarray,
        right_hand_pos: np.ndarray,
        right_hand_axisangle: np.ndarray,
        current_action_vector: Optional[np.ndarray] = None,
        q_init_left: Optional[np.ndarray] = None,
        q_init_right: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        damping: float = 1e-3,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        def _as_batch_3(x):
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x[None, :]
            if x.ndim == 3:
                x = x[:, -1, :]
            if x.ndim != 2 or x.shape[-1] != 3:
                raise ValueError(f"期望 (3,) 或 (B,3) 或 (B,T,3)，实际 shape={x.shape}")
            return x

        left_hand_pos = _as_batch_3(left_hand_pos)
        right_hand_pos = _as_batch_3(right_hand_pos)
        left_hand_axisangle = _as_batch_3(left_hand_axisangle)
        right_hand_axisangle = _as_batch_3(right_hand_axisangle)

        B = left_hand_pos.shape[0]

        # seed：优先 q_init_*；否则从 current_action_vector 里取；否则 0
        if q_init_left is None or q_init_right is None:
            if current_action_vector is None:
                q_init_left = np.zeros((B, 6), dtype=np.float64)
                q_init_right = np.zeros((B, 6), dtype=np.float64)
            else:
                a = np.asarray(current_action_vector, dtype=np.float64)
                if a.ndim == 1:
                    a = a[None, :]
                if a.ndim == 3:
                    a = a[:, -1, :]
                if a.shape[-1] != 14:
                    raise ValueError(f"current_action_vector 最后一维必须是 14，实际 shape={a.shape}")
                if a.shape[0] == 1 and B > 1:
                    a = np.repeat(a, B, axis=0)
                q_init_left = a[:, 0:6]
                q_init_right = a[:, 7:13]
        else:
            q_init_left = np.asarray(q_init_left, dtype=np.float64)
            q_init_right = np.asarray(q_init_right, dtype=np.float64)
            if q_init_left.ndim == 1:
                q_init_left = q_init_left[None, :]
            if q_init_right.ndim == 1:
                q_init_right = q_init_right[None, :]
            if q_init_left.shape[0] == 1 and B > 1:
                q_init_left = np.repeat(q_init_left, B, axis=0)
            if q_init_right.shape[0] == 1 and B > 1:
                q_init_right = np.repeat(q_init_right, B, axis=0)
            q_init_left = q_init_left.reshape(B, 6)
            q_init_right = q_init_right.reshape(B, 6)

        ql_out = np.zeros((B, 6), dtype=np.float64)
        qr_out = np.zeros((B, 6), dtype=np.float64)

        for b in range(B):
            T_cam_left = np.eye(4, dtype=np.float64)
            T_cam_left[:3, :3] = R.from_rotvec(left_hand_axisangle[b]).as_matrix()
            T_cam_left[:3, 3] = left_hand_pos[b]

            T_cam_right = np.eye(4, dtype=np.float64)
            T_cam_right[:3, :3] = R.from_rotvec(right_hand_axisangle[b]).as_matrix()
            T_cam_right[:3, 3] = right_hand_pos[b]

            T_base_left = self.T_foot_cam @ T_cam_left
            T_base_right = self.T_foot_cam @ T_cam_right

            ql = self._solve_ik_kdl("left", T_base_left, q_init_left[b].tolist(), max_iter=max_iter, tol=tol, damping=damping)
            qr = self._solve_ik_kdl("right", T_base_right, q_init_right[b].tolist(), max_iter=max_iter, tol=tol, damping=damping)

            if ql is None or qr is None:
                return None, None
            ql_out[b] = np.asarray(ql, dtype=np.float64)[:6]
            qr_out[b] = np.asarray(qr, dtype=np.float64)[:6]

        return ql_out, qr_out

    # ------------ 处理整段 episode：逐帧重投影 ------------

    def process_episode(
        self,
        video_path: Path,
        parquet_path: Path,
        output_path: Path,
        verify_ik: bool = False,
        use_state: bool = True,
    ):
        """
        处理单个 episode：对每一帧做 FK + 投影，并写出新视频

        参数:
            video_path: 输入视频路径
            parquet_path: parquet 数据文件路径
            output_path: 输出视频路径
            verify_ik: 是否执行一次 IK 验证（用第一帧数据）
            use_state: True 使用 observation.state，False 使用 action
        """
        print(f"开始处理 episode: {video_path.parent.name}")

        # 读取数据
        df = pd.read_parquet(parquet_path)
        data_key = "observation.state" if use_state else "action"

        if data_key not in df.columns:
            raise KeyError(f"数据文件中找不到 '{data_key}' 列")

        states = df[data_key].tolist()

        # 检查视频编码，如是 AV1/VP9/HEVC 则转码
        temp_video_path = None
        actual_video_path = video_path

        codec = check_video_codec(video_path)
        print(f"视频编码: {codec}")

        if codec in ["av1", "vp9", "hevc"]:
            print(f"检测到 {codec} 编码，需要转码为 H.264...")
            temp_video_path = transcode_video_to_h264(video_path)
            actual_video_path = temp_video_path

        # 打开视频
        cap = cv2.VideoCapture(str(actual_video_path))
        if not cap.isOpened():
            if temp_video_path and temp_video_path.exists():
                os.unlink(temp_video_path)
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps}fps, {num_frames} 帧")
        print(f"数据总数: {len(states)}")
    
        # 输出视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


        # 验证第70帧的IK：
        # frame_idx = 70
        """
        -0.886147 2.101878 1.805194 -1.295676 0.017645 1.012810 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
        -0.891970 2.105140 1.803110 -1.292860 0.017360 1.014650 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
        """
        history_qpos = np.array([ -0.886147, 2.101878, 1.805194, -1.295676, 0.017645, 1.012810, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000])
        current_qpos = np.array([  -0.891970, 2.105140, 1.803110, -1.292860, 0.017360, 1.014650, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000])
        (left_pos, left_rot), (right_pos, right_rot), (T_base_to_left_ee, T_base_to_right_ee) = self.process_frame_kinematics(current_qpos)
        print(f"FK: T_base_to_left_ee: {T_base_to_left_ee}")
        # 可选：先做一次 IK 验证（第一帧）
        if verify_ik and len(states) > 0:
            print("\n执行 IK 验证（使用第一帧数据）...")
            self.verify_ik_solution(history_qpos, current_qpos, left_pos, right_pos, left_rot, right_rot)

        # 逐帧处理
        for i in tqdm(range(min(num_frames, len(states))), desc="处理视频帧"):
            ok, frame = cap.read()
            if not ok:
                break

            # FK + 坐标变换：得到左右末端在相机坐标系下的位姿
            (left_pos, left_rot), (right_pos, right_rot), _ = self.process_frame_kinematics(
                np.array(states[i])
            )

            # 在图像上画投影点和小坐标轴
            self._draw_projection(frame, left_pos, left_rot, color=(0, 0, 255), frame_idx=i)   # 左手 红色
            self._draw_projection(frame, right_pos, right_rot, color=(255, 0, 0), frame_idx=i)  # 右手 蓝色

            # 帧号文字
            cv2.putText(
                frame,
                f"Frame: {i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            out.write(frame)

        cap.release()
        out.release()

        if temp_video_path and Path(temp_video_path).exists():
            os.unlink(temp_video_path)
            print("已清理临时转码文件")

        print(f"\n处理完成！输出视频: {output_path}")


# ======================= 命令行入口 =======================

def main():
    parser = argparse.ArgumentParser(
        description="ALOHA-AGILEX 双臂机器人末端执行器投影可视化（使用固定 T_foot_cam）"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/path/to/workspace/robotwin_dataset/pick_dual_bottles-aloha-agilex_clean"
        ),
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path(
            "/path/to/workspace/code/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        ),
        help="URDF 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/workspace/robotwin_dataset/output_videos"),
        help="输出目录",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="要处理的 episode 编号",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="cam_high",
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="使用的视频相机名称（决定读取哪个 mp4）",
    )
    parser.add_argument(
        "--verify_ik",
        action="store_true",
        help="是否执行 IK 验证（仅第一帧）",
        required=True,
    )
    parser.add_argument(
        "--use-action",
        action="store_true",
        help="使用 action 而非 observation.state",
    )

    # 相机内参（可选，覆盖默认）
    parser.add_argument("--fx", type=float, default=None, help="相机焦距 fx")
    parser.add_argument("--fy", type=float, default=None, help="相机焦距 fy")
    parser.add_argument("--cx", type=float, default=None, help="主点 cx")
    parser.add_argument("--cy", type=float, default=None, help="主点 cy")

    # 可选：从 RoboTwin 读取内参（只读内参，不再读外参）
    parser.add_argument(
        "--load-intrinsics-from-robotwin",
        action="store_true",
        help="从 RoboTwin 的 _camera_config.yml 自动加载相机内参",
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        default="Large_D435",
        help="RoboTwin 相机类型 (D435/Large_D435/L515/Large_L515)",
    )

    args = parser.parse_args()

    # 输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    

    # 初始化投影器（外参完全由 T_FOOT_CAM_DEFAULT 决定）
    retargeter = AlohaRetargeter(
        urdf_path=args.urdf_path,
        T_foot_cam=T_FOOT_CAM_DEFAULT,
    )
    # 构建输入/输出路径
    episode_str = f"episode_{args.episode:06d}"
    parquet_path = (
        args.dataset_root / f"data/chunk-000/{episode_str}.parquet"
    )
    video_path = (
        args.dataset_root / f"videos/chunk-000/observation.images.{args.camera}/{episode_str}.mp4"
    )
    output_path = (
        args.output_dir / f"{episode_str}_{args.camera}_projected.mp4"
    )

    # 处理 episode
    retargeter.process_episode(
        video_path=video_path,
        parquet_path=parquet_path,
        output_path=output_path,
        verify_ik=args.verify_ik,
        use_state=not args.use_action,
    )


if __name__ == "__main__":
    main()