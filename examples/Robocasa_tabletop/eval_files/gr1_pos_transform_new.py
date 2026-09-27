import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from dataclasses import dataclass, field
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import PyKDL as kdl
# import tensorflow as tf

# --- 辅助函数 (保持不变) ---


@dataclass
class GR1RetargetConfig:
    """Main configuration for GR1 retarget."""

    urdf_path: str = str(
     # (Path(__file__).resolve().parents[3] / "runtime_assets/GR1T2/GR1T2_fourier_hand_6dof.urdf")
        (Path(__file__).resolve().parents[3] / "runtime_assets/GR1T2/GR1T2_fourier_hand_6dof.urdf"))
    camera_intrinsics: dict = field(default_factory=lambda: {"fx": 502.8689, "fy": 502.8689, "cx": 640.0, "cy": 400.0})


ACTION_LAYOUT: Dict[str, tuple[int, int]] = {
    "left_arm": (0, 7),
    "left_hand": (7, 13),
    "left_leg": (13, 19),
    "neck": (19, 22),
    "right_arm": (22, 29),
    "right_hand": (29, 35),
    "right_leg": (35, 41),
    "waist": (41, 44),
}


def slice_action_vector(action: np.ndarray) -> Dict[str, np.ndarray]:
    return {group: np.asarray(action[start:end], dtype=np.float64) for group, (start, end) in ACTION_LAYOUT.items()}


def classify_joint(joint_name: str) -> str:
    name = joint_name.lower()
    if "waist" in name or "torso" in name or "pelvis" in name:
        return "waist"
    if "neck" in name or "head" in name:
        return "neck"
    if "left" in name:
        if "hand" in name or "finger" in name or "thumb" in name:
            return "left_hand"
        if "leg" in name or "hip" in name or "knee" in name or "ankle" in name:
            return "left_leg"
        return "left_arm"
    if "right" in name:
        if "hand" in name or "finger" in name or "thumb" in name:
            return "right_hand"
        if "leg" in name or "hip" in name or "knee" in name or "ankle" in name:
            return "right_leg"
        return "right_arm"
    raise KeyError(f"无法为关节 '{joint_name}' 推断分组。")


def build_joint_array(kin: KDLKinematics, action_slices: Dict[str, np.ndarray], allowed_groups: Iterable[str]) -> List[float]:
    allowed, counters = set(allowed_groups), {g: 0 for g in allowed_groups}
    joint_names = kin.get_joint_names()
    arr = [0.0] * len(joint_names)
    for idx, joint_name in enumerate(joint_names):
        group = classify_joint(joint_name)
        if group not in allowed:
            raise KeyError(f"关节 '{joint_name}' 的分组 '{group}' 不在允许范围 {allowed} 内。")
        values, offset = action_slices.get(group, []), counters.get(group, 0)
        if offset >= len(values):
            raise IndexError(f"分组 '{group}' 的动作长度不足。")
        arr[idx], counters[group] = values[offset], offset + 1
    return arr


# --- 核心重构类 (严格按照 v0.py 逻辑重写) ---
class BodyRetargeter:

    def __init__(self, urdf_path: Path, camera_intrinsics: Dict[str, float]):
        print("正在初始化 BodyRetargeter (v0 逻辑)...")
        self.camera_intrinsics = camera_intrinsics

        robot_urdf = URDF.from_xml_file(str(urdf_path))
        tree = kdl_tree_from_urdf_model(robot_urdf)
        base_link = "torso_link"

        # 创建 v0.py 中使用的所有长运动学链
        # 修改：使用 hand_base_link 作为末端，与 FK (body_retarget_robocasa_eepose_keypoints_v5.py) 保持一致
        self.kin_head = KDLKinematics(robot_urdf, base_link, "head_pitch_link", tree)
        self.kin_left_arm = KDLKinematics(robot_urdf, base_link, "left_hand_pitch_link", tree)
        self.kin_right_arm = KDLKinematics(robot_urdf, base_link, "right_hand_pitch_link", tree)

        # 直接保存 PyKDL chain 以便 IK 求解使用
        self.chain_head = tree.getChain(base_link, "head_pitch_link")
        self.chain_left_arm = tree.getChain(base_link, "left_hand_pitch_link")
        self.chain_right_arm = tree.getChain(base_link, "right_hand_pitch_link")

        # 定义相机在头部坐标系中的固定变换 (外参)
        t_cam_in_head = [2.650 - 2.65017178 + 0.23, -1.944 + 2.174 - 0.23, 1.538 - 1.4475]
        #q_cam_in_head = [-0.206, 0.677, -0.607, 0.206] # w, x, y, z
        #q_cam_in_head = [-0.676, 0.676, 0.205, -0.205]
        q_cam_in_head = [-0.205, 0.676, -0.676, 0.205]
        T_cam_in_head = self._create_transform(t_cam_in_head, q_cam_in_head)
        self.T_head_to_cam = np.linalg.inv(T_cam_in_head)
        self.hand_orientation_axes = [
            ("+x", np.array([1.0, 0.0, 0.0], dtype=np.float64), (0, 0, 255)),
            ("+y", np.array([0.0, 1.0, 0.0], dtype=np.float64), (0, 255, 0)),
            ("+z", np.array([0.0, 0.0, -1.0], dtype=np.float64), (255, 0, 0)),
        ]
        self.hand_orientation_length = 0.12

        self._chain_max_reach = {
            "head": self._compute_chain_max_reach(self.chain_head),
            "left_arm": self._compute_chain_max_reach(self.chain_left_arm),
            "right_arm": self._compute_chain_max_reach(self.chain_right_arm),
        }
        # self._ik_last_solution = {name: None for name in self._chain_max_reach}
        self._workspace_margin = 0.05
        self._ik_local_margin_max = np.deg2rad(90.0)     # 最大局部搜索范围 120度
        self._ik_local_margin = np.deg2rad(60.0)
        self._ik_last_solution = {}
        self._ik_damping = 1e-3     # IK 阻尼因子

        # === 是否使用 URDF 关节限制（设为 False 则使用宽松的 ±π 限制）===
        self._use_urdf_joint_limits = True

        # === 从 URDF 提取真实关节限制 ===
        self._joint_limits = self._extract_joint_limits_from_urdf(robot_urdf)
        self._chain_joint_limits = {
            "left_arm": self._get_chain_joint_limits(self.kin_left_arm),
            "right_arm": self._get_chain_joint_limits(self.kin_right_arm),
            "head": self._get_chain_joint_limits(self.kin_head),
        }

        # 打印关节限制信息（调试用）
        print("\n=== URDF 关节限制 ===")
        for chain_name in ["left_arm", "right_arm"]:
            if chain_name == "left_arm":
                joint_names = self.kin_left_arm.get_joint_names()
            else:
                joint_names = self.kin_right_arm.get_joint_names()
            q_min, q_max = self._chain_joint_limits[chain_name]
            print(f"\n{chain_name}:")
            for i, name in enumerate(joint_names):
                print(f"  {name}: [{np.rad2deg(q_min[i]):.1f}°, {np.rad2deg(q_max[i]):.1f}°]")
        print("=" * 30)

        print("初始化完成。")

    # 添加功能，为每个并行的n_episodes保存最后一个IK解算结果，防止因为其他一个IK求解成功后将其他解覆盖掉
    # === [FIX] helper：确保每个 env_idx 有独立 bucket，避免 batch 覆盖 ===
    def _ensure_env_bucket(self, env_idx: int):
        env_idx = int(env_idx)
        if env_idx not in self._ik_last_solution:
            # 初始化该 env 的所有链 last_solution
            self._ik_last_solution[env_idx] = {name: None for name in self._chain_max_reach}

    def _get_last_solution(self, env_idx: int, chain_label: str):
        self._ensure_env_bucket(env_idx)
        return self._ik_last_solution[int(env_idx)].get(chain_label, None)

    def _set_last_solution(self, env_idx: int, chain_label: str, sol):
        self._ensure_env_bucket(env_idx)
        self._ik_last_solution[int(env_idx)][chain_label] = sol

    # === [FIX] 添加 reset_ik_cache 方法，用于清空 IK 历史缓存（last_solution） ===
    def reset_ik_cache(self, env_idx: Optional[int] = None):
        """
        清空 IK 历史缓存（last_solution）。
        - env_idx=None: 清空所有并行 env 的缓存
        - env_idx=int : 只清空指定 env slot 的缓存
        """
        if env_idx is None:
            self._ik_last_solution.clear()
            return
        self._ik_last_solution.pop(int(env_idx), None)

    def _extract_joint_limits_from_urdf(self, robot_urdf: URDF) -> Dict[str, Tuple[float, float]]:
        """从 URDF 中提取所有关节的限制。"""
        joint_limits = {}
        for joint in robot_urdf.joints:
            if joint.type in ['revolute', 'prismatic'] and joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.pi
                upper = joint.limit.upper if joint.limit.upper is not None else np.pi
                joint_limits[joint.name] = (float(lower), float(upper))
            elif joint.type == 'continuous':
                joint_limits[joint.name] = (-2.0 * np.pi, 2.0 * np.pi)
        return joint_limits

    def _get_chain_joint_limits(self, kin: KDLKinematics) -> Tuple[np.ndarray, np.ndarray]:
        """获取运动学链中所有关节的限制。"""
        joint_names = kin.get_joint_names()
        num_joints = len(joint_names)
        q_min = np.zeros(num_joints)
        q_max = np.zeros(num_joints)

        for i, joint_name in enumerate(joint_names):
            if joint_name in self._joint_limits:
                q_min[i], q_max[i] = self._joint_limits[joint_name]
            else:
                q_min[i], q_max[i] = -np.pi, np.pi
                print(f"警告：未找到关节 '{joint_name}' 的限制，使用默认值 [-π, π]")

        return q_min, q_max

    def _create_transform(self, t: List[float], q: List[float]) -> np.ndarray:
        T = np.eye(4)
        # q 是 (w, x, y, z), scipy 需要 (x, y, z, w)
        T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = t
        return T

    def _compute_chain_max_reach(self, chain: kdl.Chain) -> float:
        reach = 0.0
        for idx in range(chain.getNrOfSegments()):
            segment = chain.getSegment(idx)
            frame = segment.getFrameToTip()
            reach += frame.p.Norm()
        return float(reach)

    def _is_target_within_workspace(self, chain_name: str, target_pose: np.ndarray) -> bool:
        max_reach = self._chain_max_reach.get(chain_name)
        if max_reach is None:
            return True
        distance = float(np.linalg.norm(target_pose[:3, 3]))
        return distance <= max_reach + self._workspace_margin

    def _prepare_fallback_solution(
        self,
        chain_name: str,
        q_init: List[float],
        num_joints: int,
        q_min: kdl.JntArray,
        q_max: kdl.JntArray,
    ) -> Tuple[List[float], str]:
        candidate = [q_init[i] if i < len(q_init) else 0.0 for i in range(num_joints)]
        reason = "使用初始关节作为兜底"

        fallback: List[float] = []
        for i in range(num_joints):
            val = candidate[i] if i < len(candidate) else 0.0
            lower = q_min[i]
            upper = q_max[i]
            fallback.append(float(max(min(val, upper), lower)))
        return fallback, reason

    # 组成44DOF向量的辅助函数
    def _build_full_44dof_vector(obs_dict):
        """
        从包含批处理数据的观测字典中构建一个 (batch_size, 44) 的完整状态向量。
        
        Args:
            obs_dict (Dict[str, np.ndarray]): 观测字典，其中 state 的形状为 (B, T, D)。
                                              B 是批次大小 (环境数量), T 是时间步, D 是特征维度。

        Returns:
            np.ndarray: 形状为 (B, 44) 的状态向量。
        """
        # 定义44-DoF向量中每个部分的起始和结束索引
        layout_44dof = {
            "left_arm": (0, 7),
            "left_hand": (7, 13),
            "left_leg": (13, 19),
            "neck": (19, 22),
            "right_arm": (22, 29),
            "right_hand": (29, 35),
            "right_leg": (35, 41),
            "waist": (41, 44),
        }

        # 从任意一个存在的状态键确定批次大小
        batch_size = 0
        for key in obs_dict:
            if key.startswith("state."):
                batch_size = obs_dict[key].shape[0]
                break

        if batch_size == 0:
            # 如果没有找到任何 state key，无法确定批次大小，返回空数组或抛出错误
            # 这里我们假设至少会有一个 state key
            # 如果可能完全没有state，则需要根据具体情况处理
            # 例如，可以尝试从 "video" key 获取 batch_size
            if "video.ego_view" in obs_dict:
                batch_size = obs_dict["video.ego_view"].shape[0]
            else:     # 默认返回一个 (1, 44) 的零向量
                return np.zeros((1, 44), dtype=np.float64)

        # 初始化一个 (batch_size, 44) 的零矩阵
        full_vector = np.zeros((batch_size, 44), dtype=np.float64)

        # 遍历布局，填充 full_vector
        for part_name, (start, end) in layout_44dof.items():
            obs_key = f"state.{part_name}"

            if obs_key in obs_dict:
                # 提取数据，形状为 (B, T, D)
                data = np.asarray(obs_dict[obs_key])

                # 我们只关心最后一个时间步的数据，其形状为 (B, D)
                last_time_step_data = data[:, -1, :]

                # 将数据填充到 full_vector 的正确位置
                full_vector[:, start:end] = last_time_step_data
            # 如果 obs_key 不在字典中，则该部分将保持为零，符合要求

        return full_vector

    def process_kinematics_dataloader(self,
                                      action_vectors: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        批量处理运动学数据，返回轴角表示。
        
        参数:
            action_vectors: shape (batch_size, 44) 或 (44,) 的动作向量
            
        返回:
            ((left_hand_pos, left_hand_axisangle), (right_hand_pos, right_hand_axisangle))
            - left_hand_pos: shape (batch_size, 3) 或 (3,)
            - left_hand_axisangle: shape (batch_size, 3) 或 (3,)
            - right_hand_pos: shape (batch_size, 3) 或 (3,)
            - right_hand_axisangle: shape (batch_size, 3) 或 (3,)
        """
        # 确保输入是 numpy array（tf.py_function 会自动转换，但这里做个保险）
        action_vectors = np.asarray(action_vectors)

        # 处理单个向量的情况
        if action_vectors.ndim == 1:
            action_vectors = action_vectors[np.newaxis, :]     # shape (1, 44)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = action_vectors.shape[0]

        # 初始化输出数组
        left_hand_positions = np.zeros((batch_size, 3), dtype=np.float64)
        right_hand_positions = np.zeros((batch_size, 3), dtype=np.float64)
        left_hand_axisangles = np.zeros((batch_size, 3), dtype=np.float64)
        right_hand_axisangles = np.zeros((batch_size, 3), dtype=np.float64)

        # 批量处理每个动作向量
        for i in range(batch_size):
            action_vector = action_vectors[i]
            action_slices = slice_action_vector(action_vector)

            # 构建 v0.py 使用的包含 waist 和 neck 的关节数组
            q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
            q_left_arm = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
            q_right_arm = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))

            # 正向运动学
            T_torso_to_head = self.kin_head.forward(q_head)
            T_torso_to_left_hand = self.kin_left_arm.forward(q_left_arm)
            T_torso_to_right_hand = self.kin_right_arm.forward(q_right_arm)

            # 坐标系转换
            T_inv_torso_to_head = np.linalg.inv(T_torso_to_head)
            T_head_to_left_hand = T_inv_torso_to_head @ T_torso_to_left_hand
            T_head_to_right_hand = T_inv_torso_to_head @ T_torso_to_right_hand

            # 最终转换到相机坐标系
            T_cam_to_left_hand = self.T_head_to_cam @ T_head_to_left_hand
            T_cam_to_right_hand = self.T_head_to_cam @ T_head_to_right_hand

            # 提取位置（展平为一维数组）
            left_hand_positions[i] = T_cam_to_left_hand[:3, 3].flatten()
            right_hand_positions[i] = T_cam_to_right_hand[:3, 3].flatten()

            # 提取旋转矩阵并转换为轴角表示
            left_hand_rot_mat = T_cam_to_left_hand[:3, :3]
            right_hand_rot_mat = T_cam_to_right_hand[:3, :3]

            # 使用 scipy 将旋转矩阵转换为轴角（rotvec）
            left_hand_axisangles[i] = R.from_matrix(left_hand_rot_mat).as_rotvec()
            right_hand_axisangles[i] = R.from_matrix(right_hand_rot_mat).as_rotvec()

        # 如果输入是单个向量，则压缩批次维度
        if squeeze_output:
            left_hand_positions = left_hand_positions[0]
            right_hand_positions = right_hand_positions[0]
            left_hand_axisangles = left_hand_axisangles[0]
            right_hand_axisangles = right_hand_axisangles[0]

        return (left_hand_positions, left_hand_axisangles), (right_hand_positions, right_hand_axisangles)

    # FK:
    def process_frame_kinematics(self, action_vector: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        action_slices = slice_action_vector(action_vector)

        # 构建 v0.py 使用的包含 waist 和 neck 的关节数组
        q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
        q_left_arm = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
        q_right_arm = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))

        # 正向运动学
        T_torso_to_head = self.kin_head.forward(q_head)
        T_torso_to_left_hand = self.kin_left_arm.forward(q_left_arm)
        T_torso_to_right_hand = self.kin_right_arm.forward(q_right_arm)

        # 坐标系转换
        T_inv_torso_to_head = np.linalg.inv(T_torso_to_head)
        T_head_to_left_hand = T_inv_torso_to_head @ T_torso_to_left_hand
        T_head_to_right_hand = T_inv_torso_to_head @ T_torso_to_right_hand

        # 最终转换到相机坐标系
        T_cam_to_left_hand = self.T_head_to_cam @ T_head_to_left_hand
        T_cam_to_right_hand = self.T_head_to_cam @ T_head_to_right_hand

        left_hand_pos = np.asarray(T_cam_to_left_hand[:3, 3]).flatten()
        right_hand_pos = np.asarray(T_cam_to_right_hand[:3, 3]).flatten()

        #import pdb; pdb.set_trace()
        left_hand_rot = np.asarray(T_cam_to_left_hand[:3, :3], dtype=np.float64)
        right_hand_rot = np.asarray(T_cam_to_right_hand[:3, :3], dtype=np.float64)

        return (left_hand_pos, left_hand_rot), (right_hand_pos, right_hand_rot)

    # FK + 转轴角表示
    def process_frame_kinematics_axisangle(
            self, action_vector: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        处理运动学数据并返回轴角表示。支持单个向量或批量处理。
        
        参数:
            action_vector: shape (batch_size, 44) 或 (44,) 的动作向量
            
        返回:
            ((left_hand_pos, left_hand_axisangle), (right_hand_pos, right_hand_axisangle))
            - left_hand_pos: shape (batch_size, 3) 或 (3,)
            - left_hand_axisangle: shape (batch_size, 3) 或 (3,)
            - right_hand_pos: shape (batch_size, 3) 或 (3,)
            - right_hand_axisangle: shape (batch_size, 3) 或 (3,)
        """
        # 确保输入是 numpy array
        action_vector = np.asarray(action_vector)

        # 处理单个向量的情况
        if action_vector.ndim == 1:
            action_vector = action_vector[np.newaxis, :]     # shape (1, 44)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = action_vector.shape[0]

        # 初始化输出数组
        left_hand_positions = np.zeros((batch_size, 3), dtype=np.float64)
        right_hand_positions = np.zeros((batch_size, 3), dtype=np.float64)
        left_hand_axisangles = np.zeros((batch_size, 3), dtype=np.float64)
        right_hand_axisangles = np.zeros((batch_size, 3), dtype=np.float64)
        left_qpos_states = np.zeros((batch_size, 7), dtype=np.float64)
        right_qpos_states = np.zeros((batch_size, 7), dtype=np.float64)

        # 批量处理每个动作向量
        for i in range(batch_size):
            action_slices = slice_action_vector(action_vector[i])

            # 新增：从切片中提取并保存左右臂的关节状态
            left_qpos_states[i] = action_slices["left_arm"]
            right_qpos_states[i] = action_slices["right_arm"]

            # 构建 v0.py 使用的包含 waist 和 neck 的关节数组
            q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
            q_left_arm = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
            q_right_arm = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))

            # 正向运动学
            T_torso_to_head = self.kin_head.forward(q_head)
            T_torso_to_left_hand = self.kin_left_arm.forward(q_left_arm)
            T_torso_to_right_hand = self.kin_right_arm.forward(q_right_arm)

            # 坐标系转换
            T_inv_torso_to_head = np.linalg.inv(T_torso_to_head)
            T_head_to_left_hand = T_inv_torso_to_head @ T_torso_to_left_hand
            T_head_to_right_hand = T_inv_torso_to_head @ T_torso_to_right_hand

            # 最终转换到相机坐标系
            T_cam_to_left_hand = self.T_head_to_cam @ T_head_to_left_hand
            T_cam_to_right_hand = self.T_head_to_cam @ T_head_to_right_hand

            # 提取位置（展平为一维数组）
            left_hand_positions[i] = T_cam_to_left_hand[:3, 3].flatten()
            right_hand_positions[i] = T_cam_to_right_hand[:3, 3].flatten()

            # 提取旋转矩阵并转换为轴角表示
            left_hand_rot = np.asarray(T_cam_to_left_hand[:3, :3], dtype=np.float64)
            right_hand_rot = np.asarray(T_cam_to_right_hand[:3, :3], dtype=np.float64)

            # 将旋转矩阵转换为轴角（rotation vector）
            left_hand_axisangles[i] = R.from_matrix(left_hand_rot).as_rotvec()
            right_hand_axisangles[i] = R.from_matrix(right_hand_rot).as_rotvec()

        # 如果输入是单个向量，则压缩批次维度
        if squeeze_output:
            left_hand_positions = left_hand_positions[0]
            right_hand_positions = right_hand_positions[0]
            left_hand_axisangles = left_hand_axisangles[0]
            right_hand_axisangles = right_hand_axisangles[0]
            left_qpos_states = left_qpos_states[0]
            right_qpos_states = right_qpos_states[0]
        return (left_hand_positions, left_hand_axisangles), (right_hand_positions, right_hand_axisangles), (left_qpos_states,
                                                                                                            right_qpos_states)

    # IK 解算 + 轴角表示：
    def inverse_kinematics_from_camera_axisangle(
            self,
            left_hand_pos: np.ndarray,
            left_hand_axisangle: np.ndarray,
            right_hand_pos: np.ndarray,
            right_hand_axisangle: np.ndarray,
            current_action_vector: Optional[np.ndarray] = None,
            q_init_left: Optional[np.ndarray] = None,
            q_init_right: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从相机坐标系下的手部位姿反向求解关节角度。支持单个样本或批量处理。
        
        参数:
            left_hand_pos: 左手位置，shape (batch_size, 3) 或 (3,) - 相机坐标系
            left_hand_axisangle: 左手轴角表示，shape (batch_size, 3) 或 (3,) - 相机坐标系
            right_hand_pos: 右手位置，shape (batch_size, 3) 或 (3,) - 相机坐标系
            right_hand_axisangle: 右手轴角表示，shape (batch_size, 3) 或 (3,) - 相机坐标系
            current_action_vector: 当前完整动作向量，shape (batch_size, 44) 或 (44,)
                                  用于获取 waist 和 neck 的角度
                                  如果为 None，则使用零角度作为初始猜测
            q_init_left: 左臂的初始关节角度猜测，使用当前step的state，shape (batch_size, num_joints) 或 (num_joints,)
            q_init_right: 右臂的初始关节角度猜测，使用当前step的state，shape (batch_size, num_joints) 或 (num_joints,)
        
        返回:
            (q_left_arm, q_right_arm): 左臂和右臂的关节角度
            - shape (batch_size, num_joints) 或 (num_joints,)，取决于输入
            - 如果 IK 求解失败，对应的返回值为 None
        """
        # 确保输入是 numpy array
        left_hand_pos = np.asarray(left_hand_pos, dtype=np.float64)
        left_hand_axisangle = np.asarray(left_hand_axisangle, dtype=np.float64)
        right_hand_pos = np.asarray(right_hand_pos, dtype=np.float64)
        right_hand_axisangle = np.asarray(right_hand_axisangle, dtype=np.float64)

        # 处理单个样本的情况
        if left_hand_pos.ndim == 1:
            left_hand_pos = left_hand_pos[np.newaxis, :]     # shape (1, 3)
            left_hand_axisangle = left_hand_axisangle[np.newaxis, :]     # shape (1, 3)
            right_hand_pos = right_hand_pos[np.newaxis, :]     # shape (1, 3)
            right_hand_axisangle = right_hand_axisangle[np.newaxis, :]     # shape (1, 3)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = left_hand_pos.shape[0]

        # 处理 current_action_vector
        if current_action_vector is not None:
            current_action_vector = np.asarray(current_action_vector, dtype=np.float64)
            if current_action_vector.ndim == 1:
                current_action_vector = current_action_vector[np.newaxis, :]     # shape (1, 44)

        # 处理初始猜测
        if q_init_left is not None:
            q_init_left = np.asarray(q_init_left, dtype=np.float64)
            if q_init_left.ndim == 1:
                q_init_left = q_init_left[np.newaxis, :]

        if q_init_right is not None:
            q_init_right = np.asarray(q_init_right, dtype=np.float64)
            if q_init_right.ndim == 1:
                q_init_right = q_init_right[np.newaxis, :]

        # 获取关节数量
        num_left_joints = len(self.kin_left_arm.get_joint_names())
        num_right_joints = len(self.kin_right_arm.get_joint_names())

        # 初始化输出数组（使用 None 填充，因为 IK 可能失败）
        q_left_arm_batch = []
        q_right_arm_batch = []

        # 批量处理每个样本
        for i in range(batch_size):
            # 提取当前样本的数据
            curr_left_pos = left_hand_pos[i]
            curr_left_axisangle = left_hand_axisangle[i]
            curr_right_pos = right_hand_pos[i]
            curr_right_axisangle = right_hand_axisangle[i]

            # === 步骤1: 将轴角表示转换为旋转矩阵，构建 4x4 变换矩阵 ===
            left_hand_rot = R.from_rotvec(curr_left_axisangle).as_matrix()
            right_hand_rot = R.from_rotvec(curr_right_axisangle).as_matrix()

            # 构建相机坐标系下的手部变换矩阵
            T_cam_to_left_hand = np.eye(4)
            T_cam_to_left_hand[:3, :3] = left_hand_rot
            T_cam_to_left_hand[:3, 3] = curr_left_pos

            T_cam_to_right_hand = np.eye(4)
            T_cam_to_right_hand[:3, :3] = right_hand_rot
            T_cam_to_right_hand[:3, 3] = curr_right_pos

            # === 步骤2: 获取 waist 和 neck 的当前角度，计算 T_torso_to_head ===
            if current_action_vector is not None:
                # 从完整动作向量中提取 waist 和 neck
                action_slices = slice_action_vector(current_action_vector[i])
                q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))

                # 如果没有提供初始猜测，从 action_vector 中提取
                if q_init_left is None:
                    curr_q_init_left = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
                else:
                    curr_q_init_left = q_init_left[i].tolist()

                if q_init_right is None:
                    curr_q_init_right = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))
                else:
                    curr_q_init_right = q_init_right[i].tolist()
            else:
                # 使用零角度作为默认值
                num_head_joints = len(self.kin_head.get_joint_names())
                q_head = [0.0] * num_head_joints

                if q_init_left is None:
                    curr_q_init_left = [0.0] * num_left_joints
                else:
                    curr_q_init_left = q_init_left[i].tolist()

                if q_init_right is None:
                    curr_q_init_right = [0.0] * num_right_joints
                else:
                    curr_q_init_right = q_init_right[i].tolist()

            # 计算 torso 到 head 的变换
            T_torso_to_head = self.kin_head.forward(q_head)

            # === 步骤3: 坐标系转换（相机 -> head -> torso）===
            T_cam_to_head = np.linalg.inv(self.T_head_to_cam)

            # 相机坐标系 -> 头部坐标系
            T_head_to_left_hand = T_cam_to_head @ T_cam_to_left_hand
            T_head_to_right_hand = T_cam_to_head @ T_cam_to_right_hand

            # 头部坐标系 -> torso 坐标系
            T_torso_to_left_hand = T_torso_to_head @ T_head_to_left_hand
            T_torso_to_right_hand = T_torso_to_head @ T_head_to_right_hand

            # === 步骤4: 使用 IK 求解器求解关节角度 ===
            # q_left_arm = self._solve_ik(self.chain_left_arm, T_torso_to_left_hand, curr_q_init_left, chain_name="left_arm")
            # q_right_arm = self._solve_ik(self.chain_right_arm, T_torso_to_right_hand, curr_q_init_right, chain_name="right_arm")
            q_left_arm = self._solve_ik(self.chain_left_arm, T_torso_to_left_hand, curr_q_init_left, chain_name="left_arm", env_idx=i)
            q_right_arm = self._solve_ik(self.chain_right_arm, T_torso_to_right_hand, curr_q_init_right, chain_name="right_arm", env_idx=i)
            # 将结果添加到批次列表
            if q_left_arm is not None:
                q_left_arm_batch.append(np.array(q_left_arm))
            else:
                q_left_arm_batch.append(None)

            if q_right_arm is not None:
                q_right_arm_batch.append(np.array(q_right_arm))
            else:
                q_right_arm_batch.append(None)

        # === 步骤5: 整理输出 ===
        # 检查是否所有样本都成功求解
        all_left_success = all(q is not None for q in q_left_arm_batch)
        all_right_success = all(q is not None for q in q_right_arm_batch)

        # 转换为 numpy 数组（如果所有样本都成功）
        if all_left_success:
            q_left_arm_result = np.array(q_left_arm_batch)
        else:
            # 有失败的情况，返回列表
            q_left_arm_result = q_left_arm_batch if not squeeze_output else (q_left_arm_batch[0] if q_left_arm_batch else None)
            if not all_left_success and squeeze_output:
                return q_left_arm_result, None

        if all_right_success:
            q_right_arm_result = np.array(q_right_arm_batch)
        else:
            # 有失败的情况，返回列表
            q_right_arm_result = q_right_arm_batch if not squeeze_output else (q_right_arm_batch[0] if q_right_arm_batch else None)
            if not all_right_success and squeeze_output:
                return None, q_right_arm_result

        # 如果输入是单个样本，则压缩批次维度
        if squeeze_output:
            q_left_arm_result = q_left_arm_result[0] if all_left_success else q_left_arm_batch[0]
            q_right_arm_result = q_right_arm_result[0] if all_right_success else q_right_arm_batch[0]

        return q_left_arm_result, q_right_arm_result

    def inverse_kinematics_from_camera(self, T_cam_to_hand: np.ndarray, initial_q: np.ndarray, kin: KDLKinematics) -> Optional[np.ndarray]:
        """
        从相机坐标系下的手部位姿反向求解关节角度。
        
        参数:
            T_cam_to_hand: 相机坐标系到手部的变换矩阵 (4x4)
            initial_q: 初始关节角度（用于IK求解的初始猜测）
            kin: 使用的运动学链（left_arm 或 right_arm）
        
        返回:
            求解的关节角度，如果求解失败则返回 None
        """
        # 步骤1: 将相机坐标系的位姿转换回 torso 坐标系
        # T_cam_to_hand -> T_head_to_hand -> T_torso_to_hand

        # 相机到头部的变换（T_head_to_cam 的逆）
        T_cam_to_head = np.linalg.inv(self.T_head_to_cam)

        # 相机 -> 头部 -> 手部
        T_head_to_hand = T_cam_to_head @ T_cam_to_hand

        # 需要知道 T_torso_to_head，但这需要当前的 neck 和 waist 关节角度
        # 这里我们使用初始关节角度来计算（应该传入当前的 q_head）
        # 暂时简化处理：假设已知完整的 torso 到 head 的变换

        # 实际上需要完整的关节配置来计算 T_torso_to_head
        # 这里返回 None，需要完整实现
        return None

    def verify_ik_solution(self, action_vector: np.ndarray, verbose: bool = True) -> Dict[str, any]:
        """
        验证 FK -> IK -> FK 的完整流程。
        
        参数:
            action_vector: 原始动作向量
            verbose: 是否打印详细信息
        
        返回:
            包含验证结果的字典
        """
        action_slices = slice_action_vector(action_vector)

        # === 步骤1: 原始 FK ===
        q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
        q_left_arm_original = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
        q_right_arm_original = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))

        # 正向运动学
        T_torso_to_head = self.kin_head.forward(q_head)
        T_torso_to_left_hand = self.kin_left_arm.forward(q_left_arm_original)
        T_torso_to_right_hand = self.kin_right_arm.forward(q_right_arm_original)

        # 转换到相机坐标系
        T_inv_torso_to_head = np.linalg.inv(T_torso_to_head)
        T_head_to_left_hand = T_inv_torso_to_head @ T_torso_to_left_hand
        T_head_to_right_hand = T_inv_torso_to_head @ T_torso_to_right_hand

        T_cam_to_left_hand = self.T_head_to_cam @ T_head_to_left_hand
        T_cam_to_right_hand = self.T_head_to_cam @ T_head_to_right_hand

        # === 步骤2: 反向变换（相机 -> torso）===
        T_cam_to_head = np.linalg.inv(self.T_head_to_cam)

        # 相机系下的手部位姿 -> 头部系下的手部位姿
        T_head_to_left_hand_recovered = T_cam_to_head @ T_cam_to_left_hand
        T_head_to_right_hand_recovered = T_cam_to_head @ T_cam_to_right_hand

        # 头部系 -> torso 系
        T_torso_to_left_hand_recovered = T_torso_to_head @ T_head_to_left_hand_recovered
        T_torso_to_right_hand_recovered = T_torso_to_head @ T_head_to_right_hand_recovered

        # === 步骤3: IK 求解 ===
        # 使用 PyKDL 的 IK 求解器
        q_left_arm_ik = self._solve_ik(self.chain_left_arm, T_torso_to_left_hand_recovered, q_left_arm_original, chain_name="left_arm")
        q_right_arm_ik = self._solve_ik(self.chain_right_arm, T_torso_to_right_hand_recovered, q_right_arm_original, chain_name="right_arm")

        # === 步骤4: 验证 FK ===
        if q_left_arm_ik is not None:
            T_torso_to_left_hand_verified = self.kin_left_arm.forward(q_left_arm_ik)
        else:
            T_torso_to_left_hand_verified = None

        if q_right_arm_ik is not None:
            T_torso_to_right_hand_verified = self.kin_right_arm.forward(q_right_arm_ik)
        else:
            T_torso_to_right_hand_verified = None

        # === 步骤5: 计算误差 ===
        results = {
            'left_arm': self._compute_errors(q_left_arm_original, q_left_arm_ik, T_torso_to_left_hand, T_torso_to_left_hand_verified),
            'right_arm': self._compute_errors(q_right_arm_original, q_right_arm_ik, T_torso_to_right_hand, T_torso_to_right_hand_verified),
        }











        if verbose:
            print("\n=== IK 验证结果 ===")
            for arm_name, result in results.items():
                print(f"\n{arm_name}:")
                if result['ik_success']:
                    print(f"  关节角度误差 (度): {result['joint_error_deg']:.4f}")
                    print(f"  位置误差 (m): {result['position_error']:.6f}")
                    print(f"  姿态误差 (度): {result['orientation_error_deg']:.4f}")
                else:
                    print(f"  IK 求解失败")

        return results

    def _solve_ik(
        self,
        chain: kdl.Chain,
        target_pose: np.ndarray,
        q_init: List[float],
        chain_name: str = "unnamed",
        env_idx: int = 0,
    ) -> Optional[List[float]]:
        """
        使用 KDL IK 求解器求解逆向运动学。
        
        参数:
            chain: 运动学链（PyKDL Chain）
            target_pose: 目标位姿 (4x4 矩阵)
            q_init: 初始关节角度
            chain_name: 用于日志与缓存的链名称
        
        返回:
            求解的关节角度；若常规 IK 失败，将返回兜底解（上一帧成功解或初始关节）

        关键改动：
            -  last_solution 按 env_idx 分桶保存：self._ik_last_solution[env_idx][chain_label]
        """

        chain_label = chain_name or "unnamed"
        env_idx = int(env_idx)
        try:
            # === [FIX] 确保该 env 的 bucket 存在 ===
            self._ensure_env_bucket(env_idx)

            if chain_label not in self._chain_max_reach:
                self._chain_max_reach[chain_label] = self._compute_chain_max_reach(chain)

            # 将 numpy 矩阵转换为 PyKDL Frame
            target_frame = self._numpy_to_kdl_frame(target_pose)

            num_joints = chain.getNrOfJoints()

            # 设置全局关节限制
            global_q_min = kdl.JntArray(num_joints)
            global_q_max = kdl.JntArray(num_joints)

            if self._use_urdf_joint_limits and chain_label in self._chain_joint_limits:
                # 使用 URDF 中的真实限制
                urdf_q_min, urdf_q_max = self._chain_joint_limits[chain_label]
                for i in range(num_joints):
                    global_q_min[i] = urdf_q_min[i] if i < len(urdf_q_min) else -np.pi
                    global_q_max[i] = urdf_q_max[i] if i < len(urdf_q_max) else np.pi
            else:
                # 使用宽松的默认限制
                for i in range(num_joints):
                    global_q_min[i] = -2.0 * np.pi
                    global_q_max[i] = 2.0 * np.pi

            q_init_full = [q_init[i] if i < len(q_init) else 0.0 for i in range(num_joints)]

            # 先取“该 env 的 last_solution”作为最终兜底候选
            last_sol = self._get_last_solution(env_idx, chain_label)

            if not self._is_target_within_workspace(chain_label, target_pose):
                # 优先用 last_solution；否则用当前 q_init 做兜底
                fallback_solution = last_sol
                if fallback_solution is None:
                    fallback_solution, reason = self._prepare_fallback_solution(chain_label, q_init, num_joints, global_q_min, global_q_max)
                else:
                    reason = "使用该 env 的上一帧 IK 解作为兜底"
                print(f"  目标超出 {chain_label} 工作空间，{reason}。")

                self._set_last_solution(env_idx, chain_label, fallback_solution)

                return fallback_solution

            # 创建求解器所需组件
            fk_solver = kdl.ChainFkSolverPos_recursive(chain)
            # ik_vel_solver = kdl.ChainIkSolverVel_pinv(chain)
            ik_vel_solver = kdl.ChainIkSolverVel_wdls(chain)     # 使用 PyKDL 内置的 WDLS（Weighted Damped Least Squares）求解器，支持阻尼
            ik_vel_solver.setLambda(self._ik_damping)     # 设置阻尼因子，避免奇异点

            # 多次尝试以提高成功率
            max_attempts = 5
            base_maxiter = 200
            eps_schedule = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
            seed_noise = 0.25     # 单位: 弧度

            for attempt in range(max_attempts):
                eps = eps_schedule[min(attempt, len(eps_schedule) - 1)]
                maxiter = base_maxiter * (attempt + 1)
                # local_margin = self._ik_local_margin * (attempt + 1)
                local_margin = min(self._ik_local_margin * (attempt + 1), self._ik_local_margin_max)

                local_q_min = kdl.JntArray(num_joints)
                local_q_max = kdl.JntArray(num_joints)
                for i in range(num_joints):
                    center = q_init_full[i]
                    lower = max(center - local_margin, global_q_min[i])
                    upper = min(center + local_margin, global_q_max[i])
                    if lower >= upper:
                        lower = global_q_min[i]
                        upper = global_q_max[i]
                    local_q_min[i] = lower
                    local_q_max[i] = upper

                ik_solver = kdl.ChainIkSolverPos_NR_JL(chain, local_q_min, local_q_max, fk_solver, ik_vel_solver, maxiter=maxiter, eps=eps)

                q_init_kdl = kdl.JntArray(num_joints)
                noise_scale = seed_noise * attempt
                for i in range(num_joints):
                    init_val = q_init_full[i]
                    if attempt > 0:
                        init_val += float(np.random.uniform(-noise_scale, noise_scale))
                    # 限制在局部关节范围内
                    init_val = max(min(init_val, local_q_max[i]), local_q_min[i])
                    q_init_kdl[i] = init_val

                q_out = kdl.JntArray(num_joints)
                ret = ik_solver.CartToJnt(q_init_kdl, target_frame, q_out)

                if ret >= 0:
                    solution = [q_out[i] for i in range(num_joints)]
                    # self._ik_last_solution[chain_label] = solution

                    # === 添加角度连续性处理，防止正pai负pai跳变 ===
                    prev_solution = self._get_last_solution(env_idx, chain_label)
                    if prev_solution is not None:
                        unwrapped_solution = []
                        for i, (curr, prev) in enumerate(zip(solution, prev_solution)):
                            # 计算角度差并归一化到 [-π, π]
                            diff = curr - prev
                            # 将差值归一化到 [-π, π] 范围
                            diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
                            # 使用修正后的差值
                            unwrapped_solution.append(prev + diff)
                        solution = unwrapped_solution

                    self._set_last_solution(env_idx, chain_label, solution)
                    return solution


                #     f"  IK 求解失败（{chain_label}），第 {attempt + 1}/{max_attempts} 次尝试，"
                #     f"返回码: {ret}，maxiter={maxiter}，eps={eps}"
                # )

            # 最终兜底策略
            # fallback_solution, reason = self._prepare_fallback_solution(
            #     chain_label, q_init, num_joints, global_q_min, global_q_max
            # )
            # self._ik_last_solution[chain_label] = fallback_solution
            # 最终兜底：优先 last_solution，否则用 q_init clip
            fallback_solution = last_sol
            if fallback_solution is None:
                fallback_solution, _ = self._prepare_fallback_solution(chain_label, q_init, num_joints, global_q_min, global_q_max)

            self._set_last_solution(env_idx, chain_label, fallback_solution)


            return fallback_solution

        except Exception as e:
            print(f"IK 求解异常（{chain_label}, env={env_idx}）: {e}")
            import traceback
            traceback.print_exc()
            num_joints = chain.getNrOfJoints()
            # 异常兜底：优先 last_solution，否则 clip(q_init) 到 URDF 限制
            last_sol = self._get_last_solution(env_idx, chain_label)
            if last_sol is not None:
                fallback = last_sol
            else:
                # 使用 URDF 限制进行 clip
                if chain_label in self._chain_joint_limits:
                    urdf_q_min, urdf_q_max = self._chain_joint_limits[chain_label]
                    fallback = [
                        float(
                            np.clip(q_init[i] if i < len(q_init) else 0.0, urdf_q_min[i] if i < len(urdf_q_min) else -np.pi,
                                    urdf_q_max[i] if i < len(urdf_q_max) else np.pi)) for i in range(num_joints)
                    ]
                else:
                    fallback = [float(np.clip(q_init[i] if i < len(q_init) else 0.0, -np.pi, np.pi)) for i in range(num_joints)]

            self._set_last_solution(env_idx, chain_label, fallback)
            return fallback

    def _numpy_to_kdl_frame(self, T: np.ndarray) -> kdl.Frame:
        """将 numpy 4x4 变换矩阵转换为 PyKDL Frame"""
        rotation = kdl.Rotation(T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2], T[2, 0], T[2, 1], T[2, 2])
        position = kdl.Vector(T[0, 3], T[1, 3], T[2, 3])
        return kdl.Frame(rotation, position)

    def _compute_errors(self, q_original: List[float], q_ik: Optional[List[float]], T_original: np.ndarray,
                        T_ik: Optional[np.ndarray]) -> Dict[str, any]:
        """计算原始值和 IK 求解值之间的误差"""
        if q_ik is None or T_ik is None:
            return {'ik_success': False, 'joint_error_deg': None, 'position_error': None, 'orientation_error_deg': None}

        # 关节角度误差（转换为度）
        q_original_arr = np.array(q_original)
        q_ik_arr = np.array(q_ik)
        joint_error = np.linalg.norm(q_original_arr - q_ik_arr)
        joint_error_deg = np.rad2deg(joint_error)

        # 位置误差
        pos_original = T_original[:3, 3]
        pos_ik = T_ik[:3, 3]
        position_error = np.linalg.norm(pos_original - pos_ik)

        # 姿态误差（使用旋转矩阵的差异）
        R_original = T_original[:3, :3]
        R_ik = T_ik[:3, :3]
        R_error = R_original.T @ R_ik
        # 计算旋转角度
        trace = np.trace(R_error)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        orientation_error_deg = np.rad2deg(angle_error)

        return {
            'ik_success': True,
            'joint_error_deg': joint_error_deg,
            'position_error': position_error,
            'orientation_error_deg': orientation_error_deg,
            'q_original': q_original,
            'q_ik': q_ik
        }

    def process_episode(self, video_path: Path, parquet_path: Path, output_path: Path, verify_ik: bool = True):
        print(f"开始处理 episode: {video_path.parent.name}")
        df = pd.read_parquet(parquet_path)
        #actions = df["action"].tolist()
        #import pdb; pdb.set_trace()
        actions = df["observation.state"].tolist()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 修改：输出视频的分辨率应为处理后的 256x256
        output_resolution = (256, 256)
        #output_resolution = (1280, 800)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, output_resolution)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频总帧数: {num_frames}, 动作数据总数: {len(actions)}")

        # === IK 验证（在第一帧上执行）===
        if verify_ik and len(actions) > 0:
            print("\n执行 IK 验证（使用第一帧数据）...")
            ik_results = self.verify_ik_solution(np.array(actions[0]), verbose=True)

        for i in tqdm(range(min(num_frames, len(actions))), desc="处理视频帧"):
            ok, frame = cap.read()
            if not ok:
                break

            # 修改：将原始帧处理为 256x256
            final_frame = process_img_cotrain(frame)

            (left_pos, left_rot), (right_pos, right_rot) = self.process_frame_kinematics(np.array(actions[i]))

            #import pdb; pdb.set_trace()

            # 修改：在处理后的 256x256 图像上绘制点
            self._draw_projection(i, final_frame, left_pos, left_rot, color=(0, 0, 255))     # 左手红色
            self._draw_projection(i, final_frame, right_pos, right_rot, color=(255, 0, 0))     # 右手蓝色

            out.write(final_frame)

        cap.release()
        out.release()
        print(f"\n处理完成！新视频已保存到: {output_path}")

    def _draw_projection(self, index, frame: np.ndarray, pos_3d: Optional[np.ndarray], orientation: Optional[np.ndarray],
                         color: Tuple[int, int, int]):
        # 修改：此方法现在包含完整的投影和坐标变换逻辑
        if pos_3d is None:
            return
        X, Y, Z = pos_3d

        u_final, v_final = -1, -1
        if Z > 0:
            # 1. 投影到虚拟的 1280x800 图像上
            u_orig = self.camera_intrinsics['fx'] * (X / Z) + self.camera_intrinsics['cx']
            v_orig = self.camera_intrinsics['fy'] * (Y / Z) + self.camera_intrinsics['cy']

            # 2. 将 2D 坐标点变换到 256x256 空间
            u_final, v_final = transform_point_cotrain(u_orig, v_orig)
            #u_final, v_final = int(u_orig), int(v_orig)

            h, w, _ = frame.shape
            #import pdb; pdb.set_trace()

            if 0 <= u_final < w and 0 <= v_final < h:
                cv2.circle(frame, (u_final, v_final), radius=5, color=color, thickness=-1)

                if orientation is not None:
                    orientation_arr = np.asarray(orientation, dtype=np.float64)
                    orientation_mat = None
                    if orientation_arr.shape == (3, 3):
                        orientation_mat = orientation_arr
                    else:
                        flat = orientation_arr.reshape(-1)
                        if flat.size == 9:
                            orientation_mat = flat.reshape(3, 3)

                    axes_to_draw: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

                    if orientation_mat is not None:
                        for _, axis_vec, axis_color in self.hand_orientation_axes:
                            axis_cam = orientation_mat @ axis_vec
                            axes_to_draw.append((axis_cam, axis_color))
                    else:
                        flat = orientation_arr.reshape(-1)
                        if flat.size == 3:
                            axes_to_draw.append((flat, color))

                    for axis_cam, axis_color in axes_to_draw:
                        axis_cam = np.asarray(axis_cam, dtype=np.float64).reshape(-1)
                        if axis_cam.size != 3:
                            continue
                        axis_norm = np.linalg.norm(axis_cam)
                        if axis_norm <= 1e-6:
                            continue
                        axis_dir = axis_cam / axis_norm
                        base_pos = np.asarray(pos_3d, dtype=np.float64)

                        length = self.hand_orientation_length
                        for _ in range(5):
                            tip_pos = base_pos + length * axis_dir
                            if tip_pos[2] <= 1e-5:
                                length *= 0.5
                                continue

                            u_tip = self.camera_intrinsics['fx'] * (tip_pos[0] / tip_pos[2]) + self.camera_intrinsics['cx']
                            v_tip = self.camera_intrinsics['fy'] * (tip_pos[1] / tip_pos[2]) + self.camera_intrinsics['cy']
                            u_tip_final, v_tip_final = transform_point_cotrain(u_tip, v_tip)

                            if 0 <= u_tip_final < w and 0 <= v_tip_final < h:
                                cv2.arrowedLine(
                                    frame,
                                    (u_final, v_final),
                                    (u_tip_final, v_tip_final),
                                    axis_color,
                                    thickness=2,
                                    tipLength=0.2,
                                )
                                break
                            length *= 0.7


def process_img_cotrain(img: np.ndarray) -> np.ndarray:
    """
    将原始图像（1280x800）处理为最终输入模型的图像（256x256）。
    这个函数与 robocasa 中的 gymnasium_groot.py 里的实现完全一致。
    """
    # 确认输入是 1280x800 的原始图像
    if not (img.shape[0] == 800 and img.shape[1] == 1280):
        # 如果视频本身就是 256x256，直接返回
        if img.shape[0] == 256 and img.shape[1] == 256:

            #import pdb; pdb.set_trace()

            # h_src, w_src = img.shape[:2]
            # target_w, target_h = 1280, 800


            # pad_w_left = (target_w - w_src) // 2
            # pad_w_right = target_w - w_src - pad_w_left
            # pad_h_top = (target_h - h_src) // 2
            # pad_h_bottom = target_h - h_src - pad_h_top


            # img = np.pad(
            #     img,
            #     ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)),
            #     mode="constant",
            #     constant_values=0,
            # )

            return img
        raise ValueError(f"输入图像尺寸 {img.shape} 不符合预期的 1280x800")

    oh, ow = 256, 256
    # 定义一个裁剪区域 (top, bottom, left, right)
    crop = (310, 770, 110, 1130)
    # 步骤 1: 裁剪原始图像
    img = img[crop[0]:crop[1], crop[2]:crop[3]]

    # 步骤 2: 第一次缩放，将裁剪后的图像缩放到 720x480
    img_resized = cv2.resize(img, (720, 480), cv2.INTER_AREA)

    # 计算填充量，使其变为正方形
    width_pad = (img_resized.shape[1] - img_resized.shape[0]) // 2
    # 步骤 3: 填充黑边，使其变为 720x720 的正方形
    img_pad = np.pad(
        img_resized,
        ((width_pad, width_pad), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # 步骤 4: 第二次缩放，将 720x720 的图像缩放到最终的 256x256
    img_resized = cv2.resize(img_pad, (oh, ow), cv2.INTER_AREA)
    return img_resized


def transform_point_cotrain(u: float, v: float) -> tuple[int, int]:
    """
    对投影到 1280x800 图像上的 2D 点应用与 process_img_cotrain 相同的几何变换。
    """
    # 步骤 1: 裁剪坐标
    crop_bounds = (310, 770, 110, 1130)     # top, bottom, left, right
    u_cropped = u - crop_bounds[2]
    v_cropped = v - crop_bounds[0]

    # 步骤 2: 第一次缩放
    original_cropped_size = (1020, 460)     # width, height
    resized1_size = (720, 480)
    scale_u1 = resized1_size[0] / original_cropped_size[0]
    scale_v1 = resized1_size[1] / original_cropped_size[1]
    u_resized1 = u_cropped * scale_u1
    v_resized1 = v_cropped * scale_v1

    # 步骤 3: 填充坐标
    pad_amount = (resized1_size[0] - resized1_size[1]) // 2     # 120
    u_padded = u_resized1
    v_padded = v_resized1 + pad_amount

    # 步骤 4: 第二次缩放
    padded_size = (720, 720)
    final_size = (256, 256)
    scale_u2 = final_size[0] / padded_size[0]
    scale_v2 = final_size[1] / padded_size[1]
    u_final = u_padded * scale_u2
    v_final = v_padded * scale_v2

    return int(u_final), int(v_final)
