import collections
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from collections import deque

import numpy as np
import cv2 as cv
import json_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from deployment_robotwin.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None

from robotwin_aloha_transform_kdl import AlohaRetargeter, RobotwinRetargetConfig

# Embodiment tag string to projector index mapping (from embodiment_tags.py)
EMBODIMENT_TAG_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "oxe_bridge": 18,
    "oxe_rt1": 19,
    "agibot_genie1": 26,
    "gr1": 24,
    "franka": 25,
    "my_robotwin": 27,
}


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        use_ee_pose: Optional[bool] = True,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        debug_save_head_images: bool = False,
        use_fixed_test_data: bool = False,
        fixed_test_data_dir: str = "../Camera/test_1",
        fixed_test_frame_idx: int = 119,
    ) -> None:
        print("######## WM4A-MODE(2/3): StarVLA/examples/Robotwin_ee/eval_files/model2robotwin_interface_submission.py")
        self.is_train_in_WM4A = True
        if self.is_train_in_WM4A:
            self.policy_ckpt_path = policy_ckpt_path
            print("######## WM4A-MODE(3/3): The EVAL with the WM4A's mode")
        else:
            print("######## JoyRA-MODE(1/1): The EVAL with the JoyRA's mode")

        print("######## Whether to use 'state' as the input is determined by the function 'eval(TASK_ENV, model, observation)' in the file 'model2robotwin_interface_submission.py'.")
        print("######## The 'use_ee_pose' value is a T/F (True/False) and is determined by hard-coding the input parameters of the ModelClient.")
        print(f"######### The Setting 'self.use_ee_pose' is {use_ee_pose}")

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        # 将embodiment_tag字符串转换为对应的index
        self.embodiment_tag = EMBODIMENT_TAG_MAPPING.get(embodiment_tag, embodiment_tag)
        self.use_ee_pose = use_ee_pose
        if self.use_ee_pose:
            robotwin_config = RobotwinRetargetConfig()
            self.retargeter = AlohaRetargeter(urdf_path=Path(robotwin_config.urdf_path))
            print(f"*** enable ee pose action space ***")
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.debug_save_head_images = debug_save_head_images
        self.use_fixed_test_data = use_fixed_test_data
        self.fixed_test_data_dir = fixed_test_data_dir
        self.fixed_test_frame_idx = fixed_test_frame_idx

        # Load fixed test data if enabled
        self.fixed_head_img = None
        self.fixed_left_img = None
        self.fixed_right_img = None
        self.fixed_state = None

        if self.use_fixed_test_data:
            self._load_fixed_test_data()

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        # self.action_chunk_size = 10
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None
        self.left_arm_joint_angles = None
        self.right_arm_joint_angles = None
        self.left_arm_actions_history = []
        self.right_arm_actions_history = []
        self.left_arm_ee_pose_history = []  # 保存左臂的ee_pose (pos + axis)
        self.right_arm_ee_pose_history = []  # 保存右臂的ee_pose (pos + axis)
        self.left_gripper_history = []  # 保存左臂夹爪动作
        self.right_gripper_history = []  # 保存右臂夹爪动作
        self.camera_images_history = {
            "head": [],
            "left": [],
            "right": []
        }  # 用于保存所有相机的图片用于debug
        self.state_history = []  # 用于保存原始state用于debug
        self.normalized_action_chunks = []  # 用于开环测试，收集每次模型生成的原始normalized_actions (50, 32)
        self._action_plot_save_dir = None  # episode结束时保存action图的目录
        self._action_plot_episode_num = None  # episode编号
        self._future_concat_frames = []  # 每次推理的当前帧+未来帧拼接图
        self._future_concat_save_dir = None
        self._future_concat_episode_num = None

    def _load_fixed_test_data(self) -> None:
        """
        加载固定的测试数据（图像和state）
        """
        test_data_dir = Path(self.fixed_test_data_dir)

        # Load fixed images
        head_img_path = test_data_dir / "head" / f"frame_{self.fixed_test_frame_idx:06d}.png"
        left_img_path = test_data_dir / "left" / f"frame_{self.fixed_test_frame_idx:06d}.png"
        right_img_path = test_data_dir / "right" / f"frame_{self.fixed_test_frame_idx:06d}.png"

        if head_img_path.exists():
            self.fixed_head_img = cv.imread(str(head_img_path))
            self.fixed_head_img = cv.cvtColor(self.fixed_head_img, cv.COLOR_BGR2RGB)
            print(f"Loaded fixed head image from {head_img_path}, shape: {self.fixed_head_img.shape}")
        else:
            print(f"Warning: Fixed head image not found at {head_img_path}")

        if left_img_path.exists():
            self.fixed_left_img = cv.imread(str(left_img_path))
            self.fixed_left_img = cv.cvtColor(self.fixed_left_img, cv.COLOR_BGR2RGB)
            print(f"Loaded fixed left image from {left_img_path}, shape: {self.fixed_left_img.shape}")
        else:
            print(f"Warning: Fixed left image not found at {left_img_path}")

        if right_img_path.exists():
            self.fixed_right_img = cv.imread(str(right_img_path))
            self.fixed_right_img = cv.cvtColor(self.fixed_right_img, cv.COLOR_BGR2RGB)
            print(f"Loaded fixed right image from {right_img_path}, shape: {self.fixed_right_img.shape}")
        else:
            print(f"Warning: Fixed right image not found at {right_img_path}")

        # Load fixed state
        state_path = test_data_dir / "state" / "state_history.npz"
        if state_path.exists():
            state_data = np.load(str(state_path))
            state_history = state_data['state']
            if self.fixed_test_frame_idx < len(state_history):
                self.fixed_state = state_history[self.fixed_test_frame_idx].copy()
                print(f"Loaded fixed state at index {self.fixed_test_frame_idx}, shape: {self.fixed_state.shape}")
            else:
                print(f"Warning: Fixed test frame index {self.fixed_test_frame_idx} out of range (state length: {len(state_history)})")
                self.fixed_state = state_history[0].copy()
                print(f"Using first state instead, shape: {self.fixed_state.shape}")
        else:
            print(f"Warning: Fixed state not found at {state_path}")

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        if self.use_ee_pose:
            self.retargeter.reset_ik_cache()
        self.num_image_history = 0
        self.raw_actions = None
        self.left_arm_joint_angles = None
        self.right_arm_joint_angles = None
        self.left_arm_actions_history = []
        self.right_arm_actions_history = []
        self.left_arm_ee_pose_history = []
        self.right_arm_ee_pose_history = []
        self.left_gripper_history = []
        self.right_gripper_history = []
        self.camera_images_history = {
            "head": [],
            "left": [],
            "right": []
        }
        self.state_history = []
        self.normalized_action_chunks = []  # 重置开环测试收集
        self._action_plot_save_dir = None
        self._action_plot_episode_num = None
        self._future_concat_frames = []
        self._future_concat_save_dir = None
        self._future_concat_episode_num = None

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state", None)
        if state is not None:
            # 保存原始state用于debug
            if self.debug_save_head_images:
                self.state_history.append(state.copy())
            if self.use_ee_pose:
                # 最初从client获取的state是14维向量：[left_arm,left_gripper,right_arm,right_gripper]
                # 这里首先将其转换为：[left_arm,right_arm,left_gripper,right_gripper]
                original_state = np.array(state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]])

                # 随后将从Robotwin环境中获取的原始state转换为ee-pose-state，
                # 似乎只需要对left_arm和right_arm做转换，保持left_gripper,right_gripper不变，同时需要保存原始的left_arm,right_arm
                self.left_arm_joint_angles = original_state[0:6]
                self.right_arm_joint_angles = original_state[6:12]
                (left_pos, left_axis), (right_pos, right_axis), (left_q, right_q) = self.retargeter.process_frame_kinematics_axisangle(state)
                # 将 left_pos, left_axis, right_pos, right_axis 与 original_state 的最后两维度拼成新的 state
                # new_state = np.concatenate([left_pos, left_axis, right_pos, right_axis, original_state[-2:].reshape(1, -1)], axis=1)
                #     state_keys = [
                #     "state.left_arm",
                #     "state.left_gripper",
                #     "state.right_arm",
                #     "state.right_gripper",
                # ] order of state

                # JoyRA模型输入的state为：[left_arm,left_gripper,right_arm,right_gripper,0*50]，共64维，前14维是有效的，
                # print('### input-state')
                new_state = np.concatenate([left_pos, left_axis, original_state[-2].reshape(1, -1), right_pos, right_axis, original_state[-1].reshape(1, -1)], axis=1)
                # print(new_state.shape)                      # (1, 14)

                # JoyRA格式的state对齐到WM4A格式
                # WM4A模型输入的state为：[left_arm,right_arm,left_gripper,right_gripper,0*50]，共64维，前14维是有效的，
                # print(self.is_train_in_WM4A)                # True
                if self.is_train_in_WM4A:
                    left_arm = new_state[:, 0:6]
                    left_gripper = new_state[:, 6:7]
                    right_arm = new_state[:, 7:13]
                    right_gripper = new_state[:, 13:14]
                    new_state = np.concatenate([left_arm, right_arm, left_gripper, right_gripper], axis=1)
                    # print(new_state.shape)                  # (1, 14)
                
                # print(self.unnorm_key)                      # robotwin
                # print(self.policy_ckpt_path)                # /path/to/policy_run/checkpoints/model.pt
                # print(len(self.action_norm_stats["min"]))   # 14
                # print(len(self.action_norm_stats["max"]))   # 14
                new_state = self.normalize_state(new_state, self.state_norm_stats)
                # print(new_state.shape)                      # (1, 14)

                # 将14维的state padding到64维
                current_state_dim = new_state.shape[1]
                target_dim = 64
                if current_state_dim < target_dim:
                    padding = np.zeros((1, target_dim - current_state_dim), dtype=new_state.dtype)
                    new_state = np.concatenate([new_state, padding], axis=1)

                example["state"] = new_state.reshape(1, -1)

            else:
                state = self.normalize_state(state, self.state_norm_stats)
                state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
                example["state"] = state.reshape(1, -1)
        else:
            print('######## state is NONE')

        task_description = example.get("lang", None)
        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = example["image"]
        # images = [self._center_crop_image(image, crop_ratio=0.95) for image in images]
        # images = [self._resize_image_pad(image) for image in images]
        concat_img = self._concat_view_images(images)
        images = [np.array(concat_img)]
        example["image"] = images
        # add embodiment_tag to example
        example["embodiment_tag"] = self.embodiment_tag
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "future_image_generation": {
                "enabled": True,
                "num_frames": 9,            # 生成 9 帧未来图像
                "return_full_video": False, # 如果为 True，data 中会包含所有帧；如果为 False，通常只返回最后一帧或指定索引帧
                "future_frame_index": -1,   # -1 通常代表最后一帧
                "height": 224,
                "width": 224,
                "num_inference_steps": 1,   # 如果模型支持，可以单独设置未来帧生成的步数
            }
        }

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            # print("Calling model to predict action")
            max_retries = 3
            retry_sleep_s = 0.3
            normalized_actions = None
            last_error_message = ""

            for attempt in range(max_retries):
                # ************** 模型前向传播
                response = self.client.predict_action(vla_input)
                data = response.get("data") if isinstance(response, dict) else None
                normalized_actions = data.get("normalized_actions") if isinstance(data, dict) else None

                if normalized_actions is not None:
                    # 收集完整的原始 normalized_actions 用于开环测试绘图
                    self.normalized_action_chunks.append(normalized_actions.copy())
                    # 收集每次推理的"当前帧+未来帧"拼接图，用于episode结束时合成视频
                    self._append_future_concat_frame(data, images[0])
                    break

                status = response.get("status") if isinstance(response, dict) else None
                error_obj = response.get("error") if isinstance(response, dict) else None
                error_message = ""
                if isinstance(error_obj, dict):
                    error_message = str(error_obj.get("message", ""))
                elif error_obj is not None:
                    error_message = str(error_obj)
                last_error_message = (
                    f"status={status}, error={error_message or 'missing data.normalized_actions'}"
                )
                print(
                    f"Policy response missing normalized_actions "
                    f"(attempt {attempt + 1}/{max_retries}): {last_error_message}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep_s)

            if normalized_actions is None:
                raise RuntimeError(
                    "Policy inference response is invalid after retries: "
                    f"{last_error_message}"
                )

            # WM4A模型输出的action为：[left_arm,right_arm,left_gripper,right_gripper,0*18]，共32维，前14维是有效的，
            # JoyRA模型输出的action为：[left_arm,left_gripper,right_arm,right_gripper,0*18]，共32维，前14维是有效的，
            # print('### output-action')
            # print(normalized_actions.shape)             # (1, 50, 32)
            normalized_actions = normalized_actions[0]
            # print(normalized_actions.shape)             # (50, 32)
            normalized_actions = normalized_actions[:, :14]
            # print(normalized_actions.shape)             # (50, 14)

            # print(self.unnorm_key)                      # robotwin
            # print(self.policy_ckpt_path)                # /path/to/policy_run/checkpoints/model.pt
            # print(len(self.action_norm_stats["min"]))   # 14
            # print(len(self.action_norm_stats["max"]))   # 14
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )
            # print(self.raw_actions.shape)               # (50, 14)
            
            # 将WM4A模型输出的格式对齐到JoyRA格式，因为这套Robotwin测试代码是适配oyRA的
            # print(self.is_train_in_WM4A)                # True
            if self.is_train_in_WM4A:
                # print('self.is_train_in_WM4A is TRUE')
                left_arm = self.raw_actions[:, 0:6]
                right_arm = self.raw_actions[:, 6:12]
                left_gripper = self.raw_actions[:, 12:13]
                right_gripper = self.raw_actions[:, 13:14]
                self.raw_actions = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                # print(self.raw_actions.shape)           # (50, 14)
            
        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        # action_keys = [
        #     "action.left_arm",
        #     "action.left_gripper",
        #     "action.right_arm",
        #     "action.right_gripper",
        # ] order of action output from model

        current_action = self.raw_actions[action_idx]
        if self.use_ee_pose:
            left_hand_pos = current_action[0:3]         # 左手位置向量
            left_hand_axis = current_action[3:6]        # 左手旋转向量
            left_hand_q = current_action[6]             # 左手夹爪
            right_hand_pos = current_action[7:10]       # 右手位置向量
            right_hand_axis = current_action[10:13]     # 右手旋转向量
            right_hand_q = current_action[13]           # 右手夹爪
            
            q_left_arm, q_right_arm = self.retargeter.inverse_kinematics_from_camera_axisangle(
                left_hand_pos, left_hand_axis, right_hand_pos, right_hand_axis, q_init_left=self.left_arm_joint_angles, q_init_right=self.right_arm_joint_angles
            )
            # 记录左右臂的所有动作（关节角度形式）
            self.left_arm_actions_history.append(q_left_arm.reshape(-1).copy())
            self.right_arm_actions_history.append(q_right_arm.reshape(-1).copy())
            # 记录左右臂的ee_pose形式（pos + axis）
            self.left_arm_ee_pose_history.append(np.concatenate([left_hand_pos, left_hand_axis]).copy())
            self.right_arm_ee_pose_history.append(np.concatenate([right_hand_pos, right_hand_axis]).copy())
            # 记录夹爪动作
            self.left_gripper_history.append(left_hand_q.copy())
            self.right_gripper_history.append(right_hand_q.copy())

            current_action = np.concatenate([q_left_arm, q_right_arm, left_hand_q.reshape(1, -1), right_hand_q.reshape(1, -1)], axis=1)
            current_action = current_action.reshape(-1)
            
        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        # 最终交给client的current_action是14维向量：[left_arm,left_gripper,right_arm,right_gripper]
        return current_action

    @staticmethod
    def normalize_state(state: dict[str, np.ndarray], state_norm_stats: Dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        mask = np.array(mask, dtype=bool)
        state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(~mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        # mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        mask = np.array(mask, dtype=bool)
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_LINEAR)
        return image

    def _resize_image_pad(
        self,
        image: np.ndarray,
        target_size: int | None = None,
        fill_value: float = 0.0,
        interpolation: int = cv.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize image keeping aspect ratio, then center pad to target square size.

        This matches the behavior of VideoResizePad in video.py:
        1. Scale the longest edge to target size
        2. Center pad the rest to make it square

        Args:
            image: Input image of shape (H, W, C) or (H, W)
            target_size: Target size for both height and width. If None, uses self.image_size[0]
            fill_value: The value to fill the padding (0.0 for black, 1.0 for white)
            interpolation: OpenCV interpolation mode (e.g., cv.INTER_LINEAR)

        Returns:
            Resized and padded image of shape (target_size, target_size, C) or (target_size, target_size)
        """
        if target_size is None:
            target_size = self.image_size[0]

        h, w = image.shape[:2]

        # Calculate scaling factor (scale longest edge to target size)
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized = cv.resize(image, (new_w, new_h), interpolation=interpolation)

        # Calculate padding (center the image)
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Convert fill_value to uint8 range if needed
        if resized.dtype == np.uint8:
            fill_color = int(fill_value * 255)
        else:
            fill_color = fill_value

        # Handle grayscale vs color images
        if len(image.shape) == 2:
            # Grayscale image
            padded = cv.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv.BORDER_CONSTANT,
                value=fill_color,
            )
        else:
            # Color image - need to provide fill value for each channel
            if isinstance(fill_color, int):
                fill_color = (fill_color, fill_color, fill_color)
            padded = cv.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv.BORDER_CONSTANT,
                value=fill_color,
            )

        return padded

    @staticmethod
    def _center_crop_image(image: np.ndarray, crop_size: tuple[int, int] | None = None, crop_ratio: float = 1.0) -> np.ndarray:
        """
        对图像进行中心裁剪

        Args:
            image: 输入图像，形状为 (H, W, C) 或 (H, W)
            crop_size: 裁剪后的目标尺寸 (height, width)，如果为 None 则使用 crop_ratio
            crop_ratio: 裁剪比例，从中心和四周裁剪的比例 (0.0-1.0)，仅在 crop_size 为 None 时使用

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]

        if crop_size is not None:
            crop_h, crop_w = crop_size
        else:
            # 使用比例裁剪
            if not (0.0 < crop_ratio <= 1.0):
                raise ValueError(f"crop_ratio must be between 0.0 and 1.0, got {crop_ratio}")
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)

        # 计算裁剪区域的起始坐标（中心点）
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        end_y = start_y + crop_h
        end_x = start_x + crop_w

        # 确保裁剪区域不超出图像边界
        if crop_h > h or crop_w > w:
            raise ValueError(f"crop_size ({crop_w}x{crop_h}) larger than image size ({w}x{h})")

        # 中心裁剪
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image

    @staticmethod
    def _as_rgb_pil(image) -> "Image.Image":
        """将各种格式的图像转换为 RGB PIL Image。"""
        from PIL import Image as PILImage
        import torch
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            if arr.size and np.nanmax(arr) <= 1.0 and np.nanmin(arr) >= 0.0:
                arr = arr * 255.0
            elif arr.size and np.nanmin(arr) < 0.0:
                arr = (arr + 1.0) * 127.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return PILImage.fromarray(arr).convert("RGB")

    @staticmethod
    def _resize_with_pad_pil(image, size=(224, 224)) -> "Image.Image":
        """等比例缩放并中心填充到指定尺寸（PIL版本）。"""
        from PIL import Image as PILImage
        image = ModelClient._as_rgb_pil(image)
        width, height = size
        scale = min(width / image.width, height / image.height)
        resized_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        resized = image.resize(resized_size, PILImage.BICUBIC)
        canvas = PILImage.new("RGB", size, 0)
        paste_xy = ((width - resized.width) // 2, (height - resized.height) // 2)
        canvas.paste(resized, paste_xy)
        return canvas

    @staticmethod
    def _crop_black_borders(image, threshold=10, min_margin=5):
        """Crop black borders from an image (common in wrist cameras).

        Exact copy of LeRobotMixtureDataset._crop_black_borders in
        starVLA/dataloader/gr00t_lerobot/datasets.py to ensure eval-time
        view concatenation matches training.
        """
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3:
            return image
        content_mask = arr.max(axis=2) > threshold
        if not content_mask.any():
            return image
        ys, xs = np.where(content_mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        margins = (x0, y0, arr.shape[1] - 1 - x1, arr.shape[0] - 1 - y1)
        if max(margins) < min_margin:
            return image
        cropped = image.crop((x0, y0, x1 + 1, y1 + 1))
        if cropped.size[0] <= 0 or cropped.size[1] <= 0:
            return image
        return cropped

    @staticmethod
    def _render_view(image, target_size):
        """Crop black borders, resize to target_size preserving aspect ratio, center-paste.

        Exact copy of LeRobotMixtureDataset._render_view in
        starVLA/dataloader/gr00t_lerobot/datasets.py to ensure eval-time
        view concatenation matches training.
        """
        from PIL import Image as PILImage
        target_h, target_w = target_size
        canvas = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
        if image is None:
            return canvas
        image = ModelClient._as_rgb_pil(image)
        image = ModelClient._crop_black_borders(image)
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return canvas
        scale = min(target_w / src_w, target_h / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = image.resize((resized_w, resized_h), PILImage.BICUBIC)
        x_offset = (target_w - resized_w) // 2
        y_offset = (target_h - resized_h) // 2
        canvas.paste(resized, (x_offset, y_offset))
        return canvas

    @staticmethod
    def _concat_view_images(images, view_size=(224, 224), gap=0):
        """Concatenate camera views into one frame — identical to training.

        Exact copy of LeRobotMixtureDataset._concat_view_images in
        starVLA/dataloader/gr00t_lerobot/datasets.py to ensure eval-time
        view concatenation matches training.

        Three-view layout with hardcoded sizes matching compose_ego_wrist:
        - Canvas: [384, 320] (H x W)
        - Ego (top): [256, 320]
        - Left wrist (bottom-left): [128, 160]
        - Right wrist (bottom-right): [128, 160]
        Pipeline: crop black borders -> resize to fixed size -> paste.
        """
        from PIL import Image as PILImage
        if not images:
            raise ValueError("No camera images provided")

        images = list(images)
        for image_idx, image in enumerate(images):
            images[image_idx] = ModelClient._as_rgb_pil(image)

        if len(images) == 3:
            # Hardcoded sizes: ego=[256,320], wrist=[128,160], canvas=[384,320]
            ego_h, ego_w = 256, 320
            wrist_h, wrist_w = 128, 160
            canvas_h, canvas_w = 384, 320

            canvas = PILImage.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            # Top: ego view
            canvas.paste(ModelClient._render_view(images[0], (ego_h, ego_w)), (0, 0))
            # Bottom-left: left wrist
            canvas.paste(ModelClient._render_view(images[1], (wrist_h, wrist_w)), (0, ego_h))
            # Bottom-right: right wrist
            canvas.paste(ModelClient._render_view(images[2], (wrist_h, wrist_w)), (wrist_w, ego_h))
            return canvas

        # 非3张图时水平拼接 (fallback)
        resized = [ModelClient._resize_with_pad_pil(image, view_size) for image in images]
        panorama = PILImage.new("RGB", (sum(image.width for image in resized), max(image.height for image in resized)), 0)
        x_offset = 0
        for image in resized:
            panorama.paste(image, (x_offset, 0))
            x_offset += image.width
        return panorama

    def _open_loop(self, save_dir: str, episode_num: int, result_suffix: Optional[str] = None) -> None:
        """开环测试可视化：拼接所有模型生成action并保存14维长图。

        result_suffix: 若提供（如 "SUCCESS" / "FAIL"），会追加到文件名末尾。
        """
        if not self.normalized_action_chunks:
            print("No normalized action chunks collected, skipping open-loop plot")
            return

        # 每次推理返回 shape (1, 50, 32)，拼接为 (50*K, 32)
        try:
            chunk_list = [np.asarray(chunk)[0] for chunk in self.normalized_action_chunks]
            all_actions = np.concatenate(chunk_list, axis=0)
        except Exception as exc:
            print(f"Warning: failed to concat normalized action chunks: {exc}")
            return

        if all_actions.ndim != 2 or all_actions.shape[1] < 14:
            print(f"Warning: invalid open-loop action shape {all_actions.shape}, expect (?, >=14)")
            return

        actions_14 = all_actions[:, :14]
        total_steps = actions_14.shape[0]
        chunk_size = chunk_list[0].shape[0] if chunk_list else 50

        dim_names = [
            "left_arm_j0", "left_arm_j1", "left_arm_j2",
            "left_arm_j3", "left_arm_j4", "left_arm_j5",
            "left_gripper",
            "right_arm_j0", "right_arm_j1", "right_arm_j2",
            "right_arm_j3", "right_arm_j4", "right_arm_j5",
            "right_gripper",
        ]

        fig, axes = plt.subplots(14, 1, figsize=(22, 2.5 * 14), sharex=True)
        axes = np.atleast_1d(axes)

        for dim_idx in range(14):
            ax = axes[dim_idx]
            ax.plot(actions_14[:, dim_idx], linewidth=0.8, color="steelblue")

            # 标记每个chunk边界，便于观察step()累积后的段落
            for k in range(1, len(chunk_list)):
                boundary = k * chunk_size
                if boundary < total_steps:
                    ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

            ax.set_ylabel(dim_names[dim_idx], fontsize=9, rotation=0, labelpad=80)
            ax.yaxis.set_ticks_position("right")
            ax.grid(True, alpha=0.3)
            if total_steps > 0:
                ax.set_xlim(0, total_steps - 1)

        axes[-1].set_xlabel("Step", fontsize=10)
        fig.suptitle(
            f"Open-loop Normalized Actions (total_steps={total_steps}, chunks={len(chunk_list)})",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{result_suffix}" if result_suffix else ""
        save_path = save_dir / f"episode{episode_num}_action_openloop{suffix}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Open-loop action plot saved to {save_path}")

    # 兼容旧调用
    def _save_action_plot(self, save_dir: str, episode_num: int) -> None:
        self._open_loop(save_dir, episode_num)

    @staticmethod
    def _as_uint8_rgb(image) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.dtype != np.uint8:
            if image.size and np.nanmax(image) <= 1.0 and np.nanmin(image) >= 0.0:
                image = image * 255.0
            elif image.size and np.nanmin(image) < 0.0:
                image = (image + 1.0) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @classmethod
    def _extract_last_future_frame(cls, pred_future_images) -> Optional[np.ndarray]:
        if pred_future_images is None:
            return None
        arr = np.asarray(pred_future_images)
        if arr.size == 0:
            return None
        if arr.ndim == 5:
            return cls._as_uint8_rgb(arr[0, -1])
        if arr.ndim == 4:
            if arr.shape[-1] in (1, 3, 4) or arr.shape[1] in (1, 3, 4):
                return cls._as_uint8_rgb(arr[0])
            return cls._as_uint8_rgb(arr[-1])
        if arr.ndim == 3:
            return cls._as_uint8_rgb(arr)
        return None

    def _append_future_concat_frame(self, response_data: dict, current_input_image: np.ndarray) -> None:
        if not isinstance(response_data, dict):
            return
        future_image = self._extract_last_future_frame(response_data.get("pred_future_images"))
        if future_image is None:
            return

        try:
            obs_image = self._as_uint8_rgb(current_input_image)
            future_image = cv.resize(
                future_image,
                (obs_image.shape[1], obs_image.shape[0]),
                interpolation=cv.INTER_AREA,
            )
            if obs_image.shape[1] >= obs_image.shape[0]:
                concat_frame = np.concatenate([obs_image, future_image], axis=0)
            else:
                concat_frame = np.concatenate([obs_image, future_image], axis=1)
            self._future_concat_frames.append(concat_frame)
        except Exception as exc:
            print(f"Warning: failed to append future concat frame: {exc}")

    def _save_future_concat_video(self, save_dir: str, episode_num: int, fps: int = 1, result_suffix: Optional[str] = None) -> None:
        if not self._future_concat_frames:
            print("No future concat frames collected, skipping future concat video")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{result_suffix}" if result_suffix else ""
        save_path = save_dir / f"episode{episode_num}_future_concat{suffix}.mp4"

        first_frame = self._future_concat_frames[0]
        h, w = int(first_frame.shape[0]), int(first_frame.shape[1])
        writer = cv.VideoWriter(str(save_path), cv.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer for {save_path}")
            return

        try:
            for frame in self._future_concat_frames:
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)
                writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        finally:
            writer.release()

        print(f"Future concat video saved to {save_path}")

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats.keys()))

        return unnorm_key

    def save_actions_history(self, output_path: str) -> None:
        """
        保存左右臂的动作历史到文件（同时保存关节角度和ee_pose形式）

        :param output_path: 输出文件路径，支持 .npy 或 .npz 格式
        """
        output_path = Path(output_path)

        # 将夹爪动作拼接在ee_pose后面
        left_arm_ee_pose_with_gripper = np.array([
            np.concatenate([ee_pose, [gripper]])
            for ee_pose, gripper in zip(self.left_arm_ee_pose_history, self.left_gripper_history)
        ])
        right_arm_ee_pose_with_gripper = np.array([
            np.concatenate([ee_pose, [gripper]])
            for ee_pose, gripper in zip(self.right_arm_ee_pose_history, self.right_gripper_history)
        ])

        if output_path.suffix == '.npz':
            np.savez(
                output_path,
                # 关节角度形式
                left_arm_actions=np.array(self.left_arm_actions_history),
                right_arm_actions=np.array(self.right_arm_actions_history),
                # 夹爪动作
                left_gripper=np.array(self.left_gripper_history),
                right_gripper=np.array(self.right_gripper_history),
                # ee_pose形式 with gripper (pos + axis + gripper)
                left_arm_ee_pose=left_arm_ee_pose_with_gripper,
                right_arm_ee_pose=right_arm_ee_pose_with_gripper
            )
            print(f"Actions history (joint angles & ee_pose & gripper) saved to {output_path}")
        else:
            # 默认保存为 .npz 格式
            output_path = output_path.with_suffix('.npz')
            np.savez(
                output_path,
                # 关节角度形式
                left_arm_actions=np.array(self.left_arm_actions_history),
                right_arm_actions=np.array(self.right_arm_actions_history),
                # 夹爪动作
                left_gripper=np.array(self.left_gripper_history),
                right_gripper=np.array(self.right_gripper_history),
                # ee_pose形式 with gripper (pos + axis + gripper)
                left_arm_ee_pose=left_arm_ee_pose_with_gripper,
                right_arm_ee_pose=right_arm_ee_pose_with_gripper
            )
            print(f"Actions history (joint angles & ee_pose & gripper) saved to {output_path}")

    def save_camera_images(self, output_dir: str, test_num: int) -> None:
        """
        将保存的所有相机图片保存到对应的子文件夹中，同时保存state数据

        :param output_dir: 输出目录的总路径，每个相机会有单独的子文件夹
        :param test_num: 测试编号，用于命名子文件夹
        """
        output_dir = Path(output_dir)
        test_dir = output_dir / f"test_{test_num}"
        test_dir.mkdir(parents=True, exist_ok=True)

        for camera_name, images in self.camera_images_history.items():
            if len(images) == 0:
                print(f"No {camera_name} images to save, skipping")
                continue

            camera_dir = test_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(images):
                # 确保图像是uint8类型
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                # 确保是BGR格式用于保存
                if len(img.shape) == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                elif img.shape[2] == 3:
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

                img_path = camera_dir / f"frame_{i:06d}.png"
                cv.imwrite(str(img_path), img)

            print(f"{camera_name.capitalize()} images saved to {camera_dir} ({len(images)} frames)")

        # 保存state数据
        if len(self.state_history) > 0:
            state_dir = test_dir / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            
            state_array = np.array(self.state_history)
            state_path = state_dir / "state_history.npz"
            np.savez(str(state_path), state=state_array)
            print(f"State data saved to {state_path} ({len(self.state_history)} frames)")
        else:
            print(f"No state data to save, skipping")

        print(f"All camera images and state saved to {test_dir}")


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    embodiment_tag = usr_args.get("embodiment_tag", None)
    debug_save_head_images = usr_args.get("debug_save_head_images", False)
    use_fixed_test_data = usr_args.get("use_fixed_test_data", False)
    fixed_test_data_dir = usr_args.get("fixed_test_data_dir", "../Camera/test_1")
    fixed_test_frame_idx = usr_args.get("fixed_test_frame_idx", 119)
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        embodiment_tag=embodiment_tag,
        debug_save_head_images=debug_save_head_images,
        use_fixed_test_data=use_fixed_test_data,
        fixed_test_data_dir=fixed_test_data_dir,
        fixed_test_frame_idx=fixed_test_frame_idx,
    )


def reset_model(model: ModelClient):
    # episode结束时保存action长图
    if model._action_plot_save_dir is not None:
        model._open_loop(model._action_plot_save_dir, model._action_plot_episode_num)

    # episode结束时保存“当前帧+未来帧”拼接视频
    if model._future_concat_save_dir is not None:
        model._save_future_concat_video(model._future_concat_save_dir, model._future_concat_episode_num)

    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    if model.use_fixed_test_data:
        # 使用固定的测试图像
        head_img = model.fixed_head_img
        left_img = model.fixed_left_img
        right_img = model.fixed_right_img
        state = model.fixed_state.copy()
        print(f"[USING FIXED TEST DATA] Frame idx: {model.fixed_test_frame_idx}")
    else:
        # 使用真实观察到的图像和state
        head_img = observation["observation"]["head_camera"]["rgb"]
        left_img = observation["observation"]["left_camera"]["rgb"]
        right_img = observation["observation"]["right_camera"]["rgb"]
        state = observation["joint_action"]["vector"]

    # Debug: save all camera images
    if model.debug_save_head_images and not model.use_fixed_test_data:
        # 只有在使用真实数据时才保存
        model.camera_images_history["head"].append(head_img.copy())
        model.camera_images_history["left"].append(left_img.copy())
        model.camera_images_history["right"].append(right_img.copy())

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]
    # images = [head_img]

    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)

    # episode结束时保存action长图与future拼接视频（与episode视频同轮次）
    # 此处不再每步保存，而是在reset_model中统一保存
    # 三个产物（action开环图、仿真录制视频、当前帧+预测帧concat视频）统一保存到
    # 同一个 eval_video_path 目录下（该目录已包含 task_name 维度），通过文件名区分：
    #   - episode{N}.mp4                  仿真录制视频（由RoboTwin框架保存）
    #   - episode{N}_action_openloop.png  action开环测试图
    #   - episode{N}_future_concat.mp4    当前帧+预测帧concat视频
    if hasattr(TASK_ENV, 'eval_video_path') and TASK_ENV.eval_video_path is not None:
        model._action_plot_save_dir = TASK_ENV.eval_video_path
        model._action_plot_episode_num = TASK_ENV.test_num

        model._future_concat_save_dir = TASK_ENV.eval_video_path
        model._future_concat_episode_num = TASK_ENV.test_num
