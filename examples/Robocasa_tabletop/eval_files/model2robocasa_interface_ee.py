from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path
from release.config import read_mode_config
class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = "gr1",
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, # @contributor
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
        prompt_speed: int = 2500,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps
        self.prompt_speed = prompt_speed
        self.use_eepose = True
        self.effective_dim = 29
        self.task_description = None
        self._has_logged_prompt = False
        self._debug_action_count = 0
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        
        print(f"DEBUG: use_eepose = {self.use_eepose}")
        if self.use_eepose:
            print("DEBUG: Entering if self.use_eepose block")
            try:
                print("DEBUG: About to import gr1_pos_transform")
                from examples.Robocasa_tabletop.eval_files.gr1_pos_transform_new import BodyRetargeter, GR1RetargetConfig
                print("DEBUG: Import successful, creating GR1RetargetConfig")
                gr1_config = GR1RetargetConfig()
                print("DEBUG: GR1RetargetConfig created, creating BodyRetargeter")
                
                # 然后，通过这个实例来访问属性
                self.body_retargeter = BodyRetargeter(
                    urdf_path=Path(gr1_config.urdf_path), 
                    camera_intrinsics=gr1_config.camera_intrinsics
                )
                print("Enabled EEPose processing in Gr00tPolicy.")
            except Exception as e:
                import traceback
                print(f"Warning: Failed to initialize EEPose processing: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                self.use_eepose = False
        else:
            print(f"DEBUG: use_eepose is False, skipping initialization")
        
    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)
    def reset(self, task_description: str or tuple) -> None:
       
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _format_prompt(self, task_text: str) -> str:
        task_text = str(task_text).strip().rstrip(".")
        return f"Task: {task_text}."
        return (
            f"Task: {task_text}. "
            f"Speed: {self.prompt_speed}. "
            f"Mistake: false. "
            f"Control Mode: ee."
        )

    def _build_instructions(self, batch_size: int) -> list[str]:
        task_description = self.task_description
        if isinstance(task_description, str):
            task_texts = [task_description] * batch_size
        elif isinstance(task_description, Sequence):
            task_texts = list(task_description)
            if len(task_texts) == 1 and batch_size > 1:
                task_texts = task_texts * batch_size
            elif len(task_texts) != batch_size:
                raise ValueError(
                    f"Task description batch size mismatch: got {len(task_texts)} tasks for batch size {batch_size}"
                )
        else:
            raise TypeError(f"Unsupported task_description type: {type(task_description)}")

        instructions = [self._format_prompt(task_text) for task_text in task_texts]
        if not self._has_logged_prompt and len(instructions) > 0:
            print(f"Inference prompt: {instructions[0]}")
            self._has_logged_prompt = True
        return instructions

    def step(
        self, 
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """
        task_description = observations['annotation.human.coarse_action'][0] # tuple       
        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        images = ego_view   # (N, 1, 3)
        
        if self.use_eepose:
            # 1. 使用 BodyRetargeter 将 EEpose 转换为标准状态表示
            obs_copy = observations.copy()
            full_44dof_vector = self._build_full_44dof_vector(obs_copy)
            (left_hand_positions, left_hand_axisangles), (right_hand_positions, right_hand_axisangles), (left_qpos_states, right_qpos_states) = self.body_retargeter.process_frame_kinematics_axisangle(full_44dof_vector)
            #print(f"Left hand positions shape: {left_hand_positions.shape}, axis-angles shape: {left_hand_axisangles.shape}")
            # 2. 将转换后的状态添加回 obs_copy
            left_arm_state = obs_copy.get("state.left_arm", None)
            right_arm_state = obs_copy.get("state.right_arm", None)
            # 拼接两个 (bs, 3) 的数组，得到一个 (bs, 6) 的二维数组
            left_eepose_2d = np.concatenate((left_hand_positions, left_hand_axisangles), axis=-1)
            right_eepose_2d = np.concatenate((right_hand_positions, right_hand_axisangles), axis=-1)
            # 使用 np.newaxis 恢复时间维度，将其从 (bs, 6) 变为 (bs, 1, 6)
            observations["state.left_arm"] = left_eepose_2d[:, np.newaxis, :]
            observations["state.right_arm"] = right_eepose_2d[:, np.newaxis, :]
            state = {}
            state['left_arm'] = observations['state.left_arm']      # (B, 1, 6)
            state['right_arm'] = observations['state.right_arm']    # (B, 1, 6)
            state['left_hand'] = observations['state.left_hand']    # (B, 1, 6)
            state['right_hand'] = observations['state.right_hand']  # (B, 1, 6)
            state['waist'] = observations['state.waist']            # (B, 1, 3)
            # 1) 先按实际使用的27维顺序拼接
            state_27 = np.concatenate([
                state['left_arm'],     # 0:6
                state['right_arm'],    # 6:12
                state['left_hand'],    # 12:18
                state['right_hand'],   # 18:24
                state['waist'],        # 24:27
            ], axis=-1)
            # 2) 用27维 stats 做 min-max normalize
            state_27 = self.normalize_state_minmax(state_27, self.state_norm_stats)
            # 3) 再插入 gripper 两维，恢复到训练时的29维布局
            bs = state_27.shape[0]
            dtype = state_27.dtype
            input_state = np.concatenate([
                state_27[:, :, 0:6],                                # left_arm
                state_27[:, :, 6:12],                               # right_arm
                np.zeros((bs, 1, 1), dtype=dtype),                  # left_gripper
                np.zeros((bs, 1, 1), dtype=dtype),                  # right_gripper
                state_27[:, :, 12:18],                              # left_hand
                state_27[:, :, 18:24],                              # right_hand
                state_27[:, :, 24:27],                              # waist
            ], axis=-1)
            # 4) pad 到 64 维
            if input_state.shape[2] != 64:
                zero_pad = np.zeros((input_state.shape[0], 1, 64 - input_state.shape[2]), dtype=input_state.dtype)
                input_state = np.concatenate([input_state, zero_pad], axis=-1)
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
        # image: Image.Image = Image.fromarray(image)
        images = [[self._resize_image(img) for img in sample] for sample in images] # (B, N_view, H, W, 3)
        # input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)
        # prepare vla input
        examples = []
        batch_size = len(images)
        instructions = self._build_instructions(batch_size)
        #print(input_state)
        #import pdb; pdb.set_trace()
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": instructions[b],
                "state": input_state[b],  # N_history, 58 #Hack BUG
                "embodiment_tag": 24,  # GR1, matches robocasa_teleop_ee training samples.
            }
            examples.append(example)
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        self._update_vla_input(vla_input)
        
        response = self.client.predict_action(vla_input)
        self._handle_model_response(response, images)
        
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D  
        # normalized_actions = normalized_actions[:, :, :self.effective_dim]
        self._log_action_debug("normalized_actions_model", normalized_actions)
        normalized_actions = normalized_actions[:, :, [i for i in range(self.effective_dim) if i not in (12, 13)]]
        self._log_action_debug("normalized_actions_exec", normalized_actions)
        
        # unnormalize actions in batch form
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        self._log_action_debug("raw_actions_exec", raw_actions)
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)
        if self.use_eepose:
            raw_action = {
                "action.left_arm": raw_actions[:, :self.n_action_steps, :6],      # (B, n_action_steps, 7)
                "action.right_arm": raw_actions[:, :self.n_action_steps, 6:12],   # (B, n_action_steps, 7)
                "action.left_hand": raw_actions[:, :self.n_action_steps, 12:18],  # (B, n_action_steps, 6)
                "action.right_hand": raw_actions[:, :self.n_action_steps, 18:24], # (B, n_action_steps, 6)
                "action.waist": raw_actions[:, :self.n_action_steps, 24:27],      # (B, n_action_steps, 3)
            }
            # 从模型输出的6-DoF EE Pose动作中提取 pos 和 axis-angle
            pred_left_eepose_seq = raw_action["action.left_arm"]
            pred_right_eepose_seq = raw_action["action.right_arm"]
            
            batch_size, horizon, _ = pred_left_eepose_seq.shape
            
            # 初始化用于存储IK结果的数组
            q_left_arm_seq = np.zeros((batch_size, horizon, 7)) # 目标是7-DoF
            q_right_arm_seq = np.zeros((batch_size, horizon, 7)) # 目标是7-DoF
            # 使用输入时的原始7-DoF手臂状态作为第一个时间步的IK初始猜测
            # 形状从 (B, 1, 7) 变为 (B, 7)
            q_init_left = left_arm_state[:, -1, :] if left_arm_state is not None else None
            q_init_right = right_arm_state[:, -1, :] if right_arm_state is not None else None
            # 遍历动作序列的每一个时间步 (从 0 到 15)
            for t in range(horizon):
                # 提取当前时间步 t 的EE Pose动作，形状为 (B, 6)
                left_eepose_t = pred_left_eepose_seq[:, t, :]
                right_eepose_t = pred_right_eepose_seq[:, t, :]
                #print(f"Time step {t}: Left EE Pose shape: {left_eepose_t.shape}, Right EE Pose shape: {right_eepose_t.shape}")
                # 将EE Pose分解为位置和轴角
                left_hand_pos = left_eepose_t[:, :3]
                left_hand_axisangle = left_eepose_t[:, 3:6]
                right_hand_pos = right_eepose_t[:, :3]
                right_hand_axisangle = right_eepose_t[:, 3:6]
                # 执行IK计算，输入是 (B, 3)，输出是 (B, 7)
                q_left_arm_t, q_right_arm_t = self.body_retargeter.inverse_kinematics_from_camera_axisangle(
                    left_hand_pos=left_hand_pos,
                    left_hand_axisangle=left_hand_axisangle,
                    right_hand_pos=right_hand_pos,
                    right_hand_axisangle=right_hand_axisangle,
                    current_action_vector=full_44dof_vector,
                    q_init_left=q_init_left,
                    q_init_right=q_init_right
                )
                # 将计算出的关节角存储到结果序列中
                if q_left_arm_t is not None:
                    q_left_arm_seq[:, t, :] = q_left_arm_t
                if q_right_arm_t is not None:
                    q_right_arm_seq[:, t, :] = q_right_arm_t
                
                # 使用当前步的IK解作为下一步的初始猜测，以保证动作的连续性
                q_init_left = q_left_arm_t
                q_init_right = q_right_arm_t
            # 将完整的关节角序列更新回 raw_action 字典
            raw_action["action.left_arm"] = q_left_arm_seq
            raw_action["action.right_arm"] = q_right_arm_seq
        
        else:
            raw_action = {
                "action.left_arm": raw_actions[:, :self.n_action_steps, :7],      # (B, n_action_steps, 7)
                "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
                "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
                "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26], # (B, n_action_steps, 6)
                "action.waist": raw_actions[:, :self.n_action_steps, 26:29],      # (B, n_action_steps, 3)
            }
        return {"actions": raw_action}

    def _handle_model_response(self, response: dict, images) -> None:
        return

    def _update_vla_input(self, vla_input: dict) -> None:
        return

    def _log_action_debug(self, name: str, actions: np.ndarray) -> None:
        if os.getenv("STARVLA_ACTION_DEBUG", "0").lower() not in {"1", "true", "yes", "on"}:
            return
        max_logs = int(os.getenv("STARVLA_ACTION_DEBUG_MAX", "20"))
        if self._debug_action_count >= max_logs:
            return
        arr = np.asarray(actions, dtype=np.float32)
        print(
            f"[ACTION_DEBUG] {name}: shape={arr.shape}, "
            f"mean={arr.mean():.4g}, std={arr.std():.4g}, "
            f"min={arr.min():.4g}, max={arr.max():.4g}, "
            f"first={arr.reshape(-1, arr.shape[-1])[0].tolist()}",
            flush=True,
        )
        if name == "raw_actions_exec":
            self._debug_action_count += 1

    @staticmethod
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
    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # Keep evaluation preprocessing aligned with the current training
        # dataset path, which uses PIL resize((224, 224)) instead of
        # aspect-ratio-preserving VideoResizePad.
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    @staticmethod
    def normalize_state_minmax(state: np.ndarray, state_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        state_min = np.asarray(state_norm_stats["min"], dtype=np.float32)
        state_max = np.asarray(state_norm_stats["max"], dtype=np.float32)
        state = 2 * (state - state_min) / (state_max - state_min) - 1
        return state
    # 组成44DOF向量的辅助函数
    # 组成44DOF向量的辅助函数
    @staticmethod
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
            "left_arm": (0, 7), "left_hand": (7, 13), "left_leg": (13, 19),
            "neck": (19, 22), "right_arm": (22, 29), "right_hand": (29, 35),
            "right_leg": (35, 41), "waist": (41, 44),
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
            else: # 默认返回一个 (1, 44) 的零向量
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
