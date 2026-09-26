

import sys
import os
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from collections import deque
import importlib
import argparse

import numpy as np
import yaml

import cv2


# 让 Python 能找到 RoboTwin 的包
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import *

# 引入你在 RoboTwin 里 cp 的 GR00T client 协议实现
from groot_client_config.service import BaseInferenceClient


# ========================= 环境与配置相关工具函数 =========================


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    """根据 task_name 实例化对应 env 类（沿用原 eval_policy_client 逻辑）"""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def get_camera_config(camera_type):
    """读取 task_config/_camera_config.yml 中的相机配置"""
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    """读取 robot embodiment 的 config.yml"""
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


# ========================= 与 GR00T server 通讯的 client =========================

class GrootPolicyClient(BaseInferenceClient):
    """
    用 GR00T 的 ZMQ 协议与远程 policy server 通讯的 client。

    对应 GR00T 侧的 RobotInferenceServer：
      - endpoint "get_action"
      - endpoint "get_modality_config"
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str | None = None,
        use_msgpack: bool = True,
    ):
        super().__init__(host=host, port=port, api_token=api_token, use_msgpack=use_msgpack)

    def get_action(self, observations: dict) -> dict:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> dict:
        return self.call_endpoint("get_modality_config", requires_input=False)


# ========================= RoboTwin <-> GR00T 适配函数 =========================

def robottwin_obs_to_groot_obs(TASK_ENV, raw_obs: dict) -> dict:
    """
    将 RoboTwin 的 TASK_ENV.get_obs() 输出转成 GR00T policy 期望的 obs 字典。
    """
    obs = {}
    obs_dict = raw_obs.get("observation", {})

    # 统一目标分辨率：与 GR00T robotwin 训练时记录的 original_resolutions 对齐
    TARGET_W, TARGET_H = 640, 480  # (width, height)

    # ====================== 1. 头部相机 ======================
    if "head_camera" in obs_dict and "rgb" in obs_dict["head_camera"]:
        head_rgb = obs_dict["head_camera"]["rgb"]  # (H, W, 3)，当前一般为 (240, 320, 3)

        # 转为 uint8
        if head_rgb.dtype != np.uint8:
            head_rgb = (head_rgb * 255).clip(0, 255).astype(np.uint8)

        # 如果分辨率不是 640x480，则先 resize 到 640x480
        h, w = head_rgb.shape[:2]
        if (w, h) != (TARGET_W, TARGET_H):
            head_rgb = cv2.resize(head_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        # 添加时间维度 T=1 -> (1, H, W, 3)
        if head_rgb.ndim == 3:
            head_rgb = head_rgb[None, ...]
        obs["video.image_high"] = head_rgb
    else:
        raise KeyError(
            f"head_camera.rgb not found in observation keys: {list(obs_dict.keys())}"
        )

    # ====================== 2. 手腕相机：兼容两种命名 ======================
    #    - 旧：cam_left_wrist / cam_right_wrist（离线数据）
    #    - 现：left_camera / right_camera（在线 RoboTwin 相机）
    left_cam_key = "cam_left_wrist" if "cam_left_wrist" in obs_dict else "left_camera"
    right_cam_key = "cam_right_wrist" if "cam_right_wrist" in obs_dict else "right_camera"

    if left_cam_key in obs_dict and "rgb" in obs_dict[left_cam_key]:
        left_rgb = obs_dict[left_cam_key]["rgb"]

        if left_rgb.dtype != np.uint8:
            left_rgb = (left_rgb * 255).clip(0, 255).astype(np.uint8)

        h, w = left_rgb.shape[:2]
        if (w, h) != (TARGET_W, TARGET_H):
            left_rgb = cv2.resize(left_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        if left_rgb.ndim == 3:
            left_rgb = left_rgb[None, ...]
        obs["video.image_left_wrist"] = left_rgb
    else:
        print(
            f"[WARN] left wrist camera '{left_cam_key}' not found in observation; "
            f"skip sending to GR00T."
        )

    if right_cam_key in obs_dict and "rgb" in obs_dict[right_cam_key]:
        right_rgb = obs_dict[right_cam_key]["rgb"]

        if right_rgb.dtype != np.uint8:
            right_rgb = (right_rgb * 255).clip(0, 255).astype(np.uint8)

        h, w = right_rgb.shape[:2]
        if (w, h) != (TARGET_W, TARGET_H):
            right_rgb = cv2.resize(right_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        if right_rgb.ndim == 3:
            right_rgb = right_rgb[None, ...]
        obs["video.image_right_wrist"] = right_rgb
    else:
        print(
            f"[WARN] right wrist camera '{right_cam_key}' not found in observation; "
            f"skip sending to GR00T."
        )

    # ====================== 3. 关节状态 ======================
    joint_vec = raw_obs["joint_action"]["vector"]  # 例如 (14,)
    obs["state.left_arm"] = joint_vec[0:6][None, ...]
    obs["state.left_gripper"] = joint_vec[6:7][None, ...]
    obs["state.right_arm"] = joint_vec[7:13][None, ...]
    obs["state.right_gripper"] = joint_vec[13:14][None, ...]

    # ====================== 4. 文本指令 ======================
    if hasattr(TASK_ENV, "get_instruction"):
        instruction = TASK_ENV.get_instruction()
    else:
        instruction = "do your thing!"
    obs["annotation.human.task_description"] = [instruction]

    return obs

def apply_groot_action_to_env(TASK_ENV, action_dict: dict, n_action_steps: int):
    """
    将 GR00T policy 返回的 action_dict 转换为 RoboTwin env 的动作，并执行若干步。

    !!! 重要：
    - GR00T policy 返回的格式是字典：{"action.left_arm": (H, D1), "action.left_gripper": (H, D2), ...}
    - 需要将这些动作合并成 joint vector 格式
    """
    # GR00T 返回的 action 格式：{"action.left_arm": (H, 6), "action.left_gripper": (H, 1), ...}
    # 需要合并成完整的 joint vector: (H, 14)
    
    if "action.left_arm" in action_dict and "action.right_arm" in action_dict:

        left_arm = np.asarray(action_dict["action.left_arm"])
        left_gripper = np.asarray(action_dict["action.left_gripper"])
        right_arm = np.asarray(action_dict["action.right_arm"])
        right_gripper = np.asarray(action_dict["action.right_gripper"])

        # 新增：处理 (B,H,D) -> (H,D)（单环境 B=1）
        if left_arm.ndim == 3 and left_arm.shape[0] == 1:
            left_arm = left_arm[0]
        if right_arm.ndim == 3 and right_arm.shape[0] == 1:
            right_arm = right_arm[0]
        if left_gripper.ndim == 3 and left_gripper.shape[0] == 1:
            left_gripper = left_gripper[0]
        if right_gripper.ndim == 3 and right_gripper.shape[0] == 1:
            right_gripper = right_gripper[0]

        # 保持 gripper 是 (H,1)
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, None]
        if right_gripper.ndim == 1:
            right_gripper = right_gripper[:, None]

        traj = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)  # (H,14)
        
        H = min(n_action_steps, traj.shape[0])
        for t in range(H):
            action_vec = traj[t]  # (14,)
            TASK_ENV.take_action(action_vec)
        return
    
    elif "actions" in action_dict:
        # 情况 2：如果返回的是统一的 "actions" 数组（向后兼容）
        traj = np.asarray(action_dict["actions"])  # (H, D)
        H = min(n_action_steps, traj.shape[0])
        for t in range(H):
            action_vec = traj[t]
            TASK_ENV.take_action(action_vec)
        return
    
    else:
        raise ValueError(f"Unknown action format: {action_dict.keys()}")


# ========================= 评测主逻辑（替代原 eval_policy_client 的 eval_policy） =========================

def eval_policy_with_groot(
    task_name,
    TASK_ENV,
    args,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    groot_host: str = "localhost",
    groot_port: int = 5555,
    groot_api_token: str | None = None,
    n_action_steps: int = 4,
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy: GR00T Remote Policy ({groot_host}:{groot_port})\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # 创建 GR00T 的 ZMQ 客户端
    groot_client = GrootPolicyClient(
        host=groot_host,
        port=groot_port,
        api_token=groot_api_token,
        use_msgpack=True,
    )

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # ========== 1. 先跑一次 expert_check（完全复用原逻辑） ==========
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # ========== 2. 正式评测 episode：设置 demo + 指令 ==========
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        # ========== 3. 如果需要，打开 ffmpeg 录制 ==========
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        # ========== 4. 主循环：get_obs -> 调 GR00T get_action -> 多步执行 ==========
        succ = False
        first_step_of_episode = True # 用于标记是否是episode的第一步

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            raw_obs = TASK_ENV.get_obs()
            groot_obs = robottwin_obs_to_groot_obs(TASK_ENV, raw_obs)

            # 第一次 rollout：强制让 server 清掉所有 env slot 的 IK 历史缓存
            if first_step_of_episode:
                print("[RobotwinClient] send meta.reset_mask=True", flush=True)
                groot_obs["meta.reset_mask"] = True  
                first_step_of_episode = False

            action_dict = groot_client.get_action(groot_obs)
            apply_groot_action_to_env(TASK_ENV, action_dict, n_action_steps=n_action_steps)

            if TASK_ENV.eval_success:
                succ = True
                break

        # ========== 5. 清理视频录制 ==========
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # ========== 6. 统计成功/失败 & 打印 ==========
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | "
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m"
            f" => \033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m,"
            f" current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


# ========================= main + 配置解析（基本沿用原 eval_policy_client） =========================

def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    instruction_type = usr_args["instruction_type"]

    # GR00T server 相关参数（可以放在 config 里）
    groot_host = usr_args.get("host", "localhost")
    groot_port = usr_args.get("port", 5555)
    groot_api_token = usr_args.get("api_token", None)
    groot_n_action_steps = usr_args.get("n_action_steps", 4)

    save_dir = usr_args.get("save_dir", None)
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_item):
        robot_file = _embodiment_types[embodiment_type_item]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # 简化 save_dir 路径，不再包含 ckpt_setting
    if save_dir is None:
        raise ValueError("save_dir is required")
    else:
        save_dir = Path(save_dir)
    save_dir = save_dir / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # 输出当前配置，完全沿用原 eval_policy_client
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print(
        "\033[95mRandom Background:\033[0m "
        + str(args["domain_randomization"]["random_background"])
    )
    if args["domain_randomization"]["random_background"]:
        print(
            " - Clean Background Rate: "
            + str(args["domain_randomization"]["clean_background_rate"])
        )
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(
            " - Crazy Random Light Rate: "
            + str(args["domain_randomization"]["crazy_random_light_rate"])
        )
    print(
        "\033[95mRandom Table Height:\033[0m "
        + str(args["domain_randomization"]["random_table_height"])
    )
    print(
        "\033[95mRandom Head Camera Distance:\033[0m "
        + str(args["domain_randomization"]["random_head_camera_dis"])
    )

    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]
    topk = 1

    st_seed, suc_num = eval_policy_with_groot(
        task_name,
        TASK_ENV,
        args,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        groot_host=groot_host,
        groot_port=groot_port,
        groot_api_token=groot_api_token,
        n_action_steps=groot_n_action_steps,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Task: {task_name}\n")
        file.write(f"Task Config: {task_config}\n")
        file.write(f"GR00T Server: {groot_host}:{groot_port}\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Success Rate: {suc_num}/{test_num} = {suc_num/test_num*100:.2f}%\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    # 在终端打印最终的成功率统计
    print("\n" + "="*60)
    print("\033[1m========== Evaluation Results ==========\033[0m")
    print("="*60)
    print(f"\033[93mTask Name:\033[0m {task_name}")
    print(f"\033[93mTask Config:\033[0m {task_config}")
    print(f"\033[93mGR00T Server:\033[0m {groot_host}:{groot_port}")
    print(f"\033[93mInstruction Type:\033[0m {instruction_type}")
    print(f"\033[93mTotal Episodes:\033[0m {test_num}")
    print(f"\033[93mSuccessful Episodes:\033[0m {suc_num}")
    print(f"\033[93mSuccess Rate:\033[0m \033[92m{suc_num}/{test_num} = {suc_num/test_num*100:.2f}%\033[0m")
    print("="*60)
    print(f"\033[94mResults saved to:\033[0m {file_path}")
    print("="*60 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T Simulation Client for RoboTwin")
    
    # ========== 必需的任务参数 ==========
    parser.add_argument("--task_name", type=str, required=True, 
                       help="Task name (e.g., beat_block_hammer)")
    parser.add_argument("--task_config", type=str, required=True,
                       help="Task config file name (e.g., demo_clean)")
    parser.add_argument("--instruction_type", type=str, default="unseen",  # 添加默认值
                       help="Instruction type (e.g., language, unseen, default: unseen)")
    parser.add_argument("--seed", type=int, required=True,
                       help="Random seed")
    
    # ========== GR00T server 连接参数 ==========
    parser.add_argument("--host", type=str, default="localhost",
                       help="GR00T server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8811,
                       help="GR00T server port (default: 8811)")
    parser.add_argument("--api_token", type=str, default=None,
                       help="API token for GR00T server (optional)")
    parser.add_argument("--n_action_steps", type=int, default=4,
                       help="Number of action steps per inference (default: 4)")
    
    # ========== 可选的保存路径 ==========
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save evaluation results (default: eval_result/...)")

    # ========== 可选的测试数量 ==========
    parser.add_argument("--test_num", type=int, default=100,
                       help="Number of episodes to test (default: 100)")
    
    args = parser.parse_args()
    
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    # 转换为字典格式供 main() 使用
    usr_args = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "instruction_type": args.instruction_type,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "api_token": args.api_token,
        "n_action_steps": args.n_action_steps,
        "save_dir": args.save_dir,
        "test_num": args.test_num,
    }
    
    main(usr_args)