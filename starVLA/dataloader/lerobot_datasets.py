# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modification: [return raw data and suport multi-dataset mixture].
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    param data_root_dir         数据集的根目录。
    param data_name             数据集的名称。
    param robot_type            要使用的机器人类型配置。
    param delete_pause_frame    是否删除暂停帧。
    param data_cfg              数据配置, 包含可选的视频后端设置。
    """

    # print("########### DATA.lerobot_datasets.make_LeRobotSingleDataset-1")
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    # print(robot_type)           # fourier_gr1_arms_waist
    # print(data_config)
    """
    <starVLA.dataloader.gr00t_lerobot.data_config.FourierGr1ArmsWaistDataConfig object at 0x7efbe15b4b80>
    """
    # print(modality_config)
    """
    {   
        'video': ModalityConfig(
            delta_indices=[0], 
            modality_keys=['video.ego_view']
        ), 
        'state': ModalityConfig(
            delta_indices=[0], 
            modality_keys=['state.left_arm', 'state.right_arm', 'state.left_hand', 'state.right_hand', 'state.waist']
        ), 
        'action': ModalityConfig(
            delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
            modality_keys=['action.left_arm', 'action.right_arm', 'action.left_hand', 'action.right_hand', 'action.waist']
        ), 
        'language': ModalityConfig(
            delta_indices=[0], 
            modality_keys=['annotation.human.coarse_action']
        )
    }
    """
    # print(transforms)
    """
        apply_to=[] 
        training=True 
        transforms=[
            StateActionToTensor(
                apply_to=['state.left_arm', 'state.right_arm', 'state.left_hand', 'state.right_hand', 'state.waist'], 
                training=True, 
                input_dtypes={}, 
                output_dtypes={}
            ), 
            StateActionTransform(
                apply_to=['state.left_arm', 'state.right_arm', 'state.left_hand', 'state.right_hand', 'state.waist'], 
                training=True, 
                normalization_modes={
                    'state.left_arm': 'min_max', 
                    'state.right_arm': 'min_max', 
                    'state.left_hand': 'min_max', 
                    'state.right_hand': 'min_max', 
                    'state.waist': 'min_max'
                }, 
                target_rotations={}, 
                normalization_statistics={}, 
                modality_metadata={}
            ), 
            StateActionToTensor(
                apply_to=['action.left_arm', 'action.right_arm', 'action.left_hand', 'action.right_hand', 'action.waist'], 
                training=True, 
                input_dtypes={}, 
                output_dtypes={}
            ), 
            StateActionTransform(
                apply_to=['action.left_arm', 'action.right_arm', 'action.left_hand', 'action.right_hand', 'action.waist'], 
                training=True, 
                normalization_modes={
                    'action.left_arm': 'min_max', 
                    'action.right_arm': 'min_max', 
                    'action.left_hand': 'min_max', 
                    'action.right_hand': 'min_max', 
                    'action.waist': 'min_max'
                }, 
                target_rotations={}, 
                normalization_statistics={}, 
                modality_metadata={}
            ), 
            GR00TTransform(
                apply_to=[], 
                training=True, 
                embodiment_tag_mapping={
                    'new_embodiment': 31, 
                    'oxe_droid': 17, 
                    'oxe_bridge': 18, 
                    'oxe_rt1': 19, 
                    'agibot_genie1': 26, 
                    'gr1': 24, 
                    'franka': 25, 
                    'action_net': 27, 
                    'agibot_beta': 28, 
                    'egodex': 29, 
                    'nvwa_f': 30, 
                    'robotwin': 32
                }, 
                language_dropout_prob=0.0, 
                default_instruction='Perform the default behavior.', 
                max_state_dim=64, 
                max_action_dim=32, 
                state_horizon=1, 
                action_horizon=16, 
                max_length=512, 
                embodiment_tag=None, 
                extra_suffix_to_canonical={}
            )
        ]
    """
    # print(dataset_path)
    """
    /path/to/workspace/datasets/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPBottleToCabinetClose_ee
    """
    
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )
    # print("########### DATA.lerobot_datasets.make_LeRobotSingleDataset-2")
    return dataset

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    # print("########### DATA.lerobot_datasets.get_vla_dataset-1")
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    # print(data_cfg.data_root_dir)   # /path/to/workspace/datasets/
    # print(data_cfg.data_mix)        # robocasa_teleop_ee
    # print(delete_pause_frame)       # False
    # print(data_mix)                 # robocasa_teleop_ee
    # print(mixture_spec)
    """
    [
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPBottleToCabinetClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPCanToDrawerClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPCupToDrawerClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPMilkToMicrowaveClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPPotatoToMicrowaveClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PnPWineToCabinetClose_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_ee', 1.0, 'fourier_gr1_arms_waist'), 
        ('PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot_eepose/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_ee', 1.0, 'fourier_gr1_arms_waist')
    ]
    """

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))
    # print(len(included_datasets))           # 24
    # print(len(filtered_mixture_spec))       # 24

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        single_dataset = make_LeRobotSingleDataset(
            Path(data_root_dir), 
            d_name, 
            robot_type, 
            delete_pause_frame=delete_pause_frame, 
            data_cfg=data_cfg
            )
        # print(type(single_dataset))         # <class 'starVLA.dataloader.gr00t_lerobot.datasets.LeRobotSingleDataset'>
        dataset_mixture.append((single_dataset, d_weight))

    dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
    # print(mode)                         # train
    # print(balance_dataset_weights)      # True
    # print(balance_trajectory_weights)   # False
    # print("########### DATA.lerobot_datasets.get_vla_dataset-2")
    return dataset



if __name__ == "__main__":

    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    # args.config_yaml = "./examples/Suqian_agibot/train_files/starvla_cotrain_robocasa_gr1.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.data_mix = "sq_egodex-f_agi_beta"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./output/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch[0].keys())
        action = batch[0]['action']
        print('action', action[0])
        print('action_mask', batch[0]['action_mask'][0])
        # print(1)
        if count > 100:
            break
        count += 1
        pass