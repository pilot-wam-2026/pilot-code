import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import torch
from .replay_dataset import ReplayDataset
import numpy as np
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"):



    # VLA_Dataset
    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        """
        dataset_py: lerobot_datasets
        data_root_dir: /path/to/local-resource
        data_mix: robocasa_teleop_ee
        action_type: delta_ee
        CoT_prompt: Your task is {instruction}. To identify the key objects for your task.
          Locate their bounding boxes in [x1,y1,x2,y2] format.
        CoT_answer: bbox
        default_image_resolution:
        - 3
        - 224
        - 224
        per_device_batch_size: 16
        load_all_data_for_training: true
        obs:
        - image_0
        image_size:
        - 224
        - 224
        delete_pause_frame: false
        include_state: true
        """
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)



        num_workers = int(getattr(cfg.datasets.vla_data, "num_workers", 8))
        pin_memory = bool(getattr(cfg.datasets.vla_data, "pin_memory", True))
        persistent_workers = bool(getattr(cfg.datasets.vla_data, "persistent_workers", True))
        prefetch_factor = int(getattr(cfg.datasets.vla_data, "prefetch_factor", 2))

        stateful = bool(cfg.trainer.get("stateful_dataloader", False))
        if stateful:
            vla_dataset = ReplayDataset(vla_dataset, seed=int(getattr(cfg, "seed", 3047)))
            # Recreate workers on epoch changes so their dataset epoch is current.
            persistent_workers = False
        dataloader_kwargs = {
            "generator": torch.Generator().manual_seed(int(getattr(cfg, "seed", 3047))),
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

        return vla_train_dataloader

    raise ValueError("This release only supports lerobot_datasets")
