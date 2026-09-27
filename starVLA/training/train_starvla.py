# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
import sys
from pathlib import Path
from starVLA.training.trainer_utils.checkpoint_state import (
    prepare_training_config, initialize_training_state, save_training_state,
    save_full_config, save_weight_metadata, save_final_checkpoint,
)
from starVLA.model.modules.world_model.reproducibility import evaluation_context
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from PIL import Image
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from starVLA.training.trainer_utils.training_reproducibility import seed_model_initialization
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

# deepspeed_plugin = DeepSpeedPlugin(
#     zero_stage=2,
#     gradient_accumulation_steps=1,
#     gradient_clipping=1.0,
#     zero3_init_flag=False,
#     hf_ds_config={
#         "bf16": {
#             "enabled": True
#         },
#         "train_micro_batch_size_per_gpu": "auto",
#         "train_batch_size": "auto",
#         "gradient_accumulation_steps": 1,
#         "zero_optimization": {
#             "stage": 2,
#             "allgather_partitions": True,
#             "allgather_bucket_size": 2e8,
#             "reduce_scatter": True,
#             "reduce_bucket_size": 2e8,
#             "overlap_comm": True,
#             "contiguous_gradients": True,
#             "round_robin_gradients": True,
#         },
#         "gradient_clipping": 1.0,
#         "communication_data_type": "bf16",
#     },
# )
# deepspeed_plugin = DeepSpeedPlugin()

# 从 CLI 参数中提前解析梯度累积步数(accelerator 为模块级对象，在 cfg 加载前创建)
def _parse_grad_accum_steps(default=1):
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--trainer.gradient_accumulation_steps" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--trainer.gradient_accumulation_steps="):
            return int(a.split("=", 1)[1])
    return default

_grad_accum_steps = _parse_grad_accum_steps(default=1)

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=_grad_accum_steps,
    gradient_clipping=1.0,
    zero3_init_flag=False,
    hf_ds_config={
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": _grad_accum_steps,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 50000000,
            "reduce_scatter": True,
            "reduce_bucket_size": 50000000,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
        "gradient_clipping": 1.0,
        "communication_data_type": "bf16",
    }
)
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_steps=_grad_accum_steps,
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)




def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    wm_cfg = cfg.framework.get("world_model", None)
    if wm_cfg is not None and wm_cfg.get("base_wm", None):
        logger.info(f"Loading Base World Model `{wm_cfg.base_wm}` from ID/Path")
    else:
        logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    # print("########### train_starvla.prepare_data-1")
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    # print(cfg.datasets.vla_data.data_mix)       # robocasa_teleop_ee
    # print(cfg.datasets.vla_data.dataset_py)     # lerobot_datasets
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    # print(type(vla_train_dataloader))           # <class 'torch.utils.data.dataloader.DataLoader'>
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    # print("########### train_starvla.prepare_data-2")
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""

    # 从cfg中读取出模型每组的学习率，同一组的所有组件共享相同的初始学习率
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    # print(len(param_groups))                    #
    # print(param_groups[0].keys())               #
    # print(len(param_groups[0]['params']))       #
    # print(param_groups[0]['params'][0].shape)   #
    # print(param_groups[0]['lr'])                #
    # print(param_groups[0]['name'])              #

    # 显式校验可训练参数，避免 DeepSpeed 在内部遇到空参数组
    total_trainable_tensors = sum(len(group["params"]) for group in param_groups)
    if total_trainable_tensors == 0:
        raise RuntimeError(
            "No trainable parameters found after grouping. "
            "Please check freeze_modules / requires_grad settings."
        )

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        foreach=False,
        fused=False,
    )
    # 打印每组权重的初始学习率
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    # INFO     | >> LR Group action_model: lr=5e-05, num_params=316                                   train_starvla.py:160
    # INFO     | >> LR Group vjepa_predictor: lr=5e-05, num_params=300                                train_starvla.py:160
    # INFO     | >> LR Group state_projector: lr=5e-05, num_params=2                                  train_starvla.py:160
    # INFO     | >> LR Group action_projector: lr=5e-05, num_params=2                                 train_starvla.py:160
    # INFO     | >> LR Group base: lr=1e-05, num_params=1976                                          train_starvla.py:160

    # 初始化学习率调度器，其中最小学习率通过cfg.trainer.scheduler_specific_kwargs传递给lr_scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing() # TODO merge with load pretrained weights

        # 根据  resume 调整 lr_scheduler
        if not self.config.trainer.get("resume_training_state"):
            self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        save_full_config(self)
        self._init_wandb()
        initialize_training_state(self)

    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")

            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()

            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps

            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        state_path = self.config.trainer.get("resume_training_state")
        if state_path:
            manifest = json.loads((Path(state_path) / "training_manifest.json").read_text())
            self.completed_steps = int(manifest["completed_steps"])
            self.resume_from_checkpoint = state_path
            return

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            # 如果同时指定了 pretrained_checkpoint，优先使用它作为 resume 的 checkpoint，
            # 并从文件名中解析 completed_steps（支持跨目录断点续训）
            if pretrained_checkpoint:
                self.resume_from_checkpoint = pretrained_checkpoint
                self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=None)
                # 从文件名中解析步数，如 steps_25000_pytorch_model.pt -> 25000
                step_match = re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint)
                if step_match:
                    self.completed_steps = int(step_match.group(1))
                else:
                    logger.warning(f"Could not parse steps from checkpoint filename: {pretrained_checkpoint}, starting from step 0")
                    self.completed_steps = 0
                logger.info(f"Resuming training from specified checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
                return None

            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)

            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            # try:
            #     self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            # except AttributeError:
            #     logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Export weights and save rank-aware native training state."""
        state_dict = self.accelerator.get_state_dict(self.model)

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            weight_path = checkpoint_path + "_pytorch_model.pt"
            torch.save(state_dict, weight_path + ".tmp")
            os.replace(weight_path + ".tmp", weight_path)
            save_weight_metadata(self, weight_path)

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json",
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml",
                    use_original_values=False
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()
        save_full_config(self)
        save_training_state(self)
        self._last_checkpoint_step = self.completed_steps

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            self.batches_in_epoch = 0
            batch_vla = next(self.vla_iter)

        self.batches_in_epoch = getattr(self, 'batches_in_epoch', 0) + 1
        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.completed_steps, self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=False, ncols=80,
            initial=self.completed_steps,
            total=self.config.trainer.max_train_steps,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                if 'future_latent_loss' in step_metrics.keys():
                    progress_bar.set_postfix(
                        {
                            # "data_t": f"{t_end_data - t_start_data:.3f}",
                            # "model_t": f"{t_end_model - t_start_model:.3f}",
                            "A": f"{step_metrics['perceiver_action_loss']:.4f}",
                            "L": f"{step_metrics['future_latent_loss']:.2f}",
                            "I": f"{step_metrics['future_image_loss']:.3f}",
                        }
                    )
                else:
                    progress_bar.set_postfix(
                        {
                            # "data_t": f"{t_end_data - t_start_data:.3f}",
                            # "model_t": f"{t_end_model - t_start_model:.3f}",
                            "A": f"{step_metrics['action_dit_loss']:.4f}",
                        }
                    )

            # evaluate model
            if self.accelerator.sync_gradients and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.accelerator.sync_gradients and self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        raw_model = self.accelerator.unwrap_model(self.model)
        with evaluation_context(raw_model):
            output_dict = raw_model.predict_action(
                examples=examples,
                use_ddim=True,
                num_ddim_steps=20,
            )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            self._save_eval_future_images(output_dict.get("pred_future_images"), examples)
            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _save_eval_future_images(self, pred_future_images, examples=None) -> None:
        if pred_future_images is None:
            return

        save_root = Path(self.config.output_dir) / "eval_future_images_latest"
        save_root.mkdir(parents=True, exist_ok=True)
        for old_file in save_root.glob("*"):
            if old_file.is_file():
                old_file.unlink()

        for image_idx, image in enumerate(pred_future_images):
            video_stem = save_root / f"step_{self.completed_steps:06d}_future_compare_{image_idx:03d}_last"
            try:
                pred_images = self._to_pil_image_list(image)
                gt_image = self._get_eval_gt_future_image(examples, image_idx)
                obs_image = self._get_eval_obs_image(examples, image_idx)
                gt_images = gt_image if isinstance(gt_image, list) else ([gt_image] if gt_image is not None else [])
                if len(pred_images) > 1 and len(gt_images) > 1:
                    pred_images[0] = gt_images[0].copy()
                self._save_future_videos(
                    pred_images=pred_images,
                    gt_images=gt_images,
                    obs_image=obs_image,
                    video_stem=video_stem,
                )
                logger.info(f"Saved eval future video to {video_stem}_compare.mp4")
            except Exception as exc:
                logger.warning(f"Failed to save eval future video {video_stem}: {exc}")

    @staticmethod
    def _to_pil_image(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if torch.is_tensor(image):
            image = image.detach().float().cpu()
            image_array = image.numpy()
        else:
            image_array = np.asarray(image)

        while image_array.ndim > 3 and image_array.shape[0] == 1:
            image_array = image_array[0]
        if image_array.ndim == 4:
            image_array = image_array[-1]
        if image_array.ndim == 3 and image_array.shape[0] in (1, 3, 4):
            image_array = np.moveaxis(image_array, 0, -1)
        if image_array.ndim == 3 and image_array.shape[-1] == 1:
            image_array = np.repeat(image_array, 3, axis=-1)
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[:, :, None], 3, axis=-1)

        if image_array.dtype != np.uint8:
            if np.nanmax(image_array) <= 1.0 and np.nanmin(image_array) >= 0.0:
                image_array = image_array * 255.0
            elif np.nanmin(image_array) < 0.0:
                image_array = (image_array + 1.0) * 127.5
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array).convert("RGB")

    def _get_eval_gt_future_image(self, examples, image_idx: int):
        if not examples or image_idx >= len(examples) or "future_image" not in examples[image_idx]:
            return None
        future_image = examples[image_idx]["future_image"]
        if isinstance(future_image, (list, tuple)):
            return [self._to_pil_image(image) for image in future_image if image is not None]
        if future_image is None:
            return None
        return self._to_pil_image(future_image)

    def _get_eval_obs_image(self, examples, image_idx: int):
        if not examples or image_idx >= len(examples) or "image" not in examples[image_idx]:
            return None
        image = examples[image_idx]["image"]
        if isinstance(image, (list, tuple)):
            return self._to_pil_image_sequence(image)
        if image is None:
            return None
        return self._to_pil_image(image)

    def _to_pil_image_sequence(self, images):
        panels = [self._to_pil_image(image) for image in images if image is not None]
        if not panels:
            return None
        width, height = panels[0].size
        panels = [panel.resize((width, height)) for panel in panels]
        canvas = Image.new("RGB", (width * len(panels), height))
        for col_idx, panel in enumerate(panels):
            canvas.paste(panel, (width * col_idx, 0))
        return canvas

    @classmethod
    def _to_pil_image_list(cls, image):
        if isinstance(image, (list, tuple)):
            return [cls._to_pil_image(item) for item in image if item is not None]
        if torch.is_tensor(image) and image.ndim == 4:
            return [cls._to_pil_image(frame) for frame in image]
        image_array = np.asarray(image)
        if image_array.ndim == 4:
            return [cls._to_pil_image(frame) for frame in image_array]
        return [cls._to_pil_image(image)]

    @staticmethod
    def _save_image_grid(images, save_path, max_cols: int = 5) -> None:
        images = [image.convert("RGB") for image in images if image is not None]
        if not images:
            return
        width, height = images[0].size
        cols = min(max_cols, len(images))
        rows = int(np.ceil(len(images) / cols))
        canvas = Image.new("RGB", (width * cols, height * rows), 0)
        for idx, image in enumerate(images):
            row, col = divmod(idx, cols)
            canvas.paste(image.resize((width, height)), (col * width, row * height))
        canvas.save(save_path)

    @staticmethod
    def _save_future_videos(pred_images, gt_images, obs_image, video_stem, fps: int = 4) -> None:
        pred_images = [image.convert("RGB") for image in pred_images if image is not None]
        gt_images = [image.convert("RGB") for image in gt_images if image is not None]
        if not pred_images:
            return

        panel_sources = pred_images + gt_images
        if obs_image is not None:
            obs_image = obs_image.convert("RGB")
            panel_sources.append(obs_image)

        panel_width = max(image.width for image in panel_sources)
        panel_height = max(image.height for image in panel_sources)

        def fit_to_panel(image):
            image = image.convert("RGB")
            scale = min(panel_width / image.width, panel_height / image.height)
            resized_size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            resized = image.resize(resized_size)
            canvas = Image.new("RGB", (panel_width, panel_height), 0)
            paste_xy = ((panel_width - resized.width) // 2, (panel_height - resized.height) // 2)
            canvas.paste(resized, paste_xy)
            return canvas

        pred_images = [fit_to_panel(image) for image in pred_images]
        gt_images = [fit_to_panel(image) for image in gt_images]
        if obs_image is not None:
            obs_image = fit_to_panel(obs_image)

        def write_video(frames, path):
            if not frames:
                return

            def write_with_codec(codec):
                import av

                frame_width, frame_height = frames[0].size
                encoded_width = frame_width + (frame_width % 2)
                encoded_height = frame_height + (frame_height % 2)
                with av.open(str(path), mode="w") as container:
                    stream = container.add_stream(codec, rate=fps)
                    stream.width = encoded_width
                    stream.height = encoded_height
                    stream.pix_fmt = "yuv420p"
                    for frame in frames:
                        if frame.size != (encoded_width, encoded_height):
                            canvas = Image.new("RGB", (encoded_width, encoded_height))
                            canvas.paste(frame, (0, 0))
                            frame = canvas
                        video_frame = av.VideoFrame.from_image(frame)
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)

            try:
                write_with_codec("h264")
            except Exception:
                write_with_codec("mpeg4")

        compare_frames = []
        length = max(len(pred_images), len(gt_images), 1)
        for frame_idx in range(length):
            panels = []
            if obs_image is not None:
                panels.append(obs_image)
            if gt_images:
                panels.append(gt_images[min(frame_idx, len(gt_images) - 1)])
            panels.append(pred_images[min(frame_idx, len(pred_images) - 1)])
            if panel_width > panel_height:
                canvas = Image.new("RGB", (panel_width, panel_height * len(panels)), 0)
                for row_idx, panel in enumerate(panels):
                    canvas.paste(panel, (0, row_idx * panel_height))
            else:
                canvas = Image.new("RGB", (panel_width * len(panels), panel_height), 0)
                for col_idx, panel in enumerate(panels):
                    canvas.paste(panel, (col_idx * panel_width, 0))
            compare_frames.append(canvas)
        write_video(compare_frames, video_stem.parent / f"{video_stem.name}_compare.mp4")

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        # ZeRO-2 下 DeepSpeed 引擎内部按 gradient_accumulation_steps 自动累积梯度，
        # 因此不能使用 accelerator.accumulate()/no_sync（与 ZeRO 梯度分区不兼容）。
        # 每个 micro-batch 正常 forward/backward/step，引擎会在累积边界才真正更新参数。
        if True:
            # VLA task forward propagation
            with self.accelerator.autocast():
                output_dict = self.model.forward(batch_vla)

                action_loss = output_dict["action_loss"]
                if not torch.is_tensor(action_loss):
                    raise RuntimeError(f"action_loss must be a torch.Tensor, got {type(action_loss)}")
                if action_loss.numel() != 1:
                    raise RuntimeError(f"action_loss must be scalar, got shape={tuple(action_loss.shape)}")
                if not torch.isfinite(action_loss.detach()):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite action_loss before update {self.completed_steps + 1}"
                    )

                for key, value in output_dict.items():
                    if key == "action_loss":
                        continue
                    if torch.is_tensor(value) and value.numel() == 1:
                        if not torch.isfinite(value.detach()):
                            raise RuntimeError(f"Non-finite scalar in output_dict[{key}]")

                total_loss = action_loss.float()

            if not torch.isfinite(total_loss.detach()):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite total_loss before update {self.completed_steps + 1}"
                )

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            # DeepSpeed 引擎会在累积边界自动执行真正的 step 并清零梯度；
            # 非累积边界时 step 为 no-op，梯度继续累积。
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        metrics = {
            "action_dit_loss": action_loss.item(),
        }
        for key, value in output_dict.items():
            if key == "action_loss":
                continue
            if torch.is_tensor(value):
                if value.numel() == 1:
                    metrics[key] = value.detach().float().item()
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    def _finalize_training(self):
        """training end processing"""
        save_final_checkpoint(self)

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")
    cfg = prepare_training_config(cfg)
    seed_model_initialization(cfg, accelerator.process_index)
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError("The repaired ZeRO-2 loop is validated only for gradient_accumulation_steps=1")
    if int(cfg.trainer.get("gradient_accumulation_steps", 1)) != accelerator.gradient_accumulation_steps:
        raise ValueError("Trainer and Accelerator gradient accumulation settings disagree")
    if cfg.trainer.get("stateful_dataloader", False):
        accelerator.dataloader_config.use_stateful_dataloader = True
        accelerator.dataloader_config.use_seedable_sampler = True
        accelerator.dataloader_config.data_seed = int(getattr(cfg, "seed", 3047))
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # 创建输出目录，保存训练配置参数和模型权重
    output_dir = setup_directories(cfg=cfg)

    # 初始化模型，并基于其创建优化器和学习率调度器
    # 根据cfg.framework.name初始化模型类
    vla = build_framework(cfg)
    # print(cfg.framework.name)   # CosmoPredict25PerceiverVJEPA2AC
    # print(type(vla))            # <class 'starVLA.model.framework.WAM_VJEPA.CosmoPredict25PerceiverVJEPA2AC.CosmoPredict25_Perceiver_VJEPA2AC'>
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # 初始化
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
