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
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from PIL import Image
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader, _sanitize_filename, _save_annotated_debug_image
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


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


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """prepare training data"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    vlm_eval_dataloader = None
    if cfg.datasets.vlm_data.dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        vlm_eval_dataloader = vlm_data_module.get("eval_dataloader")
    else:
        vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader, vlm_train_dataloader, vlm_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and learning rate scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group information
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, vlm_eval_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.vlm_eval_dataloader = vlm_eval_dataloader
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
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print trainable parameters of the model
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        components = [self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader]
        if self.vlm_eval_dataloader is not None:
            components.append(self.vlm_eval_dataloader)

        prepared = self.setup_distributed_training(self.accelerator, *components)
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.vla_train_dataloader = prepared[2]
        self.vlm_train_dataloader = prepared[3]
        if self.vlm_eval_dataloader is not None:
            self.vlm_eval_dataloader = prepared[4]

        self._init_wandb()
        self._init_checkpointing()

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
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

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

    def _log_metrics(self, metrics):
        """record training metrics"""
        if (
            self.completed_steps % self.config.trainer.logging_frequency == 0
        ):  # some parameters should be initialized for the class
            if dist.get_rank() == 0:

                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch information
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                metrics["vlm_epoch"] = round(self.completed_steps / len(self.vlm_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            # check if there is self.vla_epoch_count
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        try:
            batch_vlm = next(self.vlm_iter)
        except StopIteration:
            if not hasattr(self, "vlm_epoch_count"):
                self.vlm_epoch_count = 0
            self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.vlm_train_dataloader, self.vlm_epoch_count)
            batch_vlm = next(self.vlm_iter)

        return batch_vla, batch_vlm

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_data = time.perf_counter()
            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t_end_model = time.perf_counter()
            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)
                step_metrics = self.eval_vlm_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

                dist.barrier()  # ensure all processes are synchronized, avoid timeout

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

        if self.accelerator.is_main_process:

            examples, vlm_data = self._get_next_batch()

            score = 0.0
            num_samples = len(examples)
            actions = [example["action"] for example in examples]  # label

            # Predict actions using the model
            output_dict = self.model.predict_action(
                examples=examples,
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            self._save_eval_future_images(output_dict.get("pred_future_images"), examples)

            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        dist.barrier()
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

        width, height = pred_images[0].size
        pred_images = [image.resize((width, height)) for image in pred_images]
        gt_images = [image.resize((width, height)) for image in gt_images]
        if obs_image is not None:
            obs_image = obs_image.convert("RGB").resize((width, height))

        def write_video(frames, path):
            if not frames:
                return

            def write_with_codec(codec):
                import av

                encoded_width = width + (width % 2)
                encoded_height = height + (height % 2)
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

            width, height = frames[0].size
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
            if width > height:
                canvas = Image.new("RGB", (width, height * len(panels)), 0)
                for row_idx, panel in enumerate(panels):
                    canvas.paste(panel, (0, row_idx * height))
            else:
                canvas = Image.new("RGB", (width * len(panels), height), 0)
                for col_idx, panel in enumerate(panels):
                    canvas.paste(panel, (col_idx * width, 0))
            compare_frames.append(canvas)
        write_video(compare_frames, video_stem.parent / f"{video_stem.name}_compare.mp4")

    def eval_vlm_model(self, step_metrics: dict = None) -> dict:
        if step_metrics is None:
            step_metrics = {}
        if self.vlm_eval_dataloader is None:
            return step_metrics

        self.model.eval()
        eval_losses = []
        max_eval_batches = getattr(self.config.trainer, "vlm_eval_batches", 1)
        save_examples = getattr(self.config.trainer, "vlm_eval_save_examples", None)
        if save_examples is None:
            save_examples = getattr(
                self.vlm_eval_dataloader,
                "batch_size",
                getattr(self.config.datasets.vlm_data, "per_device_batch_size", 1),
            )
        if save_examples is None:
            save_examples = getattr(self.config.datasets.vlm_data, "per_device_batch_size", 1)
        save_examples = max(1, int(save_examples))
        tokenizer = getattr(getattr(self.model, "qwen_vl_interface", None), "processor", None)
        tokenizer = getattr(tokenizer, "tokenizer", None)

        eval_dataset = self.vlm_eval_dataloader.dataset
        dataset_len = len(eval_dataset) if hasattr(eval_dataset, "__len__") else 0
        eval_example_offset = 0
        if dataset_len > 0:
            eval_example_offset = ((self.completed_steps // max(1, self.config.trainer.eval_interval)) * save_examples) % dataset_len

        saved_example_indices = []
        with torch.no_grad():
            for batch_idx, batch_vlm in enumerate(self.vlm_eval_dataloader):
                if batch_idx >= max_eval_batches:
                    break

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vlm_output = self.model.qwen_vl_interface(**batch_vlm)
                loss = vlm_output.loss.detach().float()
                eval_losses.append(loss)

        if eval_losses:
            gathered_losses = self.accelerator.gather(torch.stack(eval_losses))
            step_metrics["eval_vlm_loss"] = gathered_losses.mean().item()

        if dataset_len > 0:
            saved_example_indices = [
                (eval_example_offset + idx) % dataset_len for idx in range(min(save_examples, dataset_len))
            ]

        if self.accelerator.is_main_process and tokenizer is not None and saved_example_indices:
            eval_records = []
            for idx, dataset_index in enumerate(saved_example_indices):
                question_text, pred_text, gt_text = self._generate_vlm_eval_example(dataset_index, tokenizer)
                eval_records.append(
                    {
                        "example_index": idx,
                        "dataset_index": dataset_index,
                        "question": question_text,
                        "prediction": pred_text,
                        "ground_truth": gt_text,
                    }
                )
            self._save_vlm_eval_examples(eval_records)

        self.model.train()
        dist.barrier()
        return step_metrics

    def _move_batch_to_device(self, batch: dict) -> dict:
        moved = {}
        model_device = self.model.qwen_vl_interface.model.device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(model_device)
            else:
                moved[key] = value
        return moved

    def _generate_vlm_eval_example(self, dataset_index: int, tokenizer) -> tuple[str, str, str]:
        dataset = self.vlm_eval_dataloader.dataset
        collate_fn = self.vlm_eval_dataloader.collate_fn
        sample = dataset[dataset_index]
        batch = collate_fn([sample])
        batch = self._move_batch_to_device(batch)

        raw_sample = dataset.list_data_dict[dataset_index]
        conversations = raw_sample.get("conversations", [])
        question_text = ""
        for conv in conversations:
            role = conv.get("role", conv.get("from"))
            if role in ("human", "user"):
                question_text = conv.get("content", conv.get("value", "")).replace("\n", " ").strip()
                break

        labels = batch["labels"][0]
        supervised_positions = torch.nonzero(labels != -100, as_tuple=False).squeeze(-1)
        if supervised_positions.numel() == 0:
            return question_text, "<no-supervised-target>", "<no-supervised-target>"

        prompt_len = int(supervised_positions[0].item())
        generation_batch = {
            "input_ids": batch["input_ids"][:, :prompt_len],
            "attention_mask": batch["attention_mask"][:, :prompt_len],
            "position_ids": batch["position_ids"][..., :prompt_len],
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in batch and batch[key] is not None:
                generation_batch[key] = batch[key]

        generated = self.model.qwen_vl_interface.generate(
            **generation_batch,
            max_new_tokens=getattr(self.config.trainer, "vlm_eval_generate_max_new_tokens", 64),
        )
        generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
        gt_ids = labels[labels != -100].detach().cpu().tolist()

        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()
        return question_text, pred_text, gt_text

    def _resolve_vlm_media_paths(self, raw_sample: dict, media_key: str) -> list[str]:
        media_files = raw_sample.get(media_key, raw_sample.get(f"{media_key}s"))
        if media_files is None:
            return []
        if not isinstance(media_files, (list, tuple)):
            media_files = [media_files]

        data_path = raw_sample.get("data_path", "")
        paths = []
        for media_file in media_files:
            if media_file is None:
                continue
            paths.append(os.path.join(data_path, media_file))
        return paths

    def _save_vlm_eval_examples(self, eval_records: list[dict]):
        if not eval_records:
            return

        dataset = self.vlm_eval_dataloader.dataset
        eval_dir = Path(self.config.output_dir) / "vlm_eval_latest"
        image_dir = eval_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for old_image_path in image_dir.glob("*"):
            if old_image_path.is_file():
                old_image_path.unlink()

        for record in eval_records:
            raw_sample = dataset.list_data_dict[record["dataset_index"]]
            sample_id = _sanitize_filename(raw_sample.get("id", f"sample_{record['dataset_index']}")).strip("_") or "sample"
            eval_conversations = [
                {"role": "question", "content": record["question"]},
                {"role": "prediction", "content": record["prediction"]},
                {"role": "ground_truth", "content": record["ground_truth"]},
            ]

            for image_idx, image_path in enumerate(self._resolve_vlm_media_paths(raw_sample, "image")):
                image_name = f"eval{record['example_index']}_sample{record['dataset_index']}_{sample_id}_image{image_idx}_qa.jpg"
                saved_image_path = image_dir / image_name
                try:
                    image = Image.open(image_path).convert("RGB")
                    _save_annotated_debug_image(image, eval_conversations, saved_image_path)
                except Exception as exc:
                    logger.warning(f"Failed to save annotated VLM eval image {image_path}: {exc}")

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm):
        """execute single training step"""
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            self.accelerator.backward(total_loss)

            # VLM task forward propagation
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vlm_output = self.model.qwen_vl_interface(**batch_vlm)
                vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm

            self.accelerator.backward(vlm_loss)

            # gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vlm_loss": vlm_loss.item(),
                }
            )
        return log_dict

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)

    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader, vlm_train_dataloader, vlm_eval_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        vlm_eval_dataloader=vlm_eval_dataloader,
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
        print(
            "🔍 Rank 0 waiting for debugger attach on port 10092..."
        )  # you may ask chatGPT what is debugger attach in vscode
        debugpy.wait_for_client()

    main(cfg)
