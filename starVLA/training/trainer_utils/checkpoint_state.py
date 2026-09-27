"""Weight exports plus explicit, atomic Accelerate training-state checkpoints."""
import importlib.metadata
import json
import os
import random
import socket
import warnings
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import RandomSampler
from starVLA.training.trainer_utils.resume_validation import validate_resume

from starVLA.model.modules.world_model.latent_contract import (
    is_cosmos25, prepare_checkpoint_config, validate_mode,
)


def plain_config(config):
    config = getattr(config, "_cfg", config)
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def prepare_training_config(cfg):
    config = plain_config(cfg)
    trainer = config.setdefault("trainer", {})
    checkpoint = trainer.get("pretrained_checkpoint")
    resume = trainer.get("resume_training_state")
    trainer.setdefault("stateful_dataloader", True)
    if resume and checkpoint:
        raise ValueError("Choose full-state resume OR weight-only initialization, not both")
    if trainer.get("is_resume") and not checkpoint and not resume:
        raise ValueError("Specify pretrained_checkpoint or resume_training_state explicitly; implicit latest is ambiguous")
    if resume:
        manifest = json.loads((Path(resume) / "training_manifest.json").read_text())
        if not manifest.get("complete"):
            raise ValueError("Refusing an incomplete training-state checkpoint")
        saved = manifest.get("latent_normalization")
        wm = config.setdefault("framework", {}).setdefault("world_model", {})
        if wm.get("latent_normalization") not in (None, saved):
            raise ValueError("Resume latent coordinates differ from the saved training state")
        if saved is not None:
            wm["latent_normalization"] = saved
        validate_resume(config, resume, manifest)
    if is_cosmos25(config):
        wm = config["framework"].setdefault("world_model", {})
        if checkpoint and not resume:
            prepare_checkpoint_config(config, checkpoint, wm.get("latent_normalization"))
        else:
            wm["latent_normalization"] = validate_mode(wm.get("latent_normalization", "canonical"))
        if config["framework"]["name"] == "CosmoPredict25PerceiverVJEPA2AC":
            generation = config["framework"].setdefault("future_image_generation", {})
            if not generation.get("allow_temporal_extrapolation", False):
                generation.update(num_frames=5, num_latent_conditional_frames=1)
    return OmegaConf.create(config)


class TrainingProgress:
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        return {
            "completed_steps": self.trainer.completed_steps,
            "epoch": getattr(self.trainer, "vla_epoch_count", 0),
            "batches_in_epoch": getattr(self.trainer, "batches_in_epoch", 0),
        }

    def load_state_dict(self, state):
        self.trainer.completed_steps = int(state["completed_steps"])
        self.trainer.vla_epoch_count = int(state["epoch"])
        self.trainer.batches_in_epoch = int(state["batches_in_epoch"])


def data_contract(trainer):
    loader = trainer.vla_train_dataloader
    config = plain_config(trainer.config)
    return {
        "num_workers": loader.num_workers,
        "batch_size": loader.batch_size,
        "dataset_length": len(loader.dataset),
        "seed": config.get("seed"),
        "datasets": config.get("datasets"),
        "replay_wrapper": type(loader.dataset).__name__ == "ReplayDataset",
    }


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(path):
    # Only load trusted, locally produced training states, never arbitrary uploads.
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def initialize_training_state(trainer):
    loader = trainer.vla_train_dataloader
    stateful = bool(getattr(loader, "use_stateful_dataloader", False))
    if stateful:
        if trainer.accelerator.num_processes != 1:
            raise ValueError(
                "Native stateful data replay is validated only at world_size=1. "
                "For distributed training explicitly disable stateful_dataloader and "
                "select resume_data_policy=epoch_restart; optimizer state is still saved."
            )
        sampler = getattr(loader.batch_sampler, "sampler", None)
        if loader.num_workers > 0 and isinstance(sampler, RandomSampler):
            raise ValueError("Multiworker shuffled native replay is unsafe in this runtime; use the WM4A sequential sampler")
        # Accelerate 1.5 accesses the single-worker snapshot shape even at factor=0.
        # There is no distributed prefetch adjustment to make with one rank.
        loader.adjust_state_dict_for_prefetch = lambda: None
    trainer._training_progress = TrainingProgress(trainer)
    trainer.accelerator.register_for_checkpointing(trainer.lr_scheduler, trainer._training_progress)
    resume = trainer.config.trainer.get("resume_training_state")
    if not resume:
        return
    manifest = json.loads((Path(resume) / "training_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Refusing an incomplete training-state checkpoint")
    validate_resume(trainer.config, resume, manifest)
    if manifest["world_size"] != trainer.accelerator.num_processes:
        raise ValueError("Exact resume requires the same distributed world size")
    exact_data = stateful and manifest.get("stateful_dataloader", False)
    policy = trainer.config.trainer.get("resume_data_policy", "exact")
    if policy not in ("exact", "epoch_restart"):
        raise ValueError(f"Unknown resume_data_policy: {policy}")
    if exact_data and manifest.get("data_contract") != data_contract(trainer):
        raise ValueError("The saved and current data contracts differ; exact replay is unsafe")
    if exact_data and trainer.vla_train_dataloader.num_workers > 0:
        if not data_contract(trainer)["replay_wrapper"]:
            raise ValueError("Multiworker exact resume requires ReplayDataset")
    if not exact_data and policy != "epoch_restart":
        raise ValueError(
            "This loader cannot exactly restore its cursor/workers. Enable a stateful "
            "loader for both save/load, or explicitly select resume_data_policy=epoch_restart."
        )
    trainer.accelerator.load_state(str(resume))
    if exact_data:
        epoch = trainer.vla_epoch_count
        if trainer.batches_in_epoch == len(loader):
            # TorchData 0.11 automatically restarts a finished snapshot, while
            # Accelerate 1.5 does not serialize DataLoaderShard.iteration.
            # Normalize this boundary before creating a new iterator.
            loader.load_state_dict({})
            loader.base_dataloader.next_iter_state = None
            epoch += 1
            trainer.vla_epoch_count = epoch
            trainer.batches_in_epoch = 0
        loader.set_epoch(epoch)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
    restore_rng(Path(resume) / f"wm4a_rng_{trainer.accelerator.process_index}.pt")
    if not exact_data:
        warnings.warn("Model/optimizer/LR/RNG restored, but data resumes at an epoch boundary, not bitwise replay")
        trainer.batches_in_epoch = 0
        trainer.vla_epoch_count += 1
        if hasattr(trainer.vla_train_dataloader, "set_epoch"):
            trainer.vla_train_dataloader.set_epoch(trainer.vla_epoch_count)


def save_full_config(trainer):
    if not trainer.accelerator.is_main_process:
        return
    config = plain_config(trainer.config)
    raw = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(raw, "backbone") and hasattr(raw.backbone, "latent_normalization"):
        config["framework"]["world_model"]["latent_normalization"] = raw.backbone.latent_normalization
    destination = Path(trainer.config.output_dir) / "config.full.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    OmegaConf.save(OmegaConf.create(config), temporary)
    os.replace(temporary, destination)


def save_training_state(trainer):
    if not trainer.config.trainer.get("save_training_state", True):
        return
    destination = Path(trainer.checkpoint_dir) / f"steps_{trainer.completed_steps}_training_state"
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite a training state: {destination}")
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.save_state(str(temporary))
    torch.save(rng_state(), temporary / f"wm4a_rng_{trainer.accelerator.process_index}.pt")
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        config = plain_config(trainer.config)
        packages = {}
        for name in ("torch", "diffusers", "transformers", "accelerate", "deepspeed", "torchdata"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        manifest = {
            "schema_version": 2, "complete": True, "host": socket.gethostname(),
            "completed_steps": trainer.completed_steps,
            "world_size": trainer.accelerator.num_processes,
            "latent_normalization": config.get("framework", {}).get("world_model", {}).get("latent_normalization"),
            "stateful_dataloader": bool(getattr(trainer.vla_train_dataloader, "use_stateful_dataloader", False)),
            "progress": trainer._training_progress.state_dict(), "packages": packages,
            "data_contract": data_contract(trainer),
        }
        (temporary / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
        OmegaConf.save(OmegaConf.create(config), temporary / "config.full.yaml")
        os.replace(temporary, destination)
    trainer.accelerator.wait_for_everyone()


def save_weight_metadata(trainer, checkpoint_path):
    if not trainer.accelerator.is_main_process:
        return
    config = plain_config(trainer.config)
    metadata = {
        "schema_version": 1, "completed_steps": trainer.completed_steps,
        "latent_normalization": config.get("framework", {}).get("world_model", {}).get("latent_normalization"),
        "training_state": (
            os.path.relpath(
                Path(trainer.checkpoint_dir) / f"steps_{trainer.completed_steps}_training_state",
                Path(checkpoint_path).parent,
            )
            if config.get("trainer", {}).get("save_training_state", True) else None
        ),
    }
    path = Path(str(checkpoint_path) + ".metadata.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, path)


def save_final_checkpoint(trainer):
    destination = Path(trainer.config.output_dir) / "final_model"
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite a final export: {destination}")
    if getattr(trainer, "_last_checkpoint_step", None) != trainer.completed_steps:
        trainer._save_checkpoint()
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    if trainer.accelerator.is_main_process:
        temporary.mkdir()
        weight_path = temporary / "pytorch_model.pt"
        torch.save(state_dict, weight_path)
        save_weight_metadata(trainer, weight_path)
        OmegaConf.save(OmegaConf.create(plain_config(trainer.config)), temporary / "config.full.yaml")
        os.replace(temporary, destination)
    trainer.accelerator.wait_for_everyone()
