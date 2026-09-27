"""Explicit Cosmos coordinates, independent of Diffusers attribute semantics."""
import json
import warnings
from pathlib import Path

import torch

MODES = ("canonical", "legacy")


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"latent_normalization must be one of {MODES}, got {mode!r}")
    return mode


def statistics(vae):
    mean = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    if mean.shape != std.shape or not torch.isfinite(mean).all():
        raise ValueError("Invalid VAE latent mean/std")
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("VAE latent std must be finite and positive")
    return mean, std


def configure_coordinates(pipe, mode):
    """Return the custom encoder divisor and align native encode/decode."""
    validate_mode(mode)
    mean, sigma = statistics(pipe.vae)
    actual = pipe.latents_std.detach().float().cpu()
    if torch.allclose(actual, sigma, rtol=1e-5, atol=1e-6):
        convention = "divisor"
    elif torch.allclose(actual, sigma.reciprocal(), rtol=1e-5, atol=1e-6):
        convention = "multiplier"
    else:
        raise ValueError("Unknown native Cosmos latent contract; refusing to guess")
    divisor = sigma if mode == "canonical" else sigma.reciprocal()
    pipe.latents_mean = mean.clone()
    pipe.latents_std = (divisor if convention == "divisor" else divisor.reciprocal()).clone()
    return mean, divisor, convention


def is_cosmos25(config):
    return str(config.get("framework", {}).get("name", "")).startswith("CosmoPredict25")


def prepare_checkpoint_config(config, checkpoint, explicit_mode=None):
    if not is_cosmos25(config):
        if explicit_mode is not None:
            raise ValueError("latent_normalization override is only supported for Cosmos-Predict2.5")
        return config
    framework = config["framework"]
    wm = framework.setdefault("world_model", {})
    declared = wm.get("latent_normalization")
    sidecar = Path(str(checkpoint) + ".metadata.json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        saved = metadata.get("latent_normalization")
        if declared is not None and saved is not None and declared != saved:
            raise ValueError("Checkpoint sidecar and run configuration disagree on latent coordinates")
        declared = saved or declared
    if declared is not None:
        validate_mode(declared)
    if explicit_mode is not None:
        validate_mode(explicit_mode)
        if declared is not None and declared != explicit_mode:
            raise ValueError("Explicit latent coordinates conflict with checkpoint metadata")
    mode = explicit_mode or declared
    if mode is None:
        raise ValueError(
            "This legacy checkpoint has no latent_normalization metadata. "
            "Pass latent_normalization='canonical' for verified pre-upgrade weights "
            "or 'legacy' for verified inverse-std-trained weights; do not guess from its name."
        )
    wm["latent_normalization"] = mode
    if framework.get("name") == "CosmoPredict25PerceiverVJEPA2AC":
        generation = framework.setdefault("future_image_generation", {})
        if not generation.get("allow_temporal_extrapolation", False):
            if generation.get("num_frames", 5) != 5:
                warnings.warn("Replacing legacy visualization length with the trained 5-frame horizon", stacklevel=2)
            generation["num_frames"] = 5
            generation["num_latent_conditional_frames"] = 1
    return config


def validate_generation_length(requested, trained, allow_extrapolation=False):
    requested, trained = int(requested), int(trained)
    if requested <= 0 or (requested - 1) % 4:
        raise ValueError("Cosmos output length must be positive and of the form 4*k+1")
    if requested != trained and not allow_extrapolation:
        raise ValueError(
            f"Requested {requested} frames but this supervision uses {trained}. "
            "Use the trained horizon, or explicitly set allow_temporal_extrapolation=True."
        )
    return requested
