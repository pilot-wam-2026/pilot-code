"""Reconstruct backbone initialization files from the complete policy checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def select(state, prefix):
    subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not subset:
        raise ValueError(f"No checkpoint parameters under {prefix!r}")
    return subset


def verify_cache(cache, stamp, manifest):
    if stamp.get("schema_version") != 2:
        raise ValueError("Runtime cache lacks per-file integrity records; choose a new cache directory.")
    if stamp.get("checkpoint_bytes") != manifest["checkpoint_bytes"]:
        raise ValueError("The runtime cache belongs to another checkpoint.")
    if stamp.get("expected_sha256") != manifest["checkpoint_sha256"]:
        raise ValueError("The runtime cache belongs to another checkpoint hash.")
    records = stamp.get("files", {})
    required = {component["output"] for component in manifest["components"]}
    for label in manifest["metadata_labels"]:
        required.update(
            "backbones/" + str(path.relative_to(ROOT / "model_metadata"))
            for path in (ROOT / "model_metadata" / label).rglob("*") if path.is_file()
        )
    if set(records) != required:
        raise ValueError("Runtime cache inventory does not match the released components.")
    for relative, expected in records.items():
        path = cache / relative
        if cache.resolve() not in path.resolve().parents or path.is_symlink():
            raise ValueError(f"External runtime cache component: {relative}")
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Missing or changed runtime cache component: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/steps_340000_pytorch_model.pt")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime")
    parser.add_argument("--skip-checksum", action="store_true", help="Only for trusted local debugging.")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    cache = args.cache.resolve()
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    if checkpoint.stat().st_size != manifest["checkpoint_bytes"]:
        raise ValueError("Checkpoint size does not match the selected historical checkpoint.")
    expected_hash = manifest.get("checkpoint_sha256")
    if not args.skip_checksum:
        if not expected_hash:
            raise ValueError("Release manifest has no pinned checkpoint SHA-256.")
        actual_hash = digest(checkpoint)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {actual_hash}")
    else:
        actual_hash = None
    cache.mkdir(parents=True, exist_ok=True)
    stamp_path = cache / "materialization.json"
    if stamp_path.exists():
        stamp = json.loads(stamp_path.read_text())
        verify_cache(cache, stamp, manifest)
        print(f"Using verified existing cache: {cache}", flush=True)
    else:
        import torch

        torch.set_num_threads(1)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if len(state) != manifest["checkpoint_keys"]:
            raise ValueError("Checkpoint key count does not match the release manifest.")
        staging = Path(tempfile.mkdtemp(prefix="materialize-", dir=cache))
        files = {}
        try:
            for label in manifest["metadata_labels"]:
                shutil.copytree(ROOT / "model_metadata" / label, staging / "backbones" / label)
            for component in manifest["components"]:
                output = staging / component["output"]
                output.parent.mkdir(parents=True, exist_ok=True)
                if "prefix" in component:
                    subset = select(state, component["prefix"])
                else:
                    subset = {name: select(state, prefix) for name, prefix in component["prefixes"].items()}
                print(f"Writing {component['output']}", flush=True)
                torch.save(subset, output)
                del subset
            for path in sorted((staging / "backbones").rglob("*")):
                if path.is_file():
                    files[path.relative_to(staging).as_posix()] = {
                        "bytes": path.stat().st_size, "sha256": digest(path),
                    }
            if (cache / "backbones").exists():
                raise FileExistsError("Refusing to overwrite an unrecognized backbone cache.")
            (staging / "backbones").rename(cache / "backbones")
            stamp = {
                "schema_version": 2,
                "expected_sha256": expected_hash,
                "verified_sha256": actual_hash,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "files": files,
                "source_checkpoint": str(checkpoint),
                "parameter_conversion": "none; original keys and tensor dtypes retained",
            }
            stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        del state

    from omegaconf import OmegaConf

    # The historical factories select model classes using canonical path names.
    for alias, label in manifest.get("backbone_aliases", {}).items():
        target = cache / "backbones" / alias
        if not target.exists():
            target.symlink_to(label, target_is_directory=True)
        if target.resolve() != (cache / "backbones" / label).resolve():
            raise ValueError(f"Unexpected backbone alias: {target}")
    config = OmegaConf.load(ROOT / "configs/pilot_340000.yaml")
    for key, relative in manifest["config_paths"].items():
        resolved = ROOT / relative if relative.startswith("starVLA/") else cache / relative
        OmegaConf.update(config, key, str(resolved), merge=False)
    config.framework.world_model.latent_normalization = "legacy"
    config.framework.future_image_generation.num_frames = 5
    config.framework.future_image_generation.num_latent_conditional_frames = 1
    config.datasets.vla_data.data_root_dir = str(cache / "datasets-not-included")
    config.run_root_dir = str(cache / "training")
    config.run_id = "pilot"
    config.output_dir = str(cache / "run")
    run = cache / "run"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = run / "checkpoints/model.pt"
    if target.is_symlink():
        if target.resolve() != checkpoint:
            raise FileExistsError(f"Runtime checkpoint link already points elsewhere: {target}")
    elif target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    else:
        target.symlink_to(checkpoint)
    OmegaConf.save(config, run / "config.yaml")
    shutil.copy2(ROOT / "configs/dataset_statistics.json", run / "dataset_statistics.json")
    Path(str(target) + ".metadata.json").write_text(json.dumps({
        "checkpoint_step": 340000, "latent_normalization": "legacy",
        "checkpoint_sha256": expected_hash, "num_frames": 5,
    }, indent=2) + "\n")
    print(f"Prepared checkpoint: {target}", flush=True)
    print("Original checkpoint and archived training config were not modified.", flush=True)


if __name__ == "__main__":
    main()
