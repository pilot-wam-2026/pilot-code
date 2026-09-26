"""Check core versions and source selection without loading a policy or rendering."""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["policy", "simulation"], required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from release.evaluate import runtime_environment
    for entry in reversed(runtime_environment(ROOT)["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, entry)
    versions = (
        {"torch": "2.8.0+cu128", "numpy": "2.2.6", "transformers": "5.8.1", "diffusers": "0.38.0", "flash_attn": "2.8.3"}
        if args.role == "policy" else
        {"torch": "2.5.1+cu124", "numpy": "1.26.4", "gymnasium": "1.0.0", "mujoco": "3.2.6", "robosuite": "1.5.1", "PyKDL": "1.5.4", "av": "12.3.0"}
    )
    records = {"python": platform.python_version(), "role": args.role, "modules": {}}
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10.")
    for name, expected in versions.items():
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        records["modules"][name] = {"version": version, "source": module.__file__}
        if version != expected:
            raise RuntimeError(f"{name}: imported {version!r}, expected {expected!r}")
    if args.role == "simulation":
        for name in ["robocasa", "robosuite"]:
            module = importlib.import_module(name)
            source = Path(module.__file__).resolve()
            if not source.is_relative_to(ROOT / "third_party" / name):
                raise RuntimeError(f"Unexpected {name} source: {source}")
            records["modules"].setdefault(name, {"source": str(source)})
        from examples.Robocasa_tabletop.eval_files.gr1_pos_transform_new import BodyRetargeter, GR1RetargetConfig
        config = GR1RetargetConfig()
        BodyRetargeter(urdf_path=Path(config.urdf_path), camera_intrinsics=config.camera_intrinsics)
        records["retarget_initialization"] = True
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
