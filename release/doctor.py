"""Check local runtime versions and importability without allocating a GPU."""
import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["policy", "simulation", "training"], required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("This package is an archive on macOS. Execution requires Linux x86_64 with NVIDIA CUDA.")
    role = "policy" if args.role == "training" else args.role
    locked = json.loads((ROOT / "environment/validated_versions.json").read_text())[role]["packages"]
    differences = []
    training_only = {"deepspeed", "torchdata"}
    for name, version in locked.items():
        if args.role == "policy" and name in training_only:
            continue
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        compatible_torch_build = name == "torch" and "+" not in version and actual and actual.split("+")[0] == version
        if actual != version and not compatible_torch_build:
            differences.append(f"{name}: expected {version}, found {actual}")
    if sys.version_info[:2] != (3, 10):
        differences.append("Python 3.10 is required.")
    if differences:
        raise SystemExit("\n".join(differences))
    if role == "simulation":
        for name in ("robocasa", "robosuite", "PyKDL",
                     "examples.Robocasa_tabletop.eval_files.model2robocasa_interface_ee_wm"):
            module = importlib.import_module(name)
            print(name, getattr(module, "__file__", "built-in"))
        for relative in ("third_party/robocasa/robocasa/models/assets",
                         "third_party/robosuite/robosuite/models/assets",
                         "runtime_assets/GR1T2/GR1T2_fourier_hand_6dof.urdf"):
            if not (ROOT / relative).exists():
                raise FileNotFoundError(f"Missing resource: {relative}")
    else:
        from starVLA.model.framework.WAM_VJEPA.CosmoPredict25PerceiverVJEPA2AC import CosmoPredict25_Perceiver_VJEPA2AC
        print(CosmoPredict25_Perceiver_VJEPA2AC.__name__)
    print("Runtime preflight passed; this is not a GPU inference or rendering test.")


if __name__ == "__main__":
    main()
