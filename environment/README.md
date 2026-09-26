# WM4A Environment Installation

Use Linux x86_64 with an NVIDIA CUDA driver and EGL rendering libraries.
Keep the policy and simulation dependencies in separate Python 3.10
environments. These installation scripts do not provide a native macOS runtime.

## Prerequisites

- Install Conda or Miniforge and make `conda` available on `PATH`.
- Provide the CUDA toolkit and C++ compiler needed to build FlashAttention.
- Provide host NVIDIA EGL libraries. `EGL_LIBRARY_PATH` can select their location.
- Supply the required RoboCasa, RoboSuite, pykdl_utils, hrl_geom, and
  Orocos KDL / PyKDL sources before using the corresponding components.
- The KDL build script expects
  `third_party/orocos_kinematics_dynamics`. This directory is absent from
  the current copy and must be supplied before the KDL build.

## Create Separate Environments

Run from the repository root with a new absolute prefix:

```bash
bash environment/create_linux_envs.sh /absolute/empty/environment/prefix
```

The script creates `policy` and `simulation` subdirectories and refuses
to overwrite existing environments. Package installation requires access
to the configured Conda and pip package indexes.

## Dependency Versions

| Component | Policy | Simulation |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.18 |
| PyTorch | 2.8.0 + CUDA 12.8 | 2.5.1 + CUDA 12.4 |
| TorchVision | 0.23.0 | 0.20.1 |
| NumPy | 2.2.6 | 1.26.4 |
| Gymnasium | Not required | 1.0.0 |
| MuJoCo | Not required | 3.2.6 |
| Transformers | 5.8.1 | Not required |
| Diffusers | 0.38.0 | Not required |
| FlashAttention | 2.8.3 | Not required |
| PyKDL | Not required | 1.5.4 |
| PyAV | 12.3.0 | 12.3.0 |

Dependency lists are stored in `policy.requirements.txt` and
`simulation.requirements.txt`. The `*.observed.json` files contain package
metadata, which can include duplicate entries; use the pinned installation
scripts and requirement files for environment creation.

## KDL Build

The environment script invokes `build_kdl.sh` for the simulation environment.
To invoke that build separately, supply the Python executable and prefix:

```bash
bash environment/build_kdl.sh /absolute/prefix/simulation/bin/python /absolute/prefix/simulation
```

Use a new build directory and the Python 3.10 interpreter from that prefix.

## Local Path Configuration

Example paths under `/path/to/` are placeholders for locally supplied resources.
Set these variables for the root `run_training.sh` entry point:

- `CUDA_VISIBLE_DEVICES`: GPU IDs allocated to this run.
- `WM4A_DATA_ROOT`: local dataset directory.
- `WM4A_COSMOS_ROOT`: local Cosmos initialization directory.
- `WM4A_VJEPA_CHECKPOINT`: local V-JEPA initialization checkpoint.
- `WM4A_RESUME_CHECKPOINT`: local policy resume checkpoint.
- `WM4A_OUTPUT_ROOT`: optional output directory.
- `WM4A_CONFIG` and `WM4A_VJEPA_REPO`: optional local overrides.

The root launcher defaults to one process and disables W&B network tracking.
No account credentials are bundled. Additional example launchers require
their placeholder paths to be configured before use.
