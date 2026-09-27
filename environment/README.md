# Runtime Environments

The resource bundle is portable source and data, not a copy of a user's
activated shell or credential cache. Absolute private paths, SSH settings,
tokens, shell history, and unrelated environments are not included.

## Platform

Use Linux x86_64 with an NVIDIA driver compatible with the chosen CUDA
wheels and working EGL. The policy uses PyTorch 2.8.0/CUDA 12.8; the
simulator uses PyTorch 2.5.1 and MuJoCo 3.2.6. Do not combine their NumPy
environments: the validated policy uses NumPy 2.2.6 and simulation uses
NumPy 1.26.4.

macOS can store and inspect this bundle. CUDA policy inference and the
supplied Linux simulator have not been ported to macOS/MPS.

## Create New Environments

```bash
bash environment/create_linux_envs.sh /absolute/new/pilot-envs
export POLICY_PYTHON=/absolute/new/pilot-envs/policy/bin/python
export SIM_PYTHON=/absolute/new/pilot-envs/simulation/bin/python
source environment/activate_paths.sh
```

The script refuses to overwrite existing environment directories. Conda,
a C++ build toolchain, a compatible CUDA development toolkit (`nvcc`) for
the FlashAttention build, and sufficient disk space are required. The PyKDL
build uses the supplied Orocos 1.5.4 source. Driver libraries remain host
dependencies; do not copy another host's driver libraries over system files.

`validated_versions.json` records the source evaluation's versions.
The requirement files describe installation inputs; a clean installation
still requires its own preflight and smoke test. Training additionally uses
`training.requirements.txt`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `PILOT_ROOT` | Root of the extracted release |
| `POLICY_PYTHON` | Policy interpreter |
| `SIM_PYTHON` | Simulator interpreter |
| `CUDA_VISIBLE_DEVICES` | Explicitly assigned CUDA device(s); never set to reserve resources |
| `MUJOCO_EGL_DEVICE_ID` | Physical EGL device index; set per simulator process |
| `MUJOCO_GL=egl` | Headless MuJoCo rendering backend |
| `PYOPENGL_PLATFORM=egl` | OpenGL backend |
| `EGL_LIBRARY_PATH` | Optional directory containing the host's NVIDIA EGL libraries |
| `PYTHONHASHSEED=0` | Fixed Python hash seed, set before interpreter startup |
| `PILOT_DATA_ROOT` | External converted dataset root for training |
| `PILOT_OUTPUT_ROOT` | New training output root, outside the release |
| `PILOT_COSMOS_ROOT` | Constructor-compatible Cosmos initialization directory |
| `PILOT_VJEPA_CHECKPOINT` | Constructor-compatible VJEPA initialization file |

`activate_paths.sh` contains the portable non-secret environment setup.
`release.evaluate` creates isolated child-process environments, explicitly
sets each CUDA/EGL assignment, and does not overwrite the user's shell.

## Storage And Verification

The checkpoint is 23.91 GB in decimal units. Simulation assets occupy
approximately 9.6 GB unpacked. Backbone materialization requires about
another checkpoint's size; environments and evaluation videos add more.
120 GiB free storage and 128 GiB host RAM are conservative planning values.

The published asset archive, when present, is restored by
`python -m release.unpack_assets`. It rejects unsafe paths and verifies
file hashes. `python -m release.verify --assets` checks both checkpoint
and unpacked-resource integrity.

Required local resources:

```text
third_party/robocasa/
third_party/robosuite/
third_party/pykdl_utils/
third_party/hrl_geom/
third_party/orocos_kinematics_dynamics/
runtime_assets/GR1T2/GR1T2_fourier_hand_6dof.urdf
```

The customized environment is not interchangeable with a generic current
RoboCasa wheel. Legal notices in these directories must remain intact.

The archived resources include the simulator code as well as its meshes,
textures, XML, and robot description. They are not only environment-variable
settings. Two unused USD conversion tools have locally parameterized path
defaults; these changes and the original/new file hashes are recorded in
`environment/resource_manifest.json`. The MuJoCo evaluation code and
assets are otherwise byte-identical to the verified source inventory.
