#!/usr/bin/env bash
# Source this file from a Linux CUDA shell; it contains no credentials.
PILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PILOT_ROOT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PILOT_ROOT:$PILOT_ROOT/third_party/robocasa:$PILOT_ROOT/third_party/robosuite:$PILOT_ROOT/third_party/pykdl_utils/src:$PILOT_ROOT/third_party/hrl_geom/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __GLX_VENDOR_LIBRARY_NAME=nvidia
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -n "${EGL_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$EGL_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID are intentionally NOT allocated here.
