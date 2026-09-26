#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Evaluation requires a Linux x86_64 NVIDIA CUDA host. The Mac copy is the portable source archive." >&2
  exit 2
fi
command -v conda >/dev/null || { echo "Install Miniforge/Conda first." >&2; exit 2; }
PREFIX="${1:-$ROOT/.runtime/envs}"
POLICY="$PREFIX/policy"
SIM="$PREFIX/simulation"
if [[ -e "$POLICY" || -e "$SIM" ]]; then
  echo "Refusing to alter existing environments. Choose a new empty prefix." >&2
  exit 2
fi
conda create -y -p "$POLICY" -c conda-forge python=3.10.19 pip
"$POLICY/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$POLICY/bin/python" -m pip install -r "$ROOT/environment/policy.requirements.txt"
"$POLICY/bin/python" -m pip install packaging ninja
MAX_JOBS="${MAX_JOBS:-4}" "$POLICY/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation

conda create -y -p "$SIM" -c conda-forge python=3.10.18 pip cmake eigen cxx-compiler
"$SIM/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
"$SIM/bin/python" -m pip install -r "$ROOT/environment/simulation.requirements.txt"
bash "$ROOT/environment/build_kdl.sh" "$SIM/bin/python" "$SIM"
echo "POLICY_PYTHON=$POLICY/bin/python"
echo "SIM_PYTHON=$SIM/bin/python"
echo "Source packages are selected through release.evaluate's isolated PYTHONPATH."
