#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:?Pass the absolute simulation Python path}"
PREFIX="${2:?Pass an installation prefix owned by this release setup}"
SOURCE="$ROOT/third_party/orocos_kinematics_dynamics"
BUILD="$PREFIX/.kdl-build"
[[ "$PREFIX" = /* && "$PYTHON" = /* ]] || { echo "Use absolute paths." >&2; exit 2; }
[[ ! -e "$BUILD" ]] || { echo "Refusing to overwrite an existing KDL build." >&2; exit 2; }
VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$VERSION" = 3.10 ]] || { echo "Use the pinned Python 3.10 interpreter." >&2; exit 2; }
export PATH="$PREFIX/bin:$PATH"
cmake -S "$SOURCE/orocos_kdl" -B "$BUILD/core" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DBUILD_TESTING=OFF
cmake --build "$BUILD/core" --parallel "${MAX_JOBS:-2}"
cmake --install "$BUILD/core"
cmake -S "$SOURCE/python_orocos_kdl" -B "$BUILD/python" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DPython3_EXECUTABLE="$PYTHON" \
  -DPYTHON_SITE_PACKAGES_INSTALL_DIR="$PREFIX/lib/python3.10/site-packages" \
  -DCMAKE_INSTALL_RPATH="$PREFIX/lib"
cmake --build "$BUILD/python" --parallel "${MAX_JOBS:-2}"
cmake --install "$BUILD/python"
PYTHONPATH="$PREFIX/lib/python3.10/site-packages" "$PYTHON" -c \
  'import PyKDL; assert PyKDL.__version__ == "1.5.4"; print(PyKDL.__file__, PyKDL.__version__)'
