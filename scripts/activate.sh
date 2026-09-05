# Source this file: source scripts/activate.sh
_geer_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_geer_root/.venv/bin/activate"
export CC=gcc-14
export CXX=g++-14
export MAX_JOBS="${MAX_JOBS:-4}"
# nvcc searches this local copy before the system CUDA headers.
export NVCC_PREPEND_FLAGS="-I$_geer_root/.cache/cuda-include"
unset _geer_root
