#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CC=gcc-14
export CXX=g++-14
export MAX_JOBS="${MAX_JOBS:-4}"
command -v uv >/dev/null
command -v nvcc >/dev/null
if ! command -v "$CXX" >/dev/null; then
    echo 'CUDA 12.8 requires GCC/G++ 14: sudo apt-get install -y g++-14' >&2
    exit 1
fi

# This checkout has .gitmodules but no tracked GLM gitlink.
glm_dir=gsplat/cuda/csrc/third_party/glm
if [[ ! -f "$glm_dir/glm/glm.hpp" ]]; then
    mkdir -p .cache "$glm_dir"
    curl -fL --retry 3 \
        https://codeload.github.com/g-truc/glm/tar.gz/refs/tags/1.0.1 \
        -o .cache/glm-1.0.1.tar.gz
    echo '9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c  .cache/glm-1.0.1.tar.gz' | sha256sum -c -
    tar -xzf .cache/glm-1.0.1.tar.gz --strip-components=1 -C "$glm_dir"
fi

uv venv --python 3.11 --allow-existing
# These legacy CUDA packages import torch even when preparing their metadata.
uv pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    'setuptools>=77,<81' wheel ninja 'numpy<2' \
    --index https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 \
    --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run --no-sync python scripts/prepare_cuda_headers.py
source scripts/activate.sh
uv sync --locked
