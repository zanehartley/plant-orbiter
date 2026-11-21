#!/usr/bin/env bash
# gsplat 1.4.0 compiles its CUDA kernels on first import (cached in ~/.cache/torch_extensions per GPU arch).
# Run once per GPU type on a GPU machine (or inside a cluster job) before training a splat.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nerfstudio
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "from gsplat.cuda._backend import _C; print('GSPLAT_CUDA_OK', _C is not None)"
