#!/usr/bin/env bash
# Build the nerfstudio env (splatfacto + COLMAP model conversion). README step 2b.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n nerfstudio python=3.8 git
conda activate nerfstudio
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11 colmap=3.8 libxcrypt
# Python 3.8 headers need crypt.h (missing from conda's gcc sysroot), and gsplat must JIT-compile with the env's
# CUDA 11.8, not the system CUDA. Saved as env vars so every `conda activate`/`conda run` gets them.
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include"
conda deactivate && conda activate nerfstudio
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r "$ORBIT_REPO/env/freeze_nerfstudio.txt"
[ -d "$NERFSTUDIO_DIR" ] || git clone https://github.com/nerfstudio-project/nerfstudio.git "$NERFSTUDIO_DIR"
git -C "$NERFSTUDIO_DIR" checkout 50e0e3c70c775e89333256213363badbf074f29d
git -C "$NERFSTUDIO_DIR" apply "$ORBIT_REPO/env/nerfstudio_splatfacto.patch"
pip install --no-deps -e "$NERFSTUDIO_DIR"
python -c "import torch, gsplat, nerfstudio; print('NS', torch.__version__, gsplat.__version__)"
# Then, on a GPU machine, compile gsplat once per GPU type: setup/build_gsplat.sh
