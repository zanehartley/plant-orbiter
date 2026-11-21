#!/usr/bin/env bash
# Build the diffsplat env (DiffSplat, chenguolin/DiffSplat): feed-forward text-to-3DGS baseline.
# Follows settings/setup.sh (torch 2.3.1+cu121, xformers 0.0.27, RaDe-GS's diff-gaussian-rasterization,
# settings/requirements.txt), with these changes: the CUDA 12.1 toolkit comes from conda (independent of the system CUDA);
# RaDe-GS is cloned into third_party/src at a pinned commit; transformers/accelerate/kiui, unpinned there, are capped at
# releases from before the code (Jan 2025) that work with the pinned diffusers 0.32; deepspeed (training only, never imported at inference) is left out;
# scikit-image and wandb are added (imported by the inference script but missing from requirements.txt).
# Weights: DiffSplat's own checkpoints go to $ORBIT_WEIGHTS/diffsplat (the repo's default is ./out in the code tree).
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
SRC="$ORBIT_REPO/third_party/src"   # CUDA extensions, at pinned commits
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"
D="$ORBIT_REPO/third_party/DiffSplat"
[ -d "$D" ] || git clone https://github.com/chenguolin/DiffSplat.git "$D"
git -C "$D" checkout a99d757   # 2026-03-19, latest commit when installed (2026-10-12)

conda env list | grep -q '^diffsplat ' || conda create -y -n diffsplat python=3.10
set +u; conda activate diffsplat; set -u   # conda's compiler/nvcc activation hooks read unset variables
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include" TORCH_CUDA_ARCH_LIST="8.6;8.9" \
  LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
set +u; conda deactivate && conda activate diffsplat; set -u
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121
pip install -r <(grep -v "^deepspeed" "$D/settings/requirements.txt" \
  | sed -e 's|^transformers$|transformers==4.48.3|' -e 's|^accelerate$|accelerate==1.3.0|' \
        -e 's|^kiui$|kiui==0.2.14|') scikit-image wandb   # kiui 0.3.5 ships an empty kiui/typing.py

mkdir -p "$SRC"
[ -d "$SRC"/RaDe-GS ] || git clone --recursive https://github.com/BaowenZ/RaDe-GS.git "$SRC"/RaDe-GS
# 5f5cd3b (2025-12-26): the last commit with the rasterizer API DiffSplat calls (require_coord); c7c56cc (2026-03-16)
# removed it.
git -C "$SRC"/RaDe-GS checkout -q 5f5cd3b && git -C "$SRC"/RaDe-GS submodule update -q --init --recursive
pip install --no-build-isolation "$SRC"/RaDe-GS/submodules/diff-gaussian-rasterization
python -c "import torch, xformers, diff_gaussian_rasterization, diffusers, kiui; print('DIFFSPLAT', torch.__version__, \
torch.cuda.is_available())"

# Weights: text-conditioned DiffSplat (SD3.5-medium, the README's "best performance") + its GSVAE/GSRecon, and the
# gated base model stabilityai/stable-diffusion-3.5-medium (needs Hub access).
(cd "$D" && python download_ckpt.py --model_type sd35m --local_dir "$ORBIT_WEIGHTS/diffsplat")
"$HF" download stabilityai/stable-diffusion-3.5-medium
"$HF" download madebyollin/taesd3   # tiny SD3 VAE the SD3 GSVAE builds on (src/models/gsvae.py TAE_DICT)
