#!/usr/bin/env bash
# Build the trellis env (TRELLIS v1 text-to-3D, microsoft/TRELLIS-text-xlarge): native text-to-3D baseline.
# TRELLIS.2 is image-only, so v1 is the newest TRELLIS that takes text. Same steps as the repo's `setup.sh --new-env
# --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast` (torch 2.4.0 + cu121, the newest
# pair the script has kaolin wheels for), with these changes: the CUDA 12.1 toolkit comes from conda (independent of the
# system CUDA), extensions are cloned into third_party/src at pinned commits instead of /tmp, and attention runs on xformers
# (ATTN_BACKEND=xformers, a documented option) instead of flash-attn, which has no prebuilt wheel here.
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
SRC="$ORBIT_REPO/third_party/src"   # CUDA extensions, at pinned commits
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"
T="$ORBIT_REPO/third_party/TRELLIS"
[ -d "$T" ] || git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$T"
git -C "$T" checkout 442aa1e   # 2025-11-05, latest commit when installed (2026-10-12)

conda env list | grep -q '^trellis ' || conda create -y -n trellis python=3.10
conda activate trellis
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11   # CUDA 12.1 supports gcc <= 12.2
# LIBRARY_PATH: extensions link -lcuda, which conda's toolkit ships only as a stub (link time only).
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include" TORCH_CUDA_ARCH_LIST="8.6;8.9" \
  LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" ATTN_BACKEND=xformers SPCONV_ALGO=native
conda deactivate && conda activate trellis
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh \
  open3d xatlas pyvista pymeshfix igraph "transformers<5" plyfile   # 5.x postdates this TRELLIS commit
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install spconv-cu120

mkdir -p "$SRC"
clone() {  # url dir commit
  [ -d "$2" ] || git clone --recurse-submodules "$1" "$2"
  git -C "$2" checkout -q "$3" && git -C "$2" submodule update -q --init --recursive
}
clone https://github.com/NVlabs/nvdiffrast.git "$SRC"/nvdiffrast v0.4.0
clone https://github.com/JeffreyXiang/diffoctreerast.git "$SRC"/diffoctreerast b09c20b84e
clone https://github.com/autonomousvision/mip-splatting.git "$SRC"/mip-splatting dda02ab5ec
pip install --no-build-isolation "$SRC"/nvdiffrast "$SRC"/diffoctreerast \
  "$SRC"/mip-splatting/submodules/diff-gaussian-rasterization
python -c "import torch, xformers, spconv, kaolin, nvdiffrast.torch, diffoctreerast, diff_gaussian_rasterization; \
print('TRELLIS', torch.__version__, torch.cuda.is_available())"

"$HF" download microsoft/TRELLIS-text-xlarge
"$HF" download openai/clip-vit-large-patch14   # text encoder named in the model's pipeline.json
# The pipeline.json loads its decoders from the old repo name (now a redirect to microsoft/TRELLIS-image-large);
# offline loads look the files up under the name asked for, so download them under that name.
"$HF" download JeffreyXiang/TRELLIS-image-large --include "ckpts/ss_dec_conv3d_16l8_fp16*" \
  --include "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16*" --include "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16*" \
  --include "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16*"
