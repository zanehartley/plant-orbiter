#!/usr/bin/env bash
# Build the trellis2 env (TRELLIS.2-4B image-to-3D): the same-input control for fruit counting.
# Same steps as the repo's `setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
# --flexgemm`, with three changes: the CUDA 12.4 toolkit comes from conda (independent of the system CUDA), extensions are cloned
# into "$ORBIT_REPO/third_party/src" at pinned commits instead of /tmp, and pillow-simd is skipped (needs sudo apt; speed only).
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
# Needs HF access to the gated facebook/dinov3-vitl16-pretrain-lvd1689m and briaai/RMBG-2.0 for the weight step.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
SRC="$ORBIT_REPO/third_party/src"   # CUDA extensions, at pinned commits
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"   # recent hub CLI
T="$ORBIT_REPO/third_party/TRELLIS.2"
[ -d "$T" ] || git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$T"
git -C "$T" checkout 75fbf01   # 2026-06-05, latest commit when installed (2026-10-11)

conda env list | grep -q '^trellis2 ' || conda create -y -n trellis2 python=3.10
conda activate trellis2
conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit
conda install -y -c conda-forge gcc=12 gxx=12
# LIBRARY_PATH: extensions link -lcuda, which conda's toolkit ships only as a stub (link time only; at run time the
# real driver library is used).
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include" TORCH_CUDA_ARCH_LIST="8.6;8.9" \
  LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
conda deactivate && conda activate trellis2
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# transformers pinned to the last 4.x: the repo's DINOv3 wrapper (2025-12-16) walks model.layer, which 5.x moved.
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.6 gradio==6.0.1 \
  tensorboard pandas lpips zstandard kornia timm
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install flash-attn==2.7.3 --no-build-isolation

mkdir -p "$SRC"
clone() {  # clone <url> <dir> <ref>
  [ -d "$2" ] || git clone --recursive "$1" "$2"
  git -C "$2" checkout "$3" && git -C "$2" submodule update --init --recursive
}
clone https://github.com/NVlabs/nvdiffrast.git "$SRC"/nvdiffrast v0.4.0
clone https://github.com/JeffreyXiang/nvdiffrec.git "$SRC"/nvdiffrec b296927cc7
clone https://github.com/JeffreyXiang/CuMesh.git "$SRC"/CuMesh 12289e1062
clone https://github.com/JeffreyXiang/FlexGEMM.git "$SRC"/FlexGEMM 6dd94a859c
for ext in nvdiffrast nvdiffrec CuMesh FlexGEMM; do pip install "$SRC"/$ext --no-build-isolation; done
pip install "$T/o-voxel" --no-build-isolation
python -c "import torch, flash_attn, nvdiffrast.torch, cumesh, flex_gemm, o_voxel; print('TRELLIS2', torch.__version__)"

# Weights into $HF_HUB_CACHE: the pipeline, the TRELLIS v1 decoder it reuses, its DINOv3 image encoder, and RMBG-2.0
# (loaded at start-up even though our RGBA inputs skip background removal).
"$HF" download microsoft/TRELLIS.2-4B
"$HF" download microsoft/TRELLIS-image-large --include "ckpts/ss_dec_conv3d_16l8_fp16*"
"$HF" download facebook/dinov3-vitl16-pretrain-lvd1689m
"$HF" download briaai/RMBG-2.0
