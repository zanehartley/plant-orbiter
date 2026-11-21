#!/usr/bin/env bash
# Build the hunyuan env (Hunyuan3D 2.1, shape + PBR texture): single-photo baseline.
# Check the Hunyuan3D 2.1 licence for your territory first. Follows the repo's install steps, with these changes: the
# CUDA 12.4 toolkit comes from conda, bpy is not installed (only needed for an optional .glb export; a one-line patch,
# env/hunyuan3d_bpy.patch, makes the import optional), and the extensions are compiled for sm_86 + sm_89.
# Run on a GPU node. Package versions used for the paper: env/freeze_hunyuan.txt.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
T="$ORBIT_REPO/third_party/Hunyuan3D-2.1"
[ -d "$T" ] || git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$T"
git -C "$T" checkout 82920d6   # 2025-10-17
git -C "$T" apply --check "$ORBIT_REPO/env/hunyuan3d_bpy.patch" 2>/dev/null && git -C "$T" apply "$ORBIT_REPO/env/hunyuan3d_bpy.patch"

conda env list | grep -q '^hunyuan ' || conda create -y -n hunyuan python=3.10
set +u; conda activate hunyuan; set -u
conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;8.9"} MAX_JOBS=8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation basicsr==1.4.2
grep -v "^bpy" "$T/requirements.txt" > "$T/requirements_nobpy.txt" && pip install -r "$T/requirements_nobpy.txt"
cd "$T/hy3dpaint/custom_rasterizer" && pip install --no-build-isolation -e .
cd "$T/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh
mkdir -p "$T/hy3dpaint/ckpt"
[ -f "$T/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || \
  wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P "$T/hy3dpaint/ckpt"
cd /tmp && python -c "import torch, custom_rasterizer; print('hunyuan', torch.__version__, torch.cuda.is_available())"
# The weights (tencent/Hunyuan3D-2.1) are downloaded into $HF_HUB_CACHE on the first run.
