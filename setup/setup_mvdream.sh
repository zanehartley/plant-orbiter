#!/usr/bin/env bash
# Build the mvdream env (MVDream in threestudio, bytedance/MVDream-threestudio): score-distillation baseline (NeRF).
# Follows the repo's README and docker/Dockerfile (torch 2.0.1+cu118, nerfacc v0.5.2, tiny-cuda-nn, requirements.txt,
# then extern/MVDream), with these changes: the CUDA 11.8 toolkit comes from conda (independent of the system CUDA); git
# dependencies are pinned (tiny-cuda-nn v1.6, nvdiffrast v0.3.1, envlight, CLIP at their commits on 2026-10-12);
# and PyPI packages the requirements leave unpinned get the last versions before the repo's final commit (2023-10-05),
# so the 2023 code runs as written.
# SD 2.1-base (the prompt processor's text encoder) is no longer on the Hub under stabilityai; the sd2-community mirror
# is used and checked against LAION's original OpenCLIP ViT-H/14 weights (baselines/text_to_3d/check_sd21_mirror.py).
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
mkdir -p "$ORBIT_DATA/tmp"
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"
M="$ORBIT_REPO/third_party/MVDream-threestudio"
[ -d "$M" ] || git clone https://github.com/bytedance/MVDream-threestudio.git "$M"
git -C "$M" checkout efcd6c3   # 2023-10-05, final commit (repo archived 2025-09-23)
[ -d "$M/extern/MVDream" ] || git clone https://github.com/bytedance/MVDream.git "$M/extern/MVDream"
git -C "$M/extern/MVDream" checkout 618f84476c

conda env list | grep -q '^mvdream ' || conda create -y -n mvdream python=3.10
set +u; conda activate mvdream; set -u   # see the re-activation below
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include" TORCH_CUDA_ARCH_LIST="8.6;8.9" \
  TCNN_CUDA_ARCHITECTURES="86;89" LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
# conda's compiler/nvcc activation hooks read unset variables, so -u is off while the env (re)activates.
set +u; conda deactivate && conda activate mvdream; set -u
pip install --upgrade pip "setuptools<70" ninja pybind11   # pybind11: pysdf builds without isolation
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch"
# requirements.txt with its git lines pinned and its unpinned PyPI packages capped at their Oct 2023 releases.
sed -e 's|nvdiffrast.git$|nvdiffrast.git@v0.3.1|' -e 's|envlight.git$|envlight.git@05b5851e85|' \
    -e 's|openai/CLIP.git$|openai/CLIP.git@d05afc436d|' -e '/nerfacc\|tiny-cuda-nn/d' \
    -e 's|^diffusers$|diffusers==0.21.4|' -e 's|^transformers$|transformers==4.34.0|' \
    -e 's|^accelerate$|accelerate==0.23.0|' -e 's|^xformers$|xformers==0.0.22|' \
    -e 's|^huggingface_hub$|huggingface_hub==0.17.3|' -e 's|^bitsandbytes$|bitsandbytes==0.41.1|' \
    -e 's|^libigl$|libigl==2.5.0|' \
    "$M/requirements.txt" > "$ORBIT_DATA/tmp/mvdream_requirements_$$.txt"
pip install --no-build-isolation -r "$ORBIT_DATA/tmp/mvdream_requirements_$$.txt"
rm "$ORBIT_DATA/tmp/mvdream_requirements_$$.txt"
pip install "numpy<2"
pip install -e "$M/extern/MVDream"
python -c "import torch, tinycudann, nerfacc, nvdiffrast.torch, mvdream; print('MVDREAM', torch.__version__, torch.cuda.is_available())"

"$HF" download MVDream/MVDream sd-v2.1-base-4view.pt
"$HF" download sd2-community/stable-diffusion-2-1-base --include "text_encoder/model.safetensors" \
  --include "text_encoder/config.json" --include "tokenizer/*" --include "model_index.json" --include "scheduler/*"
"$HF" download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin   # reference for the mirror check
