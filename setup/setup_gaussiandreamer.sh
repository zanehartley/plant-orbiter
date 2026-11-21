#!/usr/bin/env bash
# Build the gaussiandreamer env (GaussianDreamer, hustvl/GaussianDreamer): score-distillation text-to-3DGS.
# Follows the README (torch 2.0.1, requirements.txt, the repo's own diff-gaussian-rasterization and simple-knn,
# Shap-E, Cap3D's fine-tuned Shap-E), with these changes: torch 2.0.1+cu118 with the CUDA 11.8 toolkit from conda
# (the README's cu117 has no conda toolkit with gcc >= 11 support; system CUDA is 13.x); git dependencies pinned
# (tiny-cuda-nn v1.6, nvdiffrast v0.3.1, envlight, CLIP, Shap-E at their commits on 2026-10-12); unpinned PyPI packages
# capped at releases before the repo's code (Oct 2023), as for mvdream; pybind11 added (built without isolation).
# SD 2.1-base (guidance + prompt processor) is no longer hosted by stabilityai: the sd2-community copy is used; its
# UNet, VAE, text encoder and original checkpoint are byte-identical (Hub LFS sha256) to 5-6 independent copies, and
# its text encoder matches LAION's OpenCLIP weights (check_sd21_mirror.py).
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
mkdir -p "$ORBIT_DATA/tmp"
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"
G="$ORBIT_REPO/third_party/GaussianDreamer"
[ -d "$G" ] || git clone https://github.com/hustvl/GaussianDreamer.git "$G"
git -C "$G" checkout 76a38ff   # 2025-01-13, latest commit when installed (2026-10-12)
[ -d "$G/shap-e" ] || git clone https://github.com/openai/shap-e.git "$G/shap-e"
git -C "$G/shap-e" checkout -q 50131012ee
# The repo vendors diff-gaussian-rasterization without its glm submodule; use upstream's pinned commit.
GLM="$G/gaussiansplatting/submodules/diff-gaussian-rasterization/third_party/glm"
[ -d "$GLM" ] || git clone -q https://github.com/g-truc/glm.git "$GLM"
git -C "$GLM" checkout -q 5c46b9c07008ae65cb81ab79cd677ecc1934b903

conda env list | grep -q '^gaussiandreamer ' || conda create -y -n gaussiandreamer python=3.10
set +u; conda activate gaussiandreamer; set -u   # conda's compiler/nvcc activation hooks read unset variables
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CPATH="$CONDA_PREFIX/include" TORCH_CUDA_ARCH_LIST="8.6;8.9" \
  TCNN_CUDA_ARCHITECTURES="86;89" LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
set +u; conda deactivate && conda activate gaussiandreamer; set -u
pip install --upgrade pip "setuptools<70" ninja pybind11
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch"
# requirements.txt with git lines pinned, nerfacc/tcnn installed above, torch left as installed, and the unpinned
# PyPI packages capped at their Oct 2023 releases (xformers 0.0.20 is the README's torch-2.0.1 build).
sed -e 's|nvdiffrast.git$|nvdiffrast.git@v0.3.1|' -e 's|envlight.git$|envlight.git@05b5851e85|' \
    -e 's|openai/CLIP.git$|openai/CLIP.git@d05afc436d|' -e '/nerfacc\|tiny-cuda-nn\|^torch==/d' \
    -e 's|^diffusers$|diffusers==0.21.4|' -e 's|^accelerate$|accelerate==0.23.0|' \
    -e 's|^huggingface_hub$|huggingface_hub==0.17.3|' -e 's|^libigl$|libigl==2.5.0|' \
    "$G/requirements.txt" > "$ORBIT_DATA/tmp/gaussiandreamer_requirements_$$.txt"
pip install --no-build-isolation -r "$ORBIT_DATA/tmp/gaussiandreamer_requirements_$$.txt"
rm "$ORBIT_DATA/tmp/gaussiandreamer_requirements_$$.txt"
# Imported by the code but missing from requirements.txt (plyfile, open3d: GaussianDreamer system; tomesd: SD
# guidance; pysdf: threestudio's geometry registry), at their last releases before Oct 2023.
pip install --no-build-isolation open3d==0.17.0 plyfile==1.0.1 tomesd==0.1.3 pysdf==0.1.9 \
  dash==2.13.0 "pydantic<2"  # open3d imports dash; newer dash needs pydantic 2, fastapi needs < 2
pip install "numpy<2"
pip install --no-build-isolation "$G/gaussiansplatting/submodules/diff-gaussian-rasterization" \
  "$G/gaussiansplatting/submodules/simple-knn"
pip install -e "$G/shap-e"
python -c "import torch, tinycudann, nerfacc, diff_gaussian_rasterization, simple_knn, shap_e; \
print('GAUSSIANDREAMER', torch.__version__, torch.cuda.is_available())"

# Weights: Cap3D's fine-tuned Shap-E (the README's 330k-data model), Shap-E's own transmitter/text300M (downloaded by
# shap_e into ./shap_e_model_cache of the working dir, so prefetched from the weights dir), and the SD 2.1-base copy.
W="$ORBIT_WEIGHTS/gaussiandreamer"
mkdir -p "$W"
"$HF" download tiange/Cap3D --repo-type dataset misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth \
  --local-dir "$W"
(cd "$W" && python -c "from shap_e.models.download import load_model; load_model('transmitter', device='cpu'); load_model('text300M', device='cpu')")
"$HF" download sd2-community/stable-diffusion-2-1-base --include "unet/*" --include "vae/*" --include "text_encoder/*" \
  --include "tokenizer/*" --include "scheduler/*" --include "feature_extractor/*" --include "model_index.json"
