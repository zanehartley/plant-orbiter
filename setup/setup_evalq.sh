#!/usr/bin/env bash
# Build the evalq env: image-quality metrics on rendered views (evaluation/quality_metrics.py), as reported by
# DiffSplat (ImageReward) and VBench (Imaging Quality = MUSIQ-SPAQ, Aesthetic Quality = LAION aesthetic head on
# CLIP ViT-L/14). Separate from orbit_cam because ImageReward's BLIP code needs transformers 4.x. torch 2.7.0+cu128 runs
# on sm_86, sm_89 and sm_120 GPUs. torch-fidelity is included so metrics.py (view cropping) imports.
# Run on a GPU node (the CUDA extensions are compiled for the GPU it finds).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"

conda env list | grep -q '^evalq ' || conda create -y -n evalq python=3.10
set +u; conda activate evalq; set -u   # conda activation hooks read unset variables
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# image-reward 1.5 pins timm 0.6.13 and imports diffusers (0.27.2: newer need transformers > 4.37; it needs
# huggingface_hub < 0.26, which still has cached_download);
# pyiqa 0.1.13 asks for timm >= 0.8 and transformers 4.37.2. Only pyiqa's MUSIQ is used, so pyiqa goes in without its
# dependency list plus the small packages its __init__ imports; the import check below loads MUSIQ to prove that works.
pip install image-reward==1.5 "transformers==4.37.2" diffusers==0.27.2 huggingface_hub==0.25.2 torch-fidelity==0.4.0 scipy "numpy<2" \
  git+https://github.com/openai/CLIP.git@d05afc436d
pip install --no-deps pyiqa==0.1.13
pip install addict einops facexlib icecream lmdb yapf opencv-python-headless scikit-image pandas
python -c "import torch, ImageReward, clip, torch_fidelity; from pyiqa.archs.musiq_arch import MUSIQ; \
print('EVALQ', torch.__version__)"

# Weights, all under $ORBIT_WEIGHTS/vbench (VBench's own sources) or the HF cache (ImageReward).
W="$ORBIT_WEIGHTS/vbench"
mkdir -p "$W"
[ -f "$W/musiq_spaq_ckpt-358bb6af.pth" ] || wget -q -P "$W" \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
[ -f "$W/sa_0_4_vit_l_14_linear.pth" ] || wget -q -P "$W" \
  https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth
python -c "import clip; clip.load('ViT-L/14', device='cpu', download_root='$W')"
python -c "from pyiqa.archs.musiq_arch import MUSIQ; MUSIQ(pretrained_model_path='$W/musiq_spaq_ckpt-358bb6af.pth')"
"$HF" download THUDM/ImageReward
python -c "import ImageReward as RM; RM.load('ImageReward-v1.0', device='cpu', download_root='$ORBIT_WEIGHTS/imagereward')"
