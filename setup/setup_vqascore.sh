#!/usr/bin/env bash
# Build the vqascore env: VQAScore with CLIP-FlanT5-XXL (evaluation/vqascore.py; Lin et al., ECCV 2024).
# t2v-metrics 1.2 is the release of the VQAScore paper (2024-08); 3.0 imports every model family it wraps (LLaVA-OV,
# InternVideo, ...) and does not load without them. 1.2's own dependency list does not install (image-reward fails to
# build), so the package goes in without it, plus what its import needs: transformers 4.36.1 as pinned, huggingface_hub
# < 0.26 for that transformers, diffusers 0.27.2 for that hub (as the evalq env), OpenAI CLIP from git (the PyPI "clip"
# is another package). torch-fidelity so metrics.py (view cropping) imports. Built 2026-10-22; torch 2.5.1+cu124 (run on
# RTX A6000s, sm_86). The model loads in fp32 on the CPU (~46 GB RAM), then bf16 on the GPU (~23 GB).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | grep -q '^vqascore ' || conda create -y -n vqascore python=3.10
set +u; conda activate vqascore; set -u   # conda activation hooks read unset variables
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps t2v-metrics==1.2 image-reward==1.5
pip install "transformers==4.36.1" "huggingface-hub==0.25.2" "diffusers==0.27.2" "datasets<3" "numpy==1.26.4" \
  sentencepiece ftfy tqdm open-clip-torch opencv-python-headless pandas scipy omegaconf iopath fairscale tiktoken \
  openai accelerate torch-fidelity==0.4.0 git+https://github.com/openai/CLIP.git@d05afc436d
python -c "import t2v_metrics; assert 'clip-flant5-xxl' in t2v_metrics.list_all_vqascore_models(); print('VQASCORE OK')"

# Weights into $HF_HUB_CACHE (compute nodes run offline): the model, its tokenizer and its vision tower.
python - <<EOF
from huggingface_hub import snapshot_download as get
get("zhiqiulin/clip-flant5-xxl", cache_dir="$HF_HUB_CACHE")  # 22.9 GB
get("google/flan-t5-xxl", allow_patterns=["*.json", "spiece.model"], cache_dir="$HF_HUB_CACHE")  # tokenizer only
get("openai/clip-vit-large-patch14-336", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"],
    cache_dir="$HF_HUB_CACHE")
EOF
