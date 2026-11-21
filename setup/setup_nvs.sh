#!/usr/bin/env bash
# Build the nvs env shared by the SEVA (Stable Virtual Camera) and SV3D video-model baselines: Python 3.10,
# torch 2.5.1 + cu124, the seva package, and the dependencies of Stability AI's generative-models (sgm) for SV3D.
# Package versions used for the paper: env/freeze_nvs.txt. SV3D needs Hugging Face access to the gated
# stabilityai/sv3d (accept its terms first).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
SEVA="$ORBIT_REPO/third_party/stable-virtual-camera"
SGM="$ORBIT_REPO/third_party/generative-models"
[ -d "$SEVA" ] || git clone https://github.com/Stability-AI/stable-virtual-camera.git "$SEVA"
git -C "$SEVA" checkout fe19948   # 2025-06-05
[ -d "$SGM" ] || git clone https://github.com/Stability-AI/generative-models.git "$SGM"
git -C "$SGM" checkout e8cd657    # 2025-12-16

conda env list | grep -q '^nvs ' || conda create -y -n nvs python=3.10
set +u; conda activate nvs; set -u
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e "$SEVA"
pip install omegaconf pytorch-lightning==2.4.0 open-clip-torch fire imageio imageio-ffmpeg einops safetensors \
  "transformers<4.46" opencv-python-headless plyfile scipy
python -c "import torch, seva, open_clip, pytorch_lightning; print('nvs', torch.__version__, torch.cuda.is_available())"
