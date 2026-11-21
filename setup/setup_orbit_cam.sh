#!/usr/bin/env bash
# Build the orbit_cam env (Qwen-Image text-to-image + ComfyUI for MiniMax-H3 orbit generation). README step 2a.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n orbit_cam python=3.12 git
conda activate orbit_cam
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r "$ORBIT_REPO/env/freeze_orbit_cam.txt"
[ -d "$COMFYUI_DIR" ] || git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
git -C "$COMFYUI_DIR" checkout 65787d6   # ComfyUI 0.38.0, the version the paper was run with
pip install -r "$COMFYUI_DIR/requirements.txt"
python -c "import torch, diffusers; print('TORCH', torch.__version__, 'DIFFUSERS', diffusers.__version__)"
