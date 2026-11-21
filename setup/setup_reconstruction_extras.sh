#!/usr/bin/env bash
# Extra packages for the single-photo reconstruction experiments. Run after setup_nerfstudio.sh.
#   - piq (DISTS metric) in the nerfstudio env
#   - a small "masks" env with rembg, only needed to remake the PlantDreamer hybrid masks (masks/make_hybrid_masks.py)
# Versions are the ones used for the paper.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate nerfstudio
pip install piq==0.8.0
conda deactivate

conda env list | grep -q "^masks " || conda create -y -n masks python=3.10
conda activate masks
pip install rembg==2.0.65 onnxruntime==1.16.3 opencv-python numpy pillow
python -c "import rembg, cv2; print('masks env OK')"
