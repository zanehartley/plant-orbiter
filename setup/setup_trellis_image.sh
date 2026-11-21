#!/usr/bin/env bash
# TRELLIS image-to-3D (single-photo baseline). It uses the same repo commit and conda env as the text-to-3D TRELLIS
# baseline, so run setup/setup_trellis_text.sh first; this script only adds the image model's weights.
# The paper's single-photo TRELLIS runs used the same package versions built against CUDA 11.8 wheels
# (env/freeze_trellis.txt); setup_trellis_text.sh builds them against CUDA 12.1.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
HF="$(conda info --base)/envs/orbit_cam/bin/hf"
"$HF" download JeffreyXiang/TRELLIS-image-large
