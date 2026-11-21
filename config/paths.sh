# Single source of truth for where things live. Source it from the repo root before running anything:
#     source config/paths.sh
# Code stays in this folder; weights (~67 GB) and outputs go under ORBIT_DATA.
# Override any variable by exporting it before sourcing, e.g. export ORBIT_DATA=/big/disk/plant_orbiter.

export ORBIT_REPO="${ORBIT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ORBIT_DATA="${ORBIT_DATA:-$ORBIT_REPO/data}"

export ORBIT_WEIGHTS="${ORBIT_WEIGHTS:-$ORBIT_DATA/weights}"
export ORBIT_OUTPUTS="${ORBIT_OUTPUTS:-$ORBIT_DATA/outputs}"
export ORBIT_DATASETS="${ORBIT_DATASETS:-$ORBIT_DATA/datasets}"  # datasets: downloaded, converted, evaluation references
export ORBIT_INPUTS="${ORBIT_INPUTS:-$ORBIT_DATA/inputs}"  # reconstruction: prepared input photos

export COMFYUI_DIR="${COMFYUI_DIR:-$ORBIT_REPO/third_party/ComfyUI}"
export COMFY_OUTPUT="${COMFY_OUTPUT:-$COMFYUI_DIR/output}"
export NERFSTUDIO_DIR="${NERFSTUDIO_DIR:-$ORBIT_REPO/third_party/nerfstudio}"

# Model downloads by libraries (Hugging Face hub, torch) also go to the data volume.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ORBIT_WEIGHTS/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$ORBIT_WEIGHTS/torch}"
