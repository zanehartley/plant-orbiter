#!/usr/bin/env bash
# Model weights -> $ORBIT_WEIGHTS (README step 3). Already-downloaded files are skipped.
# Read the licence section of the README first (MiniMax-H3 and Qwen-Image both restrict use).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
C="$ORBIT_WEIGHTS/comfyui"
# Orbit video model (ComfyUI file names are hard-coded in pipeline/orbit.py build_workflow()).
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors vae/minimax_h3_video_vae_int8_convrot.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors --local-dir "$C"
hf download pablodawson/MiniMax-H3-360-Orbit-LoRA minimax_h3_flf2v_lora_v1.safetensors --local-dir "$C/loras"
# Background removal (cutout + per-frame masks).
hf download Comfy-Org/BiRefNet background_removal/birefnet.safetensors --local-dir "$C"
# Text-to-image (Qwen Research Licence: non-commercial research/evaluation only).
hf download Qwen/Qwen-Image-2.1 --local-dir "$ORBIT_WEIGHTS/hf/Qwen-Image-2.1"
