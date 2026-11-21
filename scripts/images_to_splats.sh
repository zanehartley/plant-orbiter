#!/usr/bin/env bash
# Input images -> 360° orbit videos (MiniMax-H3 + Orbit LoRA, own ComfyUI server) -> known-pose 3D Gaussian splats.
#   scripts/images_to_splats.sh OUT_DIR IMAGE [IMAGE ...]        (from the repo root; works inside a Slurm job too)
# Each image gets OUT_DIR/<image stem>_s<ORBIT_SEED>/ with orbit.mp4, cutout.png, frames/, masks/, rgba/ and
# 3dgs/ (known_poses.json, splat, eval.json, export/splat.ply). Finished orbits and splats are skipped, so an
# interrupted run can simply be restarted.
#
# The splat settings are the paper's recipe for text-to-3D; override any by exporting it (an empty value turns it off):
#   KP_ELEV=0:40:5          orbit elevation searched 0-40° in 5° steps (by visual-hull recall)
#   KP_SEARCH=300:6000:100  focal search range (px at 768); generated images are long-lens, near orthographic
#   KP_FALLBACK=3000        use 3000 px when the searched focal fits within 0.01 recall of it (flat recall curve)
#   KP_EXTRA=...            splatfacto: scale regularisation, max Gaussian ratio 3 (antialiased is the default)
# Other env: ORBIT_SEED (default 42), COMFY_PORT (default 8188, or derived from SLURM_JOB_ID).
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
export KP_ELEV=${KP_ELEV-0:40:5}
export KP_SEARCH=${KP_SEARCH-300:6000:100}
export KP_FALLBACK=${KP_FALLBACK-3000}
export KP_EXTRA=${KP_EXTRA-"--pipeline.model.use-scale-regularization True --pipeline.model.max-gauss-ratio 3"}
echo "KP_ELEV=$KP_ELEV KP_SEARCH=$KP_SEARCH KP_FALLBACK=$KP_FALLBACK KP_EXTRA=$KP_EXTRA"
[ $# -ge 2 ] || { echo "usage: $0 OUT_DIR IMAGE [IMAGE ...]"; exit 1; }
OUT=$(realpath -m "$1"); shift
SEED=${ORBIT_SEED:-42}
mkdir -p "$OUT"
images=(); for img in "$@"; do images+=("$(realpath "$img")"); done

todo=()
for img in "${images[@]}"; do [ -f "$OUT/$(basename "${img%.*}")_s$SEED/orbit.mp4" ] || todo+=("$img"); done
if [ ${#todo[@]} -gt 0 ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then           # one ComfyUI per job, so jobs sharing a node don't collide
    PORT=${COMFY_PORT:-$((8200 + SLURM_JOB_ID % 500))}
  else
    PORT=${COMFY_PORT:-8188}
  fi
  export COMFY="127.0.0.1:$PORT"
  export COMFY_OUTPUT="$ORBIT_DATA/tmp/comfy_${PORT}_$$"
  mkdir -p "$COMFY_OUTPUT"
  conda activate orbit_cam
  python "$COMFYUI_DIR/main.py" --listen 127.0.0.1 --port "$PORT" --output-directory "$COMFY_OUTPUT" \
    --extra-model-paths-config config/comfyui_extra_model_paths.yaml > "$COMFY_OUTPUT/comfyui.log" 2>&1 &
  COMFY_PID=$!
  trap 'kill $COMFY_PID 2>/dev/null || true' EXIT
  echo "== starting ComfyUI on $COMFY (log: $COMFY_OUTPUT/comfyui.log)"
  until curl -sf "http://$COMFY/system_stats" > /dev/null; do
    kill -0 $COMFY_PID 2>/dev/null || { echo "ComfyUI died:"; tail -20 "$COMFY_OUTPUT/comfyui.log"; exit 1; }
    sleep 5
  done
  python pipeline/orbit.py "${todo[@]}" --remove-bg --seed "$SEED" --out "$OUT"
  kill $COMFY_PID; wait $COMFY_PID 2>/dev/null || true   # frees ComfyUI's VRAM for splat training
  rm -rf "$COMFY_OUTPUT"
  conda deactivate
fi

for img in "${images[@]}"; do
  run="$OUT/$(basename "${img%.*}")_s$SEED"
  [ -f "$run/3dgs/eval.json" ] && continue
  rm -rf "$run/3dgs"                             # leftovers of an interrupted run
  echo "== $run"
  pipeline/make_splat_kp.sh "$run"
done
