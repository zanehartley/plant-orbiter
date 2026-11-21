#!/usr/bin/env bash
# Score one experiment's assets (ours + baselines) exactly as in the paper. Run after reproduce_paper.sh and
# run_baselines.sh; every step skips work already done, so a rerun continues.
#   scripts/evaluate.sh synthetic_plants   CLIP, ImageReward, VBench imaging, KD vs real photos, MMD-F / COV
#   scripts/evaluate.sh fruit_counting     CLIP, ImageReward, VQAScore (3D assets and input images), count sheets
# Needs the envs orbit_cam, nerfstudio, mvdream, evalq (+ trellis2 and vqascore for fruit_counting) and, for
# synthetic_plants, the KD reference photos: setup/download_datasets.sh, then
# evaluation/prep_reference.py --out $ORBIT_DATASETS/plant_reference (cuts them out with masks/plantdreamer_hybrid/).
# Writes $ORBIT_OUTPUTS/<experiment>/eval/{manifest*.csv, renders/, results/} (+ results_input/, counting/).
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
EXPERIMENT=${1:?usage: $0 synthetic_plants|fruit_counting}
case $EXPERIMENT in
  synthetic_plants) FLAGS=() ;;
  fruit_counting)   FLAGS=(--fruit-counting) ;;
  *) echo "unknown experiment: $EXPERIMENT (synthetic_plants|fruit_counting)"; exit 1 ;;
esac
E="$ORBIT_OUTPUTS/$EXPERIMENT/eval"
R="$E/renders"
# conda's compiler activation hooks read unset variables, so -u is off while an env (de)activates
run() { local env=$1; shift; set +u; conda activate "$env"; set -u; echo "== [$env] $*"; "$@"; set +u; conda deactivate; set -u; }

# 1. Render every asset from the same 72 cameras (3 elevations x 24 azimuths, 512 px, white background).
run orbit_cam python evaluation/build_manifest.py "$EXPERIMENT" --out "$E/manifest.csv"
run nerfstudio python evaluation/render_splats.py "$E/manifest.csv" --out "$R"
CAMERAS=$(ls "$R"/ours/*/cameras.json | head -1)   # identical for every splat asset
run mvdream python evaluation/render_threestudio.py "$E/manifest_threestudio.csv" --out "$R" --cameras "$CAMERAS"
[ "$EXPERIMENT" = synthetic_plants ] || \
  run trellis2 python evaluation/render_meshes.py "$E/manifest_meshes.csv" --out "$R" --cameras "$CAMERAS"

# 2. Appearance (and, for synthetic plants, the Inception / DINOv2 features KD needs).
if [ "$EXPERIMENT" = synthetic_plants ]; then
  run orbit_cam python evaluation/metrics.py "$R" --out "$E/results" --reference "$ORBIT_DATASETS/plant_reference"
else
  run orbit_cam python evaluation/metrics.py "$R" --out "$E/results" "${FLAGS[@]}"
fi
run evalq python evaluation/quality_metrics.py "$R" --out "$E/results" "${FLAGS[@]}"

if [ "$EXPERIMENT" = synthetic_plants ]; then
  # 3. Realism (KD, bootstrapped over assets and real plants) and shape (MMD-F / COV vs the real point clouds).
  run orbit_cam python evaluation/kd_bootstrap.py "$E/results/features" --out "$E/results"
  export OMP_NUM_THREADS=1   # one thread per worker; the pool provides the parallelism
  run nerfstudio python evaluation/shape_metrics.py "$R" --real "$ORBIT_DATASETS/PlantDreamer_Data/plants" \
    --out "$E/results" --workers "$(nproc)"
  run orbit_cam python evaluation/summarise_metrics.py "$E/results"
else
  # 3. VQAScore, the same scores on the input images (one view each), and the sheets to count fruit from.
  run vqascore python evaluation/vqascore.py "$R" --out "$E/results" "${FLAGS[@]}"
  for img in "$ORBIT_OUTPUTS/$EXPERIMENT"/images/*.png; do
    mkdir -p "$E/inputs/input/$(basename "${img%.png}")/views"
    ln -sfn "$(realpath "$img")" "$E/inputs/input/$(basename "${img%.png}")/views/input.png"
  done
  run orbit_cam python evaluation/metrics.py "$E/inputs" --out "$E/results_input" "${FLAGS[@]}"
  run evalq python evaluation/quality_metrics.py "$E/inputs" --out "$E/results_input" "${FLAGS[@]}"
  run vqascore python evaluation/vqascore.py "$E/inputs" --out "$E/results_input" "${FLAGS[@]}"
  run orbit_cam python evaluation/count_sheets.py "$R" --out "$E/counting" --add-missing
  echo "Count the fruit by eye in $E/counting/counts.csv (rules in README.md), then:"
  echo "  python evaluation/count_metrics.py $E/counting/counts.csv --quality $E/results $E/results_input"
fi
