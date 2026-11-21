#!/usr/bin/env bash
# Single-photo reconstruction of dataset plants with the paper's final settings, followed by placement, 2D scores at
# the real held-out views and (with a reference cloud) 3D geometry.
#   scripts/reconstruct_plants.sh <pd|bb|pl> <seed> <plant> [<plant> ...]
#   e.g. scripts/reconstruct_plants.sh pd 42 bean_3 kale_1
# A ComfyUI server must be running for the orbit step (see README, "Run"). Runs are written to
# $ORBIT_OUTPUTS/<dataset>/<plant>_<input view>_s<seed>/.
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
DS=${1:?usage: $0 <pd|bb|pl> <seed> <plant> [...]}; SEED=${2:?seed}; shift 2
case $DS in
  pd) export PLANT_DATA=$ORBIT_DATASETS/PlantDreamer_Data/plants; EXTRA=(--tight-masks "$ORBIT_REPO/masks/plantdreamer_hybrid") ;;
  bb|pl) export PLANT_DATA=$ORBIT_DATASETS/converted/plants; EXTRA=() ;;
  *) echo "unknown dataset: $DS (pd|bb|pl)"; exit 1 ;;
esac
# Final recipe: elevation search 0-40 deg, scale regularisation with max Gaussian ratio 3, antialiased rasterisation
# for training and for the placement renders.
export EV_RASTER=antialiased KP_ELEV=0:40:5
export KP_EXTRA="--pipeline.model.use-scale-regularization True --pipeline.model.max-gauss-ratio 3"
NS="conda run --no-capture-output -n nerfstudio python"
RUNS=$ORBIT_OUTPUTS/$DS

# Orbit video, known cameras, splat, input-view and oracle placement, 2D scores.
$NS reconstruction/batch_single_photo.py "$@" --seed "$SEED" --no-mesh "${EXTRA[@]}" \
  --inputs "$ORBIT_INPUTS/$DS" --outputs "$RUNS" --summary "summary_s$SEED.json"

# DISTS at the same held-out views, placements and colour correction as the other 2D scores.
for plant in "$@"; do
  for run in "$RUNS"/${plant}_*_s"$SEED"; do
    $NS reconstruction/score_dists.py "$plant" "$run/3dgs" "$run/3dgs/export/splat.ply"
  done
done

# 3D geometry: 2DGS mesh from the same orbit frames and cameras, placed by the oracle fit, against the reference cloud.
for plant in "$@"; do
  for run in "$RUNS"/${plant}_*_s"$SEED"; do
    g=$run/3dgs
    [ -f "$g/2dgs_v3/mesh.ply" ] || $NS reconstruction/train_2dgs.py "$run" --out 2dgs_v3 --min-component 0.005 \
      --normal-weight 0.2 --dist-weight 0.02 --smooth 3 > "$g/2dgs_v3.log" 2>&1
    ls "$PLANT_DATA/$plant"/data/*_real_pointcloud.ply > /dev/null 2>&1 || { echo "$run: no reference cloud, 3D skipped"; continue; }
    $NS reconstruction/colmap2real.py "$g" oracle > /dev/null
    G=()
    if [ "$DS" = pl ]; then  # PLANesT: plant only (pot cropped), metric tolerances from the registration scale
      G=($(python3 -c "import json; s=json.load(open('$PLANT_DATA/$plant/data/gt_registration.json'))['scale_cm_to_world']; print('--crop base --alpha', 0.4*s, '--voxel', 0.08*s)"))
    fi
    $NS reconstruction/gao_eval.py "$plant" "$g/2dgs_v3/mesh.ply" --sim3 "$g/colmap2real_oracle.json" \
      -o "$g/gao_oracle_v3.json" "${G[@]}" > /dev/null
  done
done
