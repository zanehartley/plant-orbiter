#!/usr/bin/env bash
# Datasets -> $ORBIT_DATASETS.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
mkdir -p "$ORBIT_DATASETS" && cd "$ORBIT_DATASETS"
# PlantDreamer (Hartley et al. 2025): 21 real plants + L-system meshes + their splats and results (12 GB zip).
curl -sSL -C - -o PlantDreamer_Data.zip https://plantimages.nottingham.ac.uk/datasets/PlantDreamer_Data.zip
unzip -q -n PlantDreamer_Data.zip
