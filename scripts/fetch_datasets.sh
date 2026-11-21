#!/usr/bin/env bash
# Download the public datasets used by the reconstruction experiments into $ORBIT_DATASETS and prepare them.
#   scripts/fetch_datasets.sh [botanybot] [growsplat] [planest]     (default: all three)
# Check each dataset's licence before downloading (see README, "Licences"). PlantDreamer is obtained separately
# (see README, "Datasets").
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orbit_cam  # provides `hf` (huggingface_hub)
mkdir -p "$ORBIT_DATASETS"
WHAT=("$@"); [ ${#WHAT[@]} -gt 0 ] || WHAT=(botanybot growsplat planest)

for d in "${WHAT[@]}"; do
  case $d in
    botanybot)  # Adebola et al. 2025, CC BY 4.0
      hf download --repo-type dataset SimeonOA/Botany-Bot --local-dir "$ORBIT_DATASETS/BotanyBot" ;;
    growsplat)  # Adebola et al. 2025, CC BY 4.0
      hf download --repo-type dataset SimeonOA/GrowSplat-Dataset --local-dir "$ORBIT_DATASETS/GrowSplat" ;;
    planest)  # Mertoglu et al. 2024, CC BY-NC-ND 4.0: point clouds (Aperta record 286354), images + cameras (Dropbox)
      P=$ORBIT_DATASETS/PLANesT-3D; mkdir -p "$P/pointclouds_extracted"; cd "$P"
      for f in Pepper Rose; do
        [ -f $f.zip ] || curl -sSL --retry 5 -o $f.zip "https://aperta.ulakbim.gov.tr/record/286354/files/$f.zip?download=1"
        unzip -q -o $f.zip -d pointclouds_extracted
      done
      [ -f dropbox_full.zip ] || curl -sSL --retry 5 -C - -A "Mozilla/5.0" -o dropbox_full.zip \
        "https://www.dropbox.com/scl/fo/c4vqad7ak3a9pv1bkz6x1/AGxXY1yYhWzyjIGaqDwJes8?rlkey=qsukydmtgunpqe6zg8zpsjr2c&dl=1"
      cd "$ORBIT_REPO"
      python reconstruction/planest_extract_images.py ;;  # 73 GB archive -> quarter-resolution images (about 3 GB)
    *) echo "unknown dataset: $d (botanybot|growsplat|planest)"; exit 1 ;;
  esac
done
