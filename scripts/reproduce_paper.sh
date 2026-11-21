#!/usr/bin/env bash
# Regenerate our method's assets for the paper's two text-to-3D experiments (text -> image -> orbit -> splat).
#   scripts/reproduce_paper.sh synthetic_plants   Sec. "Fully synthetic plants": bean, kale, mint x seeds 1-11, 42  (36 assets)
#   scripts/reproduce_paper.sh fruit_counting     Sec. "Synthetic fruit counting": tomato, lemon x N = 1-7 fruit
#                                                 (written as words) x seeds 42, 1, 2                          (42 assets)
# Writes $ORBIT_OUTPUTS/<experiment>/images/ (Qwen-Image inputs + prompts.json) and $ORBIT_OUTPUTS/<experiment>/splats/.
# The text-to-image seed is the asset seed; the orbit seed is always 42. About 25 min per asset on an RTX A6000.
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
EXPERIMENT=${1:?usage: $0 synthetic_plants|fruit_counting}
OUT="$ORBIT_OUTPUTS/$EXPERIMENT"
conda activate orbit_cam
case $EXPERIMENT in
  synthetic_plants) python pipeline/t2i_qwen.py --species bean kale mint --seeds 42 1 2 3 4 5 6 7 8 9 10 11 \
                      --out "$OUT/images" ;;
  fruit_counting)   python pipeline/t2i_qwen.py --species tomato lemon --counts 1 2 3 4 5 6 7 --count-words \
                      --seeds 42 1 2 --out "$OUT/images" ;;
  *) echo "unknown experiment: $EXPERIMENT (synthetic_plants|fruit_counting)"; exit 1 ;;
esac
conda deactivate
scripts/images_to_splats.sh "$OUT/splats" "$OUT"/images/*.png
