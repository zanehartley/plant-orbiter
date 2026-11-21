#!/usr/bin/env bash
# Run the paper's baselines for one experiment (each in its own conda env, built by setup/setup_<method>.sh).
#   scripts/run_baselines.sh synthetic_plants [METHOD ...]   default: trellis_text diffsplat gaussiandreamer mvdream
#   scripts/run_baselines.sh fruit_counting   [METHOD ...]   default: the same + trellis2
# trellis2 lifts our own input images, so for fruit_counting run scripts/reproduce_paper.sh fruit_counting first
# (its RGBA inputs are made here with prep_inputs.py, in the orbit_cam env).
# Writes $ORBIT_OUTPUTS/<experiment>/baselines/<method>/<asset>/. Finished assets are skipped, so a rerun continues.
# Time per asset on an RTX A6000: TRELLIS-text 14 s, DiffSplat 27 s, GaussianDreamer 13 min, MVDream 66 min.
set -euo pipefail
cd "$(dirname "$0")/.."
source config/paths.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
EXPERIMENT=${1:?usage: $0 synthetic_plants|fruit_counting [METHOD ...]}; shift
B="$ORBIT_OUTPUTS/$EXPERIMENT/baselines"
case $EXPERIMENT in
  synthetic_plants) DEFAULT="trellis_text diffsplat gaussiandreamer mvdream" ;;
  fruit_counting)   DEFAULT="trellis_text diffsplat gaussiandreamer mvdream trellis2" ;;
  *) echo "unknown experiment: $EXPERIMENT (synthetic_plants|fruit_counting)"; exit 1 ;;
esac
[ $# -gt 0 ] || set -- $DEFAULT

# conda's compiler activation hooks read unset variables, so -u is off while an env (de)activates
activate() { set +u; conda activate "$1"; set -u; }
deactivate() { set +u; conda deactivate; set -u; }
for method in "$@"; do
  echo "== $method ($EXPERIMENT)"
  case $method in
    trellis_text)    activate trellis;         python baselines/text_to_3d/run_trellis_text.py --experiment "$EXPERIMENT" --out "$B/$method" ;;
    diffsplat)       activate diffsplat;       python baselines/text_to_3d/run_diffsplat.py --experiment "$EXPERIMENT" --out "$B/$method" ;;
    gaussiandreamer) activate gaussiandreamer; python baselines/text_to_3d/run_gaussiandreamer.py --experiment "$EXPERIMENT" --out "$B/$method" ;;
    mvdream)         activate mvdream;         python baselines/text_to_3d/run_mvdream.py --experiment "$EXPERIMENT" --out "$B/$method" ;;
    trellis2)
      [ "$EXPERIMENT" = fruit_counting ] || { echo "trellis2 is the fruit-counting same-input control only"; exit 1; }
      activate orbit_cam; python baselines/text_to_3d/prep_inputs.py "$ORBIT_OUTPUTS/$EXPERIMENT/images" --out "$B/inputs"
      deactivate
      activate trellis2;  python baselines/text_to_3d/run_trellis2.py "$B/inputs" --out "$B/trellis2" ;;
    *) echo "unknown method: $method"; exit 1 ;;
  esac
  deactivate
done
