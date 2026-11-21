#!/usr/bin/env bash
# Orbit output -> KNOWN orbit poses (no COLMAP) -> splatfacto -> eval + export.
# Usage: pipeline/make_splat_kp.sh $ORBIT_OUTPUTS/<name>_s42 [experiment-name]
# The run dir needs frames/, masks/, rgba/ from orbit.py --remove-bg (symlinks to an existing run are fine).
# KP_FOCAL=<px at 768> fixes the focal; otherwise known_poses.py picks it by visual-hull consistency.
# KP_EXTRA="<ns-train options>" passes extra splatfacto options (the paper: scale regularisation, max Gaussian ratio 3).
# Antialiased rasterisation is the default (in our ablation: +0.78 dB PSNR on 8/8 plants, IoU +0.013);
# KP_CLASSIC=1 selects the classic mode. The mode used is written to 3dgs/splat_settings.json.
# KP_SEARCH=lo:hi:step overrides the focal search range (default 300:1800:25 px at 768): cropped
# inputs (GrowSplat) zoom the photo by a known factor, so the range is scaled by it.
# Text-to-image inputs are long-lens: text-to-3D uses KP_SEARCH=300:6000:100.
# KP_FALLBACK=<px> uses that focal instead when the searched one fits barely better (flat recall curve; known_poses.py
# --flat-fallback; text-to-3D uses KP_FALLBACK=3000).
# KP_ELEV=lo:hi:step also searches the orbit elevation (photos from above); off by default (the paper uses 0:40:5).
# KP_REFINE=1 lets splatfacto refine every camera (SO3xR3) during training, starting from the known poses.
# Further experiments (all off by default, not used in the paper): KP_ITERS=<n> training steps (default 30000),
# KP_METHOD=splatfacto-big, KP_DROP=<file of frame names> leaves those frames out, KP_BA=1 bundle-adjusts the known
# poses + focal with COLMAP (point_triangulator + bundle_adjuster; SfM points replace the hull seed),
# KP_UPSCALE=<dir of upscaled rgba frames> with KP_UPSCALE_F=<factor> trains on them (intrinsics scaled to match).
set -euo pipefail

SRC=$(realpath -s "$1")
EXP=${2:-$(basename "$SRC")}
ROOT=$(dirname "$(realpath "$0")")
W=$SRC/3dgs
ENV="conda run --no-capture-output -n nerfstudio"

N=$(ls "$SRC/frames" | wc -l)
LAST=$((N - 2))  # the final frame repeats frame 0 (closed loop), so drop it
# KP_CAMS=<json> (SEVA / SV3D baselines): cameras given by the video model, every frame used
[ -n "${KP_CAMS:-}" ] && LAST=$((N - 1))
mkdir -p "$W/images" "$W/images_rgba"
for i in $(seq -f "%05g" 0 $LAST); do
  cp "$SRC/frames/$i.png" "$W/images/"
  cp "$SRC/rgba/$i.png" "$W/images_rgba/"
done

echo "== known poses ($((LAST + 1)) frames)"
# KP_SPARSE=1: 3dgs/sparse/0 was written by another pose source; keep it
[ -n "${KP_SPARSE:-}" ] || $ENV python "$ROOT/known_poses.py" "$SRC" ${KP_FOCAL:+--focal "$KP_FOCAL"} ${KP_CAMS:+--cams "$KP_CAMS"} ${KP_ELEV:+--elev-search "$KP_ELEV"} ${KP_SEARCH:+--search "$KP_SEARCH"} ${KP_FALLBACK:+--flat-fallback "$KP_FALLBACK"} 2>&1 | grep -v Warning  # size read from the frames
cd "$W"
if [ -n "${KP_DROP:-}" ]; then
  python3 - "$KP_DROP" <<'PY'
import sys
from pathlib import Path
drop = set(Path(sys.argv[1]).read_text().split())
p = Path("sparse/0/images.txt")
lines = [l for l in p.read_text().splitlines() if l and not l.startswith("#")]
keep = [l for l in lines if l.split()[-1] not in drop]
p.write_text("\n".join(x for l in keep for x in (l, "")) + "\n")
for n in drop:
    for d in ("images", "images_rgba"):
        Path(d, n).unlink(missing_ok=True)
print(f"dropped {len(lines) - len(keep)} frames")
PY
fi
if [ -n "${KP_BA:-}" ]; then
  echo "== bundle adjustment from known poses"
  rm -f ba.db; mkdir -p sparse/tri sparse/ba
  $ENV colmap feature_extractor --database_path ba.db --image_path images --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_PINHOLE > ba.log 2>&1
  $ENV colmap exhaustive_matcher --database_path ba.db >> ba.log 2>&1
  $ENV colmap point_triangulator --database_path ba.db --image_path images --input_path sparse/0 --output_path sparse/tri >> ba.log 2>&1
  $ENV colmap bundle_adjuster --input_path sparse/tri --output_path sparse/ba --BundleAdjustment.refine_focal_length 1 \
    --BundleAdjustment.refine_principal_point 0 --BundleAdjustment.refine_extra_params 0 >> ba.log 2>&1
  $ENV colmap model_converter --input_path sparse/ba --output_path sparse/0 --output_type TXT >> ba.log 2>&1
  head -4 sparse/0/cameras.txt | tail -1
fi
if [ -n "${KP_UPSCALE:-}" ]; then
  cp "$KP_UPSCALE"/*.png images_rgba/
  python3 - "${KP_UPSCALE_F:-2}" <<'PY'
import sys
from pathlib import Path
f = float(sys.argv[1]); p = Path("sparse/0/cameras.txt")
out = []
for l in p.read_text().splitlines():
    if l and not l.startswith("#"):
        t = l.split(); t[2:4] = [str(int(int(x) * f)) for x in t[2:4]]; t[4:] = [str(float(x) * f) for x in t[4:]]; l = " ".join(t)
    out.append(l)
p.write_text("\n".join(out) + "\n")
PY
fi
$ENV colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type BIN > /dev/null 2>&1

echo "== splatfacto"
REFINE=(); [ -n "${KP_REFINE:-}" ] && REFINE=(--pipeline.model.camera-optimizer.mode SO3xR3)
ITERS=(); [ -n "${KP_ITERS:-}" ] && ITERS=(--max-num-iterations "$KP_ITERS")
RAST=antialiased; [ -n "${KP_CLASSIC:-}" ] && RAST=classic
EXTRA=(--pipeline.model.rasterize-mode $RAST ${KP_EXTRA:-})  # extra splatfacto options
METHOD=${KP_METHOD:-splatfacto}
$ENV ns-train "$METHOD" --output-dir runs --experiment-name "$EXP" --timestamp v1 \
  --vis tensorboard --viewer.quit-on-train-completion True "${REFINE[@]}" "${ITERS[@]}" "${EXTRA[@]}" \
  colmap --data . --colmap-path sparse/0 --images-path images_rgba --downscale-factor 1 > train.log 2>&1
CFG=$(ls runs/$EXP/*/v1/config.yml | head -1)  # splatfacto-big saves under splatfacto/
echo "{\"rasterize_mode\": \"$RAST\", \"extra\": \"${KP_EXTRA:-}\"}" > splat_settings.json

echo "== eval + export"
$ENV ns-eval --load-config "$CFG" --output-path eval.json --render-output-path eval_renders > eval.log 2>&1
python3 -c "import json; r=json.load(open('eval.json'))['results']; print({k: round(r[k], 4) for k in ('psnr', 'ssim', 'lpips')})"
$ENV ns-export gaussian-splat --load-config "$CFG" --output-dir export > export.log 2>&1
