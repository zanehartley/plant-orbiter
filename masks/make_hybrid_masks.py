"""Hybrid masks for PlantDreamer's real photos: how masks/plantdreamer_hybrid/ was made.

PlantDreamer's masks are loose around leaves (a grey band of background survives) but too tight on thin stems
(mint). Hybrid mask:
  fresh = rembg isnet-general-use cutout of the raw photo (tight edges, but keeps the cloth/turntable)
  hybrid = (fresh & dataset) | (fresh & dilate(dataset, 25 px) & upper 60% of the plant's height)
so the fresh cutout sets every edge, the dataset mask keeps the cloth out, and thin stems above the pot come back.
Every photo with a dataset mask gets one; then qa_hybrid_masks.py rejects the bad ones. CPU only; needs rembg,
onnxruntime, opencv (the "masks" env, setup/setup_reconstruction_extras.sh).
masks/plantdreamer_hybrid/ already holds the paper's masks, so this is only needed to remake them:
    python masks/make_hybrid_masks.py bean_1 bean_3 ... mint_6 \
        --plantdreamer $ORBIT_DATASETS/PlantDreamer_Data/plants --out NEW_MASKS_DIR
Writes <out>/<plant>/<view>.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU is fine and keeps the GPU free

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove



def hybrid(raw, dm, fm, dilate=25, upper=0.6):
    dd = cv2.dilate(dm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)) > 0
    ys = np.where(dm.any(1))[0]
    top, bot = ys.min(), ys.max()
    up = (np.arange(dm.shape[0]) < top + upper * (bot - top))[:, None]
    return (dm & fm) | (dd & fm & up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants", nargs="+")
    ap.add_argument("--plantdreamer", type=Path, required=True, help="PlantDreamer_Data/plants")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sess = new_session("isnet-general-use")
    for plant in a.plants:
        (a.out / plant).mkdir(parents=True, exist_ok=True)
        vs = sorted(p.name[:-8] for p in (a.plantdreamer / plant / "images" / "mask").glob("*.png.png"))
        for v in vs:
            dst = a.out / plant / f"{v}.png"
            if dst.exists():
                continue
            src = a.plantdreamer / plant / "images" / "undistorted" / f"{v}.png"
            if not src.exists():  # a few mask files have no photo (mint_4)
                print(f"{plant} {v}: no photo, skipped", flush=True)
                continue
            raw = Image.open(src).convert("RGB")
            dm = np.asarray(Image.open(a.plantdreamer / plant / "images" / "mask" / f"{v}.png.png").convert("L")) > 0
            if dm.shape != raw.size[::-1]:  # misnumbered masks (mint_4): skipped everywhere else too
                print(f"{plant} {v}: mask {dm.shape} != image {raw.size[::-1]}, skipped", flush=True)
                continue
            fm = np.asarray(remove(raw, session=sess, only_mask=True)) > 127
            m = hybrid(raw, dm, fm)
            Image.fromarray((m * 255).astype(np.uint8)).save(dst)
            print(f"{plant} {v}: dataset {dm.mean():.3f} -> hybrid {m.mean():.3f} of the image", flush=True)


if __name__ == "__main__":
    main()
