"""RGBA input photos for the baselines: the same cutout our method gets, with the mask as alpha.

TRELLIS and Hunyuan3D re-segment an opaque input with their own background removers (u2net / rembg), which on our
white-background photos cut away 40-75% of several plants (bean_5, bean_7, kale_1, kale_3, mint_5, mint_6) or kept a
grey haze (kale_3). With an alpha channel both use it as given, so every method sees the identical cutout.

    python make_rgba_inputs.py bean_3_0009_train ... [--tight-masks masks/plantdreamer_hybrid]
Writes inputs/rgba/<stem>.png (square, padded with transparent pixels like clean_input pads with white).
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from po_paths import DATASETS, HYBRID_MASKS, INPUTS

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--tight-masks", type=Path, default=HYBRID_MASKS)
    ap.add_argument("-o", "--out", type=Path, default=INPUTS / "rgba")
    ap.add_argument("--crop", type=float, default=0.0, help="same plant-centred square crop as batch_single_photo --crop")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for stem in a.stems:
        plant = stem.rsplit("_", 2)[0]
        view = stem[len(plant) + 1:]
        conv = plant.startswith(("tw_", "bb_", "tp_", "pl_", "wh_", "gs_"))
        D = (DATASETS / "converted" / "plants" if conv else DATASETS / "PlantDreamer_Data" / "plants") / plant
        im = np.asarray(Image.open(D / "images" / "undistorted" / f"{view}.png").convert("RGB"))
        mp = D / "images" / "mask" / f"{view}.png.png" if conv else a.tight_masks / plant / f"{view}.png"  # never PD masks alone
        m = np.asarray(Image.open(mp).convert("L")) > 127
        h, w = m.shape
        side, dx, dy = max(h, w), 0, 0
        if a.crop:  # identical geometry to batch_single_photo.clean_input(crop=...)
            ys, xs = np.where(m)
            side = int(round(a.crop * max(ys.max() - ys.min(), xs.max() - xs.min())))
            dx = int(round(w / 2 - (xs.min() + xs.max()) / 2))
            dy = int(round(h / 2 - (ys.min() + ys.max()) / 2))
        src = Image.fromarray(np.dstack([im, (m * 255).astype(np.uint8)]), "RGBA")
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(src, ((side - w) // 2 + dx, (side - h) // 2 + dy))
        canvas.save(a.out / f"{stem}.png")
        print(f"{stem}: alpha covers {m.mean():.3f} of the photo", flush=True)


if __name__ == "__main__":
    main()
