"""Quality gate for the hybrid PlantDreamer masks (make_hybrid_masks.py).

The fresh cutout occasionally drops a dark pot against a dark background (bean_3 low views). A hybrid mask that
keeps < --min-pot of the dataset mask's bottom quarter (the pot), or < --min-area of its total area, is moved to
<masks>_rejected/, so that view is skipped for every method (mask_path finds no file), as views
with missing masks already are. We never fall back to PlantDreamer's mask alone.

    python masks/qa_hybrid_masks.py MASKS_DIR --plantdreamer PlantDreamer_Data/plants [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masks", type=Path, help="make_hybrid_masks.py output: <plant>/<view>.png")
    ap.add_argument("--plantdreamer", type=Path, required=True, help="PlantDreamer_Data/plants")
    ap.add_argument("--min-pot", type=float, default=0.6)
    ap.add_argument("--min-area", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = bad = 0
    rejected = a.masks.with_name(a.masks.name + "_rejected")
    for f in sorted(a.masks.glob("*/*.png")):
        p, v = f.parent.name, f.stem
        h = np.asarray(Image.open(f)) > 127
        d = np.asarray(Image.open(a.plantdreamer / p / "images" / "mask" / f"{v}.png.png").convert("L")) > 0
        n += 1
        ys = np.where(d.any(1))[0]
        pot = d.copy()
        pot[: ys.min() + int(0.75 * (ys.max() - ys.min()))] = False
        pot_kept = (h & pot).sum() / max(pot.sum(), 1)
        area = h.sum() / max(d.sum(), 1)
        if pot_kept < a.min_pot or area < a.min_area:
            bad += 1
            print(f"reject {p} {v}: pot kept {pot_kept:.2f}, area {area:.2f}", flush=True)
            if not a.dry_run:
                (rejected / p).mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), rejected / p / f.name)
    print(f"{n} masks checked, {bad} rejected{' (dry run)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()
