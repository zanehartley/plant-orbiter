"""Real-photo reference for the realism metrics (KD/FD in metrics.py --reference): PlantDreamer's real plant photos.

Each photo becomes an RGBA cutout: the undistorted photo with the Plant Orbiter hybrid mask (plant + pot, the
turntable left out), <masks>/<plant>/<image>.png (masks/plantdreamer_hybrid; masks/README.md):
PlantDreamer's mask intersected with a rembg cut-out, plus thin stems back, quality-gated by its maker. The hybrid masks
replace PlantDreamer's own masks: those are loose around leaves and misaligned in bean_2 and mint_2-5.
bean_2 has no hybrid masks (part of its dataset masks outline a different plant), so it is not in the reference; a
photo without a hybrid mask (rejected by the gate, or no usable dataset mask) is dropped.
Photos are kept when:
  - the hybrid mask exists and has the photo's size, and
  - the camera elevation is within the rendered range, ELEVATION_RANGE (renders: 0, 15, 30 degrees), and
  - the photo is not one of PlantDreamer's LoRA training photos (lora_imgs/<Species>_Original_Imgs.txt), so the
    reference does not favour PlantDreamer.
Elevation comes from the photo's camera in the plant's gaussian-splatting/cameras.json (3DGS format: camera centre and
camera-to-world rotation, OpenCV axes): world up = mean camera up vector (-y column; the cameras circle the plant at
level roll), plant centre = least-squares point nearest every viewing axis, elevation = angle of (camera - centre)
above the plane normal to up.

Writes <out>/<species>/<plant>_<image>.png and <out>/reference.csv (every photo of the plants with hybrid masks, its
elevation and whether and why it was kept). CPU only; run in the orbit_cam env:
    python evaluation/prep_reference.py --out $ORBIT_DATASETS/plant_reference
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

ELEVATION_RANGE = (-5.0, 35.0)  # degrees; renders use 0 / 15 / 30


def lora_photos(lora_dir):
    """{(plant, image index)} used to train PlantDreamer's LoRAs. The lists write plants as bean_1 or bean1."""
    used = set()
    for f in lora_dir.glob("*_Original_Imgs.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                plant, idx = (x.strip() for x in line.split(","))
                used.add((re.sub(r"^([a-z]+)_?(\d+)$", r"\1_\2", plant.lower()), int(idx)))
    return used


def elevations(cameras):
    """Elevation (degrees) of each camera above the plant, from camera centres and camera-to-world rotations."""
    pos = np.array([c["position"] for c in cameras])
    rot = np.array([c["rotation"] for c in cameras])  # (N, 3, 3), columns = camera x, y (down), z (forward)
    up = -rot[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    fwd = rot[:, :, 2] / np.linalg.norm(rot[:, :, 2], axis=1, keepdims=True)
    # Point nearest all viewing rays: sum_i (I - d_i d_i^T) (x - p_i) = 0.
    proj = np.eye(3)[None] - fwd[:, :, None] * fwd[:, None, :]
    centre = np.linalg.solve(proj.sum(0), (proj @ pos[:, :, None]).sum(0))[:, 0]
    offset = pos - centre
    return np.degrees(np.arcsin(offset @ up / np.linalg.norm(offset, axis=1)))


def load_mask(plant, name, masks):
    """Hybrid mask for this photo -> (reason or "", rgb, mask)."""
    mask_file = masks / plant.name / f"{name}.png"
    if not mask_file.is_file():
        return "mask_missing", None, None
    rgb = np.asarray(Image.open(plant / "images" / "undistorted" / f"{name}.png").convert("RGB"))
    mask = np.asarray(Image.open(mask_file).convert("L")) > 127
    if mask.shape != rgb.shape[:2]:
        return "mask_size", rgb, mask
    return "", rgb, mask


def main():
    datasets = Path(os.environ.get("ORBIT_DATASETS", Path(__file__).resolve().parents[1] / "data" / "datasets"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plantdreamer", type=Path, default=datasets / "PlantDreamer_Data")
    p.add_argument("--masks", type=Path,
                   help="<plant>/<image>.png hybrid masks (default: masks/plantdreamer_hybrid)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    masks = a.masks or Path(__file__).resolve().parents[1] / "masks" / "plantdreamer_hybrid"

    lora = lora_photos(a.plantdreamer / "lora_imgs")
    rows = []
    for plant in sorted((a.plantdreamer / "plants").iterdir()):
        if not (masks / plant.name).is_dir():  # bean_2: no hybrid masks, not in the reference
            print(f"{plant.name}: no hybrid masks, skipped", flush=True)
            continue
        species = plant.name.split("_")[0]
        cameras = json.loads((plant / "gaussian-splatting" / "cameras.json").read_text())
        (a.out / species).mkdir(parents=True, exist_ok=True)
        for cam, elev in zip(cameras, elevations(cameras)):
            name = cam["img_name"]
            if (plant.name, int(name.split("_")[0])) in lora:
                reason = "lora_training"
            elif not ELEVATION_RANGE[0] <= elev <= ELEVATION_RANGE[1]:
                reason = "elevation"
            else:
                reason, rgb, mask = load_mask(plant, name, masks)
                if not reason:
                    Image.fromarray(np.dstack([rgb, mask.astype(np.uint8) * 255]), "RGBA").save(
                        a.out / species / f"{plant.name}_{name}.png")
            rows.append({"plant": plant.name, "image": name, "elevation": round(float(elev), 1),
                         "kept": int(not reason), "excluded": reason})
    with open(a.out / "reference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for species in sorted({r["plant"].split("_")[0] for r in rows}):
        sp = [r for r in rows if r["plant"].startswith(species)]
        print(f"{species}: {sum(r['kept'] for r in sp)} kept of {len(sp)} "
              f"({sum(r['excluded'] == 'lora_training' for r in sp)} LoRA, "
              f"{sum(r['excluded'] == 'elevation' for r in sp)} outside {ELEVATION_RANGE} deg, "
              f"{sum(r['excluded'].startswith('mask') for r in sp)} without a usable hybrid mask)", flush=True)


if __name__ == "__main__":
    main()
