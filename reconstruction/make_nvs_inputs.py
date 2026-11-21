"""Inputs for the SEVA / SV3D video-model baselines, made with the SAME cutout code as our runs
(batch_single_photo.clean_input: undistorted photo, dataset mask, white square canvas, photo centre kept):
  full/<stem>.png  whole photo on the square canvas (as our MiniMax input)  -> SEVA
  crop/<stem>.png  plant-centred square, 1.3 x the plant's bounding box     -> SV3D (it re-centres objects itself)
  crop/<stem>.json shift / side / crop, read by align_splat.py (copied into the run dir as input.json)

    PLANT_DATA=data/PlantDreamer_Data/plants python make_nvs_inputs.py pd -o inputs/nvs
    PLANT_DATA=data/converted/plants        python make_nvs_inputs.py bb pl wh -o inputs/nvs
    PLANT_DATA=data/converted/plants        python make_nvs_inputs.py gs --only gs_quinoa_20221122 [...] -o inputs/nvs
"""
import argparse
import json
from pathlib import Path

import batch_single_photo as bsp
from po_paths import INPUTS

# PlantDreamer evaluation plants (bean_2 excluded: its masks outline a different plant in part of the capture)
PD_PLANTS = [f"bean_{i}" for i in range(1, 13) if i != 2] + ["kale_1", "kale_2", "kale_3"] + [f"mint_{i}" for i in range(1, 7)]

PREFIX = {"bb": "bb_", "pl": ("pl_pepper", "pl_rose", "pl_ribes"), "wh": "wh_", "gs": "gs_"}
EXCLUDE = {"pl_ribes07"}  # broken mask names in the converted scan; PLANesT Ribes is not in the paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=["pd", "bb", "pl", "wh", "gs"])
    ap.add_argument("--only", nargs="+", help="plant names to make (e.g. the GrowSplat dates used in the paper)")
    ap.add_argument("-o", "--out", type=Path, default=INPUTS / "nvs")
    a = ap.parse_args()
    for sub in ("full", "crop"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        plants = PD_PLANTS if ds == "pd" else sorted(p.name for p in bsp.DATA.iterdir() if p.name.startswith(PREFIX[ds]) and p.name not in EXCLUDE)
        for plant in plants:
            if a.only and plant not in a.only:
                continue
            front = bsp.front_view(bsp.DATA / plant)
            stem = f"{plant}_{front}"
            p, _ = bsp.clean_input(plant, front, a.out / "full")
            p.rename(a.out / "full" / f"{stem}.png")
            p, side = bsp.clean_input(plant, front, a.out / "crop", crop=1.3)
            p.rename(a.out / "crop" / f"{stem}.png")
            from PIL import Image
            long_side = max(Image.open(bsp.DATA / plant / "images" / "undistorted" / f"{front}.png").size)
            info = {"pad": side / long_side - 1, "side": side, "shift_x": bsp.clean_input.shift, "shift_y": bsp.clean_input.shift_y,
                    "crop": 1.3}
            (a.out / "crop" / f"{stem}.json").write_text(json.dumps(info))
            print(ds, stem, flush=True)


if __name__ == "__main__":
    main()
