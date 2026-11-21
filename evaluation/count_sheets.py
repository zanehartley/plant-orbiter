"""Fruit counting by eye: a contact sheet and a turntable video per asset, from the fixed evaluation views.

Reads <renders>/<method>/<asset>/views/e<elev>_a<azim>.png (render_splats.py / render_threestudio.py: elevations
0/15/30, 24 azimuths) and writes, under --out:
  sheets/<method>/<asset>.jpg      every second azimuth (30 deg steps) x the 3 elevations, one image to count from
  videos/<method>/<asset>.mp4      turntable: the 24 azimuths at elevation 0, then at 30 (8 fps, 6 s)
  overview_<species>_s<seed>.jpg   methods x counts, front view, for a quick look across N
  counts.csv                       one row per asset to fill in (count, notes); ours also gets
                                   input_count, the fruit in its Qwen input image (input_image column)
Assets already done are skipped, so rerunning adds methods rendered later (MVDream, TRELLIS.2) without redoing
the rest. TRELLIS.2 lifts ours' input images, so its rows leave input_image empty (counted once, on ours' rows).
counts.csv is only written if it does not exist yet, so filled-in counts are never overwritten; with
--add-missing, rows for assets not yet in it are appended.

    python evaluation/count_sheets.py $ORBIT_OUTPUTS/fruit_counting/eval/renders \
        --out $ORBIT_OUTPUTS/fruit_counting/eval/counting
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "text_to_3d"))
from text_prompts import COUNTS, COUNT_SEEDS, COUNT_SPECIES  # noqa: E402

METHODS = ("ours", "diffsplat", "gaussiandreamer", "mvdream", "trellis_text", "trellis2")
# image-to-3D from ours' Qwen input images: named like ours, = input scored against ours' input_count
SAME_INPUT = ("ours", "trellis2")
ELEVATIONS = ("e+00", "e+15", "e+30")
TILE = 300
FIELDS = ["species", "N", "seed", "method", "asset", "sheet", "video", "input_image", "input_count", "count",
          "notes"]


def views(asset_dir, elev):
    """The asset's views at one elevation, sorted by azimuth."""
    return sorted(asset_dir.glob(f"views/{elev}_a*.png"), key=lambda v: float(v.stem.split("_a")[1]))


def contact_sheet(asset_dir, path, title):
    rows = [views(asset_dir, elev)[::2] for elev in ELEVATIONS]
    sheet = Image.new("RGB", (TILE * len(rows[0]), TILE * len(rows) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for r, row in enumerate(rows):
        for c, view in enumerate(row):
            sheet.paste(Image.open(view).convert("RGB").resize((TILE, TILE)), (c * TILE, 30 + r * TILE))
            draw.text((c * TILE + 6, 34 + r * TILE), view.stem, fill="grey")
    sheet.save(path, quality=90)


def turntable(asset_dir, path):
    frames = views(asset_dir, "e+00") + views(asset_dir, "e+30")
    listing = path.with_suffix(".txt")
    listing.write_text("".join(f"file '{f}'\nduration 0.125\n" for f in frames) + f"file '{frames[-1]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", "fps=8,format=yuv420p", "-c:v", "libx264", "-crf", "18", str(path)], check=True)
    listing.unlink()


def overview(renders, species, seed, path):
    """Rows: methods; columns: counts; the front view (elevation 15, azimuth 0) of each asset."""
    sheet = Image.new("RGB", (120 + TILE * len(COUNTS), 30 + TILE * len(METHODS)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, n in enumerate(COUNTS):
        draw.text((120 + c * TILE + TILE // 2, 10), f"N={n}", fill="black")
    for r, method in enumerate(METHODS):
        draw.text((8, 30 + r * TILE + TILE // 2), method, fill="black")
        for c, n in enumerate(COUNTS):
            front = renders / method / asset_name(method, species, n, seed) / "views" / "e+15_a000.0.png"
            if front.is_file():
                sheet.paste(Image.open(front).convert("RGB").resize((TILE, TILE)), (120 + c * TILE, 30 + r * TILE))
    sheet.save(path, quality=90)


def asset_name(method, species, count, seed):
    return f"{species}_n{count}_s{seed}_s42" if method in SAME_INPUT else f"{species}_n{count}_t{seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=Path, help="ours' Qwen input images (default: <renders>/../../images)")
    p.add_argument("--add-missing", action="store_true", help="append rows for assets not yet in counts.csv")
    a = p.parse_args()
    images = a.images or a.renders.parents[1] / "images"

    rows = []
    for species in COUNT_SPECIES:
        for seed in COUNT_SEEDS:
            for n in COUNTS:
                for method in METHODS:
                    asset = asset_name(method, species, n, seed)
                    asset_dir = a.renders / method / asset
                    if not (asset_dir / "cameras.json").is_file():
                        continue
                    sheet = a.out / "sheets" / method / f"{asset}.jpg"
                    video = a.out / "videos" / method / f"{asset}.mp4"
                    sheet.parent.mkdir(parents=True, exist_ok=True)
                    video.parent.mkdir(parents=True, exist_ok=True)
                    if not sheet.is_file():
                        contact_sheet(asset_dir, sheet, f"{method}  {asset}")
                    if not video.is_file():
                        turntable(asset_dir, video)
                    image = images / f"{species}_n{n}_s{seed}.png" if method == "ours" else None
                    rows.append({"species": species, "N": n, "seed": seed, "method": method, "asset": asset,
                                 "sheet": sheet.relative_to(a.out), "video": video.relative_to(a.out),
                                 "input_image": image or "", "input_count": "", "count": "", "notes": ""})
            overview(a.renders, species, seed, a.out / f"overview_{species}_s{seed}.jpg")

    table = a.out / "counts.csv"
    if table.exists() and not a.add_missing:
        print(f"{table} exists (not overwritten; --add-missing appends new assets)")
    else:
        # baselines share asset names, so key on (method, asset)
        done = {(r["method"], r["asset"]) for r in csv.DictReader(open(table))} if table.exists() else set()
        new = [r for r in rows if (r["method"], r["asset"]) not in done]
        with open(table, "a" if done else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not done:
                w.writeheader()
            w.writerows(new)
        print(f"{len(new)} rows -> {table}")
    print(f"{len(rows)} assets: sheets and videos under {a.out}")


if __name__ == "__main__":
    main()
