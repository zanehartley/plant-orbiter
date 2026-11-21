"""Write each converted plant's input view (front.txt) as used in the paper, from results/input_views.csv.

The dataset converters pick an input view automatically; for four plants (bb_alocasia, bb_pepper3, bb_princess_pink,
pl_pepper02) the paper uses a different view because the automatic one did not show the whole plant. Run this once
after the conversion so every run uses the paper's input photo. PlantDreamer plants need nothing: their input view
is chosen deterministically from the capture ring.

    python tools/set_input_views.py [results/input_views.csv]
"""
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = Path(os.environ.get("ORBIT_DATASETS", REPO / "data" / "datasets"))
table = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "input_views.csv"
for r in csv.DictReader(open(table)):
    if r["dataset"] == "PlantDreamer":
        continue
    d = DATASETS / "converted" / "plants" / r["plant"]
    if not d.is_dir():
        print("not converted yet:", r["plant"])
        continue
    (d / "front.txt").write_text(r["input_view"] + "\n")
    print(r["plant"], r["input_view"])
