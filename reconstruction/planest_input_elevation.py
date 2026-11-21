"""Real elevation of each PLANesT input photo: angle of the input camera above the
plant, from the dataset's own SfM poses (up = mean camera -y, centre = point nearest all optical axes), and the
angle its optical axis points down. The elevation search in known_poses.py should find roughly this.

    python planest_input_elevation.py [stems ...]   (default: every PLANesT run)
"""
import json
import os
import sys
from pathlib import Path

from po_paths import DATASETS, OUTPUTS  # before eval_real_views reads PLANT_DATA

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PLANT_DATA", str(DATASETS / "converted" / "plants"))
import numpy as np  # noqa: E402

import eval_real_views as ev  # noqa: E402
from align_splat import axes_target  # noqa: E402


def main():
    runs = Path(os.environ.get("RUNS", OUTPUTS / "pl" / "ours"))  # any dataset's seed-42 runs
    stems = sys.argv[1:] or sorted(p.name[:-4] for p in runs.glob("*_s42"))
    out = {}
    for s in stems:
        front = s[-10:]
        plant = s[:-11]
        allc = ev.cameras(plant, "all")
        tgt = axes_target([c["c2w"] for c in allc])
        up = -np.mean([c["c2w"][:3, 1] for c in allc], 0)
        up /= np.linalg.norm(up)
        cam = [c for c in allc if c["name"] == front]
        if not cam:
            print(f"{s}: input view {front} not in the dataset cameras, skipped")
            continue
        cam = cam[0]
        r = cam["c2w"][:3, 3] - tgt
        el = np.degrees(np.arcsin(r @ up / np.linalg.norm(r)))
        pitch = np.degrees(np.arcsin(-cam["c2w"][:3, 2] @ up))  # optical axis below horizontal
        kp = runs / f"{s}_s42" / "3dgs" / "known_poses.json"
        rec = json.loads(kp.read_text())["hull_recall"] if kp.exists() else float("nan")
        out[s] = {"elevation": round(float(el), 1), "pitch_down": round(float(pitch), 1), "level_recall": rec}
        print(f"{s:28s} input camera {el:5.1f} deg above the plant centre, looking {pitch:5.1f} deg down; level-model recall {rec:.3f}")
    (OUTPUTS / f"real_input_elevation_{runs.parent.name if runs.name == 'ours' else runs.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
