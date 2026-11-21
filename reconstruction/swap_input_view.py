"""Swap a converted plant's input view (the input must show the whole plant).

Rule used to pick the new view: same camera height as the old input (within 3 deg), whole plant >= 1% clear of every
photo border, smallest angle from the old input. The old input becomes a test view (_train -> _eval) and the new
one the input (_eval -> _train), in every per-view file (photo, mask, rembg, pot mask), cameras.json, pot_boxes.json
and front.txt, as make_planest_masks.py does when its choice changes. Logged in <plant>/input_swaps.json.
Standard library only (the cluster's system python3 has no numpy).

    python3 swap_input_view.py <plants dir> bb_alocasia 0017_eval [bb_pepper3 0011_eval ...]
"""
import json
import sys
from datetime import date
from pathlib import Path


def swap(D, new):
    old = (D / "front.txt").read_text().strip()
    if old == new.replace("_eval", "_train"):
        print(f"{D.name}: already swapped to {old}")
        return
    if not new.endswith("_eval") or not old.endswith("_train"):
        sys.exit(f"{D.name}: unexpected state (front {old}, new {new})")
    renames = {old: old.replace("_train", "_eval"), new: new.replace("_eval", "_train")}
    for a, b in renames.items():  # refuse to overwrite anything
        for sub, ext in (("undistorted", ".png"), ("mask", ".png.png"), ("rembg", ".png"), ("pot", ".png")):
            if (D / "images" / sub / f"{b}{ext}").exists():
                sys.exit(f"{D.name}: {sub}/{b}{ext} already exists")
    for a, b in renames.items():
        for sub, ext in (("undistorted", ".png"), ("mask", ".png.png"), ("rembg", ".png"), ("pot", ".png")):
            src = D / "images" / sub / f"{a}{ext}"
            if src.exists():
                src.rename(D / "images" / sub / f"{b}{ext}")
    cp = D / "plantdreamer" / "3dgs" / "cameras.json"
    cams = json.loads(cp.read_text())
    for c in cams:
        c["img_name"] = renames.get(c["img_name"], c["img_name"])
    cp.write_text(json.dumps(cams, indent=1))
    bp = D / "pot_boxes.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        bp.write_text(json.dumps({renames.get(k, k): v for k, v in b.items()}, indent=1))
    (D / "front.txt").write_text(renames[new] + "\n")
    lp = D / "input_swaps.json"
    log = json.loads(lp.read_text()) if lp.exists() else []
    log.append({"date": str(date.today()), "old_input": old, "new_input": renames[new], "renames": renames,
                "why": "old input clipped by the photo border; same height, whole plant in frame, nearest angle"})
    lp.write_text(json.dumps(log, indent=1))
    print(f"{D.name}: input {old} -> {renames[new]} (old input is now test view {renames[old]})")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    args = sys.argv[2:]
    for plant, new in zip(args[::2], args[1::2]):
        swap(root / plant, new)
