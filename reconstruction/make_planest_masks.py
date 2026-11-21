"""Plant + pot masks for converted PLANesT-3D scans (run in the "hunyuan" env: needs rembg; the pot step runs
planest_pot_sam.py in the "orbit_cam" env).

Protocol: plant + pot, as for Botany-Bot / PlantDreamer; not the stand under the pot or the
marker sheets on it. rembg (isnet-general-use) finds the plant reliably but is inconsistent below it (keeps the stand
and sheets in some views, drops the pot in others). Colour/darkness carving of the pot from the calibrated cameras
was tried first and was fragile (soil, shadows, dark rose foliage), so the pot comes from OWLv2 + SAM 2.1
(planest_pot_sam.py; specks in the soil filled/closed). Per view: mask = rembg, minus rembg pixels outside the SAM mask
below 60% of its height (stand, sheets), plus the SAM mask (whole-plant detections, coarse between leaves: only its
dark pot/soil pixels below that line). Pot detections off the plant's projected vertical axis (below the
centre, within a pot width) are ignored (corridor objects, pepper03); drop specks (<0.5% of the largest component). No pot found (close-ups) -> rembg.

Views: close-ups that crop the plant are still valid test views (every method is rendered at the same camera and
scored in-frame), so only the input must show the whole plant: rembg mask clear of the top/left/right edges and a
pot-shaped detection clear of the bottom edge; preference low (15-30 deg), far, best centred (files, cameras.json and
front.txt renamed if the converter's choice changes). Test views: up to 64 others, evenly spaced (scoring skips views
without a mask, as for TomatoPGT's 66). Writes images/mask/<view>.png.png (input + test), images/rembg/, images/pot/,
and a QA sheet $ORBIT_OUTPUTS/checks/<date>_planest_masks/masks_<plant>.jpg.

    python make_planest_masks.py pl_pepper01 [...]
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU keeps the GPU free (set it to use one)

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
import figs
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
DST = Path(os.environ.get("PLANT_DATA", DATASETS / "converted" / "plants"))
OC = ["conda", "run", "--no-capture-output", "-n", "orbit_cam", "python", str(ROOT / "planest_pot_sam.py")]


def clean(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    return np.isin(lab, 1 + np.where(areas >= 0.005 * areas.max())[0])


def plausible(pot, c, t, up, d):
    """A pot detection is real if it sits on the plant's vertical axis below the plant centre: project the target
    and a point below it with the calibrated camera; the bottom of the detection must lie below the target and within one
    pot width of that line (corridor objects picked up as pots are off it)."""
    R, C = np.array(c["rotation"]), np.array(c["position"])
    pr = lambda X: (lambda x: np.array([c["fx"] * x[0] / x[2] + c["width"] / 2, c["fy"] * x[1] / x[2] + c["height"] / 2]))(R.T @ (X - C))
    a, b = pr(t), pr(t - 0.3 * d * up)
    py, px = np.where(pot)
    q = np.array([px[py >= py.max() - 5].mean(), py.max()])  # bottom of the detection (whole-plant ones centre near t)
    u = (b - a) / np.linalg.norm(b - a)
    along, off = (q - a) @ u, abs(np.cross(u, q - a))
    return along > 0 and off < max(px.max() - px.min(), 40)


def load(p):
    return np.asarray(Image.open(p).convert("L")) > 127 if p.exists() else None


def rename(D, cams, renames):
    for old, new in renames.items():
        for sub, ext in (("undistorted", ".png"), ("rembg", ".png"), ("pot", ".png")):
            src = D / "images" / sub / f"{old}{ext}"
            if src.exists():
                src.rename(D / "images" / sub / f"{new}{ext}")
    for c in cams:
        c["img_name"] = renames.get(c["img_name"], c["img_name"])
    bp = D / "pot_boxes.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        bp.write_text(json.dumps({renames.get(k, k): v for k, v in b.items()}, indent=1))


def main():
    sess = new_session("isnet-general-use")
    for plant in sys.argv[1:]:
        D = DST / plant
        cp = D / "plantdreamer" / "3dgs" / "cameras.json"
        cams = json.loads(cp.read_text())
        W = json.loads((D / "world.json").read_text())
        t, up = np.array(W["target"]), np.array(W["up"])
        d = np.median([np.linalg.norm(np.array(c["position"]) - t) for c in cams])
        rdir, mdir = D / "images" / "rembg", D / "images" / "mask"
        rdir.mkdir(parents=True, exist_ok=True)
        for c in cams:  # 1. rembg on every view (needed to find the views that show the whole plant)
            p = rdir / f"{c['img_name']}.png"
            if not p.exists():
                img = Image.open(D / "images" / "undistorted" / f"{c['img_name']}.png").convert("RGB")
                remove(img, session=sess, only_mask=True).point(lambda v: 255 if v > 127 else 0).save(p)
        # 2. input candidates: plant clear of the top/left/right edges, ranked low > far > centred
        def clear(c):
            m = load(rdir / f"{c['img_name']}.png")
            return not (m[0].any() or m[:, 0].any() or m[:, -1].any())
        cand = [c for c in cams if clear(c)]
        el = np.array([c["elevation"] for c in cand])
        dist = np.array([c["distance"] for c in cand])
        off = np.array([(lambda x: np.hypot(x[0] / x[2], x[1] / x[2]))(np.array(c["rotation"]).T @ (t - np.array(c["position"])))
                        for c in cand])
        low = (el > 15) & (el < 30)
        far = dist >= np.percentile(dist[low] if low.any() else dist, 75)
        rank = np.lexsort((off, ~far, ~low))
        ranked = [cand[i]["img_name"] for i in rank[:8]]
        bp = D / "pot_boxes.json"
        if [n for n in ranked if n not in (json.loads(bp.read_text()) if bp.exists() else {})]:
            subprocess.run(OC + [plant] + ranked, check=True)
        boxes = json.loads((D / "pot_boxes.json").read_text())
        h = cams[0]["height"]
        ok = [n for n in ranked if boxes.get(n) and boxes[n]["potlike"] and boxes[n]["box"][3] < h - 3]
        front_old = (D / "front.txt").read_text().strip()
        new = ok[0] if ok else ranked[0]
        if new != front_old:
            renames = {front_old: front_old.replace("_train", "_eval"), new: new.replace("_eval", "_train")}
            rename(D, cams, renames)
            cp.write_text(json.dumps(cams, indent=1))
            (D / "front.txt").write_text(renames[new] + "\n")
            print(f"{plant}: input {front_old} -> {renames[new]}", flush=True)
        front = (D / "front.txt").read_text().strip()
        # 3. test views, pot masks for them
        others = [c["img_name"] for c in cams if c["img_name"] != front]
        test = [others[i] for i in np.unique(np.linspace(0, len(others) - 1, min(64, len(others))).round().astype(int))]
        boxes = json.loads((D / "pot_boxes.json").read_text())
        todo = [n for n in test if n not in boxes]
        if todo:
            subprocess.run(OC + [plant] + todo, check=True)
            boxes = json.loads((D / "pot_boxes.json").read_text())
        # 4. combine
        mdir.mkdir(parents=True, exist_ok=True)
        for f in mdir.glob("*.png.png"):
            f.unlink()
        sheet, nopot = [], 0
        cam = {c["img_name"]: c for c in cams}
        for k, n in enumerate([front] + test):
            m = load(rdir / f"{n}.png")
            pot = load(D / "images" / "pot" / f"{n}.png")
            if pot is not None and pot.any() and not plausible(pot, cam[n], t, up, d):
                pot = None  # a detection elsewhere in the corridor (pepper03 0046, 0216, 0262)
            if pot is not None and pot.any():
                rows = np.where(pot.any(1))[0]
                mid = int(rows.min() + 0.5 * (rows.max() - rows.min()))
                cut = int(rows.min() + 0.6 * (rows.max() - rows.min()))
                # SAM leaves specks in dark soil (also where the pot runs off the frame): fill small enclosed holes and
                # close the lower half; larger holes are gaps between leaves
                pad = np.pad(pot, 1).astype(np.uint8)
                cv2.floodFill(pad, None, (0, 0), 1)
                holes = (pad[1:-1, 1:-1] == 0).astype(np.uint8)
                nh, lab, st, cen = cv2.connectedComponentsWithStats(holes, connectivity=4)
                small = [i for i in range(1, nh) if st[i, cv2.CC_STAT_AREA] < 1e-3 * pot.size and cen[i][1] > mid]
                pot = pot | np.isin(lab, small)
                lo = cv2.morphologyEx(pot.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
                pot[mid:] |= lo[mid:]
                m[cut:] &= pot[cut:]
                if (boxes.get(n) or {}).get("potlike"):
                    m = m | pot
                else:  # whole-plant detection: coarse between leaves, so only add its dark (pot, soil) pixels low down
                    img = np.asarray(Image.open(D / "images" / "undistorted" / f"{n}.png").convert("RGB"))
                    m[cut:] |= pot[cut:] & (img[cut:].max(-1) < 90)
            else:
                nopot += 1
            m = clean(m)
            Image.fromarray((m * 255).astype(np.uint8)).save(mdir / f"{n}.png.png")
            if k == 0 or (k % 3 == 1 and len(sheet) < 24):
                img = np.asarray(Image.open(D / "images" / "undistorted" / f"{n}.png").convert("RGB")).copy()
                img[~m] = (img[~m] * 0.3).astype(np.uint8)
                if pot is not None:
                    cs, _ = cv2.findContours(pot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cs, -1, (255, 0, 255), 3)
                cv2.putText(img, n, (20, 60), 0, 1.8, (255, 255, 0), 4)
                sheet.append(cv2.resize(img, (390, 260)))
        print(f"{plant}: input {front} ({len(ok)}/{len(ranked)} top candidates show pot + plant), {len(test)} test views, "
              f"{nopot} without a pot detection", flush=True)
        out = figs.out_dir("checks", "planest_masks")
        rows = [np.hstack(sheet[i:i + 6] + [np.zeros_like(sheet[0])] * (6 - len(sheet[i:i + 6]))) for i in range(0, len(sheet), 6)]
        cv2.imwrite(str(out / f"masks_{plant}.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
