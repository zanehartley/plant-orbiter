"""Choose and/or rectify the single input photo for an orbit run.

select:  score every real view for how good a front photo it makes (side-on, little roll, plant centred, not cut
         off by the frame, plant large) and take the best.
rectify: re-render the photo as a virtual pinhole camera at the same position that looks straight at the plant,
         with no roll (world up = image up) and a focal length chosen so the plant fills --fill of the square frame.
         This is an exact rotation + zoom of the real camera (a homography), so the new camera is still a real,
         calibrated view and alignment/scoring stay valid.

Writes a derived dataset (symlinks to the original images/masks/data, plus the new view in cameras.json) whose
front.txt names the chosen view; run batch_single_photo.py on it with PLANT_DATA pointing there.

    python prep_input_view.py bb_alocasia --src data/converted/plants --mode select  --dst data/inputsel/plants
    python prep_input_view.py bb_alocasia --src data/converted/plants --mode rectify --dst data/rectified/plants
Rectified views are named 9<NNN>_train (NNN = source view id) so they keep the 4-digit naming.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent


def axes_target(c2ws):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for m in c2ws:
        c, f = m[:3, 3], m[:3, 2]  # OpenCV: camera looks down +z
        P = np.eye(3) - np.outer(f, f)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def c2w_of(c):
    m = np.eye(4)
    m[:3, :3] = np.array(c["rotation"])
    m[:3, 3] = c["position"]
    return m


def load(src, plant):
    cams = json.loads((src / plant / "plantdreamer" / "3dgs" / "cameras.json").read_text())
    c2ws = [c2w_of(c) for c in cams]
    target = axes_target(c2ws)
    up = -np.mean([m[:3, 1] for m in c2ws], 0)
    return cams, c2ws, target, up / np.linalg.norm(up)


def view_stats(src, plant, c, m, target, up):
    msk = cv2.imread(str(src / plant / "images" / "mask" / (c["img_name"] + ".png.png")), cv2.IMREAD_GRAYSCALE)
    if msk is None or not msk.any():
        return None
    ys, xs = np.nonzero(msk)
    h, w = msk.shape
    rel = m[:3, 3] - target
    elev = np.degrees(np.arcsin(rel @ up / np.linalg.norm(rel)))
    u = m[:3, :3].T @ up  # world up in camera coords (x right, y down)
    roll = np.degrees(np.arctan2(u[0], -u[1]))
    off = np.hypot((xs.min() + xs.max()) / 2 / w - 0.5, (ys.min() + ys.max()) / 2 / h - 0.5)
    cut = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2)
    fill = max(np.ptp(ys) / h, np.ptp(xs) / w)
    return {"name": c["img_name"], "elev": float(elev), "roll": float(roll), "off": float(off), "cut": cut,
            "fill": float(fill)}


def select(src, plant, cams, c2ws, target, up, max_elev=20.0):
    rows = [s for c, m in zip(cams, c2ws) if (s := view_stats(src, plant, c, m, target, up))]
    cand = [r for r in rows if 2.0 <= r["elev"] <= max_elev] or rows

    def score(r):  # lower is better: side-on-ish, upright, centred, whole plant in frame, plant large
        return abs(r["elev"] - 8) / 10 + abs(r["roll"]) / 10 + 3 * r["off"] + 5 * r["cut"] - r["fill"]
    best = min(cand, key=score)
    return best, rows


def rectify(src, plant, c, m, target, up, size=1024, fill=0.7, focal768=None, prefix="9", max_fill=0.92):
    """Homography to a virtual camera at the same centre looking at the plant, upright, plant filling `fill`."""
    name = c["img_name"]
    img = cv2.imread(str(src / plant / "images" / "undistorted" / (name + ".png")))
    msk = cv2.imread(str(src / plant / "images" / "mask" / (name + ".png.png")), cv2.IMREAD_GRAYSCALE)
    h, w = msk.shape
    K = np.array([[c["fx"], 0, w / 2], [0, c["fy"], h / 2], [0, 0, 1]])
    R_old = m[:3, :3]
    ys, xs = np.nonzero(msk)
    pts = np.stack([xs, ys, np.ones_like(xs)], 1).astype(float)[:: max(1, len(xs) // 20000)]
    rays = (R_old @ np.linalg.inv(K) @ pts.T).T  # mask pixel rays in world coords
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    z = rays.mean(0)
    for _ in range(3):  # look at the centre of the plant's angular extent, upright, then size it
        z /= np.linalg.norm(z)
        y = -up - (-up @ z) * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R_new = np.stack([x, y, z], 1)  # columns = camera axes in world (OpenCV)
        cam = rays @ R_new  # rays in the new camera frame
        u, v = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
        cu, cv_ = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
        z = R_new @ np.array([cu, cv_, 1.0])  # re-aim at the bbox centre
    ext = max(np.ptp(u), np.ptp(v))
    f = fill * size / ext
    if focal768:  # match the orbit model's own lens (focal in px at 768) instead of a fixed fill
        f = focal768 * size / 768
        if f * ext / size > max_fill:  # plant would not fit: zoom out just enough (wider than the model's lens)
            f = max_fill * size / ext
    Kn = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    H = Kn @ R_new.T @ R_old @ np.linalg.inv(K)
    im2 = cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    m2 = cv2.warpPerspective(msk, H, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    new = dict(c, img_name=f"{prefix}{int(name[:4]):03d}_train", width=size, height=size, fx=float(f), fy=float(f),
               rotation=R_new.tolist(), rectified_from=name)
    return new, im2, m2


def derive(src, dst, plant):
    """dst/<plant>: symlinks to everything in src/<plant> except cameras.json and front.txt."""
    s, d = src / plant, dst / plant
    for sub in ("images/undistorted", "images/mask"):
        (d / sub).mkdir(parents=True, exist_ok=True)
        for f in (s / sub).iterdir():
            if not (d / sub / f.name).exists():
                os.symlink(f.resolve(), d / sub / f.name)
    for f in s.iterdir():
        if f.name not in ("images", "plantdreamer", "front.txt") and not (d / f.name).exists():
            os.symlink(f.resolve(), d / f.name)
    (d / "plantdreamer" / "3dgs").mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants", nargs="+")
    ap.add_argument("--src", type=Path, default=DATASETS / "converted" / "plants")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--mode", choices=["select", "rectify", "select+rectify"], required=True)
    ap.add_argument("--view", help="rectify this view instead of selecting one")
    ap.add_argument("--fill", type=float, default=0.7)
    ap.add_argument("--focal768", type=float, help="rectify to this focal (px at 768), e.g. the orbit model's ~620")
    ap.add_argument("--max-fill", type=float, default=0.92, help="with --focal768: zoom out if the plant would fill more")
    ap.add_argument("--prefix", default="9", help="first digit of the rectified view's name (keeps variants apart)")
    a = ap.parse_args()
    for plant in a.plants:
        cams, c2ws, target, up = load(a.src, plant)
        best, rows = select(a.src, plant, cams, c2ws, target, up)
        old = (a.src / plant / "front.txt").read_text().strip() if (a.src / plant / "front.txt").exists() else None
        cur = next((r for r in rows if r["name"] == old), None)
        fmt = lambda r: f"{r['name']} elev {r['elev']:.1f} roll {r['roll']:.1f} off {r['off']:.2f} cut {r['cut']} fill {r['fill']:.2f}"
        print(f"{plant}: current {fmt(cur) if cur else old}\n{'':{len(plant)}}  selected {fmt(best)}")
        d = derive(a.src, a.dst, plant)
        cams_out = list(cams)
        front = a.view or best["name"]
        if "rectify" in a.mode:
            i = next(k for k, c in enumerate(cams) if c["img_name"] == front)
            new, im2, m2 = rectify(a.src, plant, cams[i], c2ws[i], target, up, fill=a.fill, focal768=a.focal768, prefix=a.prefix, max_fill=a.max_fill)
            # rectified view is a "train" view (never scored); keep the source's other fields
            cv2.imwrite(str(d / "images" / "undistorted" / (new["img_name"] + ".png")), im2)
            cv2.imwrite(str(d / "images" / "mask" / (new["img_name"] + ".png.png")), m2)
            # drop the source view if it is a test view: its rectified copy is now the input
            cams_out = [c for c in cams_out if c["img_name"] not in (new["img_name"], front if front.endswith("_eval") else "")] + [new]
            front = new["img_name"]
            print(f"{'':{len(plant)}}  rectified -> {front} (f = {new['fx']:.0f} px at {new['width']} px = {new['fx'] * 768 / new['width']:.0f} at 768, plant fills {m2.any(0).sum() and max(np.ptp(np.nonzero(m2.any(1))[0]), np.ptp(np.nonzero(m2.any(0))[0])) / new['width']:.2f})")
        (d / "plantdreamer" / "3dgs" / "cameras.json").write_text(json.dumps(cams_out, indent=1))
        (d / "front.txt").write_text(front + "\n")


if __name__ == "__main__":
    main()
