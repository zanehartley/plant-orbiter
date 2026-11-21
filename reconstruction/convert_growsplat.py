"""Convert GrowSplat (HF SimeonOA/GrowSplat-Dataset, CC BY 4.0; NPEC Maxi-Marvin: 15 static calibrated cameras in three
rings of five, one plant per species imaged on many dates) into the PlantDreamer-style layout, like convert_wheat.py,
for the growth case study. Run in the "nerfstudio" env (cv2, plyfile).

Per date: images_2 (540 x 960) + masks, undistorted (OPENCV k1/k2) to a pinhole with the principal point at the image
centre (eval_real_views assumes it); nerfstudio c2w (OpenGL) -> our c2w (OpenCV). World up = normal of the camera
centres' best-fit plane, signed so the cameras' image-up points along it; target = point closest to all optical axes.
Input view: the SAME camera on every date (static rig) = the lowest-elevation camera whose mask stays clear of the
image edges on the species' last (largest) date; the other 14 are test views. The dataset's plant point cloud (same
metric frame as the cameras) is the 3D / trait reference.
Writes $ORBIT_DATASETS/converted/plants/gs_<species>_<date>/ + a QA sheet.

    python convert_growsplat.py quinoa [sequoia]
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import figs
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
SRC = DATASETS / "GrowSplat"
OVER = DATASETS / "converted" / "plants"
DST = DATASETS / "converted" / "plants"
GL2CV = np.diag([1.0, -1.0, -1.0])


def load(d):
    t = json.loads((d / "transforms.json").read_text())
    cams = []
    for k, f in enumerate(t["frames"]):
        M = np.array(f["transform_matrix"])
        R, C = M[:3, :3] @ GL2CV, M[:3, 3]
        s = 0.5  # images_2
        K = np.array([[f["fl_x"] * s, 0, f["cx"] * s], [0, f["fl_y"] * s, f["cy"] * s], [0, 0, 1]])
        dist = np.array([f.get("k1", 0), f.get("k2", 0), f.get("p1", 0), f.get("p2", 0)])
        name = Path(f["file_path"]).name
        cams.append({"k": k, "R": R, "C": C, "K": K, "dist": dist, "w": round(f["w"] * s), "h": round(f["h"] * s),
                     "img": d / "images_2" / name, "mask": d / f["mask_path"]})
    return t, cams


def frame_geometry(cams):
    C = np.stack([c["C"] for c in cams])
    _, _, vt = np.linalg.svd(C - C.mean(0))
    up = vt[2]
    if np.mean([(c["R"] @ np.array([0, -1.0, 0])) @ up for c in cams]) < 0:  # image-up (-y in OpenCV) along up
        up = -up
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c in cams:
        z = c["R"][:, 2]
        P = np.eye(3) - np.outer(z, z)
        A += P
        b += P @ c["C"]
    target = np.linalg.solve(A, b)
    return up, target


def undistort(c):
    w, h = c["w"], c["h"]
    Kn = np.array([[c["K"][0, 0], 0, w / 2], [0, c["K"][1, 1], h / 2], [0, 0, 1]])
    mx, my = cv2.initUndistortRectifyMap(c["K"], c["dist"], None, Kn, (w, h), cv2.CV_32FC1)
    img = cv2.remap(cv2.imread(str(c["img"]), cv2.IMREAD_COLOR), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.imread(str(c["mask"]), cv2.IMREAD_GRAYSCALE)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.remap(m, mx, my, cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.uint8) * 255, Kn


def main():
    for sp in sys.argv[1:]:
        dates = sorted(d for d in (SRC / sp).iterdir() if (d / "transforms.json").exists()
                       and len(list((d / "images_2").glob("*.png"))) == 15 and len(list((d / "masks").glob("*.png"))) == 15)
        # input camera: lowest elevation whose mask is clear of the edges on the last date
        _, cams = load(dates[-1])
        up, target = frame_geometry(cams)
        elev = {c["k"]: np.degrees(np.arcsin(((c["C"] - target) / np.linalg.norm(c["C"] - target)) @ up)) for c in cams}
        clear = []
        for c in cams:
            _, m, _ = undistort(c)
            ys, xs = np.where(m > 0)
            if len(ys) and ys.min() > 2 and xs.min() > 2 and xs.max() < c["w"] - 3 and ys.max() < c["h"] - 3:
                clear.append(c["k"])
        front_k = min(clear or elev, key=lambda k: elev[k])
        print(f"{sp}: {len(dates)} dates, input camera {front_k} (elevation {elev[front_k]:.0f} deg; "
              f"{len(clear)}/15 cameras clear of the edges on {dates[-1].name})", flush=True)
        sheet = []
        for d in dates:
            t, cams = load(d)
            up, target = frame_geometry(cams)
            name = f"gs_{sp}_{d.name}"
            out = OVER / name
            if out.exists():
                shutil.rmtree(out)
            for sub in ("images/undistorted", "images/mask", "plantdreamer/3dgs", "data"):
                (out / sub).mkdir(parents=True, exist_ok=True)
            js = []
            for c in cams:
                img, m, Kn = undistort(c)
                n = f"{c['k']:04d}_{'train' if c['k'] == front_k else 'eval'}"
                cv2.imwrite(str(out / "images" / "undistorted" / f"{n}.png"), img)
                cv2.imwrite(str(out / "images" / "mask" / f"{n}.png.png"), m)
                rel = c["C"] - target
                js.append({"id": c["k"], "img_name": n, "src": c["img"].name, "width": c["w"], "height": c["h"],
                           "position": c["C"].tolist(), "rotation": c["R"].tolist(), "fx": float(Kn[0, 0]), "fy": float(Kn[1, 1]),
                           "elevation": float(np.degrees(np.arcsin(rel @ up / np.linalg.norm(rel)))), "distance": float(np.linalg.norm(rel))})
                if c["k"] == front_k:
                    sheet.append(cv2.resize(np.where(m[..., None] > 0, img, img // 4), (180, 320)))
            (out / "plantdreamer" / "3dgs" / "cameras.json").write_text(json.dumps(js, indent=1))
            (out / "front.txt").write_text(f"{front_k:04d}_train\n")
            (out / "world.json").write_text(json.dumps({"target": target.tolist(), "up": up.tolist()}, indent=1))
            shutil.copy(d / t["ply_file_path"], out / "data" / f"{name}_real_pointcloud.ply")
            (out / "data" / "metadata.json").write_text((d / "metadata.json").read_text())
            link = DST / name
            if not link.exists():
                link.symlink_to(out)
        rows = [np.hstack(sheet[i:i + 14] + [np.zeros_like(sheet[0])] * (14 - len(sheet[i:i + 14]))) for i in range(0, len(sheet), 14)]
        cv2.imwrite(str(figs.out_dir("checks", "growsplat_conversion") / f"inputs_{sp}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{sp}: converted {len(dates)} dates", flush=True)


if __name__ == "__main__":
    main()
