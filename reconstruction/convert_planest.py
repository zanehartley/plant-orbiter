"""Convert PLANesT-3D scans (handheld Canon 6D II around a potted plant; Metashape cameras) into the PlantDreamer-style
layout, like convert_tomatopgt.py. The paper uses the Pepper and Rose scans.

Images are the 1/4-resolution copies (1560x1040, datasets/PLANesT-3D/images_q4, made by planest_extract_images.py). Cameras come
from <Plant>_<NN>_Camera_Info.xml: per-camera camera-to-chunk transform (OpenCV axes), one adjusted Brown calibration
(f, cx/cy as offsets from the image centre, k1-k3, p1/p2 in Metashape order = OpenCV p2/p1), scaled by 1/4 here.
Per frame: undistort to a pinhole camera with the principal point at the centre. Plant centre = point closest to
all optical axes; up = normal of the camera ring, signed so the cameras look down. Input view = a far (whole plant in
frame), low (15-30 deg) camera with the plant near the image centre. Writes data/converted/plants/pl_<plant><NN>/
(no masks: make_planest_masks.py drops the views that crop the plant).

    python convert_planest.py Pepper_01 Rose_03 [...]
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
SRC = DATASETS / "PLANesT-3D" / "images_q4"
DST = DATASETS / "converted" / "plants"
SCALE = 0.25  # images_q4 are 1/4 of the 6240x4160 originals


def read_xml(p):
    """-> per-frame (label, camera-to-chunk 4x4, K, dist, (w, h)). Several calibrated sensors are allowed (Ribes 05/06:
    the zoom lens at 50 and 55 mm): each camera is undistorted with its own sensor's calibration;
    uncalibrated sensors (Pepper_06) and their cameras are skipped."""
    ch = ET.parse(p).getroot().find("chunk")
    sens = {}
    for s in ch.find("sensors").iter("sensor"):
        cal = s.find("calibration")
        if cal is None:
            continue
        g = lambda k, cal=cal: float(cal.find(k).text) if cal.find(k) is not None else 0.0
        W, H = (int(cal.find("resolution").get(k)) for k in ("width", "height"))
        w, h = round(W * SCALE), round(H * SCALE)
        f = g("f") * SCALE
        K = np.array([[f, 0, w / 2 + g("cx") * SCALE], [0, f, h / 2 + g("cy") * SCALE], [0, 0, 1]])
        dist = np.array([g("k1"), g("k2"), g("p2"), g("p1"), g("k3")])  # OpenCV order: k1 k2 p1 p2 k3
        assert not (g("b1") or g("b2")), "affinity/skew terms not handled"
        sens[s.get("id")] = (K, dist, (w, h))
    # cameras flagged as rotated (EXIF orientation != 1; 1 of 320 in Pepper_02) are skipped
    frames = [(c.get("label"), np.array(c.find("transform").text.split(), float).reshape(4, 4)) + sens[c.get("sensor_id")]
              for c in ch.find("cameras").iter("camera") if c.find("transform") is not None
              and c.get("sensor_id") in sens
              and (c.find("orientation") is None or c.find("orientation").text == "1")]
    frames.sort(key=lambda f: f[0])
    return frames


def main():
    for scan in sys.argv[1:]:
        species = scan.split("_")[0]
        frames = read_xml(SRC / species / f"{scan}_Camera_Info.xml")
        K, dist, (w, h) = frames[0][2:]
        img_dir = SRC / species / f"{scan.upper()}_IMAGES"
        c2w = np.stack([f[1] for f in frames])
        A, b = np.zeros((3, 3)), np.zeros(3)
        for m in c2w:
            P = np.eye(3) - np.outer(m[:3, 2], m[:3, 2])
            A += P
            b += P @ m[:3, 3]
        target = np.linalg.solve(A, b)
        centres = c2w[:, :3, 3]
        _, _, vt = np.linalg.svd(centres - centres.mean(0))
        up = vt[2]
        rel = centres - target
        dist_t = np.linalg.norm(rel, axis=1)
        elev = np.degrees(np.arcsin(rel @ up / dist_t))
        if np.median(elev) < 0:
            up, elev = -up, -elev
        upright = np.mean([(m[:3, :3].T @ up)[1] < 0 for m in c2w])
        assert upright > 0.9, f"{scan}: only {upright:.0%} of frames upright"
        Kn = np.array([[K[0, 0], 0, w / 2], [0, K[1, 1], h / 2], [0, 0, 1]])
        # input view: among low cameras, the farthest quartile, then the one that centres the plant best
        tc = np.einsum("nij,nj->ni", c2w[:, :3, :3].transpose(0, 2, 1), target - centres)  # target in camera coords
        off = np.hypot(tc[:, 0] / tc[:, 2], tc[:, 1] / tc[:, 2]) * Kn[0, 0] / w
        low = (elev > 15) & (elev < 30)
        far = dist_t >= np.percentile(dist_t[low], 75)
        cand = np.where(low & far)[0]
        front_i = int(cand[np.argmin(off[cand])])
        name = "pl_" + scan.lower().replace("_", "")
        out = DST / name
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("images/undistorted", "images/mask", "plantdreamer/3dgs"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        cams = []
        for i, (label, m, K, dist, (w, h)) in enumerate(frames):
            Kn = np.array([[K[0, 0], 0, w / 2], [0, K[1, 1], h / 2], [0, 0, 1]])
            src = img_dir / f"{label}.JPG"
            if not src.exists():
                print(f"  {scan}: missing image {src.name}", flush=True)
                continue
            img = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # raw sensor frame, as calibrated
            assert img.shape[:2] == (h, w), (src, img.shape)
            img = cv2.undistort(img, K, dist, None, Kn)
            n = f"{i:04d}_{'train' if i == front_i else 'eval'}"
            cv2.imwrite(str(out / "images" / "undistorted" / f"{n}.png"), img)
            cams.append({"id": i, "img_name": n, "src": src.name, "width": w, "height": h, "position": m[:3, 3].tolist(),
                         "rotation": m[:3, :3].tolist(), "fx": float(Kn[0, 0]), "fy": float(Kn[1, 1]),
                         "elevation": float(elev[i]), "distance": float(dist_t[i])})
        (out / "plantdreamer" / "3dgs" / "cameras.json").write_text(json.dumps(cams, indent=1))
        (out / "front.txt").write_text(f"{front_i:04d}_train\n")
        (out / "world.json").write_text(json.dumps({"target": target.tolist(), "up": up.tolist()}, indent=1))
        print(f"{name}: {len(cams)} views, elevation {np.median(elev):.0f} deg (input {elev[front_i]:.0f}), "
              f"input {frames[front_i][0]} at {dist_t[front_i] / np.median(dist_t):.2f}x median distance", flush=True)


if __name__ == "__main__":
    main()
