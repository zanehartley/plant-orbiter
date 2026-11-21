"""Convert Botany-Bot plants (nerfstudio format, ZED cameras) into the PlantDreamer-style layout.

Per frame: undistort (OPENCV model) the half-resolution image and the mask, shift the principal point to the
centre, then rotate by a multiple of 90 deg so the plant is upright (the cameras are mounted sideways), with
the camera updated to match. Writes data/converted/plants/bb_<plant>/ like convert_tomatowur.py (no point cloud).

    python convert_botanybot.py [pepper1 croton ...]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
SRC = DATASETS / "BotanyBot"
DST = DATASETS / "converted" / "plants"
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    plants = sys.argv[1:] or ["alocasia", "anthurium", "belize", "croton", "pepper1", "pepper2", "pepper3", "princess_pink"]
    for plant in plants:
        t = json.loads((SRC / plant / "transforms.json").read_text())
        frames = t["frames"]
        c2w = np.stack([np.array(f["transform_matrix"], float) @ GL2CV for f in frames])
        centres = c2w[:, :3, 3]
        # plant centre = point closest to all optical axes; up = turntable axis (least-variance direction)
        A, b = np.zeros((3, 3)), np.zeros(3)
        for m in c2w:
            P = np.eye(3) - np.outer(m[:3, 2], m[:3, 2])
            A += P
            b += P @ m[:3, 3]
        target = np.linalg.solve(A, b)
        _, _, vt = np.linalg.svd(centres - centres.mean(0))
        up = vt[2]
        rel = centres - target
        elev = np.degrees(np.arcsin(rel @ up / np.linalg.norm(rel, axis=1)))
        if np.median(elev) < 0:  # most cameras look down on the plant
            up, elev = -up, -elev
        out = DST / f"bb_{plant}"
        for d in ("images/undistorted", "images/mask", "plantdreamer/3dgs"):
            (out / d).mkdir(parents=True, exist_ok=True)
        # input view: lowest ring above ~3 deg elevation (closest to side-on)
        order = np.argsort(np.where(elev > 3, elev, 1e9))
        front_i = int(order[0])
        cams = []
        for i, f in enumerate(frames):
            img_p = SRC / plant / f["file_path"].replace("images/", "images_2/")
            msk_p = SRC / plant / f["mask_path"]
            if not img_p.exists() or not msk_p.exists():
                continue
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]
            s = w / f["w"]  # images_2 = half resolution
            K = np.array([[f["fl_x"] * s, 0, f["cx"] * s], [0, f["fl_y"] * s, f["cy"] * s], [0, 0, 1]])
            dist = np.array([f.get("k1", 0), f.get("k2", 0), f.get("p1", 0), f.get("p2", 0)])
            msk = cv2.resize(cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST)
            # undistort to a pinhole camera with the principal point at the image centre
            Kn = np.array([[K[0, 0], 0, w / 2], [0, K[1, 1], h / 2], [0, 0, 1]])
            img = cv2.undistort(img, K, dist, None, Kn)
            msk = cv2.undistort(msk, K, dist, None, Kn)
            fx, fy, R, C = Kn[0, 0], Kn[1, 1], c2w[i][:3, :3].copy(), c2w[i][:3, 3]
            # rotate in 90 deg steps (counter-clockwise) until world-up points up in the image
            best_k = max(range(4), key=lambda k: -(np.linalg.matrix_power(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), k).T
                                                   @ R.T @ up)[1])
            for _ in range(best_k):
                img, msk = np.rot90(img), np.rot90(msk)
                R = np.stack([R[:, 1], -R[:, 0], R[:, 2]], 1)  # rot90 CCW: x' = y, y' = -x
                fx, fy = fy, fx
            role = "train" if i == front_i else "eval"
            n = f"{i:04d}_{role}"
            cv2.imwrite(str(out / "images" / "undistorted" / f"{n}.png"), np.ascontiguousarray(img))
            cv2.imwrite(str(out / "images" / "mask" / f"{n}.png.png"), np.ascontiguousarray((msk > 127).astype(np.uint8) * 255))
            cams.append({"id": i, "img_name": n, "width": int(img.shape[1]), "height": int(img.shape[0]),
                         "position": C.tolist(), "rotation": R.tolist(), "fx": fx, "fy": fy, "elevation": float(elev[i])})
        (out / "plantdreamer" / "3dgs" / "cameras.json").write_text(json.dumps(cams, indent=1))
        (out / "front.txt").write_text(f"{front_i:04d}_train\n")
        print(f"bb_{plant}: {len(cams)} views, input {front_i:04d} at {elev[front_i]:.0f} deg elevation", flush=True)


if __name__ == "__main__":
    main()
