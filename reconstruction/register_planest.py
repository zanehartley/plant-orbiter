"""Register PLANesT-3D annotated point clouds (plant only, no pot; cm, z up, stem base at the origin) into the camera
frame of our converted scans (Metashape chunk units), so they can serve as 3D ground truth (run in "nerfstudio").

Similarity with a known up axis (cloud +z -> world.json up): unknowns yaw, scale, translation (+ a small tilt).
  1. plant-only masks per view = scoring mask minus the pot mask (make_planest_masks.py); close-ups that crop the plant
     are fine because only in-image pixels count
  2. init: scale and base from a coarse plant-only visual hull carved from those masks (a voxel survives if every view
     that sees it has it inside the mask, with 10% slack)
  3. yaw sweep every 10 deg, then Nelder-Mead on (yaw, tilt x2, log scale, translation) maximising the mean silhouette
     IoU between the projected cloud and the plant-only masks over all masked views
Writes data/converted/plants/<pl_plant>/data/<pl_plant>_real_pointcloud.ply (camera frame, xyz + rgb),
data/.../<pl_plant>_labels.npz (semantic + instance labels, same order), gt_registration.json and a QA sheet in
presentation/checks/<date>_planest_gt_registration/. 3D scoring is then plant only (the reference has no pot).

    python register_planest.py pl_pepper01 [...] [--wide-scale] [--tilt-sweep]   (wider scale / tilted starts where the default fit fails)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy.optimize import minimize

import figs
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
DATA = DATASETS / "converted" / "plants"
SRC = DATASETS / "PLANesT-3D" / "pointclouds_extracted"
DS = 4  # mask downsampling


def src_ply(plant):  # pl_pepper01 -> Pepper/Pepper_01.ply
    sp, n = plant[3:].rstrip("0123456789"), plant[3:].lstrip("abcdefghijklmnopqrstuvwxyz")
    return SRC / sp.capitalize() / f"{sp.capitalize()}_{int(n):02d}.ply"


def rot_from_up(u):
    """Rotation taking +z to u."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    c = z @ u
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def axis_rot(a, th):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


WIDE = "--wide-scale" in sys.argv
TILT = "--tilt-sweep" in sys.argv


def main():
    out_dir = figs.out_dir("checks", "planest_gt_registration")
    for plant in [x for x in sys.argv[1:] if not x.startswith("--")]:
        D = DATA / plant
        w = json.loads((D / "world.json").read_text())
        up, target = np.array(w["up"]), np.array(w["target"])
        cams = json.loads((D / "plantdreamer" / "3dgs" / "cameras.json").read_text())
        views = []
        for c in cams:
            mp = D / "images" / "mask" / f"{c['img_name']}.png.png"
            if not mp.exists():
                continue
            m = np.asarray(Image.open(mp).convert("L")) > 0
            pp = D / "images" / "pot" / f"{c['img_name']}.png"
            if pp.exists():
                pot = np.asarray(Image.open(pp).convert("L")) > 0
                if pot.shape == m.shape:
                    m = m & ~pot
            h, wd = m.shape[0] // DS, m.shape[1] // DS
            m = m[: h * DS, : wd * DS].reshape(h, DS, wd, DS).any(axis=(1, 3))  # max-pool: thin stems survive
            R = np.array(c["rotation"])
            views.append({"name": c["img_name"], "w2c_R": R.T, "C": np.array(c["position"]), "f": c["fx"] / DS,
                          "fy": c["fy"] / DS, "w": wd, "h": h, "m": torch.tensor(m, device="cuda")})
        dev = "cuda"
        Rw = torch.tensor(np.stack([v["w2c_R"] for v in views]), dtype=torch.float32, device=dev)
        Cw = torch.tensor(np.stack([v["C"] for v in views]), dtype=torch.float32, device=dev)

        fs = torch.tensor([[v["f"], v["fy"], v["w"] / 2, v["h"] / 2] for v in views], device=dev)

        def sil_iou(P):
            pc = torch.einsum("vij,vnj->vni", Rw, P[None] - Cw[:, None])
            z = pc[..., 2]
            u = (pc[..., 0] / z.clamp(min=1e-6) * fs[:, None, 0] + fs[:, None, 2]).round().long()
            v_ = (pc[..., 1] / z.clamp(min=1e-6) * fs[:, None, 1] + fs[:, None, 3]).round().long()
            ious = []
            for k, vw in enumerate(views):
                ok = (z[k] > 0) & (u[k] >= 0) & (u[k] < vw["w"]) & (v_[k] >= 0) & (v_[k] < vw["h"])
                img = torch.zeros(vw["h"], vw["w"], dtype=torch.bool, device=dev)
                img[v_[k][ok], u[k][ok]] = True
                img = torch.nn.functional.max_pool2d(img[None, None].float(), 5, 1, 2)[0, 0] > 0
                m = vw["m"]
                inter, union = (img & m).sum(), (img | m).sum()
                if union > 0 and m.sum() > 20:
                    ious.append(float(inter) / float(union))
            return float(np.mean(ious)) if ious else 0.0

        # 1. init from every view that shows the whole plant (plant-only mask clear of the image edges): plant height =
        #    mask height / f x distance to the plant centre, base = the mask's lowest point back-projected onto the
        #    vertical line through the plant centre; medians over views. (A strict visual hull fails on these thin
        #    hand-held captures; the input photo alone gave wrong scales on some plants, e.g. pepper06.)
        hs, bases = [], []
        for vw in views:
            mk = vw["m"].cpu().numpy()
            ys, xs = np.where(mk)
            if len(ys) < 50 or ys.min() <= 1 or ys.max() >= vw["h"] - 2 or xs.min() <= 1 or xs.max() >= vw["w"] - 2:
                continue
            Cv = vw["C"]
            dv = np.linalg.norm(Cv - target)
            hs.append((ys.max() - ys.min()) / vw["fy"] * dv)
            Rc = vw["w2c_R"].T
            ray = Rc @ np.array([(xs[ys == ys.max()].mean() - vw["w"] / 2) / vw["f"], (ys.max() - vw["h"] / 2) / vw["fy"], 1.0])
            k = np.linalg.lstsq(np.stack([ray, -up], 1), target - Cv, rcond=None)[0]
            bases.append(target + k[1] * up)
        h_hull = float(np.median(hs))
        base = np.median(np.array(bases), 0)
        n_full = len(hs)
        # 2. ground truth (cm, z up, base at origin)
        v = PlyData.read(str(src_ply(plant)))["vertex"]
        X0 = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
        sem, ins = np.asarray(v["scalar_semanticLabel"]), np.asarray(v["scalar_instanceLabel"])
        sub = X0[np.random.default_rng(0).choice(len(X0), min(120000, len(X0)), replace=False)]
        h_gt = np.percentile(X0[:, 2], 99) - np.percentile(X0[:, 2], 1)
        Rup = rot_from_up(up)
        e1 = np.linalg.svd(np.eye(3) - np.outer(up, up))[0][:, 0]
        e2 = np.cross(up, e1)

        def transform(p, Xs):
            yaw, tx1, tx2, ls, dx, dy, dz = p
            R = axis_rot(up, yaw) @ axis_rot(e1, tx1) @ axis_rot(e2, tx2) @ Rup
            s = np.exp(ls) * h_hull / h_gt
            return Xs @ (s * R).T + base + np.array([dx, dy, dz]) * h_hull

        def loss(p):
            return -sil_iou(torch.tensor(transform(p, sub), dtype=torch.float32, device=dev))

        # --wide-scale: the init height can be off by more than the 0.8-1.25 sweep (pl_pepper09: scale 0.024 vs ~0.06 for
        # the other peppers, IoU .11, yaw 147 deg vs ~270 for all others)
        scales = [0.4, 0.55, 0.8, 1.0, 1.25, 1.8, 2.5, 3.5] if WIDE else [0.8, 1.0, 1.25]
        # --tilt-sweep: also start from tilted clouds (the world up from a hand-held capture can be off; pl_pepper09)
        tilts = [(0, 0)] + ([(a, b) for a in (-30, -15, 0, 15, 30) for b in (-30, -15, 0, 15, 30) if (a, b) != (0, 0)]
                            if TILT else [])
        sweep = [(loss([np.radians(y), np.radians(a), np.radians(b), ls, 0, 0, 0]), y, ls, a, b)
                 for y in range(0, 360, 10) for ls in np.log(scales) for a, b in tilts]
        best0 = min(sweep)
        cands = sorted(sweep)[:4]
        best = None
        for _, y, ls, a, b in cands:
            r = minimize(loss, [np.radians(y), np.radians(a), np.radians(b), ls, 0, 0, 0], method="Nelder-Mead",
                         options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        iou = -best.fun
        Xw = transform(p, X0)
        dd = D / "data"
        dd.mkdir(exist_ok=True)
        from plyfile import PlyElement
        arr = np.empty(len(Xw), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = Xw.T
        arr["red"], arr["green"], arr["blue"] = rgb.T
        from plyfile import PlyData as PD
        PD([PlyElement.describe(arr, "vertex")], text=False).write(str(dd / f"{plant}_real_pointcloud.ply"))
        np.savez_compressed(dd / f"{plant}_labels.npz", semantic=sem.astype(np.int16), instance=ins.astype(np.int16))
        s = float(np.exp(p[3]) * h_hull / h_gt)
        (dd / "gt_registration.json").write_text(json.dumps({
            "source": str(src_ply(plant)), "units_source": "cm", "scale_cm_to_world": s, "yaw_deg": float(np.degrees(p[0])),
            "tilt_deg": [float(np.degrees(p[1])), float(np.degrees(p[2]))], "mean_sil_iou": iou,
            "yaw_sweep_best_iou": -best0[0], "n_views": len(views), "n_full_views_for_init": n_full, "plant_height_cm": float(h_gt)}, indent=1))
        # QA: overlay projected cloud (magenta) on 4 views' plant masks
        tiles = []
        for vw in [views[i] for i in np.linspace(0, len(views) - 1, 4).round().astype(int)]:
            P = torch.tensor(transform(p, sub), dtype=torch.float32, device=dev)
            pc = (torch.tensor(vw["w2c_R"], dtype=torch.float32, device=dev) @ (P - torch.tensor(vw["C"], dtype=torch.float32, device=dev)).T).T
            z = pc[:, 2]
            u = (pc[:, 0] / z * vw["f"] + vw["w"] / 2).round().long()
            v2 = (pc[:, 1] / z * vw["fy"] + vw["h"] / 2).round().long()
            ok = (z > 0) & (u >= 0) & (u < vw["w"]) & (v2 >= 0) & (v2 < vw["h"])
            im = np.where(vw["m"].cpu().numpy()[..., None], 200, 255).astype(np.uint8).repeat(3, 2)
            im[v2[ok].cpu().numpy(), u[ok].cpu().numpy()] = (200, 0, 200)
            tiles.append(np.asarray(Image.fromarray(im).resize((360, round(360 * vw["h"] / vw["w"])))))
        hmax = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in tiles]
        Image.fromarray(np.concatenate(tiles, 1)).save(out_dir / f"{plant}.jpg", quality=85)
        print(f"{plant}: mean silhouette IoU {iou:.3f} (yaw sweep {-best0[0]:.3f}), yaw {np.degrees(p[0]) % 360:.0f} deg, "
              f"scale {s:.4f} units/cm, {len(views)} views", flush=True)


if __name__ == "__main__":
    main()
