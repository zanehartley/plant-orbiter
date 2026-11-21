"""Full-capture upper bound: splatfacto on a plant's REAL photos, scored on exactly the test views
of the single-photo tables (same masks, same metrics, raw + colour-corrected on the input photo).

Training views = every real photo that is not a test view (test = `_eval` views that have a mask, as ev.score uses):
  PlantDreamer: its `_train` photos with a hybrid tight mask (views without one are skipped)
  PLANesT-3D:   the ~138 photos without a protocol mask; their mask = images/rembg (plant + sometimes stand/pot)
  wheat:        <plant>/fullcap/ (convert_wheat.py --fullcap): all 256 non-test photos of the 320 + dataset masks
  Botany-Bot:   every photo is already a test view -> 2-fold: train on the input + one half, test on the other
                (--fold 0/1; the two folds together cover every test view)
The input photo is always a training view.

    [PLANT_DATA=data/converted/plants] python fullcap_splat.py bb_croton [--fold 0] [--out-root outputs/fullcap]
Writes <out-root>/<plant>[/fold<k>]/{transforms.json, images/, run/, export/splat.ply, score.json, score_cc.json}.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import eval_real_views as ev
from po_paths import OUTPUTS

ROOT = Path(__file__).resolve().parent
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
MIN_TRAIN = 20  # fewer non-test photos than this -> 2-fold over the test views


def hull_seed(cams_masks, res=160, keep=0.8, min_views=5, down=4):
    """Visual-hull seed points (world frame) from the training photos' masks, as known_poses.py seeds our orbit splats.
    Random initialisation collapsed on ~half the plants (every Gaussian transparent, IoU 0). A voxel
    survives if, among the views it projects into, >= keep have it inside the mask (close-ups that crop the plant
    don't veto it). Returns (points, colours 0-255)."""
    import torch
    C = np.stack([c["c2w"][:3, 3] for c, _, _ in cams_masks])
    Z = np.stack([c["c2w"][:3, 2] for c, _, _ in cams_masks])
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, z in zip(C, Z):  # point closest to every optical axis = what the capture looks at
        P = np.eye(3) - np.outer(z, z)
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)
    half = 0.6 * np.median(np.linalg.norm(C - target, axis=1))
    g = torch.linspace(-half, half, res, device="cuda", dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + torch.tensor(target, device="cuda")
    seen = torch.zeros(len(pts), device="cuda")
    hit = torch.zeros(len(pts), device="cuda")
    col = torch.zeros(len(pts), 3, device="cuda")
    for c, rgb, m in cams_masks:
        h, w = m.shape
        w2c = torch.tensor(np.linalg.inv(c["c2w"]), device="cuda")
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2].clamp(min=1e-6)
        u = (c["fx"] * pc[:, 0] / z + c["w"] / 2) / down
        v = (c["fy"] * pc[:, 1] / z + c["h"] / 2) / down
        ui, vi = u.long(), v.long()
        ins = (pc[:, 2] > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        mm = torch.tensor(m, device="cuda")[vi.clamp(0, h - 1), ui.clamp(0, w - 1)] & ins
        seen += ins.float()
        hit += mm.float()
        col += torch.tensor(rgb, device="cuda", dtype=torch.float32)[vi.clamp(0, h - 1), ui.clamp(0, w - 1)] * mm[:, None]
    occ = (seen >= min_views) & (hit >= keep * seen)
    p = pts[occ].cpu().numpy()
    cl = (col[occ] / hit[occ, None].clamp(min=1)).clamp(0, 255).cpu().numpy().astype(np.uint8)
    if len(p) > 200000:
        i = np.random.default_rng(0).choice(len(p), 200000, replace=False)
        p, cl = p[i], cl[i]
    return p, cl


def write_ply(path, p, cl):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(p, cl):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("--fold", type=int, help="2-fold split for datasets whose photos are all test views (Botany-Bot)")
    ap.add_argument("--out-root", type=Path, default=OUTPUTS / "fullcap")
    ap.add_argument("--no-seed", action="store_true", help="random init (v1 behaviour; collapsed on ~half the plants)")
    ap.add_argument("--all", action="store_true", help="train on EVERY photo incl. test views (figures only; not scored) -> <plant>/all")
    ap.add_argument("--prep-only", action="store_true", help="write transforms.json + images, no training (check the split)")
    a = ap.parse_args()
    D = ev.DATA / a.plant
    from batch_single_photo import front_view  # front.txt, or PlantDreamer's ring logic (as the single-photo runs)
    front = front_view(D)
    cams = ev.cameras(a.plant, "all")
    test = [c for c in cams if c["name"].endswith("_eval") and ev.mask_path(a.plant, c["name"]).exists()]
    test_names = {c["name"] for c in test}

    # training pool: (camera, image path, mask path)
    pool = []
    fc = D / "fullcap" / "cameras.json"
    if fc.exists():
        for c in json.loads(fc.read_text()):
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = np.array(c["rotation"]), c["position"]
            pool.append(({"name": c["img_name"], "c2w": c2w, "fx": c["fx"], "fy": c["fy"], "w": c["width"], "h": c["height"]},
                         D / "fullcap" / "images" / f"{c['img_name']}.png", D / "fullcap" / "mask" / f"{c['img_name']}.png"))
    else:
        for c in cams:
            if c["name"] in test_names:
                continue
            m = ev.mask_path(a.plant, c["name"])
            if not m.exists():
                m = D / "images" / "rembg" / f"{c['name']}.png"
            if m.exists():
                pool.append((c, D / "images" / "undistorted" / f"{c['name']}.png", m))
    out = a.out_root / a.plant
    if a.all:  # every real photo, no held-out views (for figures only)
        pool += [(c, D / "images" / "undistorted" / f"{c['name']}.png", ev.mask_path(a.plant, c["name"])) for c in test]
        test = []
        out = out / "all"
    elif len(pool) < MIN_TRAIN:
        assert a.fold in (0, 1), f"{a.plant}: only {len(pool)} non-test photos -> pass --fold 0 or 1"
        held = test[a.fold::2]
        pool += [(c, D / "images" / "undistorted" / f"{c['name']}.png", ev.mask_path(a.plant, c["name"]))
                 for c in test if c not in held]
        test = held
        out = out / f"fold{a.fold}"
    if not fc.exists() and front not in {c["name"] for c, _, _ in pool}:  # (wheat's fullcap/ already has it)
        fc_ = next(c for c in cams if c["name"] == front)
        pool.append((fc_, D / "images" / "undistorted" / f"{front}.png", ev.mask_path(a.plant, front)))

    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, seedv = [], []
    for c, ip, mp in pool:
        rgb = np.asarray(Image.open(ip).convert("RGB"))
        m = np.asarray(Image.open(mp).convert("L"))
        if m.shape != rgb.shape[:2]:
            continue
        f = out / "images" / f"{c['name']}.png"
        if not f.exists():
            Image.fromarray(np.dstack([rgb, np.where(m > 127, 255, 0).astype(np.uint8)])).save(f)
        seedv.append((c, rgb[2::4, 2::4], m[2::4, 2::4] > 127))
        frames.append({"file_path": f"images/{c['name']}.png", "transform_matrix": (c["c2w"] @ GL2CV).tolist(),
                       "fl_x": c["fx"], "fl_y": c["fy"], "cx": c["w"] / 2, "cy": c["h"] / 2, "w": c["w"], "h": c["h"]})
    meta_t = {"frames": frames}
    if not a.no_seed:
        if not (out / "hull.ply").exists():
            hp, hc = hull_seed(seedv)
            if len(hp) < 500:  # inconsistent masks (PLANesT pepper07: 371 points) -> looser vote, else no seed
                hp, hc = hull_seed(seedv, keep=0.6)
                print(f"{a.plant}: hull too thin at keep 0.8, keep 0.6 -> {len(hp)} points", flush=True)
            if len(hp) >= 500:
                write_ply(out / "hull.ply", hp, hc)
            print(f"{a.plant}: hull seed {len(hp)} points", flush=True)
        if (out / "hull.ply").exists():
            meta_t["ply_file_path"] = "hull.ply"
    (out / "transforms.json").write_text(json.dumps(meta_t, indent=1))
    print(f"{a.plant}{'' if a.fold is None else f' fold {a.fold}'}: train {len(frames)} photos, test {len(test)}", flush=True)

    if a.prep_only:
        return
    ns = ["conda", "run", "--no-capture-output", "-n", "nerfstudio"]
    run = out / "run"
    if not list(run.glob("*/splatfacto/v1/config.yml")):
        subprocess.run(ns + ["ns-train", "splatfacto", "--output-dir", str(run), "--experiment-name", a.plant, "--timestamp", "v1",
                             "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True",
                             "--pipeline.model.background-color", "random",
                             # FC_EXTRA (final recipe): e.g. antialiased + scale regularisation, ratio 3;
                             # score with the matching rasteriser via EV_RASTER
                             *os.environ.get("FC_EXTRA", "").split(),
                             "nerfstudio-data", "--data", str(out), "--orientation-method", "none", "--center-method", "poses",
                             "--auto-scale-poses", "True", "--eval-mode", "all",
                             "--load-3D-points", str("ply_file_path" in meta_t)], check=True,
                       stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
    cfg = next(run.glob("*/splatfacto/v1/config.yml"))
    if not (out / "export" / "splat.ply").exists():
        subprocess.run(ns + ["ns-export", "gaussian-splat", "--load-config", str(cfg), "--output-dir", str(out / "export")],
                       check=True, stdout=open(out / "export.log", "w"), stderr=subprocess.STDOUT)
    if a.all:
        print(a.plant, "all photos:", len(frames), "->", out / "export" / "splat.ply", flush=True)
        return
    d = json.loads((cfg.parent / "dataparser_transforms.json").read_text())
    T, sc = np.array(d["transform"]), d["scale"]
    R0 = T[:, :3].T
    sim3 = {"s": 1 / sc, "R": R0.tolist(), "t": (-R0 @ T[:, 3]).tolist()}
    import torch
    gs = ev.load_ply(out / "export" / "splat.ply")
    with torch.no_grad():
        mean, rows = ev.score(a.plant, gs, test, sim3)
        col = ev.fit_colour(a.plant, gs, [c for c in cams if c["name"] == front], sim3)
        mean_cc, rows_cc = ev.score(a.plant, gs, test, sim3, col)
    meta = {"n_train": len(frames), "n_test": len(test), "input": front, "fold": a.fold}
    (out / "score.json").write_text(json.dumps({"mean": mean, "views": rows, **meta}, indent=1))
    (out / "score_cc.json").write_text(json.dumps({"mean": mean_cc, "views": rows_cc, "colour": col.tolist(), **meta}, indent=1))
    print(a.plant, "full capture:", {k: round(v, 3) for k, v in mean_cc.items()}, meta, flush=True)


if __name__ == "__main__":
    main()
