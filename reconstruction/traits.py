"""Plant traits from every method's reconstruction vs the reference.

Datasets: Nottingham wheat (scanner reference, metres) and PLANesT-3D (annotated reference, registered into the camera
frame; units -> cm via gt_registration.json). Every method is measured the same way, in the real frame with its ORACLE
placement (oracle scale):
  ours     2DGS mesh (3dgs/2dgs_v2/mesh.ply) mapped by colmap2real_oracle.json
  TRELLIS  mesh_gsframe.ply (trellis_mesh_eval.py) mapped by align_oracle.json; seeds 42 and 1
  Hunyuan  mesh.obj mapped by align_oracle.json
Meshes are sampled uniformly (dense); the reference is used as is. The pot is excluded: everything below the plant base
is dropped (PLANesT: the reference's stem foot, so the pot is cut away). Wheat: the scan shows no clear pot rim
(gao_eval.rim_height fails; the authors' scan "length" is a bounding-box side), so wheat height is measured from the pot
bottom for every method and the reference alike (a shared ~13 cm pot offset).
Traits (cm): height = 99.5th percentile of height above the base (robust to floaters); width = largest horizontal
extent (1st-99th percentile along the best of 36 directions). Both are computed on points voxel-downsampled at 5 mm
(VOX_SHAPE_CM), for the reference and every method alike: percentiles depend on point density,
and the GrowSplat sequoia clouds are so dense in the lower foliage that the raw p99.5 sat ~5 cm below the top leader;
mesh samples are even by area, so without this every method looked too tall. hull volume (cm^3); surface (leaf) area = occupied
3 mm voxels x voxel face, which counts a thin leaf once whichever way its surface is sampled (cm^2). PLANesT also gives
the reference's leaf-only area (semanticLabel 2).

    python traits.py wh wh_bc1_1033_3 [...] --ours outputs/wh/ours --baselines outputs/wh/baselines -o outputs/traits_wh.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import ConvexHull
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLANT_DATA", DATASETS / "converted" / "plants"))
VOX_CM = 0.3
VOX_SHAPE_CM = 0.5  # 5 mm: density-equalising downsample before the height / width percentiles
N_SAMPLES = 400_000


def sim3_apply(p, j):
    t = json.loads(Path(j).read_text())
    return t["s"] * p @ np.array(t["R"]).T + np.array(t["t"])


def mesh_points(path, sim3):
    m = trimesh.load(str(path), force="mesh")
    if len(m.faces) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(m, N_SAMPLES, seed=0)
    return sim3_apply(np.asarray(pts), sim3)


def voxel_down(p, size):
    """One point (the centroid) per occupied voxel, as open3d's voxel_down_sample."""
    _, inv = np.unique(np.floor(p / size).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=p[:, k]) for k in range(3)], 1) / n[:, None]


def traits(pts_cm, up, base_cm):
    h = pts_cm @ up - base_cm
    p = pts_cm[h > 0]
    if len(p) < 100:
        return None
    q = voxel_down(p, VOX_SHAPE_CM)
    hq = q @ up - base_cm
    hh = p @ up - base_cm
    e1 = np.linalg.svd(np.eye(3) - np.outer(up, up))[0][:, 0]
    e2 = np.cross(up, e1)
    width = 0.0
    for th in np.radians(np.arange(0, 180, 5)):
        d = np.cos(th) * e1 + np.sin(th) * e2
        x = q @ d
        width = max(width, np.percentile(x, 99) - np.percentile(x, 1))
    keep = hh <= np.percentile(hq, 99.5)
    try:
        vol = ConvexHull(p[keep]).volume
    except Exception:
        vol = float("nan")
    vox = np.unique(np.floor(p[keep] / VOX_CM).astype(np.int64), axis=0)
    return {"height_cm": float(np.percentile(hq, 99.5)), "width_cm": float(width), "hull_volume_cm3": float(vol),
            "surface_area_cm2": float(len(vox) * VOX_CM ** 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=["wh", "pl"])
    ap.add_argument("plants", nargs="+")
    ap.add_argument("--ours", type=Path, required=True, help="dir with <plant>_<front>_s<seed> runs")
    ap.add_argument("--baselines", type=Path, help="dir with trellis/ and hunyuan/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    for plant in a.plants:
        D = DATA / plant
        front = (D / "front.txt").read_text().strip()
        stem = f"{plant}_{front}"
        up = np.array(json.loads((D / "world.json").read_text())["up"]) if (D / "world.json").exists() else np.array([0, 0, 1.0])
        up /= np.linalg.norm(up)
        ref = np.asarray(o3d.io.read_point_cloud(str(D / "data" / f"{plant}_real_pointcloud.ply")).points)
        to_cm = 1.0 / json.loads((D / "data" / "gt_registration.json").read_text())["scale_cm_to_world"] if a.ds == "pl" else 100.0
        base = np.percentile(ref @ up, 0.2) * to_cm  # PLANesT: stem foot (no pot); wheat: pot bottom (pot included)
        row = {"base_cm": float(base), "reference": traits(ref * to_cm, up, base)}
        if a.ds == "pl":
            lab = np.load(D / "data" / f"{plant}_labels.npz")
            leaf = ref[lab["semantic"] == 2] * to_cm
            vox = np.unique(np.floor(leaf / VOX_CM).astype(np.int64), axis=0)
            row["reference"]["leaf_area_cm2"] = float(len(vox) * VOX_CM ** 2)
        methods = {}
        for seed in a.seeds:
            g = a.ours / f"{stem}_s{seed}" / "3dgs"
            if (g / "2dgs_v2" / "mesh.ply").exists() and (g / "colmap2real_oracle.json").exists():
                methods[f"ours_s{seed}"] = (g / "2dgs_v2" / "mesh.ply", g / "colmap2real_oracle.json")
        if a.baselines:
            for name, d in (("trellis_s42", a.baselines / "trellis" / stem), ("trellis_s1", a.baselines / "trellis" / f"{stem}_seed1"),
                            ("hunyuan", a.baselines / "hunyuan" / stem)):
                mesh = d / ("mesh_gsframe.ply" if name.startswith("trellis") else "mesh.obj")
                if mesh.exists() and (d / "align_oracle.json").exists():
                    methods[name] = (mesh, d / "align_oracle.json")
        for name, (mesh, sim) in methods.items():
            p = mesh_points(mesh, sim)
            row[name] = traits(p * to_cm, up, base) if p is not None else None
        res[plant] = row
        msg = " | ".join(f"{k} h {v['height_cm']:.1f} a {v['surface_area_cm2']:.0f}" for k, v in row.items()
                         if isinstance(v, dict) and v)
        print(f"{plant}: {msg}", flush=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
