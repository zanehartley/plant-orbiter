"""PLANesT traits for the paper. Every single-photo method, seeds 42/1/2, measured with traits.traits() (5 mm voxel height/width, 3 mm voxel
area; pot-free: everything below the reference stem foot dropped) under BOTH placements:
  input   the placement fitted to the input photo only (real camera distance sets the scale)
  oracle  fitted to all test views (silhouette fit)
Meshes (as in Table 3): ours / SEVA / SV3D 2DGS mesh (3dgs/2dgs_v2/mesh.ply, COLMAP frame) + colmap2real_<mode>.json
(made here from align_<mode>.json when missing); TRELLIS mesh_gsframe.ply, TRELLIS.2 mesh.glb, Hunyuan mesh.obj
+ align_<mode>.json. Reference: registered PLANesT cloud (cm via gt_registration.json), leaf area from the
leaf-labelled points (semantic 2). pl_pepper09 excluded (reference cannot be registered).
--pot-cut (reconstructions keep the pot, the reference has none, so its surface was
counted as leaf area): drop every reconstruction point that projects into the INPUT photo's pot mask (images/pot/<front>.png,
dilated 5 px). A pot is solid, so all of its 3D points fall inside its silhouette in that view; leaves in front of the pot
are not in the pot mask and survive. Uses the input photo only (no reference), same for every method.
Pot masks come from traits_pot_masks.py (images/pot_traits). Where the pot reaches the bottom edge of the photo, points projecting below the frame inside the
pot's columns (+10% each side) are cut too (pepper pots partly out of frame).

    PLANT_DATA=... python traits_planest.py <outputs/pl> -o traits_planest.json [--pot-cut] [--mesh 2dgs_v3]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import open3d as o3d

from traits import DATA, VOX_CM, mesh_points, traits

EXCLUDE = {"pl_pepper09"}
SEEDS = (42, 1, 2)
sfx = lambda s: "" if s == 42 else f"_seed{s}"


MESH = "2dgs_v2"


def methods(pl, stem, s, mode):
    g = lambda d: (d / MESH / "mesh.ply", d / f"colmap2real_{mode}.json", d)
    yield "Ours", g(pl / "final" / f"{stem}_s{s}" / "3dgs")
    yield "SEVA", g(pl / "baselines" / "seva" / f"{stem}{sfx(s)}" / "3dgs")
    yield "SV3D", g(pl / "baselines" / "sv3d" / f"{stem}{sfx(s)}" / "3dgs")
    for name, sub, mesh in (("TRELLIS", "trellis", "mesh_gsframe.ply"), ("TRELLIS.2", "trellis2", "mesh.glb"),
                            ("Hunyuan3D 2.1", "hunyuan", "mesh.obj")):
        d = pl / "baselines" / sub / f"{stem}{sfx(s)}"
        yield name, (d / mesh, d / f"align_{mode}.json", None)


def pot_filter(D, front):
    """Function: world-unit points -> points not projecting into the input photo's pot mask."""
    pm = D / "images" / "pot_traits" / f"{front}.png"
    if not pm.exists():
        return lambda p: p
    pot = binary_dilation(np.asarray(Image.open(pm).convert("L")) > 0, iterations=5)
    h, w = pot.shape
    rows = np.nonzero(pot.any(1))[0]
    # pot cut off by the frame: its lowest row lies in the bottom 4% (undistortion leaves a black band at the very bottom)
    cols = np.nonzero(pot[rows.max()])[0] if len(rows) and rows.max() > 0.96 * h else []
    if len(cols):
        y_end = rows.max()
        pad = 0.1 * (cols.max() - cols.min())
        below = (cols.min() - pad, cols.max() + pad)
    c = next(c for c in json.loads((D / "plantdreamer" / "3dgs" / "cameras.json").read_text()) if c["img_name"] == front)
    R, C = np.array(c["rotation"]), np.array(c["position"])

    def keep(p):
        q = (p - C) @ R
        z = q[:, 2]
        u = np.round(q[:, 0] / np.maximum(z, 1e-9) * c["fx"] + w / 2).astype(int)
        v = np.round(q[:, 1] / np.maximum(z, 1e-9) * c["fy"] + h / 2).astype(int)
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = np.zeros(len(p), bool)
        hit[inside] = pot[v[inside], u[inside]]
        if len(cols):
            hit |= (z > 0) & (v >= y_end) & (u >= below[0]) & (u <= below[1])
        return p[~hit]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pl", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--pot-cut", action="store_true", help="remove reconstruction points inside the input photo's pot mask")
    ap.add_argument("--mesh", default="2dgs_v2", help="2DGS mesh dir for ours/SEVA/SV3D (the paper uses 2dgs_v3)")
    a = ap.parse_args()
    global MESH
    MESH = a.mesh
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    for D in sorted(DATA.glob("pl_pepper*")) + sorted(DATA.glob("pl_rose*")):
        plant = D.name
        if plant in EXCLUDE or not (D / "data" / f"{plant}_real_pointcloud.ply").exists():
            continue
        front = (D / 'front.txt').read_text().strip()
        stem = f"{plant}_{front}"
        cut = pot_filter(D, front) if a.pot_cut else (lambda p: p)
        up = np.array(json.loads((D / "world.json").read_text())["up"]) if (D / "world.json").exists() else np.array([0, 0, 1.0])
        up /= np.linalg.norm(up)
        ref = np.asarray(o3d.io.read_point_cloud(str(D / "data" / f"{plant}_real_pointcloud.ply")).points)
        to_cm = 1.0 / json.loads((D / "data" / "gt_registration.json").read_text())["scale_cm_to_world"]
        base = float(np.percentile(ref @ up, 0.2) * to_cm)
        row = res.get(plant, {})
        row["base_cm"], row["reference"] = base, traits(ref * to_cm, up, base)
        lab = np.load(D / "data" / f"{plant}_labels.npz")
        leaf = ref[lab["semantic"] == 2] * to_cm
        row["reference"]["leaf_area_cm2"] = float(len(np.unique(np.floor(leaf / VOX_CM).astype(np.int64), axis=0)) * VOX_CM ** 2)
        for mode in ("input", "oracle"):
            for s in SEEDS:
                for name, (mesh, sim, g) in methods(a.pl, stem, s, mode):
                    key = f"{mode}|{name}|s{s}"
                    if key in row:
                        continue
                    if g is not None and not sim.exists() and (g / f"align_{mode}.json").exists() and mesh.exists():
                        subprocess.run([sys.executable, "colmap2real.py", str(g), mode], check=True, stdout=subprocess.DEVNULL)
                    if mesh.exists() and sim.exists():
                        p = mesh_points(mesh, sim)
                        row[key] = traits(cut(p) * to_cm, up, base) if p is not None else None
                    else:
                        row[key] = None
        res[plant] = row
        a.out.write_text(json.dumps(res, indent=1))
        ok = sum(1 for k, v in row.items() if "|" in k and v)
        print(f"{plant}: ref h {row['reference']['height_cm']:.1f} cm, leaf {row['reference']['leaf_area_cm2']:.0f} cm2; {ok}/36 method runs", flush=True)


if __name__ == "__main__":
    main()
