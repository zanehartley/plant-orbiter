"""Height robustness check: measure each reconstruction's height from its OWN base instead of the
reference plant base used in the paper tables (traits_planest.py / growsplat_heights.py measure every method above the
reference cloud's stem foot, so only the plant top is compared).

Own base (two rules, see own_height: 'rim' = pot rim, the one to use; 'low' = lowest non-pot point): remove the reconstruction's points that project into the input photo's pot mask (traits_planest.pot_filter,
images/pot_traits/<front>.png; same cut for every method), then base = 0.5th percentile of the remaining points along
world up (the soil line / stem foot as the reconstruction places it). Height = traits() 99.5th percentile above that base
(5 mm voxels), exactly as in the tables. The reference keeps its true height (reference base). Input-view placement.

    PLANT_DATA=... python heights_ownbase.py planest <outputs/pl> -o heights_ownbase_planest.json
    PLANT_DATA=... python heights_ownbase.py growsplat <outputs/gs> -o heights_ownbase_gs.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from traits import DATA, mesh_points, traits
import traits_planest as tp
import growsplat_heights as gh

SEEDS = (42, 1, 2)


def own_height(p_world, keep, to_cm, up):
    """-> (height, base) for two own-base rules: 'low' = 0.5th pct of the non-pot points (10-23 v1: unreliable, the
    input-view pot mask also removes the stem in front of the pot and leaves pot remnants), 'rim' = 95th pct of the
    pot points (the pot rim / soil line as the reconstruction places it; pot points = those projecting into the mask)."""
    kept = keep(p_world)
    q = kept * to_cm
    if len(q) < 100:
        return None
    kset = {tuple(r) for r in np.round(kept, 7)}
    pot = np.array([tuple(r) not in kset for r in np.round(p_world, 7)])
    out = {}
    base = float(np.percentile(q @ up, 0.5))
    out["low"] = (traits(q, up, base)["height_cm"], base)
    if pot.sum() >= 50:
        pw = p_world[pot] * to_cm
        base = float(np.percentile(pw @ up, 95))
        out["rim"] = (traits(q, up, base)["height_cm"], base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["planest", "growsplat"])
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    if a.dataset == "planest":
        tp.MESH = "2dgs_v3"
        plants = sorted(DATA.glob("pl_pepper*")) + sorted(DATA.glob("pl_rose*"))
    else:
        plants = sorted(DATA.glob("gs_quinoa_*")) + sorted(DATA.glob("gs_sequoia_*"))
    for D in plants:
        plant = D.name
        front = (D / "front.txt").read_text().strip()
        stem = f"{plant}_{front}"
        if a.dataset == "planest":
            if plant in tp.EXCLUDE or not (D / "data" / f"{plant}_real_pointcloud.ply").exists():
                continue
            to_cm = 1.0 / json.loads((D / "data" / "gt_registration.json").read_text())["scale_cm_to_world"]
        else:
            if not (a.root / "zoom" / f"{stem}_s42").exists():
                continue
            to_cm = 100.0
        up = np.array(json.loads((D / "world.json").read_text())["up"]) if (D / "world.json").exists() else np.array([0, 0, 1.0])
        up = up / np.linalg.norm(up)
        ref = np.asarray(o3d.io.read_point_cloud(str(D / "data" / f"{plant}_real_pointcloud.ply")).points) * to_cm
        rbase = float(np.percentile(ref @ up, 0.2))
        row = {"reference": traits(ref, up, rbase)["height_cm"], "reference_base_cm": rbase}
        keep = tp.pot_filter(D, front)
        if a.dataset == "planest":
            for s in SEEDS:
                for name, (mesh, sim, g) in tp.methods(a.root, stem, s, "input"):
                    if Path(mesh).exists() and Path(sim).exists():
                        p = mesh_points(mesh, sim)
                        h = own_height(p, keep, to_cm, up) if p is not None else None
                        if h:
                            row[f"{name}|s{s}"] = {k: {"height_cm": v[0], "base_cm": v[1]} for k, v in h.items()}
        else:
            M = dict(gh.METHODS)
            M["Ours"] = ("zoom/{stem}_s42/3dgs/2dgs_v3/mesh.ply", "zoom/{stem}_s42/3dgs/colmap2real_{mode}.json")
            for name, (mp, sp) in M.items():
                mesh, sim = a.root / mp.format(stem=stem), a.root / sp.format(stem=stem, mode="input")
                if mesh.exists() and sim.exists():
                    p = mesh_points(mesh, sim)
                    h = own_height(p, keep, to_cm, up) if p is not None else None
                    if h:
                        row[f"{name}|s42"] = {k: {"height_cm": v[0], "base_cm": v[1]} for k, v in h.items()}
        res[plant] = row
        print(plant, f"ref {row['reference']:.1f}", " ".join(f"{k} {v.get('rim', v['low'])['height_cm']:.1f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
