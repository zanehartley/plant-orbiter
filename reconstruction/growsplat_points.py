"""Dump the points behind growsplat_heights.py for the growth visualisation: per date and method, 20k
surface samples in the real frame (cm, chosen placement), plus the reference cloud, into one npz per species.

    python growsplat_points.py <outputs/gs dir> --mode input -o gs_points_<species>.npz --species quinoa
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from growsplat_heights import METHODS
from traits import DATA, mesh_points

ap = argparse.ArgumentParser()
ap.add_argument("gs", type=Path)
ap.add_argument("--mode", default="input", choices=["oracle", "input"])
ap.add_argument("--species", default="quinoa")
ap.add_argument("-o", "--out", type=Path, required=True)
a = ap.parse_args()
rng = np.random.default_rng(0)
out = {}
for D in sorted(DATA.glob(f"gs_{a.species}_*")):
    stem = f"{D.name}_{(D / 'front.txt').read_text().strip()}"
    if not (a.gs / "final" / f"{stem}_s42").exists():
        continue
    ref = np.asarray(o3d.io.read_point_cloud(str(D / "data" / f"{D.name}_real_pointcloud.ply")).points) * 100.0
    out[f"{D.name}|reference"] = ref[rng.choice(len(ref), min(20000, len(ref)), replace=False)].astype(np.float32)
    for m, (mp, sp3) in METHODS.items():
        mesh, sim = a.gs / mp.format(stem=stem), a.gs / sp3.format(stem=stem, mode=a.mode)
        if mesh.exists() and sim.exists():
            p = mesh_points(mesh, sim)
            if p is not None:
                out[f"{D.name}|{m}"] = (p[rng.choice(len(p), 20000, replace=False)] * 100.0).astype(np.float32)
    print(D.name, flush=True)
np.savez_compressed(a.out, **out)
