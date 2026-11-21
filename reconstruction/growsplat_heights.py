"""GrowSplat growth case study: plant height on every date for every method vs that date's reference
cloud, measured as in traits.py (oracle or input-view placement (--mode), 99.5th percentile of height above the plant base, pot cut away).
The GrowSplat cloud is the plant only, in the rig's metric frame (metres), so the base is the cloud's stem foot
(0.2th percentile along world up) and everything a reconstruction puts below it (pot, soil) is dropped.
Meshes / placements are the ones the 3D table uses:
  ours       final/<stem>_s42/3dgs/2dgs_v2/mesh.ply   + colmap2real_<mode>.json
  TRELLIS    baselines/trellis/<stem>/mesh_gsframe.ply + align_<mode>.json
  TRELLIS.2  baselines/trellis2/<stem>/mesh.glb        + align_<mode>.json
  Hunyuan3D  baselines/hunyuan/<stem>/mesh.obj         + align_<mode>.json
  SEVA/SV3D  baselines/<m>/<stem>/3dgs/2dgs_v3/mesh.ply + 3dgs/colmap2real_<mode>.json
Dates without a mesh yet are skipped (rerun later: results for existing dates are kept unless --redo).

    python growsplat_heights.py <outputs/gs dir> -o growsplat_heights.json [--species quinoa sequoia]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from traits import DATA, mesh_points, traits

METHODS = {"Ours": ("final/{stem}_s42/3dgs/2dgs_v2/mesh.ply", "final/{stem}_s42/3dgs/colmap2real_{mode}.json"),
           "TRELLIS": ("baselines/trellis/{stem}/mesh_gsframe.ply", "baselines/trellis/{stem}/align_{mode}.json"),
           "TRELLIS.2": ("baselines/trellis2/{stem}/mesh.glb", "baselines/trellis2/{stem}/align_{mode}.json"),
           "Hunyuan3D 2.1": ("baselines/hunyuan/{stem}/mesh.obj", "baselines/hunyuan/{stem}/align_{mode}.json"),
           # SEVA / SV3D: same 1.3x crop as ours, 2DGS mesh with the 0.005 small-piece filter
           "SEVA": ("baselines/seva/{stem}/3dgs/2dgs_v3/mesh.ply", "baselines/seva/{stem}/3dgs/colmap2real_{mode}.json"),
           "SV3D": ("baselines/sv3d/{stem}/3dgs/2dgs_v3/mesh.ply", "baselines/sv3d/{stem}/3dgs/colmap2real_{mode}.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gs", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--species", nargs="+", default=["quinoa", "sequoia"])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--ours-mesh", default="2dgs_v2", help="2DGS mesh folder for Ours (the paper uses 2dgs_v3: small-piece filter 0.005, keeps thin leaves)")
    ap.add_argument("--ours-dir", default="final", help="run folder for Ours under <gs> (the paper uses 'zoom': crop-aware focal search)")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "input", "input_con"],
                    help="placement: oracle (scale fitted to all test views) input (input photo only, real camera distance) or input_con (input, no sliding along the viewing axis)")
    ap.add_argument("--sv3d-dir", default="baselines", help="run folder for SV3D under <gs>")
    a = ap.parse_args()
    METHODS["SV3D"] = tuple(x.replace("baselines/", a.sv3d_dir + "/", 1) for x in METHODS["SV3D"])
    if a.ours_dir != "final":
        METHODS["Ours"] = tuple(x.replace("final/", a.ours_dir + "/", 1) for x in METHODS["Ours"])
    METHODS["Ours"] = (METHODS["Ours"][0].replace("2dgs_v2/", a.ours_mesh + "/"), METHODS["Ours"][1])
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    for sp in a.species:
        for D in sorted(DATA.glob(f"gs_{sp}_*")):
            stem = f"{D.name}_{(D / 'front.txt').read_text().strip()}"
            if not (a.gs / "final" / f"{stem}_s42").exists():
                continue  # date not in the study's 17
            row = res.get(D.name, {})
            up = np.array(json.loads((D / "world.json").read_text())["up"])
            up /= np.linalg.norm(up)
            ref = np.asarray(o3d.io.read_point_cloud(str(D / "data" / f"{D.name}_real_pointcloud.ply")).points) * 100.0
            base = float(np.percentile(ref @ up, 0.2))
            row["base_cm"], row["reference"] = base, traits(ref, up, base)
            for m, (mp, sp3) in METHODS.items():
                if row.get(m) and not a.redo:
                    continue
                mesh, sim = a.gs / mp.format(stem=stem), a.gs / sp3.format(stem=stem, mode=a.mode)
                if mesh.exists() and sim.exists():
                    p = mesh_points(mesh, sim)
                    row[m] = traits(p * 100.0, up, base) if p is not None else None
            res[D.name] = row
            print(D.name, " | ".join(f"{k} {v['height_cm']:.1f}" for k, v in row.items() if isinstance(v, dict) and v), flush=True)
            a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
