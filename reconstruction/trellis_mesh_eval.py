"""3D metrics for a TRELLIS run with the same protocol as ours (gao_eval.py).

TRELLIS writes its Gaussians and its GLB mesh in frames that differ by a 180 deg rotation about x
(verified: mesh lies on the Gaussians to 0.4% of plant size after it); the Gaussians' alignment
(align_<mode>.json from align_generic.py) then maps the mesh into the real frame.

    python trellis_mesh_eval.py <plant> outputs/baselines/trellis/<run> [--mode oracle]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parent
FLIP = np.diag([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("run", type=Path)
    ap.add_argument("--mode", default="oracle")
    a, extra = ap.parse_known_args()  # extra options go to gao_eval.py (e.g. --crop base --alpha --voxel)
    m = trimesh.load(a.run / "mesh.glb", force="mesh")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices) @ FLIP.T),
                                     o3d.utility.Vector3iVector(np.asarray(m.faces)))
    o3d.io.write_triangle_mesh(str(a.run / "mesh_gsframe.ply"), mesh)
    subprocess.run([sys.executable, str(ROOT / "gao_eval.py"), a.plant, str(a.run / "mesh_gsframe.ply"),
                    "--sim3", str(a.run / f"align_{a.mode}.json"), "-o", str(a.run / f"gao_{a.mode}.json")] + extra, check=True)


if __name__ == "__main__":
    main()
