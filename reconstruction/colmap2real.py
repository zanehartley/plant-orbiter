"""Compose the COLMAP-world -> real-world similarity for a run (2DGS meshes live in COLMAP world).

    python colmap2real.py <run>/3dgs [input|oracle|none]  -> <run>/3dgs/colmap2real_<mode>.json
"""
import json, sys
from pathlib import Path
import numpy as np

run3d, mode = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "input")
a = json.loads((run3d / f"align_{mode}.json").read_text())           # nerfstudio world -> real
d = json.loads(next((run3d / "runs").glob("*/*/v1/dataparser_transforms.json")).read_text())
T = np.array(d["transform"]); sc = d["scale"]                          # x_ns = sc (Tr x_col + Tt)
s, R, t = a["s"], np.array(a["R"]), np.array(a["t"])
out = {"s": s * sc, "R": (R @ T[:, :3]).tolist(), "t": (s * sc * R @ T[:, 3] + t).tolist()}
(run3d / f"colmap2real_{mode}.json").write_text(json.dumps(out, indent=1))
print(run3d / f"colmap2real_{mode}.json")
