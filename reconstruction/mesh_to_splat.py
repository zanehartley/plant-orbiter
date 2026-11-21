"""Turn a textured mesh into a dense splat of small opaque surface points, so mesh-based baselines
(e.g. Hunyuan3D) go through exactly the same alignment and photo scoring as splat methods.

    python mesh_to_splat.py <mesh.obj|glb> <out gaussians.ply> [--points 400000]
Points are sampled uniformly on the surface, coloured from the texture (UV lookup) or vertex colours,
with isotropic scale ~ the mean spacing between samples, full opacity, base colour only (SH degree 0).
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--points", type=int, default=400_000)
    a = ap.parse_args()
    m = trimesh.load(a.mesh, force="mesh", process=False)
    pts, face_idx = trimesh.sample.sample_surface(m, a.points, seed=0)
    col = np.full((len(pts), 3), 0.5)
    vis = m.visual
    if isinstance(vis, trimesh.visual.texture.TextureVisuals) and vis.uv is not None:
        tex = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
        if tex is not None:
            tri = m.triangles[face_idx]
            bary = trimesh.triangles.points_to_barycentric(tri, pts)
            uv = (vis.uv[m.faces[face_idx]] * bary[:, :, None]).sum(1)
            img = np.asarray(tex.convert("RGB"), np.float32) / 255
            h, w = img.shape[:2]
            x = np.clip((uv[:, 0] % 1) * (w - 1), 0, w - 1).astype(int)
            y = np.clip((1 - uv[:, 1] % 1) * (h - 1), 0, h - 1).astype(int)
            col = img[y, x]
    elif hasattr(vis, "vertex_colors") and len(vis.vertex_colors):
        vc = np.asarray(vis.vertex_colors[:, :3], np.float32) / 255
        col = vc[m.faces[face_idx]].mean(1)
    spacing = np.sqrt(m.area / a.points)
    n = len(pts)
    dt = [(k, "f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                              "rot_0", "rot_1", "rot_2", "rot_3")]
    arr = np.zeros(n, dt)
    arr["x"], arr["y"], arr["z"] = pts.T
    dc = (col - 0.5) / C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    arr["opacity"] = 6.0  # sigmoid -> ~1
    for k in ("scale_0", "scale_1", "scale_2"):
        arr[k] = np.log(spacing * 0.7)
    arr["rot_0"] = 1.0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(a.out))
    print(f"wrote {a.out}: {n} points, spacing {spacing:.4g}, textured={isinstance(vis, trimesh.visual.texture.TextureVisuals)}")


if __name__ == "__main__":
    main()
