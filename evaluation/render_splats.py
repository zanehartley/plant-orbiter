"""Render every Gaussian-splat asset in a manifest from one fixed camera set, so all methods are scored identically.

Each asset (a standard 3DGS .ply: INRIA / threestudio / nerfstudio export) is rotated so its up axis is +z, then
centred and scaled so the robust bounding box (1st-99th percentile of Gaussians with opacity > 0.5) fits a unit cube.
Views: elevations x azimuths on a sphere around the origin, fixed field of view and distance, white background.

Writes <out>/<method>/<asset>/views/<elev>_<azim>.png and cameras.json, plus points.npz: the opaque pixels of every
view unprojected at gsplat's expected depth and fused (depth_points.py; for the shape metrics). An asset that already
has views but no points.npz gets only its points; its PNGs and cameras.json are left as they are. Run in the
nerfstudio env (gsplat):
    python evaluation/render_splats.py MANIFEST.csv --out $ORBIT_OUTPUTS/synthetic_plants/eval/renders
Manifest columns: method, species, asset, ply, up   (up: +x | -x | +y | -y | +z | -z, or a vector "x y z")
                  [, rasterize_mode: classic | antialiased, the mode the splat was trained with; default classic]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from plyfile import PlyData

from depth_points import save_points, unproject_z

ELEVATIONS = (0.0, 15.0, 30.0)
N_AZIMUTH = 24
FOV_DEG = 40.0
UP_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def load_gaussians(path, device):
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
    n_coeffs = len(rest) // 3 + 1
    sh_degree = int(math.isqrt(n_coeffs)) - 1
    col = lambda keys: torch.tensor(np.stack([np.asarray(v[k], dtype=np.float32) for k in keys], 1), device=device)
    dc = col(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    sh = torch.cat([dc, col(rest).reshape(-1, 3, n_coeffs - 1).transpose(1, 2)], 1) if rest else dc
    return {"means": col(["x", "y", "z"]), "quats": col(["rot_0", "rot_1", "rot_2", "rot_3"]),
            "log_scales": col(["scale_0", "scale_1", "scale_2"]),
            "opacities": torch.sigmoid(col(["opacity"])[:, 0]), "sh": sh, "sh_degree": sh_degree}


def up_vector(up):
    """A named axis (+y) or an explicit vector ("0.02 -0.99 0.1", e.g. estimated from a method's own cameras)."""
    return UP_AXES[up] if up in UP_AXES else [float(x) for x in up.split()]


def rotation_to_z_up(up):
    """Rotation matrix taking the unit vector `up` to +z (Rodrigues)."""
    a, b = np.asarray(up, float) / np.linalg.norm(up), np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def quat_multiply(q, r):
    """Hamilton product q * r, wxyz, q broadcast over r."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def normalise(g, up):
    """Rotate up to +z, centre and scale the robust bbox of opaque Gaussians into a unit cube (in place)."""
    rot = rotation_to_z_up(up_vector(up))
    R = torch.tensor(rot, dtype=torch.float32, device=g["means"].device)
    g["means"] = g["means"] @ R.T
    w = math.sqrt(max(0.0, 1 + np.trace(rot))) / 2  # rotation as a quaternion, applied to each Gaussian's orientation
    q_rot = (w, (rot[2, 1] - rot[1, 2]) / (4 * w), (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w)) \
        if w > 1e-6 else (0.0, 1.0, 0.0, 0.0)  # 180 deg about x (only case here: up = -z)
    g["quats"] = quat_multiply(torch.tensor(q_rot, device=R.device, dtype=torch.float32),
                               torch.nn.functional.normalize(g["quats"], dim=-1))
    opaque = g["means"][g["opacities"] > 0.5]
    lo, hi = torch.quantile(opaque, 0.01, dim=0), torch.quantile(opaque, 0.99, dim=0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    g["means"] = (g["means"] - centre) / size
    g["log_scales"] = g["log_scales"] - math.log(size)
    return {"centre": centre.tolist(), "size": size, "up": up}


def look_at(eye):
    """World-to-camera (OpenCV: x right, y down, z forward) for a camera at `eye` looking at the origin, z up."""
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = -R @ eye
    return w2c


def camera_set(res):
    f = res / 2 / math.tan(math.radians(FOV_DEG) / 2)
    dist = (math.sqrt(3) / 2) / math.sin(math.radians(FOV_DEG) / 2) * 1.05  # unit cube's bounding sphere fills the view
    K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]])
    cams = []
    for el in ELEVATIONS:
        for k in range(N_AZIMUTH):
            az = 360.0 * k / N_AZIMUTH
            e, a = math.radians(el), math.radians(az)
            eye = dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            cams.append({"name": f"e{el:+03.0f}_a{az:05.1f}", "elevation": el, "azimuth": az,
                         "w2c": look_at(eye).tolist(), "K": K.tolist(), "width": res, "height": res})
    return cams


@torch.no_grad()
def render(g, cams, device, rasterize_mode):
    """Per view: RGB on white (H x W x 3, 0-1) and the unprojected opaque pixels (N x 3, unit-cube frame)."""
    viewmats = torch.tensor([c["w2c"] for c in cams], dtype=torch.float32, device=device)
    Ks = torch.tensor([c["K"] for c in cams], dtype=torch.float32, device=device)
    res = cams[0]["width"]
    images, points = [], []
    for i in range(0, len(cams), 8):  # small batches keep memory flat for large splats
        # RGB+ED: expected depth (accumulated depth / alpha) as a 4th channel; gsplat pads the background with 0 for it
        out, alpha, _ = rasterization(g["means"], g["quats"], torch.exp(g["log_scales"]), g["opacities"], g["sh"],
                                      viewmats[i:i + 8], Ks[i:i + 8], res, res, sh_degree=g["sh_degree"],
                                      rasterize_mode=rasterize_mode, render_mode="RGB+ED",
                                      backgrounds=torch.ones(len(viewmats[i:i + 8]), 3, device=device))
        images.append(out[..., :3].clamp(0, 1))
        depth, alpha = out[..., 3].cpu().numpy(), alpha[..., 0].cpu().numpy()
        points += [unproject_z(d, a, cam) for d, a, cam in zip(depth, alpha, cams[i:i + 8])]
    return torch.cat(images), points


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--res", type=int, default=512)
    args = p.parse_args()

    device = "cuda"
    cams = camera_set(args.res)
    for row in csv.DictReader(open(args.manifest)):
        out = args.out / row["method"] / row["asset"]
        if (out / "points.npz").is_file():
            continue
        have_views, asset_cams = (out / "cameras.json").is_file(), cams
        g = load_gaussians(row["ply"], device)
        frame = normalise(g, row["up"])
        if have_views:  # points for views rendered earlier: they must come from the same normalisation
            old = json.loads((out / "cameras.json").read_text())
            if not np.allclose(old["frame"]["centre"], frame["centre"], atol=1e-5) or \
                    not math.isclose(old["frame"]["size"], frame["size"], rel_tol=1e-5):
                raise SystemExit(f"{out}: normalisation differs from cameras.json; re-render this asset")
            asset_cams = old["cameras"]
        images, points = render(g, asset_cams, device, row.get("rasterize_mode") or "classic")
        if not have_views:
            (out / "views").mkdir(parents=True, exist_ok=True)
            for cam, img in zip(asset_cams, images):
                Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(
                    out / "views" / f"{cam['name']}.png")
            meta = {"asset": row, "frame": frame, "cameras": asset_cams}
            (out / "cameras.json").write_text(json.dumps(meta, indent=1))
        save_points(out / "points.npz", points, len(asset_cams))
        print(f"{row['method']}/{row['asset']}: {len(g['means'])} Gaussians, size {frame['size']:.3g}, "
              f"{sum(len(p) for p in points)} depth pixels -> {out}", flush=True)


if __name__ == "__main__":
    main()
