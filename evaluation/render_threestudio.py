"""Render threestudio NeRF assets (MVDream) with exactly the cameras render_splats.py used, so they are scored alike.

Each asset is a threestudio trial dir (configs/parsed.yaml + ckpts/last.ckpt; run_mvdream.py: <asset>/trial). Only the 3D model is rebuilt
(geometry, material, background, renderer), not the guidance diffusion model. The NeRF analogue of the splat
normalisation: density is sampled on a 256^3 grid inside the scene's bounding sphere, and the robust box (1st-99th
percentile) of points above threestudio's own isosurface threshold (the level its mesh exporter extracts) is mapped
to the unit cube. Here the cameras are moved instead of the scene: same rotation and intrinsics, camera centre
mapped back into scene units. threestudio scenes are z-up, like the splats.

Colour is the volume-rendered foreground composited on white (comp_rgb_fg + (1 - opacity)); the learned environment
background is dropped, as other methods' backgrounds are. Writes <out>/<method>/<asset>/views/<camera>.png and
cameras.json, like render_splats.py, plus points.npz (depth_points.py): each pixel with opacity >= 0.5 is placed at
its expected ray distance (threestudio's depth = sum of weight x t along the normalised ray, divided by opacity) and
mapped into the unit-cube frame. An asset that already has views but no points.npz gets only its points. Run in the
mvdream env:
    python evaluation/render_threestudio.py MANIFEST.csv --out $ORBIT_OUTPUTS/synthetic_plants/eval/renders \
        --cameras $ORBIT_OUTPUTS/synthetic_plants/eval/renders/ours/bean_s42_s42/cameras.json
Manifest columns: method, species, asset, trial
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_points import MIN_ALPHA, save_points

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "MVDream-threestudio"))
import threestudio  # noqa: E402
from threestudio.utils.config import load_config  # noqa: E402
from threestudio.utils.ops import get_ray_directions, get_rays  # noqa: E402

GRID = 256
OPENCV_TO_OPENGL = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # camera axes: (x, y down, z fwd) -> (x, y up, -z fwd)


def load_model(trial, device):
    """geometry/material/background/renderer from a trial dir -> (renderer, geometry, isosurface threshold, radius)."""
    cfg = load_config(str(trial / "configs" / "parsed.yaml")).system
    geometry = threestudio.find(cfg.geometry_type)(cfg.geometry)
    material = threestudio.find(cfg.material_type)(cfg.material)
    background = threestudio.find(cfg.background_type)(cfg.background)
    renderer = threestudio.find(cfg.renderer_type)(cfg.renderer, geometry=geometry, material=material,
                                                   background=background)
    state = torch.load(trial / "ckpts" / "last.ckpt", map_location="cpu")["state_dict"]
    # the renderer's state is its occupancy grid (nerfacc estimator); without it every ray is skipped as empty
    for name, module in [("geometry", geometry), ("material", material), ("background", background),
                         ("renderer", renderer)]:
        prefix = f"{name}."
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    for module in (geometry, material, background, renderer):
        module.to(device).eval()
    # geometry.cfg, not cfg.geometry: parsed.yaml holds only the keys set in the config, not the class defaults
    return renderer, geometry, float(geometry.cfg.isosurface_threshold), float(geometry.cfg.radius)


@torch.no_grad()
def normalise(geometry, threshold, radius, device):
    """Robust box (1-99%) of grid points with density above the isosurface threshold -> (centre, size)."""
    axis = torch.linspace(-radius, radius, GRID, device=device)
    occupied = []
    for x in axis:  # one slab at a time keeps memory flat
        pts = torch.stack(torch.meshgrid(x[None], axis, axis, indexing="ij"), -1).reshape(-1, 3)
        density = geometry.forward_density(pts)[:, 0]
        occupied.append(pts[density > threshold])
    occupied = torch.cat(occupied)
    lo, hi = torch.quantile(occupied, 0.01, dim=0), torch.quantile(occupied, 0.99, dim=0)
    return (lo + hi) / 2, float((hi - lo).max()), len(occupied)


@torch.no_grad()
def render(renderer, cam, centre, size, device):
    """RGB on white (H x W x 3, 0-1) and the opaque pixels' points in the unit-cube frame (N x 3)."""
    K, res = cam["K"], cam["width"]
    c2w_cv = torch.linalg.inv(torch.tensor(cam["w2c"], dtype=torch.float32))
    c2w = torch.eye(4)
    c2w[:3, :3] = c2w_cv[:3, :3] @ OPENCV_TO_OPENGL
    c2w[:3, 3] = centre.cpu() + size * c2w_cv[:3, 3]  # unit-cube camera centre -> scene units
    directions = get_ray_directions(res, res, (K[0][0], K[1][1]), principal=(K[0][2], K[1][2]))
    rays_o, rays_d = get_rays(directions.to(device), c2w.to(device), keepdim=True)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    out = renderer(rays_o[None], rays_d[None], light_positions=c2w[None, :3, 3].to(device))
    opacity = out["opacity"][0]
    rgb = out["comp_rgb_fg"][0] + (1 - opacity)
    keep = opacity[..., 0] >= MIN_ALPHA
    t = out["depth"][0][keep] / opacity[keep]  # expected distance along the (unit) ray, scene units
    points = (rays_o[keep] + rays_d[keep] * t - centre) / size  # scene -> unit cube
    return rgb.clamp(0, 1), points.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True, help="a cameras.json written by render_splats.py")
    args = p.parse_args()

    device = "cuda"
    cams = json.loads(args.cameras.read_text())["cameras"]
    for row in csv.DictReader(open(args.manifest)):
        out = args.out / row["method"] / row["asset"]
        if (out / "points.npz").is_file():
            continue
        have_views = (out / "cameras.json").is_file()
        renderer, geometry, threshold, radius = load_model(Path(row["trial"]), device)
        centre, size, n = normalise(geometry, threshold, radius, device)
        if have_views:  # points for views rendered earlier: they must come from the same normalisation
            old = json.loads((out / "cameras.json").read_text())["frame"]
            if not np.allclose(old["centre"], centre.tolist(), atol=1e-5) or not np.isclose(old["size"], size):
                raise SystemExit(f"{out}: normalisation differs from cameras.json; re-render this asset")
        (out / "views").mkdir(parents=True, exist_ok=True)
        points = []
        for cam in cams:
            img, cam_points = render(renderer, cam, centre, size, device)
            points.append(cam_points)
            if not have_views:
                Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(
                    out / "views" / f"{cam['name']}.png")
        if not have_views:
            frame = {"centre": centre.tolist(), "size": size, "up": "+z", "occupied_voxels": n, "threshold": threshold}
            (out / "cameras.json").write_text(json.dumps({"asset": row, "frame": frame, "cameras": cams}, indent=1))
        save_points(out / "points.npz", points, len(cams))
        print(f"{row['method']}/{row['asset']}: {n} occupied voxels, size {size:.3g}, "
              f"{sum(len(p) for p in points)} depth pixels -> {out}", flush=True)


if __name__ == "__main__":
    main()
