"""Render a smooth 360° turntable video of a trained splatfacto model.

The camera path is fitted to the training cameras: same orbit centre, distance and
field of view, starting at the first training view (the input photo for orbit
splats). An optional elevation sway shows the plant from above/below the ring.

Run in the nerfstudio env:
    conda run --no-capture-output -n nerfstudio python pipeline/render_splat.py \
        $ORBIT_OUTPUTS/<name>/3dgs/runs/<exp>/splatfacto/<ts>/config.yml [-o out.mp4] [--sway 15]
"""

import argparse
import os
from pathlib import Path

import mediapy
import numpy as np
import torch
import yaml
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup


def look_at(pos, target, up):
    """OpenGL-convention camera-to-world (camera looks down -z, y is up)."""
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z, pos], axis=1)  # 3x4


def fit_orbit(c2w):
    """Fit a sphere-ish orbit to training cameras: target, distance, basis, start angle."""
    centres = c2w[:, :3, 3]
    forwards = -c2w[:, :3, 2]
    # target = least-squares point closest to all optical axes
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, f in zip(centres, forwards):
        P = np.eye(3) - np.outer(f, f)
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)
    # plane of the camera ring; normal oriented along the cameras' up vectors
    _, _, vt = np.linalg.svd(centres - centres.mean(0))
    n = vt[2]
    if np.dot(n, c2w[:, :3, 1].mean(0)) < 0:
        n = -n
    rel = centres - target
    dist = np.linalg.norm(rel, axis=1).mean()
    elev = np.arcsin((rel @ n) / np.linalg.norm(rel, axis=1)).mean()
    # in-plane basis with u pointing at the first camera
    flat = rel - np.outer(rel @ n, n)
    u = flat[0] / np.linalg.norm(flat[0])
    v = np.cross(n, u)
    # rotate in the same direction as the training sequence
    ang = np.unwrap(np.arctan2(flat @ v, flat @ u))
    direction = 1.0 if ang[min(5, len(ang) - 1)] > 0 else -1.0
    return target, dist, elev, n, u, v * direction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("-o", "--output", type=Path, help="default: <run dir>/turntable[_sway].mp4")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sway", type=float, default=0.0, help="elevation sway amplitude in degrees")
    p.add_argument("--zoom", type=float, default=1.0, help=">1 moves the camera closer")
    p.add_argument("--bg", choices=["white", "black"], default="white")
    a = p.parse_args()

    # data paths in config.yml are relative to the directory ns-train ran in:
    # <train cwd>/<output_dir>/<experiment>/<method>/<timestamp>/config.yml
    a.config = a.config.resolve()
    if a.output:
        a.output = a.output.resolve()
    out_dir = yaml.load(a.config.read_text(), Loader=yaml.Loader).output_dir
    if not out_dir.is_absolute():
        os.chdir(a.config.parents[3 + len(out_dir.parts)])

    _, pipeline, _, _ = eval_setup(a.config, test_mode="inference")
    model = pipeline.model
    model.eval()
    model.set_background(torch.ones(3) if a.bg == "white" else torch.zeros(3))
    if hasattr(model.config, "background_color"):
        model.config.background_color = a.bg

    cams = pipeline.datamanager.train_dataset.cameras
    c2w = cams.camera_to_worlds.cpu().numpy()
    target, dist, elev0, n, u, v = fit_orbit(c2w)
    dist /= a.zoom

    # keep the training field of view, rendered as a square
    scale = a.res / float(cams.width[0])
    fx = float(cams.fx[0]) * scale
    fy = float(cams.fy[0]) * a.res / float(cams.height[0])

    n_frames = int(a.seconds * a.fps)
    out = a.output or a.config.parent / ("turntable" + (f"_sway{a.sway:g}" if a.sway else "") + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    with mediapy.VideoWriter(out, shape=(a.res, a.res), fps=a.fps, crf=16) as w:
        for i in range(n_frames):
            th = 2 * np.pi * i / n_frames
            el = elev0 + np.radians(a.sway) * np.sin(th)
            pos = target + dist * (np.cos(el) * (np.cos(th) * u + np.sin(th) * v) + np.sin(el) * n)
            cam = Cameras(
                camera_to_worlds=torch.tensor(look_at(pos, target, n), dtype=torch.float32)[None],
                fx=fx, fy=fy, cx=a.res / 2, cy=a.res / 2, width=a.res, height=a.res,
                camera_type=CameraType.PERSPECTIVE,
            ).to(model.device)
            with torch.no_grad():
                rgb = model.get_outputs_for_camera(cam)["rgb"]
            w.add_image(rgb.clamp(0, 1).cpu().numpy())
    print(f"wrote {out} ({n_frames} frames, {a.res}x{a.res}, elevation {np.degrees(elev0):.1f}° ± {a.sway:g}°)")


if __name__ == "__main__":
    main()
