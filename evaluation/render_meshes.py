"""Render textured-mesh assets (GLB) with exactly the cameras render_splats.py used, so meshes and splats are scored alike.

The camera set is read from a cameras.json written by render_splats.py (identical for every splat asset), not rebuilt
here. Each mesh gets the splat normalisation: rotate its up axis to +z, then centre and scale the robust bounding box
(1st-99th percentile of area-weighted surface samples, the mesh analogue of opaque Gaussian centres) into a unit cube.

Shading: base-colour texture (or vertex colour) x (0.7 ambient + 0.3 headlight on |normal . view|), fixed for every
view and mesh. Splats carry their lighting baked into colour; this keeps meshes close to that look without adding a
scene light as a free parameter. Rendered at 2x and box-downsampled (anti-aliasing for edges and textures), on white.

Writes <out>/<method>/<asset>/views/<camera>.png and cameras.json, like render_splats.py. Run in the trellis2 env
(nvdiffrast):
    python evaluation/render_meshes.py MANIFEST.csv --out $ORBIT_OUTPUTS/fruit_counting/eval/renders \
        --cameras $ORBIT_OUTPUTS/fruit_counting/eval/renders/ours/tomato_n5_s42_s42/cameras.json
Manifest columns: method, species, asset, mesh, up   (up: +x | -x | +y | -y | +z | -z)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

UP_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
AMBIENT, HEADLIGHT = 0.7, 0.3
SUPERSAMPLE = 2
NEAR, FAR = 0.1, 10.0


def rotation_to_z_up(up):
    """Rotation matrix taking the unit vector `up` to +z (Rodrigues); same as render_splats.py."""
    a, b = np.asarray(up, float), np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def load_mesh(path):
    """One triangle mesh with per-vertex UVs + base-colour texture, or per-vertex colours."""
    mesh = trimesh.load(str(path), force="scene").to_geometry()  # applies node transforms, merges geometries
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.uv is not None:
        material = visual.material
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        factor = np.asarray(getattr(material, "baseColorFactor", None) if getattr(material, "baseColorFactor", None)
                            is not None else [255, 255, 255, 255], float)[:3] / 255
        texture = (np.asarray(image.convert("RGB"), np.float32) / 255 * factor).astype(np.float32)
        return mesh, {"uv": np.asarray(visual.uv, np.float32), "texture": texture}
    colours = np.asarray(mesh.visual.to_color().vertex_colors, np.float32)[:, :3] / 255
    return mesh, {"colours": colours}


def normalise(mesh, up):
    """Rotate up to +z, centre and scale the robust bbox of surface samples into a unit cube (in place)."""
    rot = rotation_to_z_up(UP_AXES[up])
    mesh.vertices = mesh.vertices @ rot.T
    samples, _ = trimesh.sample.sample_surface(mesh, 200000, seed=0)
    lo, hi = np.quantile(samples, 0.01, axis=0), np.quantile(samples, 0.99, axis=0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    mesh.vertices = (mesh.vertices - centre) / size
    return {"centre": centre.tolist(), "size": size, "up": up}


def projection(K, res):
    """OpenCV intrinsics -> clip matrix with NDC y = 2v/H - 1, so nvdiffrast's row 0 is the image's top row."""
    f, cx, cy = K[0][0], K[0][2], K[1][2]
    return np.array([[2 * f / res, 0, 2 * cx / res - 1, 0], [0, 2 * f / res, 2 * cy / res - 1, 0],
                     [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)], [0, 0, 1, 0]])


@torch.no_grad()
def render(glctx, mesh, colour, cam, device):
    res = cam["width"] * SUPERSAMPLE
    K = np.array(cam["K"]) * [[SUPERSAMPLE, 1, SUPERSAMPLE], [1, SUPERSAMPLE, SUPERSAMPLE], [1, 1, 1]]
    w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    cam_pts = torch.cat([verts, torch.ones_like(verts[:, :1])], 1) @ w2c.T
    clip = (cam_pts @ torch.tensor(projection(K, res), dtype=torch.float32, device=device).T)[None]
    rast, _ = dr.rasterize(glctx, clip, faces, (res, res))

    if "uv" in colour:
        uv = torch.tensor(colour["uv"], device=device)
        uv = torch.stack([uv[:, 0], 1 - uv[:, 1]], 1)  # trimesh UVs have v up; texture row 0 is the image top
        tex = torch.tensor(colour["texture"], device=device)[None]
        albedo = dr.texture(tex, dr.interpolate(uv[None], rast, faces)[0], filter_mode="linear")
    else:
        albedo = dr.interpolate(torch.tensor(colour["colours"], device=device)[None], rast, faces)[0]

    # Headlight: |cos| between the face normal and the ray to the camera (two-sided; generated meshes are not watertight).
    tri = cam_pts[:, :3][faces.long()]
    n = torch.nn.functional.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
    view = torch.nn.functional.normalize(-tri.mean(1), dim=-1)
    lambert = (n * view).sum(-1).abs()
    face_id = rast[..., 3].long() - 1  # 0 = background
    shade = torch.where(face_id >= 0, lambert[face_id.clamp(min=0)], torch.zeros_like(rast[..., 3]))
    rgb = albedo * (AMBIENT + HEADLIGHT * shade[..., None])
    rgb = torch.where((face_id >= 0)[..., None], rgb, torch.ones_like(rgb))
    rgb = torch.nn.functional.avg_pool2d(rgb.permute(0, 3, 1, 2), SUPERSAMPLE)[0].permute(1, 2, 0)
    return rgb.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True, help="a cameras.json written by render_splats.py")
    args = p.parse_args()

    device = "cuda"
    cams = json.loads(args.cameras.read_text())["cameras"]
    glctx = dr.RasterizeCudaContext()
    for row in csv.DictReader(open(args.manifest)):
        out = args.out / row["method"] / row["asset"]
        if (out / "cameras.json").is_file():
            continue
        mesh, colour = load_mesh(row["mesh"])
        frame = normalise(mesh, row["up"])
        (out / "views").mkdir(parents=True, exist_ok=True)
        for cam in cams:
            img = render(glctx, mesh, colour, cam, device)
            Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(out / "views" / f"{cam['name']}.png")
        (out / "cameras.json").write_text(json.dumps({"asset": row, "frame": frame, "cameras": cams}, indent=1))
        print(f"{row['method']}/{row['asset']}: {len(mesh.faces)} faces, size {frame['size']:.3g} -> {out}", flush=True)


if __name__ == "__main__":
    main()
