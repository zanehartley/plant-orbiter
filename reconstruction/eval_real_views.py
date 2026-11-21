"""Score any Gaussian splat against the real held-out views of a PlantDreamer-dataset plant.

Uses the same protocol as PlantDreamer's evaluation: render at the undistorted test cameras
(`*_eval` views, cameras.json of the plant's gaussian-splatting run) on a white background and
compute PSNR over the plant mask (images/mask), plus masked SSIM / LPIPS on mask-composited images.

    python eval_real_views.py <plant> <splat.ply> [--sim3 sim3.json] [--views eval|ring]

The splat must be in the plant's real coordinate frame; --sim3 applies a similarity transform
(scale s, rotation R, translation t: x' = s R x + t) first, e.g. from align_splat.py.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from plyfile import PlyData
from po_paths import DATASETS, HYBRID_MASKS

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLANT_DATA", DATASETS / "PlantDreamer_Data" / "plants"))  # converted datasets: set PLANT_DATA
TIGHT = HYBRID_MASKS


def mask_path(plant, name):
    """Mask file for a real photo. PlantDreamer's own masks are never used alone: for
    PlantDreamer plants this is the hybrid mask (masks/README.md), and a view without one is skipped
    (the file does not exist). PLANT_DATASET_MASKS=1 restores the old masks, only to reproduce old numbers."""
    if "PlantDreamer_Data" in str(DATA.resolve()) and not os.environ.get("PLANT_DATASET_MASKS"):
        return TIGHT / plant / f"{name}.png"
    return DATA / plant / "images" / "mask" / f"{name}.png.png"


def load_ply(path, device="cuda"):
    """INRIA-format 3DGS .ply (also what ns-export gaussian-splat writes)."""
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    g = lambda k: np.asarray(v[k], np.float32)
    means = np.stack([g("x"), g("y"), g("z")], 1)
    dc = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], 1)[:, None]
    rest_n = sorted([n for n in names if n.startswith("f_rest_")], key=lambda n: int(n.split("_")[-1]))
    if rest_n:
        rest = np.stack([g(n) for n in rest_n], 1).reshape(len(means), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    t = lambda a: torch.tensor(a, device=device)
    return {"means": t(means), "quats": t(np.stack([g(f"rot_{i}") for i in range(4)], 1)),
            "scales": t(np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))),
            "opacities": t(1 / (1 + np.exp(-g("opacity")))), "sh": t(sh)}


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rot_to_quat(R):
    m = R
    w = torch.sqrt(torch.clamp(1 + m[0, 0] + m[1, 1] + m[2, 2], min=1e-12)) / 2
    return torch.stack([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])


def apply_sim3(gs, s, R, t):
    """Transform Gaussians: x' = s R x + t (R: 3x3 torch, proper rotation)."""
    out = dict(gs)
    out["means"] = s * gs["means"] @ R.T + t
    out["scales"] = gs["scales"] * s
    q = rot_to_quat(R)
    out["quats"] = quat_mul(q.expand_as(gs["quats"]), torch.nn.functional.normalize(gs["quats"], dim=-1))
    if gs["sh"].shape[1] > 1:  # rotating view-dependent colour is approximate; keep only the base colour
        out["sh"] = gs["sh"][:, :1]
    return out


# rasterisation used for every render: "classic" (default) or "antialiased" (the
# mode our final splats are trained with; EV_RASTER=antialiased or set ev.RASTER). Splats trained antialiased
# but rendered classic show needle-like edge artefacts.
RASTER = os.environ.get("EV_RASTER", "classic")


def render(gs, c2w, fx, fy, w, h, bg=1.0):
    """c2w: 4x4 OpenCV convention. Returns rgb [h,w,3], alpha [h,w]."""
    vm = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32, device=gs["means"].device))
    K = torch.tensor([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device=vm.device)
    deg = int(round(gs["sh"].shape[1] ** 0.5)) - 1
    rgb, alpha, _ = rasterization(gs["means"], torch.nn.functional.normalize(gs["quats"], dim=-1), gs["scales"],
                                  gs["opacities"], gs["sh"], vm[None], K[None], w, h, sh_degree=deg,
                                  backgrounds=torch.full((1, 3), bg, device=vm.device), rasterize_mode=RASTER)
    return rgb[0].clamp(0, 1), alpha[0, ..., 0]


def cameras(plant, which="eval"):
    cams = json.loads((DATA / plant / "plantdreamer" / "3dgs" / "cameras.json").read_text())
    out = []
    for c in cams:
        if which == "eval" and not c["img_name"].endswith("_eval"):
            continue
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(c["rotation"])
        c2w[:3, 3] = c["position"]
        out.append({"name": c["img_name"], "c2w": c2w, "fx": c["fx"], "fy": c["fy"], "w": c["width"], "h": c["height"]})
    return out


_lpips = None


def lpips_fn(a, b):
    global _lpips
    if _lpips is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    return float(_lpips(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]))


def cams_into_splat(cams, sim3):
    """Express real cameras in the splat's own frame (x_real = s R x_splat + t). Rendering is unchanged by a
    global similarity, and unlike transforming the Gaussians this keeps their view-dependent (SH) colour."""
    if sim3 is None:
        return cams
    s, R, t = sim3["s"], np.array(sim3["R"]), np.array(sim3["t"])
    out = []
    for c in cams:
        m = np.eye(4)
        m[:3, :3] = R.T @ c["c2w"][:3, :3]
        m[:3, 3] = R.T @ (c["c2w"][:3, 3] - t) / s
        out.append(dict(c, c2w=m))
    return out


def unwhite(rgb, alpha):
    """Foreground colour of a render composited on white: rgb = a*c + (1-a)."""
    a = alpha[..., None].clamp(min=1e-3)
    return ((rgb - (1 - a)) / a).clamp(0, 1)


def fit_colour(plant, gs, cams, sim3=None):
    """3x4 affine colour map (render -> photo) fitted on plant pixels of the given (input) views only."""
    X, Y = [], []
    for c, cs in zip(cams, cams_into_splat(cams, sim3)):
        if not mask_path(plant, c["name"]).exists():
            continue  # some dataset views have no mask (e.g. PlantDreamer mint)
        gt = torch.tensor(np.asarray(Image.open(DATA / plant / "images" / "undistorted" / (c["name"] + ".png")).convert("RGB"),
                                     np.float32) / 255, device="cuda")
        m = torch.tensor(np.asarray(Image.open(mask_path(plant, c["name"])).convert("L")) > 0,
                         device="cuda")
        if m.shape != gt.shape[:2]:
            continue
        pred, alpha = render(gs, cs["c2w"], c["fx"], c["fy"], gt.shape[1], gt.shape[0])
        sel = m & (alpha > 0.5)
        X.append(unwhite(pred, alpha)[sel])
        Y.append(gt[sel])
    X, Y = torch.cat(X), torch.cat(Y)
    A = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    return torch.linalg.lstsq(A, Y).solution  # 4x3


def score(plant, gs, cams, sim3=None, colour=None):
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    rows = []
    for c, cs in zip(cams, cams_into_splat(cams, sim3)):
        if not mask_path(plant, c["name"]).exists():
            continue  # some dataset views have no mask (e.g. PlantDreamer mint)
        gt = torch.tensor(np.asarray(Image.open(DATA / plant / "images" / "undistorted" / (c["name"] + ".png")).convert("RGB"),
                                     np.float32) / 255, device="cuda")
        m = torch.tensor(np.asarray(Image.open(mask_path(plant, c["name"])).convert("L")) > 0,
                         device="cuda")
        if m.shape != gt.shape[:2]:  # a few dataset views have a mask from a different undistortion: skip them
            print(f"warning: {plant} {c['name']}: mask {tuple(m.shape)} != image {tuple(gt.shape[:2])}, view skipped")
            continue
        pred, alpha = render(gs, cs["c2w"], c["fx"], c["fy"], gt.shape[1], gt.shape[0])
        if colour is not None:  # colour correction fitted on the input view(s) only, applied to the rendered
            # plant colour only (not to empty background, which must stay white)
            fg = unwhite(pred, alpha)
            fg = (torch.cat([fg, torch.ones_like(fg[..., :1])], -1) @ colour).clamp(0, 1)
            pred = alpha[..., None] * fg + (1 - alpha[..., None])
        mse = ((pred - gt)[m] ** 2).mean()
        psnr_mask = float(10 * torch.log10(1 / mse))
        # masked composites on white for SSIM / LPIPS
        gt_c = torch.where(m[..., None], gt, torch.ones_like(gt))
        pr_c = torch.where(m[..., None], pred, torch.ones_like(pred))
        s = float(ssim(pr_c.permute(2, 0, 1)[None], gt_c.permute(2, 0, 1)[None], data_range=1.0))
        rows.append({"view": c["name"], "psnr_mask": psnr_mask, "ssim_mask": s, "lpips_mask": lpips_fn(pr_c, gt_c),
                     "sil_iou": float(((alpha > 0.5) & m).sum() / ((alpha > 0.5) | m).sum())})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "view"}
    return mean, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("ply", type=Path)
    ap.add_argument("--sim3", type=Path)
    ap.add_argument("--views", default="eval", choices=["eval", "all"])
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    gs = load_ply(a.ply)
    sim3 = json.loads(a.sim3.read_text()) if a.sim3 else None
    with torch.no_grad():
        mean, rows = score(a.plant, gs, cameras(a.plant, a.views), sim3)
    print(json.dumps({k: round(v, 4) for k, v in mean.items()}), f"({len(rows)} views)")
    if a.out:
        a.out.write_text(json.dumps({"mean": mean, "views": rows}, indent=1))


if __name__ == "__main__":
    main()
