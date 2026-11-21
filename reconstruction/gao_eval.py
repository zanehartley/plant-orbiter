"""Geometry metrics following Gao, Hartley & French (Plant Methods 2026), re-implemented from the paper.

Their protocol: ground-truth mesh = alpha shapes on the PlantDreamer point cloud; pots and bases cropped
from both meshes; alignment = centroid + bounding-box scale, then ICP; metrics = Chamfer distance,
normal consistency, IoU and F-score (threshold 1% of the bounding-box diagonal).
Details the paper doesn't give (alpha, crop height, normalisation, IoU grid, CD definition) are chosen
here and written to the output so they can be matched to their scripts later.

    python gao_eval.py <plant> <our mesh.ply> --sim3 <colmap->real json> [-o out.json]
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLANT_DATA", DATASETS / "PlantDreamer_Data" / "plants"))  # converted datasets: set PLANT_DATA
N_SAMPLES = 100_000


def up_axis(plant):
    cams = json.loads((DATA / plant / "plantdreamer" / "3dgs" / "cameras.json").read_text())
    up = -np.mean([np.array(c["rotation"])[:, 1] for c in cams], 0)  # OpenCV: -y is up
    return up / np.linalg.norm(up)


def rim_height(points, up, n_bins=40):
    """Top of the pot: first height (from the bottom) where the cloud's width drops below half the pot's."""
    h = points @ up
    bins = np.linspace(h.min(), h.max(), n_bins + 1)
    widths = []
    for i in range(n_bins):
        s = (h >= bins[i]) & (h < bins[i + 1])
        if s.sum() < 20:
            widths.append(0.0)
            continue
        q = points[s] - np.outer(points[s] @ up, up)
        widths.append(np.ptp(q, 0).max())
    pot = max(widths[: n_bins // 3])
    for i in range(1, n_bins):
        if widths[i] < 0.5 * pot and i > 2:
            return bins[i]
    return bins[n_bins // 3]


def crop_above(mesh, up, h0):
    v = np.asarray(mesh.vertices)
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_vertices_by_mask(v @ up < h0)
    return mesh


def gt_mesh(plant, alpha, voxel, ref=None):
    pc = o3d.io.read_point_cloud(str(ref or next((DATA / plant / "data").glob("*_real_pointcloud.ply"))))
    pc = pc.voxel_down_sample(voxel)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha)
    mesh.compute_vertex_normals()
    return mesh, np.asarray(pc.points)


def sample(mesh, n=N_SAMPLES):
    mesh.compute_triangle_normals()
    p = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    return np.asarray(p.points), np.asarray(p.normals)


def metrics(pa, na, pb, nb, tau, grid):
    da, ia = cKDTree(pb).query(pa)  # pred -> gt
    db, ib = cKDTree(pa).query(pb)  # gt -> pred
    prec, rec = (da < tau).mean(), (db < tau).mean()
    nc = 0.5 * (np.abs((na * nb[ia]).sum(1)).mean() + np.abs((nb * na[ib]).sum(1)).mean())
    lo = np.minimum(pa.min(0), pb.min(0))
    size = (np.maximum(pa.max(0), pb.max(0)) - lo).max() / grid
    va = {tuple(x) for x in np.floor((pa - lo) / size).astype(int)}
    vb = {tuple(x) for x in np.floor((pb - lo) / size).astype(int)}
    return {"chamfer_l2_mean": float(0.5 * (da.mean() + db.mean())),
            "chamfer_sq_sum": float((da ** 2).mean() + (db ** 2).mean()),
            "normal_consistency": float(nc), "fscore": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "precision": float(prec), "recall": float(rec), "surface_voxel_iou": len(va & vb) / len(va | vb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--sim3", type=Path, required=True, help="similarity mapping the mesh into the real frame")
    ap.add_argument("--alpha", type=float, default=0.004, help="alpha-shape radius (scene units, m)")
    ap.add_argument("--voxel", type=float, default=0.0008)
    ap.add_argument("--grid", type=int, default=128, help="voxel grid for surface IoU")
    ap.add_argument("--crop", choices=["none", "rim", "base"], default="none",
                    help="none (default, as in the paper): score plant + pot; rim: old Gao et al. crop above the "
                         "detected pot rim (its detector fails on leafy plants); base: drop everything below the reference's "
                         "lowest point (PLANesT: the reference has no pot, so the pot is removed from the prediction)")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--dump", type=Path, help="also save the normalised reference + ICP-aligned prediction points (.npz)")
    ap.add_argument("--ref", type=Path, help="reference cloud to use instead of data/<plant>_real_pointcloud.ply")
    a = ap.parse_args()

    up = up_axis(a.plant)
    gt, gt_pts = gt_mesh(a.plant, a.alpha, a.voxel, a.ref)
    if a.crop == "rim":
        h0 = rim_height(gt_pts, up)
        gt = crop_above(gt, up, h0)
    elif a.crop == "base":
        hh = gt_pts @ up
        h0 = np.percentile(hh, 0.2) - 0.01 * np.ptp(hh)  # the plant's base (stem foot), 1% slack
    else:
        h0 = -np.inf

    pred = o3d.io.read_triangle_mesh(str(a.mesh))
    t = json.loads(a.sim3.read_text())
    T = np.eye(4)
    T[:3, :3] = t["s"] * np.array(t["R"])
    T[:3, 3] = t["t"]
    pred.transform(T)
    if a.crop in ("rim", "base"):
        pred = crop_above(pred, up, h0)

    # normalise by the ground truth's bounding-box diagonal, centred on its centroid
    gv = np.asarray(gt.vertices)
    c, diag = gv.mean(0), np.linalg.norm(np.ptp(gv, 0))
    for m in (gt, pred):
        m.translate(-c)
        m.scale(1 / diag, center=np.zeros(3))
    pg, ng = sample(gt)
    pp, npred = sample(pred)

    # ICP with scale (as in their protocol), starting from our camera-based alignment
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pp))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pg))
    # rigid ICP: scale is already set from the cameras, and ICP with scaling shrinks a partly wrong shape
    # onto the ground truth (bean_3 test: recall 0.18 -> 0.02)
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, 0.05, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    M = reg.transformation
    pp_icp = pp @ M[:3, :3].T + M[:3, 3]
    n_icp = npred @ (M[:3, :3] / np.cbrt(np.linalg.det(M[:3, :3]))).T

    out = {"before_icp": metrics(pp, npred, pg, ng, 0.01, a.grid),
           "after_icp": metrics(pp_icp, n_icp, pg, ng, 0.01, a.grid),
           "settings": {"alpha": a.alpha, "voxel": a.voxel, "crop": a.crop, "rim_height": float(h0) if a.crop == "rim" else None, "gt_diag_m": float(diag),
                        "tau": "1% of GT bbox diagonal", "iou_grid": a.grid, "samples": N_SAMPLES}}
    print(json.dumps({k: {m: round(v, 5) for m, v in d.items()} if isinstance(d, dict) and k != "settings" else d
                      for k, d in out.items()}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    if a.dump:  # for figures (make_3d_video.py); same units as the metrics (GT bbox diagonal = 1)
        np.savez_compressed(a.dump, ref=pg.astype(np.float32), pred=pp_icp.astype(np.float32), up=up, tau=0.01,
                            fscore=out["after_icp"]["fscore"], chamfer=out["after_icp"]["chamfer_l2_mean"])


if __name__ == "__main__":
    main()
