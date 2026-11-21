"""2D Gaussian Splatting (2DGS) on an orbit run, for clean surfaces and meshes.

Uses the COLMAP poses and RGBA frames that make_splat.sh already produced (<src>/3dgs/sparse/0,
images_rgba/). 2DGS uses flat discs instead of 3D blobs and is trained with depth-normal
consistency + depth distortion losses, so its depth is a surface. The mesh is TSDF-fused from
median depth rendered on rings at several heights.

Run in the nerfstudio env (needs gsplat >= 1.4, open3d):
    conda run --no-capture-output -n nerfstudio python train_2dgs.py outputs/<name>_s<seed> [--steps 30000]
Writes <src>/3dgs/2dgs/: ckpt.pt, eval.json, mesh.ply, mesh.glb
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from po_paths import OUTPUTS

C0 = 0.28209479177387814  # SH band 0


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_colmap(sparse):
    """COLMAP text model -> intrinsics K, {image name: world-to-camera 4x4}, points, colours."""
    if not (sparse / "images.txt").exists():
        import subprocess
        subprocess.run(["colmap", "model_converter", "--input_path", str(sparse), "--output_path", str(sparse),
                        "--output_type", "TXT"], check=True, capture_output=True)
    cam = [l for l in (sparse / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()
    model, w, h, prm = cam[1], int(cam[2]), int(cam[3]), list(map(float, cam[4:]))
    assert model in ("SIMPLE_PINHOLE", "PINHOLE"), model
    fx, fy = (prm[0], prm[0]) if model == "SIMPLE_PINHOLE" else (prm[0], prm[1])
    cx, cy = prm[-2], prm[-1]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    lines = [l for l in (sparse / "images.txt").read_text().splitlines() if not l.startswith("#")]
    views = {}
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rot(list(map(float, p[1:5])))
        w2c[:3, 3] = list(map(float, p[5:8]))
        views[p[9]] = w2c
    pts, cols = [], []
    for l in (sparse / "points3D.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            p = l.split()
            pts.append(list(map(float, p[1:4])))
            cols.append(list(map(float, p[4:7])))
    return K, (w, h), views, np.array(pts, np.float32), np.array(cols, np.float32) / 255


def ssim(a, b):
    """a, b: [1,3,H,W]."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    win = g.expand(3, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, win, padding=5, groups=3), F.conv2d(b, win, padding=5, groups=3)
    s_aa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))).mean()


def render(p, viewmat, K, wh, sh_degree, bg):
    return rasterization_2dgs(
        means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
        opacities=torch.sigmoid(p["opacities"]), colors=torch.cat([p["sh0"], p["shN"]], 1),
        viewmats=viewmat[None], Ks=K[None], width=wh[0], height=wh[1], sh_degree=sh_degree,
        backgrounds=torch.cat([bg, bg.new_zeros(1)])[None], render_mode="RGB+ED", distloss=True, depth_mode="median")


def orbit_views(w2cs, elevs_deg, n_az):
    """Camera-to-world (OpenCV) poses on rings around the subject, fitted to the training cameras."""
    c2w = np.linalg.inv(np.stack(w2cs))
    centres, fwd = c2w[:, :3, 3], c2w[:, :3, 2]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, f in zip(centres, fwd):
        P = np.eye(3) - np.outer(f, f)
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)
    _, _, vt = np.linalg.svd(centres - centres.mean(0))
    n = vt[2] if np.dot(vt[2], -c2w[:, :3, 1].mean(0)) > 0 else -vt[2]  # OpenCV up is -y
    rel = centres - target
    dist = np.linalg.norm(rel, axis=1).mean()
    flat = rel[0] - np.dot(rel[0], n) * n
    u = flat / np.linalg.norm(flat)
    v = np.cross(n, u)
    out = []
    for el in np.radians(elevs_deg):
        for th in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            d = np.cos(el) * (np.cos(th) * u + np.sin(th) * v) + np.sin(el) * n
            pos = target + dist * d
            z = -d  # look at target
            up = n if abs(np.sin(el)) < 0.99 else -(np.cos(th) * u + np.sin(th) * v)
            x = np.cross(z, up)  # OpenCV: x right, y down, z forward
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            m = np.eye(4)
            m[:3, :3] = np.stack([x, y, z], 1)
            m[:3, 3] = pos
            out.append(m)
    return out, dist, target, n


def load_real(plant, grid=96):
    """Real photos + known cameras of a converted plant; initial points from a vote-based visual hull."""
    import eval_real_views as ev
    cams = ev.cameras(plant, "all")
    views, Ks, imgs, masks = {}, [], [], []
    wh = None
    for c in sorted(cams, key=lambda c: c["name"]):
        rgb = np.asarray(Image.open(ev.DATA / plant / "images" / "undistorted" / (c["name"] + ".png")).convert("RGB"), np.float32) / 255
        m = np.asarray(Image.open(ev.mask_path(plant, c["name"])).convert("L"), np.float32)[..., None] / 255
        if m.shape[:2] != rgb.shape[:2] or (wh and (rgb.shape[1], rgb.shape[0]) != wh):
            continue
        wh = (rgb.shape[1], rgb.shape[0])
        views[c["name"]] = np.linalg.inv(c["c2w"]).astype(np.float32)
        Ks.append(np.array([[c["fx"], 0, wh[0] / 2], [0, c["fy"], wh[1] / 2], [0, 0, 1]], np.float32))
        imgs.append(torch.tensor(rgb))
        masks.append(torch.tensor(m))
    names = sorted(views)
    c2ws = [np.linalg.inv(views[n]) for n in names]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for m_ in c2ws:
        P = np.eye(3) - np.outer(m_[:3, 2], m_[:3, 2])
        A += P
        b += P @ m_[:3, 3]
    t = np.linalg.solve(A, b)
    r = 0.35 * np.median([np.linalg.norm(m_[:3, 3] - t) for m_ in c2ws])
    g = np.linspace(-r, r, grid)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + t
    votes, seen, colsum = np.zeros(len(X)), np.zeros(len(X)), np.zeros((len(X), 3))
    for n, K_, im, mk in zip(names, Ks, imgs, masks):
        w2c = views[n]
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        u, v = K_[0, 0] * Xc[:, 0] / Xc[:, 2] + K_[0, 2], K_[1, 1] * Xc[:, 1] / Xc[:, 2] + K_[1, 2]
        ok = (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1]) & (Xc[:, 2] > 0)
        ins = np.zeros(len(X), bool)
        ins[ok] = mk.numpy()[v[ok].astype(int), u[ok].astype(int), 0] > 0.5
        votes += ins
        seen += ok
        colsum[ins] += im.numpy()[v[ins].astype(int), u[ins].astype(int)]
    occ = (votes >= 0.85 * seen) & (seen > 5)
    pts = X[occ].astype(np.float32)
    cols = (colsum[occ] / np.maximum(votes[occ], 1)[:, None]).astype(np.float32)
    if len(pts) > 60000:
        sel = np.random.default_rng(0).choice(len(pts), 60000, replace=False)
        pts, cols = pts[sel], cols[sel]
    print(f"{plant}: {len(names)} real views, hull init {len(pts)} points", flush=True)
    return Ks[0], wh, views, Ks, pts, cols, torch.stack(imgs), torch.stack(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, nargs="?", help="orbit output dir (contains 3dgs/ from make_splat.sh)")
    ap.add_argument("--real-plant", help="instead: train on a dataset plant's REAL photos with its known cameras "
                    "(PLANT_DATA layout) -> full-capture 3D reference in outputs/realcap/<plant>/<out>")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--voxels", type=int, default=384)
    ap.add_argument("--elevations", default="-5,10,25,40,55,70,85")
    ap.add_argument("--normal-weight", type=float, default=0.05, help="depth-normal consistency (higher = smoother)")
    ap.add_argument("--dist-weight", type=float, default=0.01, help="depth distortion (higher = thinner surfaces)")
    ap.add_argument("--smooth", type=int, default=0, help="Taubin smoothing iterations on the mesh")
    ap.add_argument("--min-component", type=float, default=0.05,
                    help="drop mesh pieces smaller than this fraction of the largest piece (floater removal; the paper uses "
                         "0.005: 0.05 also deleted thin leaves that the TSDF leaves disconnected from the stem (pl_pepper06)")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="cut this fraction of the height off the bottom (removes the frayed edge where the photo ends)")
    ap.add_argument("--out", default="2dgs", help="output folder name under <src>/3dgs/")
    ap.add_argument("--mesh-only", action="store_true", help="re-mesh from <out>/ckpt.pt without retraining")
    a = ap.parse_args()
    dev = "cuda"
    if a.real_plant:
        K, wh, views, Ks, pts, cols, imgs, masks = load_real(a.real_plant)
        names = sorted(views)
        out = OUTPUTS / "realcap" / a.real_plant / a.out
        out.mkdir(parents=True, exist_ok=True)
        imgs, masks = imgs.to(dev), masks.to(dev)
    else:
        W3 = a.src / "3dgs"
        out = W3 / a.out
        out.mkdir(exist_ok=True)
        K, wh, views, pts, cols = load_colmap(W3 / "sparse" / "0")
        names = sorted(views)
        Ks = [K] * len(names)
        imgs, masks = [], []
        for nm in names:
            im = np.asarray(Image.open(W3 / "images_rgba" / nm), np.float32) / 255
            imgs.append(torch.tensor(im[..., :3]))
            masks.append(torch.tensor(im[..., 3:]))
        imgs, masks = torch.stack(imgs).to(dev), torch.stack(masks).to(dev)
    Kt = torch.tensor(K, device=dev)
    Kv = torch.tensor(np.stack(Ks), dtype=torch.float32, device=dev)  # per-view intrinsics
    vms = torch.tensor(np.stack([views[n] for n in names]), dtype=torch.float32, device=dev)
    test = [i for i in range(len(names)) if i % 8 == 0]
    train = [i for i in range(len(names)) if i % 8 != 0]

    centres = np.linalg.inv(np.stack([views[n] for n in names]))[:, :3, 3]
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1

    # init from the COLMAP points
    nn_ = NearestNeighbors(n_neighbors=4).fit(pts)
    d, _ = nn_.kneighbors(pts)
    s0 = np.log(np.clip(d[:, 1:].mean(1), 1e-7, None))
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(np.repeat(s0[:, None], 3, 1), dtype=torch.float32, device=dev)),
        "quats": torch.nn.Parameter(torch.rand(N, 4, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(cols, device=dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(key_for_gradient="gradient_2dgs", verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    for step in range(0 if a.mesh_only else a.steps):
        i = train[np.random.randint(len(train))]
        bg = torch.rand(3, device=dev)
        gt = imgs[i] * masks[i] + bg * (1 - masks[i])
        sh_deg = min(step // 1000, 3)
        colors, alphas, normals, normals_d, distort, median, info = render(params, vms[i], Kv[i], wh, sh_deg, bg)
        strategy.step_pre_backward(params, opts, state, step, info)
        rgb = colors[0, ..., :3]
        loss = 0.8 * (rgb - gt).abs().mean() + 0.2 * (1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss = loss + 0.1 * F.binary_cross_entropy(alphas[0].clamp(1e-5, 1 - 1e-5), masks[i])
        if step > 7000:  # depth-normal consistency: rendered normals should match normals of the depth map
            nl = (1 - (normals[0] * normals_d).sum(-1, keepdim=True)) * alphas[0].detach()
            loss = loss + a.normal_weight * nl.mean()
        if step > 3000:
            loss = loss + a.dist_weight * distort.mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 2000 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  gaussians {len(params['means'])}", flush=True)

    if a.mesh_only:
        params = {k: v.to(dev) for k, v in torch.load(out / "ckpt.pt").items()}
    else:
        torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "ckpt.pt")

    # held-out PSNR on subject pixels (same split/metric as the splatfacto runs)
    res, ious = [], []
    with torch.no_grad():
        for i in test:
            bg = torch.ones(3, device=dev)
            colors, alphas, *_ = render(params, vms[i], Kv[i], wh, 3, bg)
            gt = imgs[i] * masks[i] + (1 - masks[i])
            m = masks[i][..., 0] > 0.5
            res.append(float(10 * torch.log10(1 / ((colors[0, ..., :3] - gt) ** 2)[m].mean())))
            al = alphas[0, ..., 0] > 0.5
            ious.append(float((al & m).sum() / (al | m).sum()))
    ev = {"heldout_subject_psnr": float(np.mean(res)), "heldout_sil_iou": float(np.mean(ious)), "per_view": res, "gaussians": len(params["means"]),
          "normal_weight": a.normal_weight, "dist_weight": a.dist_weight}
    if not a.mesh_only:
        (out / "eval.json").write_text(json.dumps(ev, indent=2))
    print(f"held-out subject PSNR {ev['heldout_subject_psnr']:.2f} dB, silhouette IoU {ev['heldout_sil_iou']:.3f}")

    # mesh: TSDF over median depth from rings at several heights
    elevs = [float(e) for e in a.elevations.split(",")]
    poses, dist, target, up = orbit_views([views[n] for n in names], elevs, 36)
    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) > 0.5
        p = params["means"][keep].cpu().numpy()
    voxel = float(np.max(np.percentile(p, 99, 0) - np.percentile(p, 1, 0))) / a.voxels
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(wh[0], wh[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    with torch.no_grad():
        for c2w in poses:
            w2c = np.linalg.inv(c2w).astype(np.float32)
            colors, alphas, _, _, _, median, _ = render(params, torch.tensor(w2c, device=dev), Kt, wh, 3,
                                                        torch.ones(3, device=dev))
            rgb = (colors[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            depth = median[0, ..., 0].cpu().numpy().astype(np.float32)
            depth[alphas[0, ..., 0].cpu().numpy() < 0.5] = 0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(depth),
                depth_scale=1.0, depth_trunc=3 * dist, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, w2c)
    mesh = vol.extract_triangle_mesh()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < a.min_component * counts.max())
    mesh.remove_unreferenced_vertices()
    if a.trim > 0:  # flat cut perpendicular to the orbit's up axis
        h = (np.asarray(mesh.vertices) - target) @ up
        mesh.remove_vertices_by_mask(h < h.min() + a.trim * (h.max() - h.min()))
        # the cut can leave slivers that were only attached below it
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[labels] < 0.01 * counts.max())
        mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if a.smooth:
        colors = np.asarray(mesh.vertex_colors).copy()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=a.smooth)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        # isolated vertices along the cut can come out of the smoother as NaN
        mesh.remove_vertices_by_mask(np.isnan(np.asarray(mesh.vertices)).any(1))
    if len(mesh.triangles) > 500_000:
        mesh = mesh.simplify_quadric_decimation(500_000)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(str(out / "mesh.glb"), mesh)
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {out / 'mesh.ply'}")


if __name__ == "__main__":
    main()
