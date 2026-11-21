"""Replace COLMAP for orbit videos: the orbit's cameras are known by construction.

The MiniMax-H3 orbit LoRA renders a level camera (no pitch, no roll) circling the image's vertical axis,
aimed at that axis, moving right by 360/72 deg per frame (measured on COLMAP reconstructions of bean_3,
bean_9, kale_3, mint_6: pitch 0-2 deg, roll <1 deg, optical axes hit the rotation axis). With the radius fixed
to 1 (global scale is free), the only unknown is the focal length. We pick it by silhouette consistency: carve a
visual hull from the orbit masks at each candidate focal and keep the focal whose hull best reproduces the
masks (wrong focal lengths make the views disagree and the hull erodes). The hull also seeds the splat.

Writes <run>/3dgs/sparse/0/{cameras,images,points3D}.txt in COLMAP format, so splatfacto, align_splat.py,
train_2dgs.py and colmap2real.py work unchanged.

    python pipeline/known_poses.py $ORBIT_OUTPUTS/<run>_s42 [--focal F | --search 300:1600:25]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STEP = 2 * np.pi / 72  # 73 frames, the last repeats the first


def poses(n, step=STEP, elev=0.0):
    """World z up, orbit radius 1 around the origin; frame 0 at +x. Returns OpenCV c2w (n, 4, 4).
    elev (deg): the circle is raised by this angle and every camera looks down at the origin (a photo
    taken from above gives an orbit that stays above the plant). elev=0 is exactly the original level circle."""
    e = np.radians(elev)
    out = []
    for k in range(n):
        phi = step * k  # increasing phi moves the camera to its right (as in the orbit videos)
        C = np.array([np.cos(phi) * np.cos(e), np.sin(phi) * np.cos(e), np.sin(e)])
        z = -C
        x = np.cross([0.0, 0.0, -1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # = (0, 0, -1) when level
        m = np.eye(4)
        m[:3, :3] = np.stack([x, y, z], 1)
        m[:3, 3] = C
        out.append(m)
    return np.stack(out)


def load_masks(run, n, size):
    ms = []
    for k in range(n):
        m = Image.open(run / "masks" / f"{k:05d}.png").convert("L")
        if m.size != (size, size):
            m = m.resize((size, size), Image.NEAREST)
        ms.append(np.asarray(m) > 127)
    return torch.tensor(np.stack(ms), device="cuda")


def project(pts, c2w, f, size):
    """pts (P,3) -> pixel coords (n,P,2) and depth (n,P) for all cameras."""
    w2c = torch.linalg.inv(c2w)
    pc = torch.einsum("nij,pj->npi", w2c[:, :3, :3], pts) + w2c[:, None, :3, 3]
    uv = f * pc[..., :2] / pc[..., 2:3].clamp(min=1e-6) + size / 2
    return uv, pc[..., 2]


def carve(masks, c2w, f, size, res=128, keep=0.97):
    """Visual hull on a res^3 grid; a voxel survives if it projects inside the mask in >= keep of the views."""
    half = 0.5 * size / f  # half-width of the view frustum at the axis (distance 1)
    g = torch.linspace(-half, half, res, device="cuda")
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    n = masks.shape[0]
    votes = torch.zeros(len(pts), device="cuda")
    for s in range(0, n, 8):
        uv, d = project(pts, c2w[s:s + 8], f, size)
        u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size) & (d > 0)
        hit = masks[s:s + 8][torch.arange(uv.shape[0], device="cuda")[:, None], v.clamp(0, size - 1), u.clamp(0, size - 1)]
        votes += (inside & hit).float().sum(0)
    occ = votes >= keep * n
    return pts[occ], 2 * half / (res - 1)


def recall(masks, c2w, pts, f, size, vox):
    """Mean fraction of mask pixels covered by the hull's projection (hull points splatted at voxel size)."""
    if len(pts) == 0:
        return 0.0
    n = masks.shape[0]
    rs = []
    rad = max(1, int(np.ceil(vox * f / 2)))  # voxel footprint in pixels at distance ~1
    for s in range(0, n, 8):
        uv, _ = project(pts, c2w[s:s + 8], f, size)
        for j in range(uv.shape[0]):
            img = torch.zeros(size, size, device="cuda", dtype=torch.bool)
            u = uv[j, :, 0].round().long().clamp(0, size - 1)
            v = uv[j, :, 1].round().long().clamp(0, size - 1)
            img[v, u] = True
            img = torch.nn.functional.max_pool2d(img[None, None].float(), 2 * rad + 1, 1, rad)[0, 0] > 0
            m = masks[s + j]
            rs.append(float((img & m).sum() / m.sum().clamp(min=1)))
    return float(np.mean(rs))


def colours(run, pts, c2w, f, size, n):
    """Average the orbit frames' colour at each hull point over the views where it projects inside the image."""
    acc = torch.zeros(len(pts), 3, device="cuda")
    cnt = torch.zeros(len(pts), 1, device="cuda")
    for k in range(0, n, 6):
        im = torch.tensor(np.asarray(Image.open(run / "frames" / f"{k:05d}.png").convert("RGB").resize((size, size))),
                          device="cuda").float()
        uv, d = project(pts, c2w[k:k + 1], f, size)
        u, v = uv[0, :, 0].round().long(), uv[0, :, 1].round().long()
        ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        acc[ok] += im[v[ok], u[ok]]
        cnt[ok] += 1
    return (acc / cnt.clamp(min=1)).clamp(0, 255).byte().cpu().numpy()


def write_colmap(out, c2w, f, size, names, pts, cols):
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.txt").write_text(f"# known orbit poses\n1 SIMPLE_PINHOLE {size} {size} {f:.6f} {size / 2} {size / 2}\n")
    lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for i, (m, nm) in enumerate(zip(c2w, names), start=1):
        w2c = np.linalg.inv(m)
        R, t = w2c[:3, :3], w2c[:3, 3]
        qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        if qw > 1e-6:
            qx, qy, qz = (R[2, 1] - R[1, 2]) / (4 * qw), (R[0, 2] - R[2, 0]) / (4 * qw), (R[1, 0] - R[0, 1]) / (4 * qw)
        else:  # 180 deg rotation: pick the largest diagonal
            i_ = int(np.argmax(np.diag(R)))
            j_, k_ = (i_ + 1) % 3, (i_ + 2) % 3
            q = np.zeros(4)
            q[i_ + 1] = np.sqrt(max(0.0, 1 + R[i_, i_] - R[j_, j_] - R[k_, k_])) / 2
            q[j_ + 1] = (R[j_, i_] + R[i_, j_]) / (4 * q[i_ + 1])
            q[k_ + 1] = (R[k_, i_] + R[i_, k_]) / (4 * q[i_ + 1])
            q[0] = (R[k_, j_] - R[j_, k_]) / (4 * q[i_ + 1])
            qw, qx, qy, qz = q
        lines.append(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {nm}")
        lines.append("")
    (out / "images.txt").write_text("\n".join(lines) + "\n")
    pl = ["# POINT3D_ID X Y Z R G B ERROR TRACK[]"]
    for i, (p, c) in enumerate(zip(pts, cols), start=1):
        pl.append(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0")
    (out / "points3D.txt").write_text("\n".join(pl) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--focal", type=float, help="focal in px at 768 (skip the search)")
    ap.add_argument("--search", default="300:1800:25", help="lo:hi:step focal search range (px at 768)")
    ap.add_argument("--size", type=int, help="frame size in px (default: read from frames/00000.png)")
    ap.add_argument("--max-points", type=int, default=100_000)
    ap.add_argument("--cams", type=Path, help="other video models (SEVA / SV3D baselines): JSON with 'c2w' "
                    "(OpenCV, one per frame, orbit target at the origin, radius ~1), optional 'focal' (px at --size; "
                    "otherwise searched as for the MiniMax orbit). All frames are used (no repeated last frame).")
    ap.add_argument("--elev-search", help="lo:hi:step (deg): also search the orbit elevation by hull recall ("
                    "photos taken from above); default: level orbit only")
    ap.add_argument("--flat-fallback", type=float, metavar="F",
                    help="after the search, use focal F instead if the searched focal fits barely better "
                         "(near-orthographic inputs give a flat recall curve, so the searched value is noise)")
    ap.add_argument("--flat-tol", type=float, default=0.01, help="recall margin for --flat-fallback")
    a = ap.parse_args()
    run = a.run.resolve()
    size = a.size or Image.open(run / "frames" / "00000.png").size[0]  # 768 for MiniMax, 576 for SEVA / SV3D
    if a.cams:
        cj = json.loads(a.cams.read_text())
        c2w_np = np.asarray(cj["c2w"], float)
        n = len(c2w_np)
        if cj.get("focal") and not a.focal:
            a.focal = float(cj["focal"])
        if cj.get("search"):
            a.search = cj["search"]
    else:
        n = len(list((run / "masks").glob("*.png"))) - 1  # last frame repeats frame 0
        c2w_np = poses(n)
    c2w = torch.tensor(c2w_np, device="cuda", dtype=torch.float32)
    masks = load_masks(run, n, size)
    scores = {}
    elev, elev_scores = 0.0, {}
    if a.elev_search and not a.cams:  # coarse focal sweep per elevation; the best elevation then gets the usual search
        lo, hi, st = map(float, a.search.split(":"))
        elo, ehi, est = map(float, a.elev_search.split(":"))
        for e in np.arange(elo, ehi + 1e-6, est):
            ct = torch.tensor(poses(n, elev=e), device="cuda", dtype=torch.float32)
            sc = []
            for f in ([a.focal] if a.focal else np.arange(lo, hi + 1e-6, st * 2)):  # --focal (known camera): elevation only
                pts, vox = carve(masks, ct, f, size, res=96)
                sc.append((recall(masks, ct, pts, f, size, vox), float(f)))
            elev_scores[f"{e:.1f}"] = {"recall": round(max(sc)[0], 4), "focal": max(sc)[1]}
        elev = float(max(elev_scores, key=lambda k: elev_scores[k]["recall"]))
        c2w_np = poses(n, elev=elev)
        c2w = torch.tensor(c2w_np, device="cuda", dtype=torch.float32)
        print("elevation search: " + ", ".join(f"{k}: {v['recall']:.3f}" for k, v in elev_scores.items()) + f" -> {elev:.1f} deg")
    alt = np.asarray(cj["alt_c2w"], float) if a.cams and cj.get("alt_c2w") else None
    if alt is not None:  # pick the camera set (e.g. turning direction) whose coarse focal sweep fits the masks best
        lo, hi, st = map(float, a.search.split(":"))
        best_set = []
        for cand in (c2w_np, alt):
            ct = torch.tensor(cand, device="cuda", dtype=torch.float32)
            sc = [recall(masks, ct, *carve(masks, ct, f, size, res=96)[:1], f, size, carve(masks, ct, f, size, res=96)[1])
                  for f in np.arange(lo, hi + 1e-6, st * 2)]
            best_set.append(max(sc))
        if best_set[1] > best_set[0]:
            c2w_np = alt
        c2w = torch.tensor(c2w_np, device="cuda", dtype=torch.float32)
        print(f"camera set: {'alt_c2w' if best_set[1] > best_set[0] else 'c2w'} (coarse recall {best_set[0]:.3f} vs {best_set[1]:.3f})")
    if a.focal:
        best = a.focal
    else:
        lo, hi, st = map(float, a.search.split(":"))
        for f in np.arange(lo, hi + 1e-6, st):  # coarse sweep at low resolution
            pts, vox = carve(masks, c2w, f, size, res=96)
            scores[float(f)] = recall(masks, c2w, pts, f, size, vox)
        fc = max(scores, key=scores.get)
        for f in np.arange(fc - st, fc + st + 1e-6, st / 5):  # refine around the best
            pts, vox = carve(masks, c2w, f, size, res=128)
            scores[float(f)] = recall(masks, c2w, pts, f, size, vox)
        best = max(scores, key=scores.get)
    pts, vox = carve(masks, c2w, best, size, res=160)
    rec = recall(masks, c2w, pts, best, size, vox)
    fallback = {}
    if a.flat_fallback and not a.focal:  # text-to-3D; compared at the chosen elevation
        pts_fb, vox_fb = carve(masks, c2w, a.flat_fallback, size, res=160)  # same resolution as the searched focal
        rec_fb = recall(masks, c2w, pts_fb, a.flat_fallback, size, vox_fb)
        use = rec - rec_fb <= a.flat_tol
        fallback = {"focal_rule": "fallback" if use else "search", "focal_searched": float(best),
                    "recall_searched": rec, "recall_fallback": rec_fb, "flat_tol": a.flat_tol}
        if use:
            best, pts, rec = a.flat_fallback, pts_fb, rec_fb
    if len(pts) > a.max_points:
        pts = pts[torch.randperm(len(pts), device="cuda")[:a.max_points]]
    cols = colours(run, pts, c2w, best, size, n)
    names = [f"{k:05d}.png" for k in range(n)]
    out = run / "3dgs" / "sparse" / "0"
    write_colmap(out, c2w_np, best, size, names, pts.cpu().numpy(), cols)
    info = {"focal": float(best), "hull_recall": rec, "hull_points": int(len(pts)), "frames": n,
            "cams": str(a.cams) if a.cams else "minimax orbit model", **fallback,
            "search": {f"{k:.1f}": round(v, 4) for k, v in sorted(scores.items())},
            "elevation": elev, "elevation_search": elev_scores}
    (run / "3dgs" / "known_poses.json").write_text(json.dumps(info, indent=1))
    rule = f" ({fallback['focal_rule']})" if fallback else ""
    print(f"{run.name}: focal {best:.1f} px{rule}, hull recall {rec:.3f}, {len(pts)} points -> {out}")


if __name__ == "__main__":
    main()
