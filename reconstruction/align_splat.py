"""Align an orbit splat to a PlantDreamer plant's real cameras, then score it at the real test views.

Coordinate chain: nerfstudio world --(dataparser transform)--> COLMAP world of the generated frames
--(similarity)--> real world. The similarity is initialised by putting generated frame 0's camera on
the real camera of the input photo, with scale from the ratio of camera-to-plant distances, and then
refined by matching rendered silhouettes (and colour) to real masks:
  --refine input   only the photo(s) given to the pipeline (front, + pinned views)  [honest]
  --refine oracle  all real test views (upper bound, like ICP-to-ground-truth in Gao et al. 2026)

    python align_splat.py <plant> <orbit run dir> [--refine input|oracle|none]
Writes <run>/3dgs/align_<mode>.json (sim3 for splat.ply) and prints the eval_real_views scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import eval_real_views as ev
from train_2dgs import load_colmap
from po_paths import OUTPUTS

# nerfstudio's dataparser_transforms.json already includes its COLMAP axis convention (verified: COLMAP
# points land on the splat to 0.07% of scene size with this mapping, vs 0.3% for a random rotation)
P = np.eye(3)


def ns_to_colmap(run3d):
    """Similarity mapping nerfstudio-world points (splat.ply) back to the generated COLMAP world."""
    cfg = next((run3d / "runs").glob("*/*/v1/dataparser_transforms.json"))
    d = json.loads(cfg.read_text())
    T = np.array(d["transform"])
    Tr, Tt, sc = T[:, :3], T[:, 3], d["scale"]
    R0 = P.T @ Tr.T
    return 1 / sc, R0, -R0 @ Tt


def axes_target(c2ws):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for m in c2ws:
        c, f = m[:3, 3], m[:3, 2]  # OpenCV: camera looks down +z
        Pm = np.eye(3) - np.outer(f, f)
        A += Pm
        b += Pm @ c
    return np.linalg.solve(A, b)


def so3_exp(w):
    th = torch.linalg.norm(w) + 1e-12
    k = w / th
    K = torch.zeros(3, 3, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def refine(gs, s, R, t, cams, plant, iters=300, scale_down=4, axis=None, ray=None):
    """Optimise a correction (log-scale, rotation, translation) about the plant centre to match real masks.
    axis (unit 3-vector): rotate only about it; ray (unit 3-vector): no translation along it (--constrain)."""
    tgt = []
    for c in cams:
        mp = ev.mask_path(plant, c["name"])
        if not mp.exists():  # some plants (mint_4) lack masks for later eval views; skipped, as eval_real_views does
            continue
        m = Image.open(mp).convert("L")
        im = Image.open(ev.DATA / plant / "images" / "undistorted" / (c["name"] + ".png")).convert("RGB")
        if m.size != im.size:  # misnumbered masks (mint_4): eval_real_views skips these views too
            continue
        w, h = m.size[0] // scale_down, m.size[1] // scale_down
        mt = torch.tensor(np.asarray(m.resize((w, h))) > 0, device="cuda").float()
        it = torch.tensor(np.asarray(im.resize((w, h)), np.float32) / 255, device="cuda")
        tgt.append((c, mt, it, w, h))
    base = ev.apply_sim3(gs, s, torch.tensor(R, dtype=torch.float32, device="cuda"),
                         torch.tensor(t, dtype=torch.float32, device="cuda"))
    centre = base["means"].median(0).values.detach()
    ls = torch.zeros(1, device="cuda", requires_grad=True)
    w = torch.full((3,), 1e-4, device="cuda", requires_grad=True)
    dt = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([{"params": [ls, w], "lr": 3e-3}, {"params": [dt], "lr": 3e-3}])
    ax = None if axis is None else torch.tensor(axis, dtype=torch.float32, device="cuda")
    ry = None if ray is None else torch.tensor(ray, dtype=torch.float32, device="cuda")
    wv = (lambda: (w @ ax) * ax) if ax is not None else (lambda: w)
    tv = (lambda: dt - (dt @ ry) * ry) if ry is not None else (lambda: dt)
    for i in range(iters):
        Rc = so3_exp(wv())
        sc = torch.exp(ls)
        g = dict(base)
        g["means"] = sc * (base["means"] - centre) @ Rc.T + centre + tv()
        g["scales"] = base["scales"] * sc
        g["quats"] = ev.quat_mul(ev.rot_to_quat(Rc).expand_as(base["quats"]), base["quats"])
        loss = 0
        for c, mt, it, ww, hh in tgt:
            f = 1 / scale_down
            rgb, a = ev.render(g, c["c2w"], c["fx"] * f, c["fy"] * f, ww, hh)
            inter = (a * mt).sum()
            loss = loss + 1 - inter / (a.sum() + mt.sum() - inter + 1e-6)
            loss = loss + 0.5 * ((rgb - it).abs() * mt[..., None]).sum() / (mt.sum() * 3 + 1e-6)
        opt.zero_grad()
        (loss / len(tgt)).backward()
        opt.step()
    with torch.no_grad():
        Rc, sc, dtv = so3_exp(wv()).cpu().numpy(), float(torch.exp(ls)), tv().cpu().numpy()
    c0 = centre.cpu().numpy()
    # x' = sc Rc (s R x + t - c0) + c0 + dt
    return sc * s, Rc @ R, sc * Rc @ (t - c0) + c0 + dtv, float(loss) / len(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("run", type=Path, help="orbit output dir, e.g. outputs/bean_3_0009_train_s42")
    ap.add_argument("--refine", default="input", choices=["none", "input", "oracle"])
    ap.add_argument("--pinned", nargs="+", required=True, help="real view name(s) given as input, front photo first")
    ap.add_argument("--constrain", action="store_true",
                    help="constrained fit: rotate only about the real up axis and never move along the "
                         "input camera's viewing ray (one view can't fix depth); outputs get the suffix _con")
    ap.add_argument("--suffix", default="", help="added to align/score file names")
    a = ap.parse_args()
    run3d = a.run / "3dgs"
    excl = set()
    inputs = a.pinned  # the real photo(s) given to the pipeline; never scored

    # cameras
    K, wh, views, _, _ = load_colmap(run3d / "sparse" / "0")
    gen_c2w = {n: np.linalg.inv(m) for n, m in views.items()}
    real = {c["name"]: c for c in ev.cameras(a.plant, "all")}
    f_cam = real[a.pinned[0]]  # --pinned lists the front photo first

    # nerfstudio world -> generated COLMAP world
    s0, R0, t0 = ns_to_colmap(run3d)
    # generated COLMAP world -> real world: camera of frame 0 onto the real front camera
    Rg, Cg = gen_c2w["00000.png"][:3, :3], gen_c2w["00000.png"][:3, 3]
    Rr, Cr = f_cam["c2w"][:3, :3], f_cam["c2w"][:3, 3]
    ipj = a.run / "input.json"
    shift = json.loads(ipj.read_text()).get("shift_x", 0) if ipj.exists() else 0
    if shift:  # centred input (batch_single_photo --center): frame 0 looks at photo column w/2 - shift, not the centre
        phi = np.arctan(-shift / f_cam["fx"])
        Rr = Rr @ np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    ipd = json.loads(ipj.read_text()) if ipj.exists() else {}
    crop_ray = None
    if "shift_y" in ipd and (ipd.get("shift_y") or ipd.get("shift_x")) and ipd.get("crop"):
        # cropped input (--crop, GrowSplat): a crop keeps the camera's orientation and only moves the plant off the
        # optical axis, so undo any yaw applied above and put the orbit target on the ray through the crop centre
        # (photo pixel (w/2 - shift_x, h/2 - shift_y)) instead of rotating the camera (a 17 deg pitch tilted the orbit)
        Rr = f_cam["c2w"][:3, :3]
        crop_ray = Rr @ np.array([-ipd["shift_x"] / f_cam["fx"], -ipd["shift_y"] / f_cam["fy"], 1.0])
        crop_ray /= np.linalg.norm(crop_ray)
    Tg = axes_target(list(gen_c2w.values()))
    d_gen = np.linalg.norm(Cg - Tg)
    d_real = np.linalg.norm(Cr - axes_target([c["c2w"] for c in real.values()]))
    R1 = Rr @ Rg.T
    # the orbit's focal can differ from the real camera's (known-pose runs use the orbit's own focal): keep the
    # plant at the real distance and scale it so it covers the same part of the input photo. Equal focals
    # (COLMAP runs, focal fixed to the real camera) reduce to s1 = d_real / d_gen, t1 = Cr - s1 R1 Cg.
    f_real = f_cam["fx"] * wh[0] / max(f_cam["w"], f_cam["h"])  # real focal in px at the orbit resolution
    ip = a.run / "input.json"  # padded input (batch_single_photo --pad): the orbit canvas spans (1 + pad) x the photo
    if ip.exists():
        f_real /= 1 + json.loads(ip.read_text()).get("pad", 0.0)
    s1 = d_real / d_gen * K[0, 0] / f_real
    t1 = Cr + d_real / d_gen * R1 @ (Tg - Cg) - s1 * R1 @ Tg
    if crop_ray is not None:  # generated target onto the crop-centre ray at the real distance
        t1 = Cr + d_real * crop_ray - s1 * R1 @ Tg
    s, R, t = s1 * s0, R1 @ R0, s1 * R1 @ t0 + t1

    gs = ev.load_ply(run3d / "export" / "splat.ply")
    tag = ("_con" if a.constrain else "") + a.suffix
    loss = None
    if a.refine != "none":
        test = [c for c in ev.cameras(a.plant, "eval") if c["name"] not in inputs and c["name"] not in excl]
        cams = [real[n] for n in inputs] if a.refine == "input" else test
        kw = {}
        if a.constrain:
            up = -np.mean([c["c2w"][:3, 1] for c in real.values()], 0)
            kw = {"axis": up / np.linalg.norm(up), "ray": f_cam["c2w"][:3, 2] / np.linalg.norm(f_cam["c2w"][:3, 2])}
        s, R, t, loss = refine(gs, s, R, t, cams, a.plant, **kw)
    out = {"s": float(s), "R": R.tolist(), "t": list(map(float, t)), "mode": a.refine, "inputs": inputs, "loss": loss}
    (run3d / f"align_{a.refine}{tag}.json").write_text(json.dumps(out, indent=1))
    sim3 = {"s": float(s), "R": R.tolist(), "t": list(map(float, t))}
    with torch.no_grad():
        # never score on a view that was given to the pipeline as input
        mean, rows = ev.score(a.plant, gs, [c for c in ev.cameras(a.plant, "eval") if c["name"] not in inputs and c["name"] not in excl], sim3)
    (run3d / f"score_{a.refine}{tag}.json").write_text(json.dumps({"mean": mean, "views": rows}, indent=1))
    print(f"{a.run.name} align={a.refine}: " + json.dumps({k: round(v, 4) for k, v in mean.items()}))
    with torch.no_grad():  # same, with a colour correction fitted on the input photo(s) only
        col = ev.fit_colour(a.plant, gs, [real[n] for n in inputs], sim3)
        mean_cc, rows_cc = ev.score(a.plant, gs, [c for c in ev.cameras(a.plant, "eval") if c["name"] not in inputs and c["name"] not in excl], sim3, col)
    outdir = run3d
    (outdir / f"score_{a.refine}{tag}_cc.json").write_text(json.dumps({"mean": mean_cc, "views": rows_cc, "colour": col.tolist()}, indent=1))
    print("  colour-corrected: " + json.dumps({k: round(v, 4) for k, v in mean_cc.items()}))


if __name__ == "__main__":
    main()
