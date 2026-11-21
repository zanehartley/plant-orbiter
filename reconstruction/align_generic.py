"""Align a camera-less splat (e.g. TRELLIS output, canonical frame) to a PlantDreamer plant and score it.

No cameras come with the splat, so: centre it on the plant (optical-axis intersection of the real cameras),
search over up-axis choice x yaw x scale for the best silhouette match at the reference views, then run the
same silhouette/colour refinement as align_splat.py:
  --refine input   reference = the input photo only  [honest]
  --refine oracle  reference = all real test views    [upper bound, like ICP-to-GT in Gao et al. 2026]

    python align_generic.py <plant> <splat.ply> --input-view 0009_train --refine oracle -o out_dir
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import eval_real_views as ev
from align_splat import axes_target, refine


def rot_axis(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def sil_iou(gs, s, R, t, cams, masks):
    g = ev.apply_sim3(gs, s, torch.tensor(R, dtype=torch.float32, device="cuda"),
                      torch.tensor(t, dtype=torch.float32, device="cuda"))
    tot = 0
    for c, m in zip(cams, masks):
        f = m.shape[1] / c["w"]
        _, a = ev.render(g, c["c2w"], c["fx"] * f, c["fy"] * f, m.shape[1], m.shape[0])
        a = a > 0.5
        tot += float((a & m).sum() / max((a | m).sum(), 1))
    return tot / len(cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("ply", type=Path)
    ap.add_argument("--input-view", required=True)
    ap.add_argument("--suffix", default="", help="added to align/score file names")
    ap.add_argument("--refine", default="input", choices=["input", "oracle"])
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--up", help="source up axis as x,y,z when the generator has a fixed convention (default: guess)")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="ignore this percentile at each end when measuring the source height (stray floaters)")
    ap.add_argument("--constrain", action="store_true",
                    help="as align_splat --constrain (GrowSplat seedlings): refine rotates only about the real "
                         "up axis and never moves along the input camera's viewing axis (one view can't fix depth), so "
                         "the plant stays at the rig distance it was started at; outputs get the suffix _con")
    ap.add_argument("--tilt-search", action="store_true",
                    help="also try source up axes tilted 8 and 16 deg (8 directions each) around --up; used where a "
                         "hand-set per-plant up axis is visibly off (e.g. Hunyuan bean_1, bb_pepper1)")
    a = ap.parse_args()
    # per-generator defaults (e.g. outputs/baselines/hunyuan/align_hints.json: fixed up axis, floaters to trim)
    hints = a.ply.resolve().parent.parent / "align_hints.json"
    if hints.exists():
        h = json.loads(hints.read_text())
        # per_plant: meshes the generator emitted in another orientation (checked visually, see the hints' "why")
        # per_run (by run dir, e.g. bean_3_0009_train_seed1) beats per_plant, which was checked on the seed-42 outputs
        # only and so applies to those (dirs without _seed<N>); other seeds orient independently
        run = a.ply.resolve().parent.name
        a.up = (a.up or h.get("per_run", {}).get(run)
                or (h.get("per_plant", {}).get(a.plant) if "_seed" not in run else None) or h.get("up"))
        a.trim = a.trim or h.get("trim", 0.0)
        print("align hints:", hints, h)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.constrain:
        a.suffix = "_con" + a.suffix

    real = {c["name"]: c for c in ev.cameras(a.plant, "all")}
    inputs = [a.input_view]
    excl = set()
    test = [c for c in ev.cameras(a.plant, "eval") if c["name"] not in inputs and c["name"] not in excl]
    ref = [real[n] for n in inputs] if a.refine == "input" else test
    # some plants (mint_4) have no masks for later eval views; skip them, as eval_real_views does
    ref = [c for c in ref if ev.mask_path(a.plant, c["name"]).exists()]
    masks = []
    for c in ref:
        m = Image.open(ev.mask_path(a.plant, c["name"])).convert("L")
        masks.append(torch.tensor(np.asarray(m.resize((c["w"] // 4, c["h"] // 4))) > 0, device="cuda"))

    gs = ev.load_ply(a.ply)
    pts = gs["means"].cpu().numpy()[gs["opacities"].cpu().numpy() > 0.3]
    centre_src = np.median(pts, 0)
    target = axes_target([c["c2w"] for c in real.values()])
    up = -np.mean([c["c2w"][:3, 1] for c in real.values()], 0)
    up /= np.linalg.norm(up)
    # real plant size from the input view's mask (projected height at the plant's distance)
    c0 = real[a.input_view]
    m0 = np.asarray(Image.open(ev.mask_path(a.plant, a.input_view)).convert("L")) > 0
    rows = np.where(m0.any(1))[0]
    real_h = (rows.max() - rows.min()) / c0["fy"] * np.linalg.norm(c0["c2w"][:3, 3] - target)
    # start points for the plant's centre: the rig's aim point (all cameras' axes), and the point at that
    # distance on the input photo's ray through the plant mask's centre. In GrowSplat the rig aims ~30 cm above a
    # seedling, so starts at the aim point never overlapped it (search IoU 0, honest IoU 0); the search keeps the start
    # that fits best, so where the aim point already worked nothing gets worse.
    ys0, xs0 = np.where(m0)
    ray = c0["c2w"][:3, :3] @ np.array([(xs0.mean() - c0["w"] / 2) / c0["fx"], (ys0.mean() - c0["h"] / 2) / c0["fy"], 1.0])
    centres = [target, c0["c2w"][:3, 3] + ray / np.linalg.norm(ray) * np.linalg.norm(c0["c2w"][:3, 3] - target)]

    best = None
    # Plants grow upward: take the source's long axis, oriented from the wide end (pot) to the narrow end
    # (foliage/stem), as its up direction. A silhouette-only search can otherwise accept an upside-down fit.
    ext = np.ptp(pts, 0)
    ax = int(np.argmax(ext))
    h = pts[:, ax]
    lo, hi = pts[h < np.percentile(h, 20)], pts[h > np.percentile(h, 80)]
    spread = lambda q: np.ptp(np.delete(q, ax, 1), 0).prod()
    up_auto = np.eye(3)[ax] * (1 if spread(lo) >= spread(hi) else -1)
    if a.up:
        up_auto = np.asarray([float(x) for x in a.up.split(",")])
        up_auto /= np.linalg.norm(up_auto)
    print("source up axis:", up_auto.tolist())
    cands = []
    ups = [up_auto]
    if a.tilt_search:
        perp = np.linalg.svd(up_auto[None])[2][1:]  # two axes perpendicular to up_auto
        for tilt in np.radians([8, 16]):
            for phi in np.radians(np.arange(0, 360, 45)):
                ax = np.cos(phi) * perp[0] + np.sin(phi) * perp[1]
                ups.append(rot_axis(ax, tilt) @ up_auto)
    for up_src in [u.tolist() for u in ups]:
        u = np.asarray(up_src, float)
        hu = pts @ u
        h_src = np.percentile(hu, 100 - a.trim) - np.percentile(hu, a.trim)
        # rotation taking the source up axis onto the real up axis
        v = np.cross(u, up)
        R_up = np.eye(3) if np.linalg.norm(v) < 1e-8 else rot_axis(v, np.arccos(np.clip(u @ up, -1, 1)))
        for yaw in np.radians(np.arange(0, 360, 10)):
            R = rot_axis(up, yaw) @ R_up
            for k in (0.8, 1.0, 1.25):
                s = k * real_h / h_src
                for cen in centres:
                    t = cen - s * R @ centre_src
                    cands.append((sil_iou(gs, s, R, t, ref[:6], masks[:6]), s, R, t))
    cands.sort(key=lambda x: -x[0])
    best = cands[0]
    # refine the top 3 starting points fully and keep the best: the splat has no cameras, so be generous
    kw = {"axis": up, "ray": c0["c2w"][:3, 2] / np.linalg.norm(c0["c2w"][:3, 2])} if a.constrain else {}
    fits = [refine(gs, s0, R0, t0, ref, a.plant, iters=600, **kw) for _, s0, R0, t0 in cands[:3]]
    s, R, t, loss = min(fits, key=lambda f: f[3])
    sim3 = {"s": float(s), "R": R.tolist(), "t": list(map(float, t))}
    with torch.no_grad():
        mean, rows_ = ev.score(a.plant, gs, test, sim3)
    (a.out / f"align_{a.refine}{a.suffix}.json").write_text(json.dumps({"s": float(s), "R": R.tolist(), "t": list(map(float, t)),
                                                             "search_iou": best[0], "loss": loss}, indent=1))
    (a.out / f"score_{a.refine}{a.suffix}.json").write_text(json.dumps({"mean": mean, "views": rows_}, indent=1))
    print(f"{a.ply} align={a.refine}: " + json.dumps({k: round(v, 4) for k, v in mean.items()}))
    with torch.no_grad():  # same, with a colour correction fitted on the input photo(s) only
        col = ev.fit_colour(a.plant, gs, [real[n] for n in inputs], sim3)
        mean_cc, rows_cc = ev.score(a.plant, gs, test, sim3, col)
    outdir = a.out
    (outdir / f"score_{a.refine}{a.suffix}_cc.json").write_text(json.dumps({"mean": mean_cc, "views": rows_cc, "colour": col.tolist()}, indent=1))
    print("  colour-corrected: " + json.dumps({k: round(v, 4) for k, v in mean_cc.items()}))


if __name__ == "__main__":
    main()
