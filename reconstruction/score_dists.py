"""DISTS (Ding et al. 2020, perceptual structure + texture similarity; piq implementation) for existing scored runs.
Re-renders each run exactly as eval_real_views.score did: the same
held-out views (read from the run's score_<mode>_cc.json), its placement (align_<mode>.json), its input-view colour
correction and the rasteriser it was scored with (splat_settings.json, else classic), plant masked onto white, full
resolution, as LPIPS. Also recomputes LPIPS as a check that the renders match the stored scores.

    python score_dists.py <plant> <dir with align/score json> <ply> [--modes input oracle]
        -> <dir>/dists_<mode>_cc.json  {"mean": {"dists", "lpips_check"}, "views": [...]}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import piq
import torch
from PIL import Image

import eval_real_views as ev

_dists = None


def dists_fn(a, b):
    global _dists
    if _dists is None:
        _dists = piq.DISTS(reduction="none").cuda()
    return float(_dists(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("dir", type=Path)
    ap.add_argument("ply", type=Path)
    ap.add_argument("--modes", nargs="+", default=["input", "oracle"])
    a = ap.parse_args()
    st = a.dir / "splat_settings.json"
    ev.RASTER = json.loads(st.read_text()).get("rasterize_mode", "classic") if st.exists() else "classic"
    gs = ev.load_ply(a.ply)
    cams = {c["name"]: c for c in ev.cameras(a.plant, "all")}
    for mode in a.modes:
        sc = json.loads((a.dir / f"score_{mode}_cc.json").read_text())
        sim3 = json.loads((a.dir / f"align_{mode}.json").read_text())
        colour = torch.tensor(sc["colour"], device="cuda", dtype=torch.float32) if sc.get("colour") else None
        stored = {v["view"]: v for v in sc["views"]}
        rows = []
        for name in stored:
            c = cams[name]
            cs = ev.cams_into_splat([c], sim3)[0]
            gt = torch.tensor(np.asarray(Image.open(ev.DATA / a.plant / "images" / "undistorted" / (name + ".png")).convert("RGB"),
                                         np.float32) / 255, device="cuda")
            m = torch.tensor(np.asarray(Image.open(ev.mask_path(a.plant, name)).convert("L")) > 0, device="cuda")
            with torch.no_grad():
                pred, alpha = ev.render(gs, cs["c2w"], c["fx"], c["fy"], gt.shape[1], gt.shape[0])
                if colour is not None:
                    fg = ev.unwhite(pred, alpha)
                    fg = (torch.cat([fg, torch.ones_like(fg[..., :1])], -1) @ colour).clamp(0, 1)
                    pred = alpha[..., None] * fg + (1 - alpha[..., None])
                gt_c = torch.where(m[..., None], gt, torch.ones_like(gt))
                pr_c = torch.where(m[..., None], pred, torch.ones_like(pred)).clamp(0, 1)
                rows.append({"view": name, "dists": dists_fn(pr_c, gt_c), "lpips_check": ev.lpips_fn(pr_c, gt_c),
                             "lpips_stored": stored[name]["lpips_mask"]})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("dists", "lpips_check", "lpips_stored")}
        (a.dir / f"dists_{mode}_cc.json").write_text(json.dumps({"mean": mean, "views": rows}, indent=1))
        print(f"{a.plant} {a.dir.name} {mode}: DISTS {mean['dists']:.4f}  LPIPS {mean['lpips_check']:.4f} (stored {mean['lpips_stored']:.4f})", flush=True)


if __name__ == "__main__":
    main()
