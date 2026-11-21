"""DISTS for the full-capture upper bound (Table 2). Same protocol as
score_dists.py, but for fullcap_splat.py runs, which have no align_<mode>.json / score_<mode>_cc.json: the placement
is nerfstudio's dataparser transform (exactly as fullcap_splat.py scores), the views and colour correction come from
score_cc.json. Rasteriser from EV_RASTER (the final full-capture runs are antialiased). LPIPS is recomputed as a check.

    EV_RASTER=antialiased [PLANT_DATA=...] python fullcap_dists.py <plant> <run dir (the one with score_cc.json)>
        -> <run dir>/dists_cc.json  {"mean": {"dists", "lpips_check", "lpips_stored"}, "views": [...]}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import eval_real_views as ev
from score_dists import dists_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plant")
    ap.add_argument("dir", type=Path)
    a = ap.parse_args()
    sc = json.loads((a.dir / "score_cc.json").read_text())
    d = json.loads(next((a.dir / "run").glob("*/splatfacto/v1/dataparser_transforms.json")).read_text())
    T, s = np.array(d["transform"]), d["scale"]
    R0 = T[:, :3].T
    sim3 = {"s": 1 / s, "R": R0.tolist(), "t": (-R0 @ T[:, 3]).tolist()}
    gs = ev.load_ply(a.dir / "export" / "splat.ply")
    cams = {c["name"]: c for c in ev.cameras(a.plant, "all")}
    colour = torch.tensor(sc["colour"], device="cuda", dtype=torch.float32)
    rows = []
    for v in sc["views"]:
        c = cams[v["view"]]
        cs = ev.cams_into_splat([c], sim3)[0]
        gt = torch.tensor(np.asarray(Image.open(ev.DATA / a.plant / "images" / "undistorted" / (v["view"] + ".png")).convert("RGB"),
                                     np.float32) / 255, device="cuda")
        m = torch.tensor(np.asarray(Image.open(ev.mask_path(a.plant, v["view"])).convert("L")) > 0, device="cuda")
        with torch.no_grad():
            pred, alpha = ev.render(gs, cs["c2w"], c["fx"], c["fy"], gt.shape[1], gt.shape[0])
            fg = ev.unwhite(pred, alpha)
            fg = (torch.cat([fg, torch.ones_like(fg[..., :1])], -1) @ colour).clamp(0, 1)
            pred = alpha[..., None] * fg + (1 - alpha[..., None])
            gt_c = torch.where(m[..., None], gt, torch.ones_like(gt))
            pr_c = torch.where(m[..., None], pred, torch.ones_like(pred)).clamp(0, 1)
            rows.append({"view": v["view"], "dists": dists_fn(pr_c, gt_c), "lpips_check": ev.lpips_fn(pr_c, gt_c),
                         "lpips_stored": v["lpips_mask"]})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("dists", "lpips_check", "lpips_stored")}
    (a.dir / "dists_cc.json").write_text(json.dumps({"mean": mean, "views": rows}, indent=1))
    print(f"{a.plant} {a.dir.name}: DISTS {mean['dists']:.4f}  LPIPS {mean['lpips_check']:.4f} (stored {mean['lpips_stored']:.4f})", flush=True)


if __name__ == "__main__":
    main()
