"""GrowSplat height summary: per species and method, height error over the season vs each date's reference
cloud, from growsplat_heights.py output (5 mm voxel heights, zoom runs for ours). Phases by the reference height:
seedling < 10 cm, plateau >= 95% of the final reference height, growth in between. Excluded dates: the
first quinoa date (2.8 cm, 77 px in the input: too small to reconstruct).

    python gs_height_stats.py outputs/gs_heights/gs_heights_vox_input.json [more.json ...]
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

EXCLUDE = {"gs_quinoa_20221117"}
METHODS = ["Ours", "TRELLIS", "TRELLIS.2", "Hunyuan3D 2.1", "SEVA", "SV3D"]

for f in sys.argv[1:]:
    r = json.loads(open(f).read())
    print(f"== {f}")
    for sp in ("quinoa", "sequoia"):
        ks = sorted(k for k in r if f"_{sp}_" in k and k not in EXCLUDE)
        ref = np.array([r[k]["reference"]["height_cm"] for k in ks])
        phase = np.where(ref < 10, "seedling", np.where(ref >= 0.95 * ref[-1], "plateau", "growth"))
        cnt = {p: int((phase == p).sum()) for p in ("seedling", "growth", "plateau")}
        print(f"  {sp}: {len(ks)} dates  phases {cnt}")
        for m in METHODS:
            ok = np.array([bool(r[k].get(m)) for k in ks])
            h = np.array([r[k][m]["height_cm"] if r[k].get(m) else np.nan for k in ks])
            e = np.abs(h - ref)[ok]
            ph = " ".join(f"{p} {np.abs(h - ref)[ok & (phase == p)].mean():.2f}" if (ok & (phase == p)).any() else f"{p} -"
                          for p in ("seedling", "growth", "plateau"))
            rho = spearmanr(h[ok], ref[ok])[0]
            print(f"    {m:14s} n={ok.sum():2d}  MAE {e.mean():5.2f} cm  MAPE {100 * (e / ref[ok]).mean():5.1f}%  "
                  f"bias {(h - ref)[ok].mean():+5.2f}  rho {rho:.3f} | {ph}")
