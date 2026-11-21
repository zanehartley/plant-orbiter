"""Summary of traits_planest.py: per species and placement, for every method, height and leaf-area error
against the PLANesT reference over the plants (method value = mean over seeds 42/1/2 per plant; plants with all 3
seeds only). MAPE and MAE, mean signed error (bias), and Spearman rho across the plants of the species (does the
method rank the plants correctly). Leaf area: the method's occupied 3 mm voxel area (whole reconstruction above the
plant base; it has no leaf labels) vs the reference's leaf-labelled points.

    python traits_stats.py traits_planest.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

METHODS = ["TRELLIS", "TRELLIS.2", "Hunyuan3D 2.1", "SEVA", "SV3D", "Ours"]
r = json.loads(open(sys.argv[1]).read())
for mode in ("input", "oracle"):
    print(f"== {mode} placement")
    for sp in ("pepper", "rose"):
        plants = sorted(p for p in r if p.startswith(f"pl_{sp}"))
        print(f"  {sp}: {len(plants)} plants")
        for m in METHODS:
            rows = []
            for p in plants:
                v = [r[p].get(f"{mode}|{m}|s{s}") for s in (42, 1, 2)]
                if all(v):
                    rows.append((r[p]["reference"]["height_cm"], np.mean([x["height_cm"] for x in v]),
                                 r[p]["reference"]["leaf_area_cm2"], np.mean([x["surface_area_cm2"] for x in v])))
            if len(rows) < 3:
                print(f"    {m:14s} n={len(rows)} (too few)")
                continue
            hr, hm, ar, am = map(np.array, zip(*rows))
            print(f"    {m:14s} n={len(rows):2d}  height MAPE {100 * np.mean(np.abs(hm - hr) / hr):5.1f}%  MAE {np.mean(np.abs(hm - hr)):5.2f} cm"
                  f"  bias {np.mean(hm - hr):+6.2f}  rho {spearmanr(hm, hr)[0]:+.2f} | leaf area MAPE {100 * np.mean(np.abs(am - ar) / ar):6.1f}%"
                  f"  bias {100 * np.mean((am - ar) / ar):+6.1f}%  rho {spearmanr(am, ar)[0]:+.2f}")
