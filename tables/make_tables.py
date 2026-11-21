"""Rebuild every results table of the paper from results/ (no experiment outputs needed).

    python tables/make_tables.py [results_dir]        -> prints all tables as markdown

Conventions (as in the paper): a method's value per plant = mean over seeds 42, 1, 2, then the mean over plants;
only plants with all 3 seeds are used. 2D scores are masked and colour-corrected, at real held-out views.
"""
import csv, json, os, sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
METHODS = ["TRELLIS", "TRELLIS.2", "Hunyuan3D 2.1", "SEVA", "SV3D", "Ours"]
SEEDS = ("42", "1", "2")


def rows(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))


def per_plant(rs, keys, value):
    """{key tuple: {plant: [values over seeds]}} -> {key: [plant means]} (plants with all 3 seeds)."""
    acc = defaultdict(lambda: defaultdict(dict))
    for r in rs:
        if r[value] == "":
            continue
        acc[tuple(r[k] for k in keys)][r["plant"]][r["seed"]] = float(r[value])
    return {k: {p: np.mean([v[s] for s in SEEDS]) for p, v in d.items() if all(s in v for s in SEEDS)} for k, d in acc.items()}


def table(title, header, body, note=""):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for b in body:
        print("| " + " | ".join(b) + " |")
    if note:
        print(f"\n{note}")


# ---------------------------------------------------------------- Table: single-photo 2D (main)
s2d = rows("scores_2d.csv")
for r in s2d:
    r["group"] = "PLANesT" if r["dataset"] == "PLANesT-3D" else {"PlantDreamer": "PD", "Botany-Bot": "BB"}[r["dataset"]]
METRICS = [("sil_iou", "IoU", 3), ("dists", "DISTS", 3), ("psnr_mask", "PSNR", 2), ("lpips_mask", "LPIPS", 3)]
G = ["PD", "BB", "PLANesT"]
pp = {m: per_plant(s2d, ("group", "method", "placement"), m) for m, _, _ in METRICS}
full = rows("full_capture.csv")
fc = defaultdict(lambda: defaultdict(list))
for r in full:
    g = "PLANesT" if r["dataset"] == "PLANesT-3D" else {"PlantDreamer": "PD", "Botany-Bot": "BB"}[r["dataset"]]
    for m in ("sil_iou", "psnr_mask", "lpips_mask"):
        fc[(g, m)][r["plant"]].append(float(r[m]))
    if r["dists_plant_mean"] != "":
        fc[(g, "dists")][r["plant"]] = [float(r["dists_plant_mean"])]
for place in ("input", "oracle"):
    body = []
    for meth in METHODS:
        line = [meth]
        for m, _, d in METRICS:
            for g in G:
                v = pp[m].get((g, meth, place), {})
                line.append(f"{np.mean(list(v.values())):.{d}f}" if v else "-")
        body.append(line)
    line = ["Full capture"]
    for m, _, d in METRICS:
        for g in G:
            v = [np.mean(x) for x in fc[(g, m)].values()]
            line.append(f"{np.mean(v):.{d}f}" if v else "-")
    body.append(line)
    n = {g: len(pp["sil_iou"].get((g, "Ours", place), {})) for g in G}
    table(f"Single-photo reconstruction at real held-out views, {place}-view placement (n: PD {n['PD']}, BB {n['BB']}, PLANesT {n['PLANesT']})",
          ["Method"] + [f"{lab} {g}" for _, lab, _ in METRICS for g in G], body)

# ---------------------------------------------------------------- Supplementary: per PLANesT species
for place in ("input", "oracle"):
    body = []
    for meth in METHODS:
        line = [meth]
        for m, _, d in METRICS:
            for sp in ("pepper", "rose"):
                v = per_plant([r for r in s2d if r["species"] == sp], ("method", "placement"), m).get((meth, place), {})
                line.append(f"{np.mean(list(v.values())):.{d}f}" if v else "-")
        body.append(line)
    table(f"PLANesT per species, {place}-view placement (supplement)", ["Method"] + [f"{lab} {sp}" for _, lab, _ in METRICS for sp in ("pepper", "rose")], body)

# ---------------------------------------------------------------- Table: 3D geometry
geo = [r for r in rows("geometry_3d.csv") if "pepper09" not in r["plant"]]
for r in geo:
    r["group"] = {"PlantDreamer": "PlantDreamer", "Botany-Bot": "Botany-Bot"}.get(r["dataset"], r["species"])
GG = ["PlantDreamer", "Botany-Bot", "pepper", "rose"]
cd, fs = per_plant(geo, ("group", "method"), "chamfer_l2_x1e3"), per_plant(geo, ("group", "method"), "fscore")
body = [[meth] + sum([[f"{np.mean(list(cd[(g, meth)].values())):.1f}", f"{np.mean(list(fs[(g, meth)].values())):.3f}"]
                      if (g, meth) in cd else ["-", "-"] for g in GG], []) for meth in METHODS]
table("3D geometry (oracle placement + ICP; CD x1e3, F at 1% of the diagonal; pl_pepper09 excluded)",
      ["Method"] + [f"{x} {g} (n={len(cd.get((g, 'Ours'), {}))})" for g in GG for x in ("CD", "F")], body)

# ---------------------------------------------------------------- Table: placement of camera-less generators
pl = [r for r in rows("placement_generators.csv") if r["dataset"] in ("PlantDreamer", "Botany-Bot")]
body = []
for meth in ("TRELLIS", "TRELLIS.2", "Hunyuan3D 2.1"):
    a = np.array([float(r["input_vs_oracle_deg"]) for r in pl if r["method"] == meth])
    body.append([meth, f"{np.median(a):.0f}", f"{(a > 90).sum()}/{len(a)}"])
table("Placement: rotation between input-view and oracle fit (PlantDreamer + Botany-Bot, 3 seeds)", ["Method", "Median (deg)", ">90 deg"], body)

# ---------------------------------------------------------------- Traits (PLANesT, input-view placement)
tr = json.load(open(os.path.join(R, "traits_planest_potcut2_v3.json")))
body = []
for sp in ("pepper", "rose"):
    plants = sorted(p for p in tr if p.startswith(f"pl_{sp}") and "pepper09" not in p)
    for meth in METHODS:
        hr, hm, ar, am = [], [], [], []
        for p in plants:
            v = [tr[p].get(f"input|{meth}|s{s}") for s in SEEDS]
            if all(v):
                hr.append(tr[p]["reference"]["height_cm"]); hm.append(np.mean([x["height_cm"] for x in v]))
                ar.append(tr[p]["reference"]["leaf_area_cm2"]); am.append(np.mean([x["surface_area_cm2"] for x in v]))
        hr, hm, ar, am = map(np.array, (hr, hm, ar, am))
        body.append([f"{sp} (n={len(hr)})", meth, f"{100 * np.mean(np.abs(hm - hr) / hr):.1f}", f"{np.mean(np.abs(hm - hr)):.2f}",
                     f"{np.mean(hm - hr):+.2f}", f"{spearmanr(hm, hr)[0]:.2f}", f"{100 * np.mean(np.abs(am - ar) / ar):.1f}", f"{spearmanr(am, ar)[0]:.2f}"])
table("PLANesT traits, input-view placement (pl_pepper09 excluded)",
      ["Species", "Method", "Height MAPE %", "Height MAE cm", "Height bias cm", "Height rho", "Leaf area MAPE %", "Leaf area rho"], body)

# ---------------------------------------------------------------- GrowSplat: height over the season + IoU
gh = json.load(open(os.path.join(R, "heights_input_zoom_v3.json")))
giou = rows("growsplat_iou.csv")
body = []
for sp in ("quinoa", "sequoia"):
    dates = sorted(d for d in gh if f"_{sp}_" in d and "20221117" not in d)  # first quinoa date excluded (plant ~3 cm)
    for meth in METHODS:
        ref = np.array([gh[d]["reference"]["height_cm"] for d in dates if meth in gh[d]])
        est = np.array([gh[d][meth]["height_cm"] for d in dates if meth in gh[d]])
        iou = {pl_: np.mean([float(r["sil_iou"]) for r in giou if r["method"] == meth and r["placement"] == pl_
                             and r["species"] == sp and r["excluded_in_paper"] == "0"]) for pl_ in ("input", "oracle")}
        if len(ref) == 0:
            body.append([sp, meth, "-", "-", "-", "-", "-", "-"]); continue
        body.append([f"{sp} (n={len(ref)})", meth, f"{np.mean(np.abs(est - ref)):.2f}", f"{100 * np.mean(np.abs(est - ref) / ref):.1f}",
                     f"{spearmanr(est, ref)[0]:.2f}", f"{iou['input']:.3f}", f"{iou['oracle']:.3f}"])
table("GrowSplat: height over the season (input-view placement, seed 42) + silhouette IoU",
      ["Species", "Method", "Height MAE cm", "Height MAPE %", "rho", "IoU input", "IoU oracle"], body)

# ---------------------------------------------------------------- Ablations (supplement)
ab = rows("ablations.csv")
for name, a, b in (("elevation", "elevation_search", "level"), ("focal", "estimated_focal", "known_focal")):
    body = []
    for ds in ("PlantDreamer", "Botany-Bot", "PLANesT-3D"):
        for place in ("input", "oracle"):
            sel = [r for r in ab if r["ablation"] == name and r["dataset"] == ds and r["placement"] == place]
            if not sel:
                continue
            for setting in (a, b):
                v = [r for r in sel if r["setting"] == setting]
                body.append([ds, place, setting, str(len(v))] + [f"{np.mean([float(r[k]) for r in v]):.{d}f}" for k, d in (("sil_iou", 3), ("psnr_mask", 2), ("lpips_mask", 3))])
    table(f"Ablation: {name} (runs)", ["Dataset", "Placement", "Setting", "n runs", "IoU", "PSNR", "LPIPS"], body)

# ---------------------------------------------------------------- Paper Table 3: height MAPE / rho, GrowSplat + PLANesT
def height_stats(ref, est):
    ref, est = np.asarray(ref), np.asarray(est)
    return 100 * np.mean(np.abs(est - ref) / ref), spearmanr(est, ref)[0]


body = []
for meth in METHODS:
    cells, mapes = [], []
    for sp in ("quinoa", "sequoia"):
        dates = sorted(d for d in gh if f"_{sp}_" in d and "20221117" not in d and meth in gh[d])
        m, rho = height_stats([gh[d]["reference"]["height_cm"] for d in dates], [gh[d][meth]["height_cm"] for d in dates])
        cells += [f"{m:.1f}", f"{rho:.2f}"]; mapes.append(m)
    for sp in ("pepper", "rose"):
        plants = [p for p in sorted(tr) if p.startswith(f"pl_{sp}") and "pepper09" not in p
                  and all(tr[p].get(f"input|{meth}|s{s}") for s in SEEDS)]
        m, rho = height_stats([tr[p]["reference"]["height_cm"] for p in plants],
                              [np.mean([tr[p][f"input|{meth}|s{s}"]["height_cm"] for s in SEEDS]) for p in plants])
        cells += [f"{m:.1f}", f"{rho:.2f}"]; mapes.append(m)
    body.append([meth] + cells + [f"{np.mean(mapes):.1f}"])
table("Paper Table 3: plant height from one photo (input-view placement)",
      ["Method", "Quinoa MAPE", "Quinoa rho", "Sequoia MAPE", "Sequoia rho", "Pepper MAPE", "Pepper rho", "Rose MAPE", "Rose rho", "Mean MAPE"], body)
