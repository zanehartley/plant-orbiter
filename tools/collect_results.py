"""Collect the per-run result files behind the paper's tables into a few CSVs (run once, on the machine that holds the
experiment outputs). Everything in results/ is produced by this script; make_tables.py reads only results/.

    python tools/collect_results.py /path/to/orbit_cam/outputs results/

Layout of the outputs it reads (as produced by the pipeline scripts):
    <ds>/final/<plant>_s<seed>/3dgs/            ours (ds = pd PlantDreamer, bb Botany-Bot, pl PLANesT-3D)
    <ds>/baselines/<method>/<plant>[_seed<n>]/  TRELLIS, TRELLIS.2, Hunyuan3D 2.1 (+ /3dgs for SEVA, SV3D)
    <ds>/final_level/<run>/3dgs/                ablation: level orbit cameras (no elevation search)
    ../archive/outputs/kfocal/<ds>/<run>/3dgs/  ablation: known focal length
    fullcap_final/<plant>/fold<k>/              full-capture splats (upper bound)
    gs/zoom/<stem>_s42/3dgs, gs/baselines/...   GrowSplat
"""
import csv, glob, gzip, json, os, sys
import numpy as np

OUT_SRC, DST = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(DST, exist_ok=True)
os.chdir(OUT_SRC)
SEEDS = (42, 1, 2)
DATASETS = {"pd": "PlantDreamer", "bb": "Botany-Bot", "pl": "PLANesT-3D"}
BASE = {"TRELLIS": ("trellis", ""), "TRELLIS.2": ("trellis2", ""), "Hunyuan3D 2.1": ("hunyuan", ""),
        "SEVA": ("seva", "/3dgs"), "SV3D": ("sv3d", "/3dgs")}


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def run_dir(ds, method, plant, seed):
    if method == "Ours":
        return "%s/final/%s_s%d/3dgs" % (ds, plant, seed)
    dn, sub = BASE[method]
    return "%s/baselines/%s/%s%s%s" % (ds, dn, plant, "" if seed == 42 else "_seed%d" % seed, sub)


def species(ds, plant):
    if ds == "pl":
        return "pepper" if "pepper" in plant else "rose"
    if ds == "pd":
        return plant.split("_")[0]
    return ""


def rot_angle(a, b):
    Ra, Rb = np.array(a["R"], float), np.array(b["R"], float)
    Ra /= np.cbrt(np.linalg.det(Ra)); Rb /= np.cbrt(np.linalg.det(Rb))
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def plants(ds):
    out = sorted({os.path.basename(p)[:-4] for p in glob.glob("%s/final/*_s42" % ds)})
    return [p for p in out if "ribes" not in p]  # redcurrant: not in the paper


scores, views, geo, place, cams = [], [], [], [], []
for ds in DATASETS:
    for plant in plants(ds):
        for method in ["Ours"] + list(BASE):
            for seed in SEEDS:
                d = run_dir(ds, method, plant, seed)
                for ref in ("input", "oracle"):
                    s = load("%s/score_%s_cc.json" % (d, ref))
                    if s is None:
                        continue
                    dd = load("%s/dists_%s_cc.json" % (d, ref))
                    m = s["mean"]
                    scores.append(dict(dataset=DATASETS[ds], species=species(ds, plant), plant=plant, method=method,
                                       seed=seed, placement=ref, sil_iou=m["sil_iou"], psnr_mask=m["psnr_mask"],
                                       ssim_mask=m["ssim_mask"], lpips_mask=m["lpips_mask"],
                                       dists=dd["mean"]["dists"] if dd else ""))
                    for v in s.get("views", []):
                        views.append(dict(dataset=DATASETS[ds], plant=plant, method=method, seed=seed, placement=ref,
                                          view=v["view"], sil_iou=v["sil_iou"], psnr_mask=v["psnr_mask"],
                                          ssim_mask=v["ssim_mask"], lpips_mask=v["lpips_mask"]))
                # 3D: meshes of ours / SEVA / SV3D are the fixed 2DGS meshes (gao_oracle_v3); generators use their own
                g = load(d + ("/gao_oracle_v3.json" if method in ("Ours", "SEVA", "SV3D") else "/gao_oracle.json"))
                if g:
                    g = g.get("after_icp", g)
                    geo.append(dict(dataset=DATASETS[ds], species=species(ds, plant), plant=plant, method=method,
                                    seed=seed, chamfer_l2_x1e3=g["chamfer_l2_mean"] * 1e3, fscore=g["fscore"]))
                if method in ("TRELLIS", "TRELLIS.2", "Hunyuan3D 2.1"):
                    a, b = load(d + "/align_input.json"), load(d + "/align_oracle.json")
                    if a and b:
                        place.append(dict(dataset=DATASETS[ds], plant=plant, method=method, seed=seed,
                                          input_vs_oracle_deg=rot_angle(a, b)))
                if method == "Ours":
                    k = load(d + "/known_poses.json")
                    if k:
                        cams.append(dict(dataset=DATASETS[ds], plant=plant, seed=seed, focal_px=k.get("focal"),
                                         elevation_deg=k.get("elevation", 0.0)))

# ablations: level cameras (runs where the elevation search picked > 0) and known focal length
abl = []
for ds in DATASETS:
    for d in sorted(glob.glob("%s/final_level/*_s*" % ds)):
        run = os.path.basename(d); plant, seed = run.rsplit("_s", 1)
        if "ribes" in plant:
            continue
        for ref in ("input", "oracle"):
            for setting, p in (("elevation_search", "%s/final/%s/3dgs" % (ds, run)), ("level", d + "/3dgs")):
                s = load("%s/score_%s_cc.json" % (p, ref))
                if s:
                    m = s["mean"]
                    abl.append(dict(ablation="elevation", dataset=DATASETS[ds], plant=plant, seed=int(seed), setting=setting,
                                    placement=ref, sil_iou=m["sil_iou"], psnr_mask=m["psnr_mask"], lpips_mask=m["lpips_mask"]))
    for d in sorted(glob.glob("../archive/outputs/kfocal/%s/*_s42" % ds)):
        run = os.path.basename(d); plant = run.rsplit("_s", 1)[0]
        for ref in ("input", "oracle"):
            for setting, p in (("estimated_focal", "%s/final/%s/3dgs" % (ds, run)), ("known_focal", d + "/3dgs")):
                s = load("%s/score_%s_cc.json" % (p, ref))
                if s:
                    m = s["mean"]
                    abl.append(dict(ablation="focal", dataset=DATASETS[ds], plant=plant, seed=42, setting=setting,
                                    placement=ref, sil_iou=m["sil_iou"], psnr_mask=m["psnr_mask"], lpips_mask=m["lpips_mask"]))

# full capture (upper bound): per plant, mean over folds
full = []
for d in sorted(glob.glob("fullcap_final/*")):
    plant = os.path.basename(d)
    ds = "bb" if plant.startswith("bb_") else "pl" if plant.startswith("pl_") else "pd" if plant.split("_")[0] in ("bean", "kale", "mint") else None
    if ds is None or "ribes" in plant:
        continue  # wheat / redcurrant: not in the paper
    for f in sorted(glob.glob(d + "/fold*/score_cc.json")) or ([d + "/score_cc.json"] if os.path.exists(d + "/score_cc.json") else []):
        m = load(f)["mean"]
        full.append(dict(dataset=DATASETS[ds], species=species(ds, plant), plant=plant, fold=os.path.basename(os.path.dirname(f)),
                         sil_iou=m["sil_iou"], psnr_mask=m["psnr_mask"], ssim_mask=m["ssim_mask"], lpips_mask=m["lpips_mask"]))
dist_full = {}
for f in glob.glob("fullcap_final/*/dists_cc.json") + glob.glob("fullcap_final/*/fold*/dists_cc.json"):
    dist_full.setdefault(f.split("/")[1], []).append(load(f)["mean"]["dists"])
for r in full:
    r["dists_plant_mean"] = float(np.mean(dist_full[r["plant"]])) if r["plant"] in dist_full else ""

# GrowSplat silhouette IoU (seed 42; first quinoa date excluded in the paper, kept here and flagged)
gs = []
GS = {"Ours": "gs/zoom/{s}_s42/3dgs", "TRELLIS": "gs/baselines/trellis/{s}", "TRELLIS.2": "gs/baselines/trellis2/{s}",
      "Hunyuan3D 2.1": "gs/baselines/hunyuan/{s}", "SEVA": "gs/baselines/seva/{s}/3dgs", "SV3D": "gs/baselines/sv3d/{s}/3dgs"}
for stem in sorted(os.path.basename(p)[:-4] for p in glob.glob("gs/zoom/*_s42")):
    for method, pat in GS.items():
        for ref in ("input", "oracle"):
            s = load(pat.format(s=stem) + "/score_%s_cc.json" % ref)
            if s:
                gs.append(dict(stem=stem, species="quinoa" if "_quinoa_" in stem else "sequoia" if "_sequoia_" in stem else "",
                               excluded_in_paper=int("20221117" in stem), method=method, placement=ref, sil_iou=s["mean"]["sil_iou"]))


def write(name, rows, gz=False):
    if not rows:
        print("EMPTY", name); return
    op = gzip.open if gz else open
    with op(os.path.join(DST, name), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("%-28s %6d rows" % (name, len(rows)))


write("scores_2d.csv", scores)
write("scores_2d_per_view.csv.gz", views, gz=True)
write("geometry_3d.csv", geo)
write("placement_generators.csv", place)
write("orbit_cameras.csv", cams)
write("ablations.csv", abl)
write("full_capture.csv", full)
write("growsplat_iou.csv", gs)
# trait / height results are already compact JSON: copied as they are
for src in ("traits_planest_potcut2_v3.json", "gs_heights/heights_input_zoom_v3.json"):
    if os.path.exists(src):
        with open(src) as f, open(os.path.join(DST, os.path.basename(src)), "w") as g:
            g.write(f.read())
        print("copied", src)
    else:
        print("MISSING", src)
