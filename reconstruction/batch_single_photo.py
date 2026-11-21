"""Single-photo reconstruction of many plants (PlantDreamer layout; other datasets after conversion, PLANT_DATA):
orbit video, known-pose splat, then scores at the real test views with both placements (input and oracle;
eval_real_views via align_splat) and, unless --no-mesh, the 3D metrics of Gao et al. on a 2DGS mesh (gao_eval).

    python batch_single_photo.py bean_1 bean_3 ... [--seed 42]
Results: <outputs>/<plant>_<front>_s<seed>/3dgs/{score_*.json, gao.json}; summary in <outputs>/summary.json.
scripts/reconstruct_plants.sh runs this with the paper's settings, followed by DISTS and the final mesh.
"""

import argparse
import os
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import ring_views
from po_paths import DATASETS, INPUTS, OUTPUTS, PIPELINE

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PLANT_DATA", DATASETS / "PlantDreamer_Data" / "plants"))  # converted datasets: set PLANT_DATA
NS = ["conda", "run", "--no-capture-output", "-n", "nerfstudio", "python"]


def sh(cmd, log=None, env=None):
    print(" ".join(map(str, cmd)), flush=True)
    import os
    e = dict(os.environ, **(env or {}))
    with open(log, "a") if log else open("/dev/null", "w") as f:
        subprocess.run(list(map(str, cmd)), check=True, stdout=f if log else None, stderr=subprocess.STDOUT if log else None, env=e)


def clean_input(plant, name, out_dir, tight=None, pad=0.0, center=False, crop=0.0):
    D = DATA / plant
    im = np.asarray(Image.open(D / "images" / "undistorted" / (name + ".png")).convert("RGB")).copy()
    tp = tight / plant / f"{name}.png" if tight else None
    if tp is not None and tp.exists():  # hybrid tight mask (masks/README.md): no grey halo, stems kept
        m = np.asarray(Image.open(tp).convert("L")) > 127
    else:
        m = np.asarray(Image.open(D / "images" / "mask" / (name + ".png.png")).convert("L")) > 0
    im[~m] = 255
    h, w = m.shape
    side = int(round(max(h, w) * (1 + pad)))  # pad > 0: white margin round the photo, centre kept (intrinsics unchanged)
    dx = 0
    if center:  # pot on the image's vertical centre line: the orbit model spins about it (turntable, not "earth orbit")
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        base = np.where(m[int(y1 - 0.15 * (y1 - y0)):y1 + 1].any(0))[0]  # bottom 15% of the plant: the pot
        dx = int(round(w / 2 - (base.min() + base.max()) / 2))  # shift right by dx (photo px)
        # grow the canvas (symmetrically, centre kept) if the shifted plant would leave it
        need = 2 * max(w / 2 - (xs.min() + dx), (xs.max() + dx) - w / 2) + 2
        side = max(side, int(np.ceil(need)))
    dy = 0
    if crop:  # crop > 0: square canvas = crop x the plant's bounding box, centred on it (GrowSplat: tall
        # portrait photos with the pot at the bottom edge, where a young plant is a speck on the canvas's bottom edge)
        ys, xs = np.where(m)
        side = int(round(crop * max(ys.max() - ys.min(), xs.max() - xs.min())))
        dx = int(round(w / 2 - (xs.min() + xs.max()) / 2))
        dy = int(round(h / 2 - (ys.min() + ys.max()) / 2))
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(Image.fromarray(im), ((side - w) // 2 + dx, (side - h) // 2 + dy))
    p = out_dir / f"{plant}_{name}.png"
    sq.save(p)
    clean_input.shift = dx  # read by main() for input.json
    clean_input.shift_y = dy
    return p, side


def front_view(pdir):
    """The input view: front.txt for converted datasets, else PlantDreamer's ring logic (as for the seed-42 runs)."""
    if (pdir / "front.txt").exists():  # converted datasets name their input view explicitly
        return (pdir / "front.txt").read_text().strip()
    names, c2w = ring_views.load_views(pdir / "transforms" / "original" / "transforms.json")
    keep = [i for i, n in enumerate(names) if (pdir / "images" / "mask" / (n + ".png")).exists()]
    names, c2w = [names[i] for i in keep], c2w[keep]
    try:
        r = ring_views.pick_ring_auto(names, c2w, exclude=ring_views.lora_views(pdir))
    except RuntimeError:  # spiral captures (e.g. bean_1): only a near side-on front view is needed
        r = ring_views.pick_ring(names, c2w, ring_views.lora_views(pdir), tol=12, band=8, min_views=10,
                                 min_elev=0, max_elev=40)
    return Path(r["front"]).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plants", nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mesh", action="store_true", help="skip the 2DGS mesh + geometry metrics")
    ap.add_argument("--summary", default="summary.json")
    ap.add_argument("--tight-masks", type=Path, help="dir of hybrid input masks (masks/plantdreamer_hybrid)")
    ap.add_argument("--inputs", type=Path, default=INPUTS / "pd_clean", help="where the cleaned input photos go")
    ap.add_argument("--outputs", type=Path, default=OUTPUTS / "clean", help="where the runs go")
    ap.add_argument("--orbit-only", action="store_true", help="stop after generating the orbit video")
    ap.add_argument("--center", action="store_true", help="shift the input so the pot sits on the image's vertical centre line "
                    "(the orbit model's spin axis); recorded in <run>/input.json for align_splat.py")
    ap.add_argument("--crop", type=float, default=0.0, help="square input = CROP x the plant's bounding box, centred on "
                    "it (e.g. 1.3; GrowSplat's tall photos); input.json records the shifts for the honest alignment")
    ap.add_argument("--pad", type=float, default=0.0, help="white margin round the input photo, as a fraction of its "
                    "long side (0.3 = canvas 1.3x; for plants the photo clips, e.g. wheat)")
    a = ap.parse_args()
    inp = a.inputs
    inp.mkdir(parents=True, exist_ok=True)
    outs = a.outputs
    outs.mkdir(parents=True, exist_ok=True)
    summary_p = outs / a.summary
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    for plant in a.plants:
        try:
            pdir = DATA / plant
            front = front_view(pdir)
            photo, side = clean_input(plant, front, inp, a.tight_masks, a.pad, a.center, a.crop)
            shift, shift_y = getattr(clean_input, "shift", 0), getattr(clean_input, "shift_y", 0)
            cams = {c["img_name"]: c for c in json.loads((pdir / "plantdreamer" / "3dgs" / "cameras.json").read_text())}
            focal = cams[front]["fx"] * 768 / side
            run = outs / f"{plant}_{front}_s{a.seed}"
            if not (run / "masks" / "00072.png").exists():
                sh(["conda", "run", "--no-capture-output", "-n", "orbit_cam", "python", PIPELINE / "orbit.py", photo,
                    "--seed", a.seed, "--remove-bg", "--out", outs])
            if a.pad or a.center or a.crop:  # align_splat.py needs the canvas-to-photo ratio and the shift
                long_side = max(cams[front]["width"], cams[front]["height"])
                (run / "input.json").write_text(json.dumps({"pad": side / long_side - 1, "side": side, "shift_x": shift,
                                                            "shift_y": shift_y, "crop": a.crop}))
            log = run / "batch.log"
            if a.orbit_only:
                print(plant, "orbit done:", run, flush=True)
                continue
            if not (run / "3dgs" / "export" / "splat.ply").exists():
                try:
                    sh([PIPELINE / "make_splat_kp.sh", run, f"{plant}_kp"], log)
                except subprocess.CalledProcessError:
                    if not (run / "3dgs" / "export" / "splat.ply").exists():
                        raise  # only the optional video step at the end may fail
            for m in ("input", "oracle"):
                if not (run / "3dgs" / f"score_{m}.json").exists():
                    sh(NS + [ROOT / "align_splat.py", plant, run, "--refine", m, "--pinned", front], log)
            if a.no_mesh:
                res = {m: json.loads((run / "3dgs" / f"score_{m}.json").read_text())["mean"] for m in ("input", "oracle")}
                res["front"] = front
                summary[plant] = res
                summary_p.write_text(json.dumps(summary, indent=1))
                print(plant, "photo(input):", {k: round(v, 3) for k, v in res["input"].items()}, flush=True)
                continue
            if not (run / "3dgs" / "2dgs_v2" / "mesh.ply").exists():
                sh(NS + [ROOT / "train_2dgs.py", run, "--out", "2dgs_v2", "--normal-weight", "0.2",
                         "--dist-weight", "0.02", "--smooth", "3"], log)
            sh(["python3", ROOT / "colmap2real.py", run / "3dgs", "input"], log)
            sh(NS + [ROOT / "gao_eval.py", plant, run / "3dgs" / "2dgs_v2" / "mesh.ply", "--sim3",
                     run / "3dgs" / "colmap2real_input.json", "-o", run / "3dgs" / "gao.json"], log)
            res = {m: json.loads((run / "3dgs" / f"score_{m}.json").read_text())["mean"] for m in ("input", "oracle")}
            res["gao"] = json.loads((run / "3dgs" / "gao.json").read_text())["after_icp"]
            res["front"] = front
            summary[plant] = res
            summary_p.write_text(json.dumps(summary, indent=1))
            print(plant, "photo(input):", {k: round(v, 3) for k, v in res["input"].items()},
                  "gao:", {k: round(res["gao"][k], 4) for k in ("chamfer_l2_mean", "normal_consistency", "fscore")}, flush=True)
        except Exception as e:  # keep going with the other plants
            print("FAILED", plant, repr(e), flush=True)


if __name__ == "__main__":
    main()
