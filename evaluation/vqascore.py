"""VQAScore (Lin et al., ECCV 2024) on rendered views: P("Yes") from CLIP-FlanT5-XXL to 'Does this figure show
"<prompt>"? Please answer yes or no.', the question template of t2v_metrics. Per asset, mean over its views.

The VQAScore paper checked this model against human ratings of text-to-3D assets (multi-view renders), so it is
used as is (t2v_metrics 1.2, the release of that paper; 3.0 imports every model family it wraps). The prompt is
metrics.eval_prompts (--fruit-counting: the count prompt without its number, as for CLIP and ImageReward). Each view goes
through metrics.py's load_render first (crop to the subject plus 10% margin, padded square on white, 512 px), so
VQAScore sees the same images as the CLIP and ImageReward scores.

Writes <out>/vqascore.csv (method, species, asset, vqascore) and <out>/vqascore_per_view.csv. Assets already in
vqascore_per_view.csv are skipped, so a job that hits its time limit resumes on resubmit. Run in the vqascore env
(scripts/setup/setup_vqascore.sh); the model loads in fp32 on the CPU (~46 GB RAM) and runs in bf16 (~23 GB GPU):
    python evaluation/vqascore.py RENDERS_DIR --out RESULTS_DIR [--fruit-counting]
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import t2v_metrics

from metrics import eval_prompts, load_render

MODEL = "clip-flant5-xxl"
FIELDS = ["method", "species", "asset", "vqascore"]
VIEW_FIELDS = ["method", "species", "asset", "view", "vqascore"]


def load_scorer(device="cuda"):
    scorer = t2v_metrics.VQAScore(model=MODEL, device=device, cache_dir=os.environ.get("HF_HUB_CACHE"))
    # t2v_metrics opens image paths itself; we pass the cropped PIL views instead
    scorer.model.image_loader = lambda image: image
    return scorer


def score_views(scorer, views, prompt, batch_size):
    """VQAScore of each view (PIL images) against one prompt."""
    scores = []
    for i in range(0, len(views), batch_size):
        batch = views[i:i + batch_size]
        scores += scorer.model.forward(batch, [prompt] * len(batch)).float().cpu().tolist()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders", type=Path, help="<renders>/<method>/<asset>/views/*.png")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fruit-counting", action="store_true", help="fruit-counting species and prompts")
    p.add_argument("--batch-size", type=int, default=12)
    a = p.parse_args()
    prompts = eval_prompts(a.fruit_counting)
    a.out.mkdir(parents=True, exist_ok=True)
    out_csv, view_csv = a.out / "vqascore.csv", a.out / "vqascore_per_view.csv"
    done = set()
    if view_csv.exists():
        with open(view_csv, newline="") as fh:
            done = {(r["method"], r["asset"]) for r in csv.DictReader(fh)}
    new_means, new_views = not out_csv.exists(), not view_csv.exists()
    scorer = load_scorer()

    with open(out_csv, "a", newline="") as fh, open(view_csv, "a", newline="") as fv:
        w, wv = csv.DictWriter(fh, fieldnames=FIELDS), csv.DictWriter(fv, fieldnames=VIEW_FIELDS)
        if new_means:
            w.writeheader()
        if new_views:
            wv.writeheader()
        for method_dir in sorted(d for d in a.renders.iterdir() if d.is_dir()):
            for asset in sorted(d for d in method_dir.iterdir() if d.is_dir()):
                if (method_dir.name, asset.name) in done:
                    continue
                species = asset.name.split("_")[0]
                paths = sorted((asset / "views").glob("*.png"))
                scores = score_views(scorer, [load_render(f) for f in paths], prompts[species], a.batch_size)
                ids = {"method": method_dir.name, "species": species, "asset": asset.name}
                wv.writerows({**ids, "view": path.stem, "vqascore": round(s, 4)} for path, s in zip(paths, scores))
                fv.flush()
                w.writerow({**ids, "vqascore": round(float(np.mean(scores)), 4)})
                fh.flush()
                print(f"{method_dir.name}/{asset.name}: {np.mean(scores):.4f}", flush=True)
    print("wrote", out_csv, view_csv)


if __name__ == "__main__":
    main()
