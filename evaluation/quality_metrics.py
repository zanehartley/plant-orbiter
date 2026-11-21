"""Image-quality metrics on rendered views, as the text-to-3DGS papers report them. Per asset, mean over its views:
  imagereward     ImageReward-v1.0 score of each view against the prompt (DiffSplat: averaged over views)
  vbench_imaging  VBench "imaging quality": MUSIQ trained on SPAQ (pyiqa arch, VBench's checkpoint), score / 100;
                  views with a longer side above 512 px are downscaled to 512 (VBench's default 'longer' mode)
  vbench_aesthetic VBench "aesthetic quality": LAION aesthetic linear head (sa_0_4_vit_l_14_linear) on L2-normalised
                  OpenAI CLIP ViT-L/14 image features, bicubic resize + centre crop to 224, score / 10
Each view goes through metrics.py's load_render first (crop to the subject plus 10% margin, padded square on white,
512 px), so these metrics see exactly the images the CLIP metrics see. VBench's own code, read 2026-10-12
(vbench/aesthetic_quality.py, imaging_quality.py, utils.py), is reproduced, not imported: vbench reads videos.

Writes <out>/quality.csv (method, species, asset + the three metrics); summarise_metrics.py merges it with
per_asset.csv. Also <out>/quality_per_view.csv: the same metrics for every view (view names give elevation and
azimuth, for score vs azimuth). Rows are appended as each asset finishes and assets already in quality_per_view.csv
are skipped, so a job that hits its time limit resumes on resubmit and a new method is scored by rerunning on the
whole renders dir; an asset already in quality.csv (scored before per-view rows existed) only gets its per-view rows.
After re-rendering a method, delete its rows from both files to rescore it. Run in the evalq env:
    python evaluation/quality_metrics.py RENDERS_DIR --out RESULTS_DIR [--fruit-counting]
--fruit-counting scores ImageReward against the fruit-counting prompt (metrics.eval_prompts: the count prompt without its number).
"""
import argparse
import csv
import os
from pathlib import Path

import clip
import ImageReward
import numpy as np
import torch
import torch.nn.functional as F
from pyiqa.archs.musiq_arch import MUSIQ
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from metrics import eval_prompts, load_render

METRICS = ["imagereward", "vbench_imaging", "vbench_aesthetic"]
FIELDS = ["method", "species", "asset"] + METRICS
VIEW_FIELDS = ["method", "species", "asset", "view"] + METRICS
WEIGHTS = Path(os.environ.get("ORBIT_WEIGHTS", Path(__file__).resolve().parents[1] / "data" / "weights"))
VBENCH = WEIGHTS / "vbench"
# VBench utils.clip_transform(224), applied to uint8 tensors
CLIP_224 = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=False), CenterCrop(224),
                    lambda x: x.float().div(255.0),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


class Quality:
    def __init__(self, device):
        self.device = device
        self.reward = ImageReward.load("ImageReward-v1.0", device=device, download_root=str(WEIGHTS / "imagereward"))
        self.musiq = MUSIQ(pretrained_model_path=str(VBENCH / "musiq_spaq_ckpt-358bb6af.pth")).to(device)
        self.musiq.training = False  # as VBench (it does not call .eval())
        self.clip, _ = clip.load("ViT-L/14", device=device, download_root=str(VBENCH))
        self.aesthetic = torch.nn.Linear(768, 1)
        self.aesthetic.load_state_dict(torch.load(VBENCH / "sa_0_4_vit_l_14_linear.pth", map_location="cpu"))
        self.aesthetic.to(device).eval()

    @torch.no_grad()
    def asset(self, views, prompt):
        """Each metric for each of an asset's views (PIL images): {metric: list over views}."""
        u8 = torch.from_numpy(np.stack([np.asarray(v) for v in views])).permute(0, 3, 1, 2).to(self.device)  # N,3,H,W
        reward = self.reward.score(prompt, views)
        reward = reward if isinstance(reward, list) else [reward]
        imaging = []
        for img in u8:  # one view at a time, as VBench
            x = img[None].float()
            if max(x.shape[-2:]) > 512:  # VBench 'longer' mode
                scale = 512 / max(x.shape[-2:])
                x = F.interpolate(x, size=(round(x.shape[-2] * scale), round(x.shape[-1] * scale)), mode="bilinear",
                                  antialias=False)
            imaging.append(self.musiq(x / 255.0).item())
        feats = F.normalize(self.clip.encode_image(CLIP_224(u8)).float(), dim=-1, p=2)
        aesthetic = self.aesthetic(feats).squeeze(-1) / 10
        return {"imagereward": [float(r) for r in reward], "vbench_imaging": [i / 100 for i in imaging],
                "vbench_aesthetic": aesthetic.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders", type=Path, help="<renders>/<method>/<asset>/views/*.png")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fruit-counting", action="store_true", help="fruit-counting species and prompts")
    a = p.parse_args()
    prompts = eval_prompts(a.fruit_counting)
    a.out.mkdir(parents=True, exist_ok=True)
    out_csv, view_csv = a.out / "quality.csv", a.out / "quality_per_view.csv"

    def assets_in(path):
        if not path.exists():
            return set()
        with open(path, newline="") as fh:
            return {(r["method"], r["asset"]) for r in csv.DictReader(fh)}

    have_means, done = assets_in(out_csv), assets_in(view_csv)
    new_means, new_views = not out_csv.exists(), not view_csv.exists()
    quality = Quality("cuda")

    with open(out_csv, "a", newline="") as fh, open(view_csv, "a", newline="") as fv:
        w, wv = csv.DictWriter(fh, fieldnames=FIELDS), csv.DictWriter(fv, fieldnames=VIEW_FIELDS)
        if new_means:
            w.writeheader()
        if new_views:
            wv.writeheader()
        for method_dir in sorted(d for d in a.renders.iterdir() if d.is_dir()):
            for asset in sorted(d for d in method_dir.iterdir() if d.is_dir()):
                key = (method_dir.name, asset.name)
                if key in done:
                    continue
                species = asset.name.split("_")[0]
                paths = sorted((asset / "views").glob("*.png"))
                scores = quality.asset([load_render(f) for f in paths], prompts[species])
                ids = {"method": method_dir.name, "species": species, "asset": asset.name}
                wv.writerows({**ids, "view": path.stem, **{m: round(scores[m][i], 4) for m in METRICS}}
                             for i, path in enumerate(paths))
                fv.flush()
                row = {**ids, **{m: round(float(np.mean(scores[m])), 4) for m in METRICS}}
                if key not in have_means:
                    w.writerow(row)
                    fh.flush()
                print(f"{method_dir.name}/{asset.name}: {row}", flush=True)
    print("wrote", out_csv, view_csv)


if __name__ == "__main__":
    main()
