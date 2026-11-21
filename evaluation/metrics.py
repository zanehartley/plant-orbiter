"""Text-to-3D metrics on rendered views (from render_splats.py), synthetic plants.

Per asset (mean over its views):
  clip_b32, clip_l14   CLIP score = 100 * cos(image, prompt) with openai CLIP ViT-B/32 and ViT-L/14
  rprec_l14            CLIP R-precision: fraction of views where the true species' prompt scores highest
                       among the three species prompts (ViT-L/14)
Per method x species, against real reference photos (only with --reference DIR, holding DIR/<species>/*.png as
RGBA cutouts of plant + pot; preparing them is a separate step: PlantDreamer's own segmented images include the
checkerboard turntable, and the reference should exclude PlantDreamer's LoRA training photos and match the render
elevations):
  kd_incep, fd_incep   kernel / Frechet distance on Inception-v3 pool features (torch-fidelity, FID-standard weights)
  kd_dino, fd_dino     same on DINOv2-base CLS features
KD is the main realism number: FD is biased when there are fewer samples than feature dimensions (2048 / 768).
For KD with bootstrap intervals over assets and reference plants, run kd_bootstrap.py on the cached features.

Also writes per_view.csv (CLIP B/32 and L/14 of every view; view names give elevation and azimuth, for score vs
azimuth) and caches the Inception / DINOv2 features: features/<method>/<asset>.npz (incep, dino, views) and, with
--reference, features/reference/<species>.npz (incep, dino, plant, image; plant = "<species>_<n>" from the file name).

Every rendered view and reference photo is cropped to the subject's bounding box plus a 10% margin, padded square on
white, so framing does not differ between methods. Run in the orbit_cam env:
    python evaluation/metrics.py RENDERS_DIR --out RESULTS_DIR [--reference REFERENCE_DIR]
--fruit-counting scores fruit-counting renders: tomato / lemon against text_prompts.count_eval_prompt ("A photo of
tomato plants with tomatoes", the count prompt without its number); R-precision is then between those two prompts.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "text_to_3d"))
from text_prompts import COUNT_SPECIES, count_eval_prompt  # noqa: E402

PROMPT = "A photo of {species} plants"  # PlantDreamer's guidance prompt: the content we score against
SPECIES = ("bean", "kale", "mint")
CROP_SIZE = 512


def eval_prompts(fruit_counting=False):
    """{species: prompt} the views are scored against: synthetic plants PROMPT, fruit counting the count prompt without its number."""
    if fruit_counting:
        return {s: count_eval_prompt(s) for s in COUNT_SPECIES}
    return {s: PROMPT.format(species=s) for s in SPECIES}


def crop_subject(rgb, mask):
    """Square crop around the mask's bounding box with a 10% margin, padded with white, resized to CROP_SIZE."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return Image.fromarray(rgb).resize((CROP_SIZE, CROP_SIZE))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.1 / 2
    canvas = Image.new("RGB", (int(2 * half) + 1,) * 2, "white")
    canvas.paste(Image.fromarray(rgb), (int(round(half - cx)), int(round(half - cy))))
    return canvas.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)


def load_render(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return crop_subject(rgb, rgb.min(-1) < 240)  # renders are on pure white


def load_real(path):
    rgba = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
    alpha = rgba[..., 3:]
    rgb = (rgba[..., :3] * alpha + (1 - alpha)) * 255
    return crop_subject(rgb.round().astype(np.uint8), alpha[..., 0] > 0.5)


class Features:
    def __init__(self, device):
        self.device = device
        self.clip = {k: (CLIPModel.from_pretrained(m).to(device).eval(), CLIPProcessor.from_pretrained(m))
                     for k, m in [("b32", "openai/clip-vit-base-patch32"), ("l14", "openai/clip-vit-large-patch14")]}
        self.inception = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    @torch.no_grad()
    def text(self, key, prompts):
        model, proc = self.clip[key]
        inputs = proc(text=list(prompts), return_tensors="pt", padding=True).to(self.device)
        t = model.get_text_features(**inputs).pooler_output  # transformers 5: projected embedding is pooler_output
        return torch.nn.functional.normalize(t, dim=-1).float().cpu()  # matches images(): scored on CPU

    @torch.no_grad()
    def images(self, imgs):
        out = {}
        for key, (model, proc) in self.clip.items():
            f = model.get_image_features(**proc(images=imgs, return_tensors="pt").to(self.device)).pooler_output
            out[key] = torch.nn.functional.normalize(f, dim=-1)
        u8 = torch.tensor(np.stack([np.asarray(i) for i in imgs])).permute(0, 3, 1, 2).to(self.device)
        out["incep"] = self.inception(u8)[0]
        out["dino"] = self.dino(**self.dino_proc(images=imgs, return_tensors="pt").to(self.device)).pooler_output
        return {k: v.float().cpu() for k, v in out.items()}


def batched(feats, paths, loader, bs=64):
    chunks = defaultdict(list)
    for i in range(0, len(paths), bs):
        for k, v in feats.images([loader(p) for p in paths[i:i + bs]]).items():
            chunks[k].append(v)
    return {k: torch.cat(v) for k, v in chunks.items()}


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)  # scipy >= 1.18 dropped disp= and returns only the root
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def kernel_distance(a, b, n_subsets=100, subset_size=1000, seed=0):
    """Unbiased MMD^2 with a cubic polynomial kernel, averaged over random subsets (as in torch-fidelity's KID)."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(a), len(b))
    d = a.shape[1]
    vals = []
    for _ in range(n_subsets):
        x, y = a[rng.choice(len(a), m, replace=False)], b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = ((x @ x.T) / d + 1) ** 3, ((y @ y.T) / d + 1) ** 3, ((x @ y.T) / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders", type=Path, help="<renders>/<method>/<asset>/views/*.png")
    p.add_argument("--reference", type=Path, help="DIR/<species>/*.png cleaned real cutouts; omit to skip realism")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fruit-counting", action="store_true", help="fruit-counting species and prompts (see above)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    feats = Features(a.device)
    cache = a.out / "features"

    real = {}
    if a.reference:
        for d in sorted(x for x in a.reference.iterdir() if x.is_dir()):  # skip e.g. prep_reference.py's reference.csv
            photos = sorted(d.glob("*.png"))
            real[d.name] = batched(feats, photos, load_real)
            (cache / "reference").mkdir(parents=True, exist_ok=True)
            np.savez(cache / "reference" / f"{d.name}.npz", incep=real[d.name]["incep"].numpy(),
                     dino=real[d.name]["dino"].numpy(), image=[ph.stem for ph in photos],
                     plant=["_".join(ph.stem.split("_")[:2]) for ph in photos])  # bean_10_0006_train -> bean_10
        print("reference photos:", {sp: len(v["incep"]) for sp, v in real.items()}, flush=True)

    # Generated views: per-asset CLIP metrics, and pooled features per method x species.
    prompts = eval_prompts(a.fruit_counting)
    text = {k: feats.text(k, list(prompts.values())) for k in ("b32", "l14")}
    gen, asset_rows, view_rows = defaultdict(lambda: defaultdict(list)), [], []
    for method_dir in sorted(d for d in a.renders.iterdir() if d.is_dir()):
        for asset in sorted(d for d in method_dir.iterdir() if d.is_dir()):
            species = asset.name.split("_")[0]
            views = sorted((asset / "views").glob("*.png"))
            f = batched(feats, views, load_render)
            s_idx = list(prompts).index(species)
            sims = {k: f[k] @ text[k].T for k in ("b32", "l14")}  # views x species
            view_rows += [{"method": method_dir.name, "species": species, "asset": asset.name, "view": v.stem,
                           "clip_b32": round(100 * sims["b32"][i, s_idx].item(), 3),
                           "clip_l14": round(100 * sims["l14"][i, s_idx].item(), 3)} for i, v in enumerate(views)]
            (cache / method_dir.name).mkdir(parents=True, exist_ok=True)
            np.savez(cache / method_dir.name / f"{asset.name}.npz", incep=f["incep"].numpy(), dino=f["dino"].numpy(),
                     views=[v.stem for v in views])
            asset_rows.append({"method": method_dir.name, "species": species, "asset": asset.name,
                               "views": len(f["incep"]),
                               "clip_b32": round(100 * sims["b32"][:, s_idx].mean().item(), 3),
                               "clip_l14": round(100 * sims["l14"][:, s_idx].mean().item(), 3),
                               "rprec_l14": round((sims["l14"].argmax(1) == s_idx).float().mean().item(), 4)})
            if real:
                for k in ("incep", "dino"):
                    gen[(method_dir.name, species)][k].append(f[k])
            print(f"{method_dir.name}/{asset.name}: clip_l14 {asset_rows[-1]['clip_l14']}", flush=True)

    dist_rows = []
    for (method, species), fk in sorted(gen.items()):
        row = {"method": method, "species": species}
        for k in ("incep", "dino"):
            g, r = torch.cat(fk[k]).numpy().astype(np.float64), real[species][k].numpy().astype(np.float64)
            row.update({"n_gen": len(g), "n_real": len(r), f"kd_{k}": round(kernel_distance(g, r), 5),
                        f"fd_{k}": round(frechet(g, r), 3)})
        dist_rows.append(row)

    for name, rows in [("per_asset.csv", asset_rows), ("per_view.csv", view_rows), ("distribution.csv", dist_rows)]:
        if not rows:
            continue
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
