"""Realism with uncertainty: kernel distance (KD) to the real reference photos, bootstrapped over whole assets and
whole reference plants.

metrics.py reports one KD per method x species, but its 864 generated views are 12 assets and the reference photos
are 12 / 3 / 6 real bean / kale / mint plants, so views and photos are not independent samples. Each bootstrap
replicate resamples generated assets and reference plants with replacement (all views / photos of a drawn cluster
come along) and recomputes KD.

  kd            MMD^2 with the cubic polynomial kernel k(x, y) = (x.y / d + 1)^3 (the KID kernel), where the
                within-set terms use only pairs from different clusters (views of different assets, photos of
                different plants). This estimates KD between "a view of a random asset" and "a photo of a random
                plant" without the within-cluster pairs, which are much more alike than independent samples. It is
                what makes the bootstrap unbiased: resampling duplicates whole clusters, and pairs between copies of
                the same cluster are left out like any within-cluster pair (with metrics.py's sample-level estimator
                they inflated every replicate). With one sample per cluster it is metrics.py's unbiased KD; with
                clusters it differs from metrics.py's distribution.csv value, so the main table uses this one.
  ci_lo, ci_hi  95% percentile interval over the replicates. A replicate that draws a single reference plant
                every time (kale: 3 plants, 1 in 9 replicates) has no between-plant pairs and is dropped;
                n_boot_used says how many remain
  diff_vs_ref   kd minus the reference method's kd (default: ours); p_vs_ref is the two-sided bootstrap p of that
                difference. Within a replicate every method sees the same reference-plant draw, so the comparison is
                paired on the reference; generated assets are drawn independently per method.
species "all" is the mean over the three species (the main-table number), computed per replicate, for methods with
every species. Values are raw (KD-Inception is usually shown x100; KD-DINOv2 as is).

Reads the features metrics.py caches: <features>/<method>/<asset>.npz and <features>/reference/<species>.npz.
Writes realism.csv. CPU only (orbit_cam env):
    python evaluation/kd_bootstrap.py $ORBIT_OUTPUTS/synthetic_plants/eval/results/features \
        --out $ORBIT_OUTPUTS/synthetic_plants/eval/results
"""
import argparse
import csv
from pathlib import Path

import numpy as np

FEATURES = ("incep", "dino")
SPECIES = ("bean", "kale", "mint")


class ClusterKD:
    """KD between generated samples x and reference samples y, grouped into clusters (assets / plants).

    The kernel matrices are summed into cluster x cluster blocks once, so any resample of whole clusters, given as a
    count per cluster, is a few small quadratic forms. Within-set terms leave out every pair inside one original
    cluster, including pairs between copies of it in a resample.
    """

    def __init__(self, x, x_clusters, y, y_clusters):
        d = x.shape[1]
        kernel = lambda a, b: (a @ b.T / d + 1) ** 3  # noqa: E731
        onehot = lambda c: (np.unique(c)[:, None] == np.asarray(c)[None]).astype(np.float64)  # noqa: E731
        ax, ay = onehot(x_clusters), onehot(y_clusters)
        self.sxx, self.syy, self.sxy = ax @ kernel(x, x) @ ax.T, ay @ kernel(y, y) @ ay.T, ax @ kernel(x, y) @ ay.T
        self.nx, self.ny = ax.sum(1), ay.sum(1)  # samples per cluster

    def kd_many(self, count_x, count_y):
        """KD for each row of cluster counts (replicates x clusters)."""
        cx, cy = np.atleast_2d(count_x).astype(np.float64), np.atleast_2d(count_y).astype(np.float64)

        def within(c, s, n):
            """Mean kernel over pairs from different original clusters; NaN if the resample has only one."""
            pairs = np.einsum("ra,ab,rb->r", c, s, c) - (c ** 2) @ np.diag(s)
            count = (c @ n) ** 2 - (c ** 2) @ (n ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(count > 0, pairs / count, np.nan)

        cross = np.einsum("ra,ap,rp->r", cx, self.sxy, cy) / ((cx @ self.nx) * (cy @ self.ny))
        return within(cx, self.sxx, self.nx) + within(cy, self.syy, self.ny) - 2 * cross

    def kd(self, count_x=None, count_y=None):
        """KD of one resample; by default every cluster once (the point estimate)."""
        count_x = np.ones(len(self.nx)) if count_x is None else count_x
        count_y = np.ones(len(self.ny)) if count_y is None else count_y
        return float(self.kd_many(count_x, count_y)[0])


def draw(rng, n_clusters, n_boot):
    """Bootstrap resamples of whole clusters, as counts: replicates x clusters, each row summing to n_clusters."""
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)


def two_sided_p(diffs):
    """Bootstrap p of a difference: twice the smaller tail mass at 0, capped at 1 (NaN replicates dropped)."""
    diffs = diffs[~np.isnan(diffs)]
    return float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))


def load_method(method_dir, species, feature):
    """Stacked features of one method's assets of a species, and the asset index of each row."""
    assets = sorted(method_dir.glob(f"{species}_*.npz"))
    feats = [np.load(a)[feature].astype(np.float64) for a in assets]
    return (np.concatenate(feats), np.repeat(np.arange(len(feats)), [len(f) for f in feats])) if feats else (None, None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features", type=Path, help="metrics.py's features dir: <method>/<asset>.npz + reference/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref", default="ours", help="method every other method is compared with")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    methods = sorted(d for d in a.features.iterdir() if d.is_dir() and d.name != "reference")
    rows, samples = [], {}  # samples[(method, species, feature)] = KD per replicate
    for si, species in enumerate(SPECIES):
        if not (a.features / "reference" / f"{species}.npz").is_file():
            continue
        real = np.load(a.features / "reference" / f"{species}.npz")
        plants = real["plant"]
        for fi, feature in enumerate(FEATURES):
            y = real[feature].astype(np.float64)
            # one reference-plant draw per replicate, shared by every method (paired comparison on the reference)
            draw_y = draw(np.random.default_rng([a.seed, si, fi]), len(np.unique(plants)), a.boot)
            for mi, method_dir in enumerate(methods):
                x, x_clusters = load_method(method_dir, species, feature)
                if x is None:
                    continue
                kd = ClusterKD(x, x_clusters, y, plants)
                draw_x = draw(np.random.default_rng([a.seed, si, fi, mi + 1]), len(kd.nx), a.boot)
                samples[method_dir.name, species, feature] = kd.kd_many(draw_x, draw_y)
                rows.append({"method": method_dir.name, "species": species, "feature": feature,
                             "n_assets": len(kd.nx), "n_views": len(x), "n_plants": len(kd.ny), "n_real": len(y),
                             "kd": kd.kd()})
            print(f"{species} {feature}: {len(methods)} methods", flush=True)

    for method_dir in methods:  # species mean, per replicate
        for feature in FEATURES:
            per_species = [samples.get((method_dir.name, sp, feature)) for sp in SPECIES]
            if any(s is None for s in per_species):
                continue
            samples[method_dir.name, "all", feature] = np.mean(per_species, axis=0)
            parts = [r for r in rows if r["method"] == method_dir.name and r["feature"] == feature]
            rows.append({"method": method_dir.name, "species": "all", "feature": feature,
                         **{k: sum(r[k] for r in parts) for k in ("n_assets", "n_views", "n_plants", "n_real")},
                         "kd": float(np.mean([r["kd"] for r in parts]))})

    points = {(r["method"], r["species"], r["feature"]): r["kd"] for r in rows}
    for r in rows:
        key, ref_key = (r["method"], r["species"], r["feature"]), (a.ref, r["species"], r["feature"])
        r["ci_lo"], r["ci_hi"] = np.nanpercentile(samples[key], [2.5, 97.5])
        r["n_boot_used"] = int((~np.isnan(samples[key])).sum())
        if ref_key in samples and r["method"] != a.ref:
            r["diff_vs_ref"] = points[key] - points[ref_key]
            r["p_vs_ref"] = f"{two_sided_p(samples[key] - samples[ref_key]):.3g}"
        for k in ("kd", "ci_lo", "ci_hi", "diff_vs_ref"):
            if k in r:
                r[k] = f"{r[k]:.6g}"

    fields = ["method", "species", "feature", "n_assets", "n_views", "n_plants", "n_real", "kd", "ci_lo", "ci_hi",
              "n_boot_used", "diff_vs_ref", "p_vs_ref"]
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "realism.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out / "realism.csv")


if __name__ == "__main__":
    main()
