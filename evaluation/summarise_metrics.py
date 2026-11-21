"""Summarise metrics.py / quality_metrics.py results: per-asset means per method x species, plus significance tests.

Reads per_asset.csv (CLIP score, R-precision), and if present quality.csv (ImageReward, VBench imaging / aesthetic
quality; merged by method + asset) and distribution.csv (realism: KD/FD against real photos, one value per method x
species, so reported without tests).

Methods evaluated on the same assets get paired Wilcoxon tests; methods on different assets (every baseline here:
they share no input with ours) get unpaired Mann-Whitney U tests against --ref, per species and pooled ("all").
Writes summary.md next to per_asset.csv.

    python evaluation/summarise_metrics.py $ORBIT_OUTPUTS/synthetic_plants/eval/results
"""
import argparse
import csv
from pathlib import Path
from statistics import mean

from scipy.stats import mannwhitneyu, wilcoxon

METRICS = ["clip_b32", "clip_l14", "rprec_l14"]
QUALITY_METRICS = ["imagereward", "vbench_imaging", "vbench_aesthetic"]
REALISM = ["n_gen", "n_real", "kd_incep", "fd_incep", "kd_dino", "fd_dino"]


def summary_table(rows, metrics):
    lines = ["| Method | Species | n | " + " | ".join(metrics) + " |",
             "| --- " * (3 + len(metrics)) + "|"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for method in methods:
        for sp in sorted({r["species"] for r in rows}) + ["all"]:
            sel = [r for r in rows if r["method"] == method and sp in ("all", r["species"])]
            if sel:
                lines.append(f"| {method} | {sp} | {len(sel)} | "
                             + " | ".join(f"{mean(r[m] for r in sel):.3f}" for m in metrics) + " |")
    return lines


def tests(rows, ref, metrics):
    by = {(r["method"], r["asset"]): r for r in rows}
    species = sorted({r["species"] for r in rows})
    lines = ["| ref vs other | species | test | n | metric | mean ref | mean other | p |", "| --- " * 8 + "|"]
    for other in dict.fromkeys(r["method"] for r in rows):
        if other == ref:
            continue
        shared = sorted({a for m, a in by if m == ref} & {a for m, a in by if m == other})
        for m in metrics:
            if shared:  # same assets: paired
                xa, xb = [by[ref, a][m] for a in shared], [by[other, a][m] for a in shared]
                p = wilcoxon(xa, xb).pvalue if any(x != y for x, y in zip(xa, xb)) else 1.0
                lines.append(f"| {ref} vs {other} | all | Wilcoxon | {len(shared)} | {m} | {mean(xa):.3f} "
                             f"| {mean(xb):.3f} | {p:.3g} |")
                continue
            # Different assets: test within each species too, since species mixes differ and species dominates scores.
            for sp in species + ["all"]:
                xa = [r[m] for r in rows if r["method"] == ref and sp in ("all", r["species"])]
                xb = [r[m] for r in rows if r["method"] == other and sp in ("all", r["species"])]
                if min(len(xa), len(xb)) < 5:
                    continue
                p = mannwhitneyu(xa, xb).pvalue
                lines.append(f"| {ref} vs {other} | {sp} | Mann-Whitney | {len(xa)}/{len(xb)} | {m} | {mean(xa):.3f} "
                             f"| {mean(xb):.3f} | {p:.3g} |")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="directory holding per_asset.csv")
    p.add_argument("--ref", default="ours", help="method every other method is tested against")
    args = p.parse_args()

    rows = [{**r, **{m: float(r[m]) for m in METRICS}} for r in csv.DictReader(open(args.results / "per_asset.csv"))]
    metrics = list(METRICS)
    if (args.results / "quality.csv").is_file():
        quality = {(r["method"], r["asset"]): r for r in csv.DictReader(open(args.results / "quality.csv"))}
        missing = [(r["method"], r["asset"]) for r in rows if (r["method"], r["asset"]) not in quality]
        if missing:
            raise SystemExit(f"quality.csv lacks {len(missing)} assets of per_asset.csv, e.g. {missing[:3]}: rerun")
        rows = [{**r, **{m: float(quality[r["method"], r["asset"]][m]) for m in QUALITY_METRICS}} for r in rows]
        metrics += QUALITY_METRICS
    md = ["# Text-to-3D metrics", "", "CLIP score = 100 x cosine(view, prompt), mean over views; R-precision = fraction "
          "of views where the true species' prompt ranks first among bean/kale/mint. ImageReward and VBench imaging "
          "(MUSIQ-SPAQ / 100) / aesthetic (LAION head / 10) quality: mean over views.", ""]
    md += summary_table(rows, metrics)
    if (args.results / "distribution.csv").is_file():
        dist = list(csv.DictReader(open(args.results / "distribution.csv")))
        md += ["", "## Realism vs real photos (lower is better)", "",
               "| Method | Species | " + " | ".join(REALISM) + " |", "| --- " * (2 + len(REALISM)) + "|"]
        md += [f"| {r['method']} | {r['species']} | " + " | ".join(r[k] for k in REALISM) + " |" for r in dist]
    md += ["", "## Tests", ""] + tests(rows, args.ref, metrics)
    (args.results / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
