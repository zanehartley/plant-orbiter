"""Fruit-counting inter-rater agreement: the second counter's subset vs the first counter's counts.

The second counter, blind to method, asset name and N, recounted a stratified subset: one asset per (method,
species, N), drawn with seed 2026, 84 of the 252 assets, plus 14 of the 42 input images (kind "input": the count is
of ours' input image). Joins the second counter's sheet (species, N, seed, method, asset, kind, count) with the first
counter's counts.csv (count for 3D assets, input_count for input images). Per group, over the items both counted:
  exact agreement (%)          counter 1 = counter 2
  mean absolute difference     mean |counter 1 - counter 2|
  Krippendorff's alpha         interval metric, with a 95% bootstrap interval over items
Groups: all 3D assets, per method, N <= 5 / N >= 6, and the input images (kept separate: one 2D view, not an asset).
Also Exact (%) on the subset by each counter, to show whether the method ranking depends on who counted.
Writes agreement.md next to the second counter's sheet.

    python evaluation/count_agreement.py results/fruit_counting/counts_counter2.csv \
        --counts results/fruit_counting/counts_counter1.csv
"""
import argparse
import csv
import random
from pathlib import Path
from statistics import mean

from count_metrics import fmt
from count_sheets import METHODS

BOOT, SEED = 2000, 2026


def alpha_interval(units):
    """Krippendorff's alpha, interval metric, for units given as lists of values (missing values left out).

    alpha = 1 - D_o / D_e on the pairable values (units with >= 2 values); None when every value is the same."""
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    n = len(values)
    if n == 0:
        return None
    # within-unit disagreement, each ordered pair weighted 1 / (m_u - 1)
    observed = sum(sum((a - b) ** 2 for a in u for b in u) / (len(u) - 1) for u in units)
    # all ordered pairs of pooled values: sum_{i != j} (v_i - v_j)^2 = 2 n sum (v - mean)^2
    v_bar = mean(values)
    expected = 2 * n * sum((v - v_bar) ** 2 for v in values) / (n - 1)
    return None if expected == 0 else 1 - observed / expected


def bootstrap_alpha(pairs, boot=BOOT, seed=SEED):
    """95% percentile interval of alpha, resampling items with replacement (resamples with no variance dropped)."""
    rng = random.Random(seed)
    alphas = sorted(a for a in (alpha_interval(rng.choices(pairs, k=len(pairs))) for _ in range(boot))
                    if a is not None)
    return alphas[int(0.025 * len(alphas))], alphas[int(0.975 * len(alphas)) - 1]


def agreement(pairs):
    """Exact agreement, mean absolute difference and alpha over (counter 1, counter 2) pairs."""
    lo, hi = bootstrap_alpha(pairs)
    return {"n": len(pairs), "exact": 100 * mean(a == b for a, b in pairs),
            "mad": mean(abs(a - b) for a, b in pairs), "alpha": alpha_interval(pairs), "lo": lo, "hi": hi}


def load(second_path, counts_path):
    """One dict per subset item: kind, method, asset, N, first (counter 1), second (counter 2)."""
    first = {(r["method"], r["asset"]): r for r in csv.DictReader(open(counts_path))}
    items = []
    for r2 in csv.DictReader(open(second_path)):
        r1 = first[(r2["method"], r2["asset"])]
        c1, c2 = (r1["count"] if r2["kind"] == "3d" else r1["input_count"]).strip(), r2["count"].strip()
        if c1 == "" or c2 == "":
            raise SystemExit(f"{r2['method']} {r2['asset']} ({r2['kind']}): missing count (counter 1 {c1!r}, 2 {c2!r})")
        items.append({"kind": r2["kind"], "method": r2["method"], "asset": r2["asset"], "N": int(r2["N"]),
                      "first": int(c1), "second": int(c2)})
    return items


def groups(items):
    """(label, items) in table order."""
    d3 = [i for i in items if i["kind"] == "3d"]
    return ([("3D assets, all", d3)] + [(m, [i for i in d3 if i["method"] == m]) for m in METHODS]
            + [("3D, N ≤ 5", [i for i in d3 if i["N"] <= 5]), ("3D, N ≥ 6", [i for i in d3 if i["N"] >= 6]),
               ("Input images (2D)", [i for i in items if i["kind"] == "input"])])


def markdown(items):
    lines = ["## Agreement (counter 1 = counts.csv, counter 2 = subset)", "",
             "| Items | n | Exact agreement (%) | Mean abs. difference | Krippendorff's α (interval) [95% CI] |",
             "| --- " * 5 + "|"]
    for label, group in groups(items):
        s = agreement([(i["first"], i["second"]) for i in group])
        lines.append(f"| {label} | {s['n']} | {fmt(s['exact'])} | {fmt(s['mad'], 2)} "
                     f"| {fmt(s['alpha'], 3)} [{fmt(s['lo'], 2)}, {fmt(s['hi'], 2)}] |")

    lines += ["", "## Exact (%) on the subset, by counter", "", "| Items | n | Counter 1 | Counter 2 |",
              "| --- " * 4 + "|"]
    for label, group in groups(items)[1:len(METHODS) + 1] + groups(items)[-1:]:
        lines.append(f"| {label} | {len(group)} | {fmt(100 * mean(i['first'] == i['N'] for i in group))} "
                     f"| {fmt(100 * mean(i['second'] == i['N'] for i in group))} |")

    lines += ["", "## Disagreements", "", "| method | asset | N | counter 1 | counter 2 |", "| --- " * 5 + "|"]
    lines += [f"| {i['method']}{' (input)' if i['kind'] == 'input' else ''} | {i['asset']} | {i['N']} "
              f"| {i['first']} | {i['second']} |"
              for i in sorted(items, key=lambda i: (i["kind"], i["method"], i["asset"])) if i["first"] != i["second"]]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("second", type=Path, help="second counter's sheet (counts_counter2.csv)")
    p.add_argument("--counts", type=Path, required=True, help="first counter's counts.csv")
    a = p.parse_args()
    items = load(a.second, a.counts)
    md = markdown(items)
    (a.second.parent / "agreement.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
