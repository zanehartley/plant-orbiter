"""Fruit-counting table from counts.csv (count_sheets.py), once it is filled in by eye.

Per method, over its counted assets (every asset counts, no validity filter):
  Exact (%)   count = N                                       MAE      mean |count - N|
  Under / Over (%), Exact for N <= 5 and N >= 6               for the text, not table columns
  Spearman rho between N and count (does the count rise with N), for the text and fig_counts.py, not a table column
  = input (%) ours and TRELLIS.2 (SAME_INPUT, both lift ours' Qwen images): 3D count = count in the input image
              (ours' input_count of the same asset), the disagreements split into 3D fewer / 3D more
A row "input" scores ours' input images (input_count) against N: the 2D row of the table.
Tests vs ours are unpaired as in the synthetic-plant experiment (seeds do not match across methods): Fisher's exact test on Exact,
Mann-Whitney U on the absolute error; TRELLIS.2 also Fisher on = input. Rows with an empty count are skipped and
reported.
With --quality (results folders of metrics.py, quality_metrics.py and vqascore.py, e.g. the renders' and the input
images'), the table also gets CLIP L/14, ImageReward and VQAScore: per-method means, Mann-Whitney U vs ours; the
input-image row is not tested (one 2D view, not a 3D asset).
Writes count_metrics.md (paper table, then the numbers for the text) and count_table.tex (the table's rows for
tab:t23d_counting; * = p < 0.05 vs ours, uncorrected) next to counts.csv.

    python evaluation/count_metrics.py $ORBIT_OUTPUTS/fruit_counting/eval/counting/counts.csv \
        --quality $ORBIT_OUTPUTS/fruit_counting/eval/results $ORBIT_OUTPUTS/fruit_counting/eval/results_input
From the packaged results:
    python evaluation/count_metrics.py results/fruit_counting/counts_counter1.csv \
        --quality results/fruit_counting results/fruit_counting/input_image
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from count_sheets import METHODS, SAME_INPUT

REF = "ours"
FIELDS_IN = ["method", "species", "asset", "N", "count", "input_count"]
# (column, file in a results folder, header in the table); per-asset values, one row per (method, asset)
QUALITY = (("clip_l14", "per_asset.csv", "CLIP L/14"), ("imagereward", "quality.csv", "ImageReward"),
           ("vqascore", "vqascore.csv", "VQAScore"))
# table row order and names (text-to-3D baselines, the same-input block, then the 2D reference)
ROWS = (("gaussiandreamer", "GaussianDreamer"), ("diffsplat", "DiffSplat"), ("trellis_text", "TRELLIS-text"),
        ("mvdream", "MVDream$^\\dagger$"), ("trellis2", "TRELLIS.2$^\\ddagger$"), (REF, "Ours"),
        ("input", "\\textit{Input image (2D)}"))
ALPHA = 0.05


def load(path):
    """Counted rows ({method, species, N, count, input_count (None if empty)}) and uncounted rows per method."""
    rows, skipped = [], Counter()
    for r in csv.DictReader(open(path)):
        if r["count"].strip() == "":
            skipped[r["method"]] += 1
            continue
        rows.append({"method": r["method"], "species": r["species"], "asset": r["asset"], "N": int(r["N"]),
                     "count": int(r["count"]),
                     "input_count": int(r["input_count"]) if r["input_count"].strip() else None})
    return rows, dict(skipped)


def pct(flags):
    return 100 * mean(flags) if flags else None


def rho(pairs):
    """Spearman rho between N and count; None if either is constant (no ranking, scipy would return nan)."""
    ns, counts = zip(*pairs) if pairs else ((), ())
    return None if len(set(ns)) < 2 or len(set(counts)) < 2 else spearmanr(ns, counts).statistic


def scores(pairs):
    """Exact / Under / Over (%), MAE, Spearman rho and the N split over (N, count) pairs."""
    return {"n": len(pairs),
            "exact": pct([c == n for n, c in pairs]),
            "exact_le5": pct([c == n for n, c in pairs if n <= 5]),
            "exact_ge6": pct([c == n for n, c in pairs if n >= 6]),
            "under": pct([c < n for n, c in pairs]),
            "over": pct([c > n for n, c in pairs]),
            "mae": mean(abs(c - n) for n, c in pairs),
            "rho": rho(pairs)}


def fisher(ref_flags, flags):
    return fisher_exact([[sum(ref_flags), len(ref_flags) - sum(ref_flags)],
                         [sum(flags), len(flags) - sum(flags)]]).pvalue


def tests(ref_pairs, pairs):
    """Fisher on Exact, Mann-Whitney on absolute error: method vs REF, unpaired."""
    p_exact = fisher([c == n for n, c in ref_pairs], [c == n for n, c in pairs])
    err_ref, err = [abs(c - n) for n, c in ref_pairs], [abs(c - n) for n, c in pairs]
    # all errors equal on both sides: no ranking to test (scipy would return nan)
    p_abs = 1.0 if len(set(err_ref + err)) == 1 else mannwhitneyu(err_ref, err).pvalue
    return {"p_exact": p_exact, "p_abs": p_abs}


def summarise(rows):
    """One dict per method present (METHODS order, unknown methods after), then the input-image row."""
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    methods += sorted({r["method"] for r in rows} - set(methods))
    pairs = {m: [(r["N"], r["count"]) for r in rows if r["method"] == m] for m in methods}
    # the input image is counted once, on ours' rows; same-input methods look it up by asset name
    input_count = {r["asset"]: r["input_count"] for r in rows if r["method"] == REF and r["input_count"] is not None}
    vs_input = {m: [(r["count"], input_count[r["asset"]]) for r in rows
                    if r["method"] == m and r["asset"] in input_count] for m in methods if m in SAME_INPUT}
    out = []
    for m in methods:
        s = {"method": m, **scores(pairs[m])}
        if m in vs_input:
            s["eq_input"] = pct([c == i for c, i in vs_input[m]])
            s["fewer_input"] = pct([c < i for c, i in vs_input[m]])
            s["more_input"] = pct([c > i for c, i in vs_input[m]])
            s["n_input"] = len(vs_input[m])
        if m != REF and REF in pairs:
            s.update(tests(pairs[REF], pairs[m]))
            if vs_input.get(m) and vs_input.get(REF):
                s["p_eq_input"] = fisher([c == i for c, i in vs_input[REF]], [c == i for c, i in vs_input[m]])
        out.append(s)
    inputs = [(r["N"], r["input_count"]) for r in rows if r["method"] == REF and r["input_count"] is not None]
    if inputs:
        out.append({"method": "input", **scores(inputs)})
    return out


def load_quality(folders):
    """{column: {method: [per-asset values]}} from the QUALITY files in each results folder."""
    values = {column: {} for column, _, _ in QUALITY}
    for folder in folders:
        for column, name, _ in QUALITY:
            with open(Path(folder) / name, newline="") as fh:
                for r in csv.DictReader(fh):
                    values[column].setdefault(r["method"], []).append(float(r[column]))
    return values


def quality_summary(values):
    """{method: {column: (mean, p vs REF or None)}}; REF and the input-image row are not tested."""
    out = {}
    for column, per_method in values.items():
        for m, v in per_method.items():
            test = m not in (REF, "input") and REF in per_method
            out.setdefault(m, {})[column] = (mean(v), mannwhitneyu(per_method[REF], v).pvalue if test else None)
    return out


def fmt(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}"


def star(p):
    return "*" if p is not None and p < ALPHA else ""


def paper_cells(s, quality):
    """The 6 table cells of one summary row as (text, starred): Exact, MAE, = input, then the QUALITY columns."""
    q = quality.get(s["method"], {})
    cells = [(fmt(s["exact"]), star(s.get("p_exact"))), (fmt(s["mae"], 2), star(s.get("p_abs"))),
             (fmt(s.get("eq_input")), star(s.get("p_eq_input")))]
    for column, _, _ in QUALITY:
        m, p = q.get(column, (None, None))
        cells.append((fmt(m, 3 if column == "vqascore" else 2), star(p)))
    return cells


def ordered(summary):
    by = {s["method"]: s for s in summary}
    return [(by[m], name) for m, name in ROWS if m in by]


def ranks(rows):
    """{(row, column): 1 or 2}: best / second-best value per column among the 3D methods (not the input image),
    lower is better for MAE (column 1). = input (column 2) has only ours and TRELLIS.2: the better one is bold, with
    no second-best (it would mark the worse of two), and a single value is not ranked. Ties share a rank."""
    out = {}
    methods = [i for i, (s, _, _) in enumerate(rows) if s["method"] != "input"]
    for col in range(6):
        values = {i: float(rows[i][2][col][0]) for i in methods if rows[i][2][col][0] != "–"}
        if col == 2 and len(values) < 2:
            continue
        distinct = sorted(set(values.values()), reverse=col != 1)
        for i, v in values.items():
            if distinct.index(v) < (1 if col == 2 else 2):
                out[(i, col)] = distinct.index(v) + 1
    return out


def latex_rows(summary, quality):
    """Body rows of tab:t23d_counting, rules between baselines / same-input block (TRELLIS.2, ours) / input image.
    \\textbf / \\underline: best / second-best 3D method, as in the synthetic-plant table."""
    rows = [(s, name, paper_cells(s, quality)) for s, name in ordered(summary)]
    rank = ranks(rows)
    lines = []
    for i, (s, name, cells) in enumerate(rows):
        if s["method"] in SAME_INPUT and (i == 0 or rows[i - 1][0]["method"] not in SAME_INPUT):
            lines.append("    \\cmidrule(l){1-7}")
        if s["method"] == "input":
            lines.append("    \\midrule")
        tex = []
        for col, (text, starred) in enumerate(cells):
            text = text.replace("–", "--") if text == "–" else text.replace("-", "$-$")
            text = {1: f"\\textbf{{{text}}}", 2: f"\\underline{{{text}}}"}.get(rank.get((i, col)), text)
            tex.append(text + ("$^{*}$" if starred else ""))
        lines.append(f"    {name:<26} & " + " & ".join(tex) + " \\\\")
    return lines


def paper_markdown(summary, quality):
    head = ["Method", "Exact (%)↑", "MAE↓", "= input (%)↑"] + [f"{h}↑" for _, _, h in QUALITY]
    lines = ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"]
    for s, name in ordered(summary):
        name = (name.replace("$^\\dagger$", "†").replace("$^\\ddagger$", "‡")
                .replace("\\textit{", "*").replace("}", "*"))
        lines.append(f"| {name} | " + " | ".join(t + st for t, st in paper_cells(s, quality)) + " |")
    return lines


def markdown(summary, skipped):
    lines = ["| Method | n | Exact (%) | MAE | = input (%) | 3D fewer / more than input (%) | Exact N≤5 / N≥6 (%) "
             "| Under / Over (%) | Spearman ρ | p Exact (Fisher) | p abs. error (Mann-Whitney) | p = input (Fisher) |",
             "| --- " * 12 + "|"]
    for s in summary:
        name = "Input image (2D)" if s["method"] == "input" else s["method"]
        lines.append(f"| {name} | {s['n']} | {fmt(s['exact'])} | {fmt(s['mae'], 2)} | {fmt(s.get('eq_input'))} "
                     f"| {fmt(s.get('fewer_input'))} / {fmt(s.get('more_input'))} "
                     f"| {fmt(s['exact_le5'])} / {fmt(s['exact_ge6'])} | {fmt(s['under'])} / {fmt(s['over'])} "
                     f"| {fmt(s['rho'], 2)} "
                     f"| {fmt(s.get('p_exact'), 3)} | {fmt(s.get('p_abs'), 3)} | {fmt(s.get('p_eq_input'), 3)} |")
    if skipped:
        lines.append("\nNot counted yet (skipped): " + ", ".join(f"{m} {k}" for m, k in sorted(skipped.items())))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("counts", type=Path, help="counts.csv from count_sheets.py")
    p.add_argument("--quality", type=Path, nargs="+", default=[], help="results folders with the QUALITY files")
    a = p.parse_args()
    rows, skipped = load(a.counts)
    summary = summarise(rows)
    quality = quality_summary(load_quality(a.quality)) if a.quality else {}
    md = (["## Table (* = p < 0.05 vs ours, uncorrected)", ""] + paper_markdown(summary, quality)
          + ["", "## For the text", ""] + markdown(summary, skipped))
    (a.counts.parent / "count_metrics.md").write_text("\n".join(md) + "\n")
    (a.counts.parent / "count_table.tex").write_text("\n".join(latex_rows(summary, quality)) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
