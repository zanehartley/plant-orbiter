"""Checks for count_metrics.py on small hand-made count tables.

    python evaluation/test_count_metrics.py      (or pytest)
"""
import csv
import itertools
import math
import tempfile
from pathlib import Path

from count_metrics import FIELDS_IN, latex_rows, load, load_quality, quality_summary, summarise


_assets = itertools.count()


def row(method, n, count, input_count=None, species="tomato", asset=None):
    """asset defaults to a fresh name; pass it to pair a same-input method with ours' row of the same image."""
    return {"method": method, "species": species, "asset": asset or f"a{next(_assets)}", "N": n, "count": count,
            "input_count": input_count}


def write_csv(rows):
    path = Path(tempfile.mkdtemp()) / "counts.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_IN + ["notes"])
        w.writeheader()
        w.writerows({**r, "notes": ""} for r in rows)
    return path


def by_method(rows):
    return {s["method"]: s for s in summarise(rows)}


def test_exact_mae_under_over():
    s = by_method([row("ours", 3, 3), row("ours", 4, 2), row("ours", 5, 6), row("ours", 7, 7)])["ours"]
    assert s["n"] == 4 and s["exact"] == 50 and s["under"] == 25 and s["over"] == 25
    assert s["mae"] == (0 + 2 + 1 + 0) / 4


def test_exact_split_at_six():
    s = by_method([row("ours", 5, 5), row("ours", 5, 4), row("ours", 6, 6), row("ours", 7, 1)])["ours"]
    assert s["exact_le5"] == 50 and s["exact_ge6"] == 50


def test_spearman_rho():
    s = by_method([row("ours", 1, 1), row("ours", 2, 3), row("ours", 3, 2), row("ours", 4, 4)])["ours"]
    assert math.isclose(s["rho"], 0.8)   # ranks 1 3 2 4 against 1 2 3 4: 1 - 6 * 2 / (4 * 15)
    # every count the same (e.g. no fruit on any asset): no ranking, so None rather than nan
    assert by_method([row("ours", n, 0) for n in range(1, 8)])["ours"]["rho"] is None


def test_input_agreement_and_input_row():
    rows = [row("ours", 3, 3, 3), row("ours", 4, 3, 4), row("ours", 5, 6, 5), row("ours", 6, 5, 5)]
    s = by_method(rows)
    assert s["ours"]["eq_input"] == 50 and s["ours"]["fewer_input"] == 25 and s["ours"]["more_input"] == 25
    # the input row scores the input counts (3, 4, 5, 5) against N
    assert s["input"]["n"] == 4 and s["input"]["exact"] == 75 and s["input"]["mae"] == 0.25
    assert "eq_input" not in s["input"]


def test_baselines_have_no_input_columns_and_get_tests():
    rows = [row("ours", n, n) for n in range(1, 8)] + [row("diffsplat", n, 1) for n in range(1, 8)]
    s = by_method(rows)
    assert "eq_input" not in s["diffsplat"] and "p_exact" not in s["ours"]
    assert math.isclose(s["diffsplat"]["exact"], 100 / 7)
    assert s["diffsplat"]["p_exact"] < 0.05 and s["diffsplat"]["p_abs"] < 0.05


def test_identical_methods_p_one():
    rows = [row(m, n, n) for m in ("ours", "gaussiandreamer") for n in range(1, 8)]
    s = by_method(rows)["gaussiandreamer"]
    assert s["p_exact"] == 1.0 and s["p_abs"] == 1.0


def test_same_input_method_scored_against_ours_input_counts():
    # trellis2 lifts the same images as ours: its = input uses ours' input_count of the same asset
    ours = [row("ours", 3, 3, 3, asset="x1"), row("ours", 4, 4, 4, asset="x2"), row("ours", 5, 5, 6, asset="x3"),
            row("ours", 6, 6, 6, asset="x4")]
    trellis = [row("trellis2", 3, 3, asset="x1"), row("trellis2", 4, 2, asset="x2"),
               row("trellis2", 5, 7, asset="x3"), row("trellis2", 6, 6, asset="x4")]
    s = by_method(ours + trellis)
    t = s["trellis2"]
    assert t["eq_input"] == 50 and t["fewer_input"] == 25 and t["more_input"] == 25 and t["n_input"] == 4
    assert {"p_exact", "p_abs", "p_eq_input"} <= set(t) and "p_eq_input" not in s["ours"]
    assert "eq_input" not in by_method(ours + [row("diffsplat", 3, 3)])["diffsplat"]


def test_same_input_fisher_on_input_agreement():
    ours = [row("ours", 3, 3, 3, asset=f"y{i}") for i in range(10)]
    trellis = [row("trellis2", 3, 5, asset=f"y{i}") for i in range(10)]
    t = by_method(ours + trellis)["trellis2"]
    assert t["eq_input"] == 0 and t["p_eq_input"] < 0.05


def test_load_skips_uncounted_rows():
    path = write_csv([row("ours", 3, 3, 3), row("ours", 4, "", ""), row("diffsplat", 2, 0)])
    rows, skipped = load(path)
    assert [(r["method"], r["count"]) for r in rows] == [("ours", 3), ("diffsplat", 0)]
    assert skipped == {"ours": 1}


def test_load_keeps_3d_count_when_input_count_missing():
    rows, _ = load(write_csv([row("ours", 3, 2, "")]))
    assert rows[0]["count"] == 2 and rows[0]["input_count"] is None
    s = by_method(rows)
    assert "input" not in s and s["ours"]["eq_input"] is None


def write_quality(rows_by_file):
    """A results folder with per_asset.csv / quality.csv / vqascore.csv from {file: [(method, value), ...]}."""
    folder = Path(tempfile.mkdtemp())
    for name, column, values in rows_by_file:
        with open(folder / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "asset", column])
            w.writeheader()
            w.writerows({"method": m, "asset": f"a{i}", column: v} for i, (m, v) in enumerate(values))
    return folder


def test_quality_means_and_tests_vs_ours():
    ours = [("ours", v) for v in (30, 31, 32, 33, 34, 35, 36)]
    base = [("diffsplat", v) for v in (20, 21, 22, 23, 24, 25, 26)]
    folder = write_quality([("per_asset.csv", "clip_l14", ours + base),
                            ("quality.csv", "imagereward", ours + base),
                            ("vqascore.csv", "vqascore", ours + base)])
    q = quality_summary(load_quality([folder]))
    assert q["ours"]["clip_l14"] == (33, None)
    mean_d, p_d = q["diffsplat"]["vqascore"]
    assert mean_d == 23 and p_d < 0.05


def test_quality_input_row_not_tested():
    folder = write_quality([("per_asset.csv", "clip_l14", [("ours", 1), ("input", 2)]),
                            ("quality.csv", "imagereward", [("ours", 1), ("input", 2)]),
                            ("vqascore.csv", "vqascore", [("ours", 1), ("input", 2)])])
    q = quality_summary(load_quality([folder]))
    assert q["input"]["imagereward"] == (2, None)


def test_latex_stars_and_dashes():
    rows = [row("ours", n, n, n) for n in range(1, 8)] + [row("diffsplat", n, 1) for n in range(1, 8)]
    summary = summarise(rows)
    quality = {m: {k: (1.0, 0.01 if m == "diffsplat" else None) for k in ("clip_l14", "imagereward", "vqascore")}
               for m in ("ours", "diffsplat", "input")}
    tex = latex_rows(summary, quality)
    diffsplat = next(line for line in tex if line.strip().startswith("DiffSplat"))
    assert diffsplat.count("$^{*}$") == 5 and "& -- &" in diffsplat   # Exact, MAE, 3 quality columns; no = input
    ours = next(line for line in tex if line.strip().startswith("Ours"))
    assert "$^{*}$" not in ours and "100.0" in ours


def test_latex_same_input_block():
    ours = [row("ours", n, n, n, asset=f"z{n}") for n in range(1, 8)]
    trellis = [row("trellis2", n, n + 2, asset=f"z{n}") for n in range(1, 8)]
    tex = latex_rows(summarise(ours + trellis + [row("diffsplat", n, 1) for n in range(1, 8)]), {})
    names = [line.split("&")[0].strip() if "&" in line else line.strip() for line in tex]
    # baselines, then the same-input block (TRELLIS.2, ours) after one rule, then the input image
    assert names[names.index("DiffSplat") + 1] == "\\cmidrule(l){1-7}"
    assert names[names.index("DiffSplat") + 2].startswith("TRELLIS.2") and names[names.index("DiffSplat") + 3] == "Ours"
    trellis_line = tex[names.index("DiffSplat") + 2]
    assert trellis_line.split("&")[3].strip() == "0.0$^{*}$"   # = input, Fisher vs ours



def test_latex_trellis_text_in_baseline_block():
    # TRELLIS-text is a text-to-3D baseline: between DiffSplat and MVDream, before the same-input rule, no = input
    ours = [row("ours", n, n, n, asset=f"v{n}") for n in range(1, 8)]
    baselines = [row(m, n, 1) for m in ("diffsplat", "trellis_text", "mvdream") for n in range(1, 8)]
    tex = latex_rows(summarise(ours + baselines), {})
    names = [line.split("&")[0].strip() if "&" in line else line.strip() for line in tex]
    i = names.index("TRELLIS-text")
    assert names[i - 1] == "DiffSplat" and names[i + 1].startswith("MVDream")
    assert names.index("\\cmidrule(l){1-7}") > i
    cells = [c.strip() for c in tex[i].split("&")]
    assert cells[3] == "--" and cells[1].endswith("$^{*}$")   # no = input; Exact 14.3 vs ours 100, Fisher

def test_latex_input_agreement_bold_only():
    # = input has two methods: the better one is bold, the other gets no "second-best" underline
    ours = [row("ours", n, n, n, asset=f"w{n}") for n in range(1, 8)]
    trellis = [row("trellis2", n, n + 2, asset=f"w{n}") for n in range(1, 8)]
    tex = {line.split("&")[0].strip(): line for line in latex_rows(summarise(ours + trellis), {}) if "&" in line}
    assert tex["Ours"].split("&")[3].strip() == "\\textbf{100.0}"
    assert "underline" not in tex["TRELLIS.2$^\\ddagger$"].split("&")[3]
    # ours alone: nothing to compare, so no bold
    alone = [line for line in latex_rows(summarise(ours), {}) if line.strip().startswith("Ours")][0]
    assert alone.split("&")[3].strip() == "100.0"


def test_latex_best_and_second_best():
    rows = ([row("ours", n, n, n) for n in range(1, 8)] + [row("diffsplat", n, 1) for n in range(1, 8)]
            + [row("mvdream", n, n + 3) for n in range(1, 8)])
    quality = {"ours": {k: (-0.5, None) for k in ("clip_l14", "imagereward", "vqascore")},
               "diffsplat": {k: (-1.0, 0.2) for k in ("clip_l14", "imagereward", "vqascore")},
               "mvdream": {k: (-2.0, 0.2) for k in ("clip_l14", "imagereward", "vqascore")},
               "input": {k: (9.0, None) for k in ("clip_l14", "imagereward", "vqascore")}}
    tex = {line.split("&")[0].strip(): line for line in latex_rows(summarise(rows), quality) if "&" in line}
    assert "\\textbf{100.0}" in tex["Ours"] and "\\textbf{0.00}" in tex["Ours"]   # Exact, MAE (lower is better)
    assert "\\textbf{$-$0.50}" in tex["Ours"] and "\\underline{$-$1.00}" in tex["DiffSplat"]
    assert "textbf" not in tex["\\textit{Input image (2D)}"].split("&", 1)[1]   # the 2D row is not ranked


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
