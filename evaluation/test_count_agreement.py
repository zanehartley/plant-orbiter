"""Checks for count_agreement.py: Krippendorff's alpha against the published worked example.

    python evaluation/test_count_agreement.py      (or pytest)
"""
import math

from count_agreement import agreement, alpha_interval

# Krippendorff (2011), "Computing Krippendorff's alpha-reliability": 4 coders x 12 units, missing values left out;
# published interval alpha = 0.849
EXAMPLE = [[1, 1, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4],
           [1, 1, 2, 1], [2, 2, 2, 2], [5, 5, 5], [1, 1], [3]]


def test_published_example():
    assert round(alpha_interval(EXAMPLE), 3) == 0.849


def test_perfect_and_no_variance():
    assert alpha_interval([[1, 1], [3, 3], [6, 6]]) == 1
    assert alpha_interval([[2, 2], [2, 2]]) is None


def test_two_counter_closed_form():
    # two counters, no missing: alpha = 1 - (n - 1) sum (a - b)^2 / sum_{i != j} (v_i - v_j)^2, with n = 2 * items
    pairs = [(1, 2), (3, 3), (5, 4), (6, 7)]
    values = [v for p in pairs for v in p]
    pooled = sum((a - b) ** 2 for a in values for b in values)
    assert math.isclose(alpha_interval(pairs), 1 - (len(values) - 1) * 2 * sum((a - b) ** 2 for a, b in pairs) / pooled)


def test_agreement_summary():
    s = agreement([(1, 1), (3, 3), (5, 4), (6, 6)])
    assert s["n"] == 4 and s["exact"] == 75 and s["mad"] == 0.25
    assert s["lo"] <= s["alpha"] <= s["hi"] <= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
