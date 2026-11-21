"""Checks for shape_metrics.py: normalisation, yaw-sweep + ICP alignment, Chamfer / F-score and the set metrics.

    python evaluation/test_shape_metrics.py      (or pytest; nerfstudio env, needs open3d)
"""
import numpy as np

from shape_metrics import align, chamfer_fscore, normalise_unit_height, set_metrics, uses_excluded_plant


def asymmetric_shape(rng, n=4000):
    """A 'plant': a pot (cylinder) with three arms of different lengths, so every yaw looks different."""
    pot = np.c_[0.2 * np.cos(t := rng.uniform(0, 2 * np.pi, n)), 0.2 * np.sin(t), rng.uniform(0, 0.3, n)]
    arms = [np.c_[np.outer(s := rng.uniform(0, length, n), [np.cos(a), np.sin(a)]), 0.3 + 0.8 * s]
            for a, length in ((0.0, 0.9), (2.0, 0.5), (4.0, 0.3))]
    return np.concatenate([pot, *arms])


def rotate_z(points, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def test_normalise_gives_unit_robust_height_centred_on_the_robust_box():
    rng = np.random.default_rng(0)
    p = normalise_unit_height(asymmetric_shape(rng) * 0.17 + [3.0, -2.0, 5.0])
    lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)
    assert np.isclose(hi[2] - lo[2], 1.0)
    assert np.allclose((lo + hi) / 2, 0, atol=1e-9)


def test_identical_clouds_score_perfectly_and_far_clouds_score_zero():
    p = normalise_unit_height(asymmetric_shape(np.random.default_rng(1)))
    cd, f = chamfer_fscore(p, p, (0.01, 0.05))
    assert cd == 0 and f == [1.0, 1.0]
    cd, f = chamfer_fscore(p, p + [3.0, 0, 0], (0.01, 0.05))
    assert f == [0.0, 0.0] and cd > 1


def test_chamfer_is_the_mean_of_both_directions():
    a, b = np.array([[0.0, 0, 0], [1, 0, 0]]), np.array([[0.0, 0, 0.5]])
    # a -> b: 0.5 and sqrt(1.25); b -> a: 0.5
    cd, (f,) = chamfer_fscore(a, b, (0.6,))
    assert np.isclose(cd, 0.5 * ((0.5 + np.sqrt(1.25)) / 2 + 0.5))
    assert np.isclose(f, 2 * 0.5 * 1.0 / 1.5)  # precision 1/2, recall 1


def test_align_recovers_a_yaw_and_offset_anywhere_on_the_circle():
    rng = np.random.default_rng(2)
    real = normalise_unit_height(asymmetric_shape(rng))
    for yaw in (37.0, 140.0, 255.0):
        gen = normalise_unit_height(rotate_z(asymmetric_shape(rng), yaw) + [0.05, -0.03, 0.0])
        aligned = align(gen, real)
        _, (f,) = chamfer_fscore(aligned, real, (0.02,))
        assert f > 0.95, (yaw, f)


def test_set_metrics_follow_achlioptas():
    # 3 generated x 2 real; lower is better for cd
    cd = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, 0.9]])
    out = set_metrics(cd, lower_is_better=True)
    assert np.isclose(out["mmd"], (0.1 + 0.4) / 2)  # per real plant: best generated asset
    assert np.isclose(out["cov"], 1 / 2)  # every generated asset's nearest real plant is the first
    assert np.isclose(out["nn_mean"], (0.1 + 0.2 + 0.3) / 3)  # per generated asset: nearest real plant
    f = 1 - cd
    out = set_metrics(f, lower_is_better=False)
    assert np.isclose(out["mmd"], (0.9 + 0.6) / 2) and np.isclose(out["cov"], 1 / 2)


def test_set_metrics_leave_one_out_ignores_the_diagonal():
    d = np.array([[0.0, 0.3, 0.5], [0.3, 0.0, 0.2], [0.5, 0.2, 0.0]])
    out = set_metrics(d, lower_is_better=True, leave_one_out=True)
    assert np.isclose(out["nn_mean"], (0.3 + 0.2 + 0.2) / 3)
    assert np.isclose(out["cov"], 2 / 3)  # nearest others: 1, 2, 1


def test_excluded_plant_drops_its_column_and_its_real_row_but_not_a_generated_asset_of_that_name():
    assert uses_excluded_plant({"method": "ours", "asset": "bean_s1_s42", "plant": "bean_2"})
    assert uses_excluded_plant({"method": "real", "asset": "bean_2", "plant": "bean_1"})
    assert not uses_excluded_plant({"method": "plantdreamer", "asset": "bean_2", "plant": "bean_1"})
    assert not uses_excluded_plant({"method": "ours", "asset": "bean_s1_s42", "plant": "bean_1"})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
