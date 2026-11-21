"""Checks for kd_bootstrap.py against brute-force kernel distances on random features.

    python evaluation/test_kd_bootstrap.py      (or pytest)
"""
import numpy as np

from kd_bootstrap import ClusterKD, draw
from metrics import kernel_distance


def brute_kd(x, y, x_orig, y_orig):
    """MMD^2 with the cubic kernel; within-set pairs from the same original cluster are left out."""
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3  # noqa: E731
    same_x, same_y = x_orig[:, None] == x_orig[None], y_orig[:, None] == y_orig[None]
    return k(x, x)[~same_x].mean() + k(y, y)[~same_y].mean() - 2 * k(x, y).mean()


def random_sets(rng, n_assets=5, n_plants=3, d=16, spread=0.0):
    """Clustered features: each asset / plant has its own offset (spread), its views scatter around it."""
    sizes_x, sizes_y = rng.integers(3, 9, n_assets), rng.integers(4, 12, n_plants)
    cx, cy = np.repeat(np.arange(n_assets), sizes_x), np.repeat(np.arange(n_plants), sizes_y)
    x = rng.normal(size=(sizes_x.sum(), d)) + spread * rng.normal(size=(n_assets, d))[cx]
    y = rng.normal(0.3, 1.1, size=(sizes_y.sum(), d)) + spread * rng.normal(size=(n_plants, d))[cy]
    return x, cx, y, cy


def test_point_estimate_uses_between_cluster_pairs():
    x, cx, y, cy = random_sets(np.random.default_rng(0))
    assert np.isclose(ClusterKD(x, cx, y, cy).kd(), brute_kd(x, y, cx, cy))


def test_one_sample_per_cluster_is_metrics_py_kd():
    """With every sample its own cluster this is the usual unbiased KD (metrics.kernel_distance on whole sets)."""
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(30, 16)), rng.normal(0.2, 1, size=(30, 16))
    kd = ClusterKD(x, np.arange(30), y, np.arange(30)).kd()
    assert np.isclose(kd, kernel_distance(x, y, n_subsets=2, subset_size=30))


def test_resampled_clusters_match_explicit_copies():
    rng = np.random.default_rng(2)
    x, cx, y, cy = random_sets(rng)
    kd = ClusterKD(x, cx, y, cy)
    tested = 0
    for _ in range(40):
        count_x, count_y = draw(rng, 5, 1)[0], draw(rng, 3, 1)[0]
        if count_y.max() == 3:  # one plant drawn three times: no between-plant pairs, KD undefined
            assert np.isnan(kd.kd(count_x, count_y))
            continue
        ix = np.concatenate([np.flatnonzero(cx == a) for a in range(5) for _ in range(count_x[a])])
        iy = np.concatenate([np.flatnonzero(cy == p) for p in range(3) for _ in range(count_y[p])])
        assert np.isclose(kd.kd(count_x, count_y), brute_kd(x[ix], y[iy], cx[ix], cy[iy]))
        tested += 1
    assert tested > 20


def test_bootstrap_is_centred_on_the_estimate():
    """Strongly clustered data: duplicated clusters must not inflate KD (the bug the between-cluster form fixes)."""
    rng = np.random.default_rng(3)
    x, cx, y, cy = random_sets(rng, n_assets=12, n_plants=12, spread=2.0)
    kd = ClusterKD(x, cx, y, cy)
    samples = kd.kd_many(draw(rng, 12, 2000), draw(rng, 12, 2000))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    assert lo <= kd.kd() <= hi
    assert abs(np.nanmean(samples) - kd.kd()) < 0.15 * np.nanstd(samples)


def test_same_draws_give_zero_difference():
    rng = np.random.default_rng(4)
    x, cx, y, cy = random_sets(rng, n_assets=8, n_plants=4)
    kd = ClusterKD(x, cx, y, cy)
    draws_x, draws_y = draw(rng, 8, 200), draw(rng, 4, 200)
    a, b = kd.kd_many(draws_x, draws_y), kd.kd_many(draws_x, draws_y)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
