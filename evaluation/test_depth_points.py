"""Checks for depth_points.py: unprojecting a depth image and projecting the points back gives the same pixels.

    python evaluation/test_depth_points.py      (or pytest)
"""
import numpy as np

from depth_points import unproject_z, voxel_downsample
from render_splats import camera_set


def project(points, cam):
    """Pixel column, row (pixel centres at +0.5) and camera z of world points."""
    K, w2c = np.asarray(cam["K"]), np.asarray(cam["w2c"])
    x = points @ w2c[:3, :3].T + w2c[:3, 3]
    return x[:, 0] / x[:, 2] * K[0, 0] + K[0, 2] - 0.5, x[:, 1] / x[:, 2] * K[1, 1] + K[1, 2] - 0.5, x[:, 2]


def test_unproject_then_project_is_identity_for_every_shared_camera():
    rng = np.random.default_rng(0)
    res = 64
    for cam in camera_set(res):
        pixels = rng.choice(res * res, 50, replace=False)
        v, u = pixels // res, pixels % res
        depth, alpha = np.zeros((res, res)), np.zeros((res, res))
        depth[v, u], alpha[v, u] = rng.uniform(2.0, 4.0, 50), 1.0
        pu, pv, pz = project(unproject_z(depth, alpha, cam), cam)
        order = np.lexsort((u, v))  # unproject_z returns pixels in row-major order
        assert np.allclose(pu, u[order]) and np.allclose(pv, v[order])
        assert np.allclose(pz, depth[v[order], u[order]])


def test_points_land_in_the_unit_cube_frame():
    """A pixel at the image centre of a shared camera looks at the origin: at the camera's distance it is the origin."""
    for cam in camera_set(32):
        w2c = np.asarray(cam["w2c"])
        dist = np.linalg.norm(-w2c[:3, :3].T @ w2c[:3, 3])  # camera centre's distance from the origin
        depth, alpha = np.full((32, 32), dist), np.zeros((32, 32))
        alpha[15:17, 15:17] = 1.0  # the 4 pixels around the image centre
        assert np.allclose(unproject_z(depth, alpha, cam).mean(0), 0, atol=1e-9)


def test_transparent_pixels_are_dropped():
    cam = camera_set(16)[0]
    depth, alpha = np.full((16, 16), 3.0), np.zeros((16, 16))
    alpha[4, 5], alpha[6, 7] = 0.49, 0.5
    assert len(unproject_z(depth, alpha, cam)) == 1


def test_voxel_downsample_keeps_one_mean_point_per_voxel():
    pts = np.array([[0.001, 0.001, 0.001], [0.003, 0.003, 0.003], [0.5, 0.5, 0.5]])
    out = voxel_downsample(pts, voxel=0.01)
    assert len(out) == 2
    assert np.allclose(sorted(out.tolist()), [[0.002, 0.002, 0.002], [0.5, 0.5, 0.5]], atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
