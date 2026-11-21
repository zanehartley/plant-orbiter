"""Point clouds from rendered depth, for the shape metrics: every method's asset becomes points in the same way.

Following TRELLIS / Point-E (points from unprojected multi-view depth), each renderer unprojects the opaque pixels
(alpha >= MIN_ALPHA) of all 72 shared views at their expected depth and fuses them in the unit-cube frame the views
use (z up). The fused cloud is voxel-downsampled (one point per occupied voxel, at the voxel's mean position) and
saved as <asset>/points.npz: points (N x 3, float32), voxel, min_alpha, n_views. Floaters are kept on purpose: they
are geometry the asset really has. numpy only, so the splat (nerfstudio) and NeRF (mvdream) renderers share it.
"""
import numpy as np

MIN_ALPHA = 0.5
VOXEL = 1 / 256  # of the unit cube the asset is normalised into


def unproject_z(depth, alpha, cam, min_alpha=MIN_ALPHA):
    """World points of the pixels with alpha >= min_alpha. depth: camera-z per pixel (H x W); cam: as in cameras.json
    (OpenCV w2c, K). Pixel centres are at +0.5."""
    v, u = np.nonzero(alpha >= min_alpha)
    z = depth[v, u]
    K, w2c = np.asarray(cam["K"]), np.asarray(cam["w2c"])
    x_cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (x_cam - t) @ R  # R^T (x - t), row-vector form


def voxel_downsample(points, voxel=VOXEL):
    """One point per occupied voxel, at the mean of the points in it."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


def save_points(path, points, n_views, voxel=VOXEL, min_alpha=MIN_ALPHA):
    np.savez(path, points=voxel_downsample(np.concatenate(points), voxel), voxel=voxel, min_alpha=min_alpha,
             n_views=n_views)
