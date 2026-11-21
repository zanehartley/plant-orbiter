"""Shape: how close generated plants are to real plants of the same species, as point clouds (Achlioptas et al. 2018).

A generated plant has no real twin, so each asset is compared with every real plant of its species (PlantDreamer's
real point clouds: 11 bean / 3 kale / 6 mint, bean_2 excluded) and the set metrics are taken over that asset x plant
matrix.

Points: the asset's points.npz (render_splats.py / render_threestudio.py: depth of the 72 shared views, unit-cube
frame, z up) and the real cloud (*_real_pointcloud.ply, z up to within 3 degrees by PlantDreamer's cameras). Both keep
the pot: every method generates one and the scans include it (same choice as the image-to-3D table, gao_eval.py
--crop none). Each cloud is scaled to unit robust height (1st-99th percentile of z), centred on its robust box and
voxel-downsampled to FINE_VOXEL, so point density and size do not depend on the source.
Alignment, per pair: a yaw sweep (N_YAW starts about z, rigid ICP on COARSE_VOXEL clouds, keep the lowest Chamfer),
then rigid ICP on MID_VOXEL clouds (6x faster than on the fine clouds, F@0.02 within 0.001 on a test pair).
No scaling (as gao_eval.py) and no mirroring. Distances are then taken on the fine clouds.
Distances, as gao_eval.py: Chamfer = mean of the two directions' mean nearest-neighbour distance; F-score at each
tau (fraction of unit height) from precision (generated -> real) and recall (real -> generated).

Set metrics per method x species and distance (shape.csv), with D the asset x plant matrix:
  mmd      per real plant the best generated asset, averaged over plants (MMD-CD; MMD-F with the F-score)
  cov      fraction of real plants that are some asset's nearest plant (COV)
  nn_mean  per asset the nearest real plant, averaged over assets (one value per asset: shape_pairs.csv gives the
           per-asset numbers for significance tests)
species "all" is the mean over the three species, for methods with all three. Method "real" compares each real
plant with the other plants of its species (leave-one-out): the level of real-to-real variation, and what tau is
judged against (it uses no generated asset).
No intervals or tests: mmd and cov are set-level best-of statistics, for which a bootstrap is biased (in a
trial, MMD-F intervals fell below the point value and COV intervals above it).

Writes shape_pairs.csv (one row per asset x plant; resumable) and shape.csv. CPU only; nerfstudio env (open3d):
    python evaluation/shape_metrics.py $ORBIT_OUTPUTS/synthetic_plants/eval/renders \
        --real $ORBIT_DATASETS/PlantDreamer_Data/plants --out $ORBIT_OUTPUTS/synthetic_plants/eval/results
"""
import argparse
import csv
import json
import multiprocessing
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from depth_points import voxel_downsample

SPECIES = ("bean", "kale", "mint")
TAUS = (0.01, 0.02, 0.05)  # F-score thresholds, fraction of unit height
FINE_VOXEL = 0.005
MID_VOXEL = 0.01
COARSE_VOXEL = 0.02
N_YAW = 24
ICP_COARSE, ICP_FINE = 0.1, 0.05  # ICP correspondence distance, fraction of unit height
REAL_VOXEL_M = 0.0008  # real clouds are first thinned in metres, as gao_eval.py
REAL_GLOB = "*/data/*_real_pointcloud.ply"
# Real plants left out of the reference, as in the realism reference (prep_reference.py): bean_2's photo masks are
# misnumbered after view 160, so it is excluded from every metric. Its scan itself looks normal; rows
# with it may remain in shape_pairs.csv but are ignored. PlantDreamer's generated bean_2 asset is still scored.
EXCLUDED_PLANTS = ("bean_2",)


def normalise_unit_height(points):
    """Scale to unit robust height (1st-99th percentile of z) and centre on the robust box."""
    lo, hi = np.percentile(points, 1, 0), np.percentile(points, 99, 0)
    return (points - (lo + hi) / 2) / (hi[2] - lo[2])


def chamfer_fscore(a, b, taus):
    """a: generated, b: real. Chamfer (mean of both directions) and the F-score at each tau."""
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    f = []
    for tau in taus:
        precision, recall = (da < tau).mean(), (db < tau).mean()
        f.append(float(2 * precision * recall / max(precision + recall, 1e-9)))
    return float(0.5 * (da.mean() + db.mean())), f


def _cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))


def _icp(src, dst, init, threshold):
    return o3d.pipelines.registration.registration_icp(
        src, dst, threshold, init, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)).transformation


def _apply(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def align(gen, real, n_yaw=N_YAW):
    """gen rigidly aligned to real: best of n_yaw yaw starts (ICP on coarse clouds), then ICP on mid clouds."""
    gen_c, real_c = voxel_downsample(gen, COARSE_VOXEL), voxel_downsample(real, COARSE_VOXEL)
    src, dst = _cloud(gen_c), _cloud(real_c)
    best, best_cd = None, np.inf
    for k in range(n_yaw):
        c, s = np.cos(2 * np.pi * k / n_yaw), np.sin(2 * np.pi * k / n_yaw)
        init = np.eye(4)
        init[:2, :2] = [[c, -s], [s, c]]
        T = _icp(src, dst, init, ICP_COARSE)
        cd, _ = chamfer_fscore(_apply(T, gen_c), real_c, ())
        if cd < best_cd:
            best, best_cd = T, cd
    T = _icp(_cloud(voxel_downsample(gen, MID_VOXEL)), _cloud(voxel_downsample(real, MID_VOXEL)), best, ICP_FINE)
    return _apply(T, gen)


def set_metrics(d, lower_is_better, leave_one_out=False):
    """d: assets x real plants (a distance, or a similarity with lower_is_better=False). leave_one_out: d is
    plants x plants and the diagonal (a plant with itself) is ignored."""
    d = np.asarray(d, dtype=float)
    if not lower_is_better:
        d = -d
    if leave_one_out:
        d = d + np.diag(np.full(len(d), np.inf))
    sign = 1 if lower_is_better else -1
    return {"mmd": sign * d.min(0).mean(), "cov": len(set(d.argmin(1))) / d.shape[1], "nn_mean": sign * d.min(1).mean()}


def uses_excluded_plant(row):
    """A shape_pairs.csv row against an excluded real plant (or, for method 'real', from one)."""
    return row["plant"] in EXCLUDED_PLANTS or (row["method"] == "real" and row["asset"] in EXCLUDED_PLANTS)


def load_real(real_root):
    """{plant: (species, normalised fine cloud)} for every *_real_pointcloud.ply."""
    real = {}
    for ply in sorted(real_root.glob(REAL_GLOB)):
        plant = ply.parent.parent.name
        if plant in EXCLUDED_PLANTS:
            continue
        pts = np.asarray(o3d.io.read_point_cloud(str(ply)).voxel_down_sample(REAL_VOXEL_M).points)
        real[plant] = (plant.split("_")[0], voxel_downsample(normalise_unit_height(pts), FINE_VOXEL))
    return real


def load_asset(asset_dir):
    species = json.loads((asset_dir / "cameras.json").read_text())["asset"]["species"]
    pts = np.load(asset_dir / "points.npz")["points"].astype(np.float64)
    return species, voxel_downsample(normalise_unit_height(pts), FINE_VOXEL)


REAL = {}  # the real clouds, per process (main, and each worker via _init_worker); read by score_asset


def _init_worker(real):
    REAL.update(real)


def score_asset(job):
    """All rows of one asset (or, for method 'real', one real plant vs the other plants of its species)."""
    method, asset, asset_dir = job
    species, gen = REAL[asset] if method == "real" else load_asset(Path(asset_dir))
    rows = []
    for plant, (plant_species, real) in REAL.items():
        if plant_species != species or (method == "real" and plant == asset):
            continue
        cd, f = chamfer_fscore(align(gen, real), real, TAUS)
        rows.append({"method": method, "species": species, "asset": asset, "plant": plant, "n_gen": len(gen),
                     "n_real": len(real), "cd": cd, **{f"f@{tau}": v for tau, v in zip(TAUS, f)}})
    return rows


def pair_matrix(pairs, method, species, distance):
    """The asset x plant matrix of one method and species (None if it has no rows)."""
    rows = [r for r in pairs if r["method"] == method and r["species"] == species]
    if not rows:
        return None
    assets = sorted({r["asset"] for r in rows})
    plants = sorted({r["plant"] for r in rows} | (set(assets) if method == "real" else set()))
    d = np.full((len(assets), len(plants)), np.nan)
    for r in rows:
        d[assets.index(r["asset"]), plants.index(r["plant"])] = float(r[distance])
    if method == "real":
        np.fill_diagonal(d, 0)  # a plant with itself is not scored; set_metrics masks the diagonal
    if np.isnan(d).any():
        raise SystemExit(f"{method}/{species}: incomplete asset x plant matrix in shape_pairs.csv")
    return d


def summarise(pairs):
    out, keys = [], ("mmd", "cov", "nn_mean")
    methods = sorted({r["method"] for r in pairs})
    for name, lower in [("cd", True)] + [(f"f@{tau}", False) for tau in TAUS]:
        for method in methods:
            per_species = []
            for species in SPECIES:
                d = pair_matrix(pairs, method, species, name)
                if d is not None:
                    per_species.append({"method": method, "species": species, "distance": name, "n_assets": len(d),
                                        "n_plants": d.shape[1], **set_metrics(d, lower, method == "real")})
            out += per_species
            if len(per_species) == len(SPECIES):
                out.append({"method": method, "species": "all", "distance": name,
                            "n_assets": sum(r["n_assets"] for r in per_species),
                            "n_plants": sum(r["n_plants"] for r in per_species),
                            **{k: float(np.mean([r[k] for r in per_species])) for k in keys}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders", type=Path, help="<method>/<asset>/points.npz + cameras.json (e.g. renders)")
    p.add_argument("--real", type=Path, required=True, help="PlantDreamer plants dir: <plant>/data/*_real_pointcloud.ply")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    pairs_path = a.out / "shape_pairs.csv"
    pairs = list(csv.DictReader(open(pairs_path))) if pairs_path.is_file() else []
    n_rows = len(pairs)
    pairs = [r for r in pairs if not uses_excluded_plant(r)]
    print(f"ignoring {n_rows - len(pairs)} rows with excluded plants {EXCLUDED_PLANTS}", flush=True)
    done = {(r["method"], r["asset"]) for r in pairs}
    jobs = [("real", ply.parent.parent.name, "") for ply in sorted(a.real.glob(REAL_GLOB))
            if ply.parent.parent.name not in EXCLUDED_PLANTS]
    jobs += [(m.name, d.name, str(d)) for m in sorted(a.renders.iterdir()) if m.is_dir()
             for d in sorted(m.iterdir()) if (d / "points.npz").is_file()]
    jobs = [j for j in jobs if j[:2] not in done]
    print(f"{len(jobs)} assets to score", flush=True)
    if jobs:  # loading the real clouds takes minutes; a rerun of the summary alone skips it
        REAL.update(load_real(a.real))
        print(f"{len(REAL)} real plants", flush=True)

    fields = ["method", "species", "asset", "plant", "n_gen", "n_real", "cd"] + [f"f@{tau}" for tau in TAUS]
    a.out.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        if not pairs:
            writer.writeheader()
        # spawn, not fork: open3d's thread pool, started by load_real in this process, deadlocks forked workers
        with multiprocessing.get_context("spawn").Pool(a.workers, _init_worker, (REAL,)) as pool:
            for rows in pool.imap_unordered(score_asset, jobs):
                writer.writerows(rows)
                fh.flush()
                pairs += rows
                print(f"{rows[0]['method']}/{rows[0]['asset']}: best F@0.02 "
                      f"{max(r['f@0.02'] for r in rows):.3f}", flush=True)

    summary = summarise(pairs)
    with open(a.out / "shape.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {pairs_path} and {a.out / 'shape.csv'}")


if __name__ == "__main__":
    main()
