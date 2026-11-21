"""List one experiment's assets to evaluate: the inputs of render_splats.py, render_meshes.py and render_threestudio.py.

Methods (one row per finished asset):
  ours            our splats ($ORBIT_OUTPUTS/<experiment>/splats/<image>_s42/3dgs/export/splat.ply); z-up; rendered
                  in the rasterize mode it was trained with (antialiased, read from its config.yml)
  trellis_text    TRELLIS-text-xlarge (run_trellis_text.py); -y up (save_ply maps TRELLIS's z-up frame to
                  OpenCV/COLMAP axes, y down); antialiased (TRELLIS renders with mip-splatting's rasterizer)
  diffsplat       DiffSplat SD3.5-medium (run_diffsplat.py); +y up (OpenGL world, kiui orbit cameras; save_ply writes
                  positions unchanged); classic (plain 3DGS rasterizer)
  gaussiandreamer GaussianDreamer (run_gaussiandreamer.py); +z up (threestudio world, Shap-E points used as is);
                  classic (original 3DGS rasterizer)
Each row carries the rasterize mode its own method renders with, so render_splats.py renders every splat that way.
Text-only baselines are named <...>_t<seed>: they share no input with ours, so tests against them are unpaired.

Second manifest next to --out, manifest_threestudio.csv (input to render_threestudio.py):
  mvdream         MVDream (threestudio NeRF, run_mvdream.py)
Third manifest, manifest_meshes.csv (input to render_meshes.py), fruit counting only:
  trellis2        TRELLIS.2-4B on our RGBA input images, seed 42 (run_trellis2.py); glTF, so y-up. Named
                  <image>_s42 like ours (<image>_s<orbit seed>), so the same asset name means the same input image.

    python evaluation/build_manifest.py synthetic_plants --out $ORBIT_OUTPUTS/synthetic_plants/eval/manifest.csv
    python evaluation/build_manifest.py fruit_counting --out $ORBIT_OUTPUTS/fruit_counting/eval/manifest.csv
"""
import argparse
import csv
import os
import re
from pathlib import Path

SPLAT_BASELINES = (("trellis_text", "-y", "antialiased"), ("diffsplat", "+y", "classic"),
                   ("gaussiandreamer", "+z", "classic"))


def main():
    default_outputs = Path(__file__).resolve().parents[1] / "data" / "outputs"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=("synthetic_plants", "fruit_counting"))
    p.add_argument("--outputs", type=Path, default=Path(os.environ.get("ORBIT_OUTPUTS", default_outputs)))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.outputs / a.experiment

    rows = []
    for run in runs(root / "splats", "3dgs/export/splat.ply"):
        rows.append({"method": "ours", "species": run.name.split("_")[0], "asset": run.name,
                     "ply": str(run / "3dgs" / "export" / "splat.ply"), "up": "+z",
                     "rasterize_mode": trained_rasterize_mode(run / "3dgs")})
    for method, up, mode in SPLAT_BASELINES:
        for run in runs(root / "baselines" / method, "gaussians.ply"):
            rows.append({"method": method, "species": run.name.split("_")[0], "asset": run.name,
                         "ply": str(run / "gaussians.ply"), "up": up, "rasterize_mode": mode})

    nerf_rows = [{"method": "mvdream", "species": run.name.split("_")[0], "asset": run.name, "trial": str(run / "trial")}
                 for run in runs(root / "baselines" / "mvdream", "run.json")]  # run.json: written on success
    mesh_rows = [{"method": "trellis2", "species": run.name.split("_")[0], "asset": f"{run.name}_s42",
                  "mesh": str(run / "mesh.glb"), "up": "+y"}
                 for run in runs(root / "baselines" / "trellis2", "mesh.glb")]

    write_manifest(a.out, rows)
    if nerf_rows:
        write_manifest(a.out.with_name("manifest_threestudio.csv"), nerf_rows)
    if mesh_rows:
        write_manifest(a.out.with_name("manifest_meshes.csv"), mesh_rows)


def runs(method_dir, done_file):
    """Finished run dirs of a method (those holding done_file), sorted; none if the method has not run."""
    return [r for r in sorted(method_dir.iterdir()) if (r / done_file).is_file()] if method_dir.is_dir() else []


def trained_rasterize_mode(splat_dir):
    """The splatfacto rasterize mode recorded in the run's config.yml (YAML with Python tags, so read as text)."""
    config = next(splat_dir.glob("runs/*/*/v1/config.yml")).read_text()
    return re.search(r"^\s*rasterize_mode: (\w+)$", config, re.M).group(1)


def write_manifest(path, rows):
    if not rows:
        raise SystemExit(f"no finished assets for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["method"]] = counts.get(r["method"], 0) + 1
    print(counts, "->", path)


if __name__ == "__main__":
    main()
