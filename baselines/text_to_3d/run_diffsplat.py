"""DiffSplat (SD3.5-medium, text-conditioned) on the paper's prompts (text_prompts.py): one Gaussian splat per species x seed.

Runs the repo's own src/infer_gsdiff_sd3.py with the README's SD3.5m command (config gsdiff_sd35m_80g.yaml, tag
gsdiff_gobj83k_sd35m__render, default sampler settings, full precision) plus --save_ply. One call per seed, with the
three species prompts in a prompt file: the script seeds one generator per call, so each (seed, species) sample is
reproducible. The repo's checkpoint dir (--output_dir) is $ORBIT_WEIGHTS/diffsplat, where the script also writes its
inference outputs; each PLY is moved to <out>/<asset>/gaussians.ply. DiffSplat's PLY is standard 3DGS in an OpenGL
y-up world, rendered by DiffSplat with a plain 3DGS rasterizer (RaDe-GS), so it is scored y-up in classic mode.

Run in the diffsplat env:
    python baselines/text_to_3d/run_diffsplat.py --experiment synthetic_plants --out $ORBIT_OUTPUTS/synthetic_plants/baselines/diffsplat
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from text_prompts import EXPERIMENTS, jobs

DIFFSPLAT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "DiffSplat"
CKPT_DIR = Path(os.environ.get("ORBIT_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights")) / "diffsplat"
CONFIG, TAG = "configs/gsdiff_sd35m_80g.yaml", "gsdiff_gobj83k_sd35m__render"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", help="default: every seed of the experiment")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True, help="prompt set (text_prompts.py)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    prompt_file = a.out / "prompts.txt"
    infer_dir = CKPT_DIR / TAG / "inference"

    for seed in a.seeds or sorted({s for _, _, s in jobs(a.experiment)}):
        batch = [(a.out / asset, prompt) for asset, prompt, s in jobs(a.experiment) if s == seed]  # one run per seed
        if all((asset / "gaussians.ply").exists() for asset, _ in batch):
            continue
        assets, prompts = map(list, zip(*batch))
        prompt_file.write_text("\n".join(prompts) + "\n")
        for stale in infer_dir.glob("000_*"):  # outputs are named by prompt index only, so clear the last seed's
            stale.unlink()
        start = time.time()
        subprocess.run([sys.executable, "src/infer_gsdiff_sd3.py", "--config_file", CONFIG, "--tag", TAG,
                        "--allow_tf32", "--output_dir", str(CKPT_DIR), "--prompt_file", str(prompt_file),
                        "--seed", str(seed), "--gpu_id", "0", "--save_ply", "--name_by_id"],
                       cwd=DIFFSPLAT_DIR, check=True)
        secs = round((time.time() - start) / len(batch), 1)  # per asset, model loading included
        for j, (asset, prompt) in enumerate(zip(assets, prompts)):  # --name_by_id: <image>_<prompt>_<ckpt step>
            ply, = infer_dir.glob(f"000_{j:03d}_*.ply")  # exactly one per prompt
            asset.mkdir(exist_ok=True)
            shutil.move(str(ply), asset / "gaussians.ply")
            shutil.move(str(ply.with_name(f"{ply.stem}_gs.png")), asset / "input_views.png")
            (asset / "run.json").write_text(json.dumps({"model": f"DiffSplat {TAG}", "prompt": prompt, "seed": seed,
                                                        "seconds": secs}, indent=2))
        print(f"seed {seed}: {', '.join(x.name for x in assets)}, {secs} s each", flush=True)


if __name__ == "__main__":
    main()
