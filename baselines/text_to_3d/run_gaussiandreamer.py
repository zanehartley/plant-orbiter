"""GaussianDreamer (hustvl, SDS on 3D Gaussians initialised from Shap-E) on the paper's prompts (text_prompts.py): one splat per
species x seed.

Runs the repo's own launch.py with the README's text-to-3D command (configs/gaussiandreamer-sd.yaml: 1200 steps,
sh_degree 0) and default settings; only the prompt, the seed and the output location are set. The seed drives both
the Shap-E initialisation and the SDS run (threestudio seeds everything). SD 2.1-base, used for the prompt processor
and the guidance, is no longer hosted by stabilityai; the sd2-community mirror is used (check_sd21_mirror.py).
The splat stays in threestudio's z-up world (Shap-E's points are z-up and used as is; cameras orbit +z) and is
rendered by the original 3DGS rasterizer, so it is scored +z up in classic mode.

The repo reads its Shap-E checkpoint from ./load/ and Shap-E reads its base models from ./shap_e_model_cache, both
relative to the working directory, so it runs from <out> with load/ (the repo's load/ entries plus Cap3D's fine-tuned
Shap-E) and shap_e_model_cache linked in from $ORBIT_WEIGHTS/gaussiandreamer (scripts/setup/setup_gaussiandreamer.sh).

Writes <out>/<asset>/trial/ (threestudio's trial dir; save/last_3dgs.ply), <out>/<asset>/gaussians.ply (a copy of
last_3dgs.ply) and <out>/<asset>/run.json, written last, so it marks a finished run.
Run in the gaussiandreamer env:
    python baselines/text_to_3d/run_gaussiandreamer.py --experiment synthetic_plants \
        --out $ORBIT_OUTPUTS/synthetic_plants/baselines/gaussiandreamer
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

GD_DIR = Path(__file__).resolve().parents[2] / "third_party" / "GaussianDreamer"
CONFIG = GD_DIR / "configs" / "gaussiandreamer-sd.yaml"
WEIGHTS = Path(os.environ.get("ORBIT_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights")) / "gaussiandreamer"
SHAPE_CKPT = WEIGHTS / "misc" / "our_finetuned_models" / "shapE_finetuned_with_330kdata.pth"
SD21_MIRROR = "sd2-community/stable-diffusion-2-1-base"


def link_working_dir(out):
    """<out>/load/ (repo load/ entries + the Shap-E checkpoint) and <out>/shap_e_model_cache, as the code expects."""
    load = out / "load"
    load.mkdir(exist_ok=True)
    for src in [*(GD_DIR / "load").iterdir(), SHAPE_CKPT]:
        if not (load / src.name).exists():
            (load / src.name).symlink_to(src)
    if not (out / "shap_e_model_cache").exists():
        (out / "shap_e_model_cache").symlink_to(WEIGHTS / "shap_e_model_cache")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="asset names to run (default: all)")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True, help="prompt set (text_prompts.py)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    link_working_dir(a.out)

    for asset, prompt, seed in jobs(a.experiment):
        run = a.out / asset
        if (a.only and asset not in a.only) or (run / "run.json").exists():
            continue
        start = time.time()
        subprocess.run([sys.executable, str(GD_DIR / "launch.py"), "--config", str(CONFIG), "--train", "--gpu", "0",
                        f"system.prompt_processor.prompt={prompt}", f"seed={seed}",
                        f"system.prompt_processor.pretrained_model_name_or_path={SD21_MIRROR}",
                        f"system.guidance.pretrained_model_name_or_path={SD21_MIRROR}",
                        f"exp_root_dir={a.out}", f"name={asset}", "tag=trial", "use_timestamp=false"],
                       cwd=a.out, check=True)
        secs = round(time.time() - start, 1)
        shutil.copyfile(run / "trial" / "save" / "last_3dgs.ply", run / "gaussians.ply")
        (run / "run.json").write_text(json.dumps({"config": CONFIG.name, "prompt": prompt, "seed": seed,
                                                  "sd21": SD21_MIRROR, "seconds": secs}, indent=2))
        print(f"{asset}: {secs} s", flush=True)


if __name__ == "__main__":
    main()
