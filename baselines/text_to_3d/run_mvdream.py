"""MVDream (threestudio, SDS on a NeRF) on the paper's prompts (text_prompts.py): one trial per species x seed.

Runs the repo's own launch.py with the README's recommended config (configs/mvdream-sd21.yaml: no shading, 10k
steps) and default settings; only the prompt, the seed and the output location are set. The prompt processor's text
encoder is SD 2.1-base, which stabilityai no longer hosts; the sd2-community mirror is used instead (checked against
LAION's original OpenCLIP weights by check_sd21_mirror.py).

Writes <out>/<asset>/trial/ (threestudio's trial dir: configs/parsed.yaml, ckpts/last.ckpt, save/) and
<out>/<asset>/run.json. threestudio caches text embeddings under its working directory, so it runs from <out>, with
the repo's load/ (prompt library etc., read by relative path) linked in.
Run in the mvdream env:
    python baselines/text_to_3d/run_mvdream.py --experiment synthetic_plants --out $ORBIT_OUTPUTS/synthetic_plants/baselines/mvdream [--only ...]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from text_prompts import EXPERIMENTS, jobs

MVDREAM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "MVDream-threestudio"
CONFIG = MVDREAM_DIR / "configs" / "mvdream-sd21.yaml"
SD21_MIRROR = "sd2-community/stable-diffusion-2-1-base"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="asset names to run (default: all)")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True, help="prompt set (text_prompts.py)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not (a.out / "load").exists():
        (a.out / "load").symlink_to(MVDREAM_DIR / "load")

    for asset, prompt, seed in jobs(a.experiment):
        trial = a.out / asset / "trial"  # threestudio's <exp_root_dir>/<name>/<tag>
        # run.json is written last; a crashed run also leaves ckpts/last.ckpt, so that is no done marker
        if (a.only and asset not in a.only) or (trial.parent / "run.json").exists():
            continue
        start = time.time()
        subprocess.run([sys.executable, str(MVDREAM_DIR / "launch.py"), "--config", str(CONFIG), "--train", "--gpu", "0",
                        f"system.prompt_processor.prompt={prompt}", f"seed={seed}",
                        f"system.prompt_processor.pretrained_model_name_or_path={SD21_MIRROR}",
                        f"exp_root_dir={a.out}", f"name={asset}", "tag=trial", "use_timestamp=false"],
                       cwd=a.out, check=True)
        secs = round(time.time() - start, 1)
        (trial.parent / "run.json").write_text(json.dumps({"config": CONFIG.name, "prompt": prompt, "seed": seed,
                                                           "sd21_text_encoder": SD21_MIRROR, "seconds": secs},
                                                          indent=2))
        print(f"{asset}: {secs} s", flush=True)


if __name__ == "__main__":
    main()
