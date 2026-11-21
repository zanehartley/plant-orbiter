"""TRELLIS-text-xlarge (TRELLIS v1, text-conditioned) on the paper's prompts (text_prompts.py): one Gaussian splat per species x seed.

Settings follow the repo's example_text.py at the pinned commit: default sampler settings, one sample per seed. Only
the Gaussian decoder runs (formats=["gaussian"]); the mesh and radiance-field decoders read the same structured latent
and are not scored. The splat is saved with the repo's save_ply, which maps TRELLIS's z-up frame to OpenCV/COLMAP
axes (y down), so the PLY's up is -y. TRELLIS renders its splats with mip-splatting's rasterizer (2D filter with
opacity compensation), so they are scored with gsplat's antialiased mode.

Writes <out>/<asset>/{gaussians.ply, run.json}. Run in the trellis env:
    python baselines/text_to_3d/run_trellis_text.py --experiment synthetic_plants --out $ORBIT_OUTPUTS/synthetic_plants/baselines/trellis_text \
        [--only bean_t42 ...]
"""
import os

os.environ.setdefault("SPCONV_ALGO", "native")  # as in example_text.py: no autotuning for one-off runs

import argparse
import json
import sys
import time
from pathlib import Path

from text_prompts import EXPERIMENTS, jobs

TRELLIS_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_DIR))
from trellis.pipelines import TrellisTextTo3DPipeline  # noqa: E402

MODEL = "microsoft/TRELLIS-text-xlarge"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="asset names to run (default: all)")
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True, help="prompt set (text_prompts.py)")
    a = p.parse_args()

    pipeline = TrellisTextTo3DPipeline.from_pretrained(MODEL)
    pipeline.cuda()
    for asset, prompt, seed in jobs(a.experiment):
        out = a.out / asset
        if (a.only and asset not in a.only) or (out / "gaussians.ply").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        gaussians = pipeline.run(prompt, seed=seed, formats=["gaussian"])["gaussian"][0]
        gaussians.save_ply(str(out / "gaussians.ply"))
        secs = round(time.time() - start, 1)
        (out / "run.json").write_text(json.dumps({"model": MODEL, "prompt": prompt, "seed": seed, "seconds": secs},
                                                 indent=2))
        print(f"{asset}: {secs} s", flush=True)


if __name__ == "__main__":
    main()
