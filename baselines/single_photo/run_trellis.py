"""TRELLIS (microsoft/TRELLIS-image-large) image-to-3D on the same RGBA cutouts as our method (run in the "trellis"
conda env; install: setup/setup_trellis_text.sh, then setup/setup_trellis_image.sh).

    python baselines/single_photo/run_trellis.py inputs/rgba/bean_3_0009_train.png [...] [--seed 42]
    python baselines/single_photo/run_trellis.py front.png,rear.png [...] --multi --tag _2view   # not used in the paper
Writes $ORBIT_OUTPUTS/baselines/trellis/<photo stem>/{gaussians.ply, mesh.glb}. --multi: each argument is a comma-separated
group (front photo first) given to TRELLIS's own multi-image mode (run_multi_image, mode "stochastic" as in its
example); the output is named after the front photo + --tag.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "TRELLIS"))

from PIL import Image  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import postprocessing_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="photos, or comma-separated groups with --multi")
    ap.add_argument("--multi", action="store_true", help="each argument is a group of photos of one plant (front first)")
    ap.add_argument("--tag", default="", help="output-name suffix (e.g. _exact2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, help="parent dir for the per-photo outputs (default outputs/baselines/trellis)")
    a = ap.parse_args()
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    for arg in a.images:
        group = [Path(x) for x in arg.split(",")] if a.multi else [Path(arg)]
        out = (a.out_root or Path(os.environ.get("ORBIT_OUTPUTS", ROOT / "data" / "outputs")) / "baselines" / "trellis") / (group[0].stem + a.tag + (f"_seed{a.seed}" if a.seed != 42 else ""))
        out.mkdir(parents=True, exist_ok=True)
        if a.multi:
            res = pipe.run_multi_image([Image.open(x) for x in group], seed=a.seed, mode="stochastic")
        else:
            res = pipe.run(Image.open(group[0]), seed=a.seed)  # TRELLIS does its own background removal / cropping
        res["gaussian"][0].save_ply(str(out / "gaussians.ply"))
        glb = postprocessing_utils.to_glb(res["gaussian"][0], res["mesh"][0], simplify=0.95, texture_size=1024)
        glb.export(str(out / "mesh.glb"))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
