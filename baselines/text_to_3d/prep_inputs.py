"""RGBA inputs for the image-to-3D baseline (TRELLIS.2): each Qwen image with the subject mask from our own BiRefNet.

The orbit pipeline cuts the subject out with ComfyUI's BiRefNet (pipeline/orbit.py --remove-bg); the same weights run
here, directly through ComfyUI's loader, on the full-resolution image. Giving every method the same mask keeps the
baseline's own background remover (TRELLIS.2: RMBG-2.0) out of the comparison; TRELLIS.2 uses the alpha channel as
given when the input is RGBA.

Writes <out>/<plant>.png (RGBA, same size as the input). Run in the orbit_cam env:
    python baselines/text_to_3d/prep_inputs.py $ORBIT_OUTPUTS/fruit_counting/images \
        --out $ORBIT_OUTPUTS/fruit_counting/baselines/inputs
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get("COMFYUI_DIR", str(Path(__file__).resolve().parents[2] / "third_party" / "ComfyUI")))
import comfy.bg_removal_model  # noqa: E402  (ComfyUI is not an installed package)

BIREFNET = Path(os.environ.get("ORBIT_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights")) / "comfyui" / "background_removal" \
    / "birefnet.safetensors"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, help="directory of <plant>.png text-to-image outputs")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    model = comfy.bg_removal_model.load(str(BIREFNET))
    for path in sorted(a.images.glob("*.png")):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        with torch.no_grad():
            mask = model.encode_image(torch.from_numpy(rgb).float()[None] / 255)[0]  # ComfyUI IMAGE: BHWC in [0, 1]
        alpha = (mask.float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(a.out / path.name)
        print(f"{path.name}: subject fraction {(alpha > 127).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
