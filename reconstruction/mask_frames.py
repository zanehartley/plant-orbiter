"""BiRefNet foreground masks for generated frames, exactly as orbit.py makes them for our MiniMax orbits (ComfyUI's
LoadBackgroundRemovalModel + RemoveBackground: soft sigmoid mask saved as 8-bit grey, RGBA alpha = that mask), so the
SEVA / SV3D video-model baselines get the same masker as ours. Run in the orbit_cam env.

    python mask_frames.py <run dir> [...]     reads <run>/frames/*.png, writes <run>/masks/ and <run>/rgba/
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", REPO / "third_party" / "ComfyUI"))  # set by config/paths.sh
WEIGHTS = Path(os.environ.get("ORBIT_WEIGHTS", REPO / "data" / "weights"))
sys.path.insert(0, str(COMFYUI_DIR))
import comfy.bg_removal_model as bgm  # noqa: E402


def main():
    model = bgm.load(str(WEIGHTS / "comfyui" / "background_removal" / "birefnet.safetensors"))
    for run in map(Path, sys.argv[1:]):
        (run / "masks").mkdir(exist_ok=True)
        (run / "rgba").mkdir(exist_ok=True)
        frames = sorted((run / "frames").glob("*.png"))
        for f in frames:
            im = Image.open(f).convert("RGB")
            t = torch.from_numpy(np.asarray(im, np.float32) / 255)[None]
            with torch.no_grad():
                m = model.encode_image(t)[0].float().clamp(0, 1).cpu().numpy()
            mask = Image.fromarray((m * 255).round().astype(np.uint8), "L")
            mask.save(run / "masks" / f.name)
            rgba = im.copy()
            rgba.putalpha(mask)
            rgba.save(run / "rgba" / f.name)
        print(f"{run}: {len(frames)} frames masked", flush=True)


if __name__ == "__main__":
    main()
