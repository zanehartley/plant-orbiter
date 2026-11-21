"""Hunyuan3D 2.1 (shape + PBR texture) on the same RGBA cutouts as our method (run in the "hunyuan" env; install:
setup/setup_hunyuan.sh). Check the Hunyuan3D 2.1 licence for your territory before use; in the paper its outputs are
used for comparison only.

    python baselines/single_photo/run_hunyuan.py inputs/rgba/bean_3_0009_train.png [...] [--seed 42] [--no-texture]
Writes $ORBIT_OUTPUTS/baselines/hunyuan/<photo stem>/{mesh_shape.glb, mesh.glb}
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HY = ROOT / "third_party" / "Hunyuan3D-2.1"
sys.path[:0] = [str(HY), str(HY / "hy3dshape"), str(HY / "hy3dpaint")]
os.chdir(HY)  # the paint pipeline loads configs/checkpoints by relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from torchvision_fix import apply_fix  # repo helper for torchvision API changes
    apply_fix()
except Exception:
    pass
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.rembg import BackgroundRemover  # noqa: E402


def paint_pipeline():
    """Hunyuan3D 2.1 PBR texture pipeline (also used by run_hunyuan_mv.py)."""
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    # upstream defaults mix repo-root and hy3dpaint-relative paths; same overrides as the official demo.py
    conf.realesrgan_ckpt_path = str(HY / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = str(HY / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = str(HY / "hy3dpaint" / "hunyuanpaintpbr")
    return Hunyuan3DPaintPipeline(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, help="parent dir for the per-photo outputs (default outputs/baselines/hunyuan)")
    ap.add_argument("--no-texture", action="store_true")
    a = ap.parse_args()
    images = [p if p.is_absolute() else (ROOT / p) for p in a.images]
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    rembg = BackgroundRemover()
    outs = []
    for img in images:
        out = (a.out_root or Path(os.environ.get("ORBIT_OUTPUTS", ROOT / "data" / "outputs")) / "baselines" / "hunyuan") / (img.stem + (f"_seed{a.seed}" if a.seed != 42 else ""))
        out.mkdir(parents=True, exist_ok=True)
        im = Image.open(img)
        if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 255:
            pass  # the shared cutout (make_rgba_inputs.py): every method gets the same plant mask
        else:
            im = rembg(im.convert("RGB"))  # opaque input: Hunyuan's own background removal
        mesh = shape(image=im, generator=torch.Generator("cuda").manual_seed(a.seed))[0]
        mesh.export(str(out / "mesh_shape.glb"))
        im.save(out / "input_rgba.png")
        outs.append(out)
        print("shape", out, flush=True)
    if a.no_texture:
        return
    del shape
    torch.cuda.empty_cache()
    paint = paint_pipeline()
    for out in outs:
        paint(str(out / "mesh_shape.glb"), image_path=str(out / "input_rgba.png"), output_mesh_path=str(out / "mesh.obj"), save_glb=False)  # glb export needs Blender
        print("texture", out, flush=True)


if __name__ == "__main__":
    main()
