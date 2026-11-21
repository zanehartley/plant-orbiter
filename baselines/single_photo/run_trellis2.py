"""TRELLIS.2 (microsoft/TRELLIS.2-4B, MIT) image-to-3D on the same RGBA cutouts as the other baselines (run in the
conda env trellis2; install: setup/setup_trellis2.sh).

    python baselines/single_photo/run_trellis2.py inputs/rgba/bean_3_0009_train.png [...] [--seed 42]
Writes <out-root>/<photo stem>[_seed<N>]/{mesh.glb, input_rgba.png, run.json}; then reconstruction/mesh_to_splat.py and
reconstruction/align_generic.py place and score it like Hunyuan3D. The cutout's hidden (alpha 0) pixels are zeroed first: TRELLIS.2
premultiplies by alpha but resizes inputs over 1024 px before that, which would blend the hidden photo into leaf edges.
Only the shape + texture pipeline runs; its background remover is never used (inputs have alpha).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "TRELLIS.2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import o_voxel  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402
from trellis2.pipelines import rembg as _rembg  # noqa: E402


class _NoRembg:
    """Stand-in for the pipeline's background remover (briaai/RMBG-2.0: gated, CC BY-NC): our inputs are RGBA cutouts,
    so preprocess_image never calls it; this keeps that model out of the pipeline (and fails loudly if it is ever used)."""
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("input has no alpha channel: TRELLIS.2 background removal is disabled in this baseline")


_rembg.BiRefNet = _NoRembg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, default=Path(os.environ.get("ORBIT_OUTPUTS", ROOT / "data" / "outputs")) / "baselines" / "trellis2")
    ap.add_argument("--pipeline-type", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--texture-size", type=int, default=2048, help="GLB texture size (2048 in the paper)")
    ap.add_argument("--decimation", type=int, default=1000000, help="GLB face target (full run: 500000, disk)")
    a = ap.parse_args()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    for img in a.images:
        out = a.out_root / (img.stem + (f"_seed{a.seed}" if a.seed != 42 else ""))
        if (out / "mesh.glb").exists():
            print("skip", out, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        im = np.asarray(Image.open(img).convert("RGBA")).copy()
        im[im[..., 3] == 0, :3] = 0
        im = Image.fromarray(im)
        im.save(out / "input_rgba.png")
        t0 = time.time()
        mesh = pipe.run(im, seed=a.seed, pipeline_type=a.pipeline_type)[0]
        mesh.simplify(16777216)  # nvdiffrast limit (as example.py)
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                         attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                         decimation_target=a.decimation, texture_size=a.texture_size, remesh=True, remesh_band=1, remesh_project=0)
        glb.export(str(out / "mesh.glb"))
        (out / "run.json").write_text(json.dumps({"seed": a.seed, "pipeline_type": a.pipeline_type, "seconds": round(time.time() - t0, 1),
                                                  "model": "microsoft/TRELLIS.2-4B", "texture_size": a.texture_size, "decimation": a.decimation, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))
        print("wrote", out, f"{time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
