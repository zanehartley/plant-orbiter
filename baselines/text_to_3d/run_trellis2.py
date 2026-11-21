"""TRELLIS.2-4B on our fruit-counting input images (RGBA from prep_inputs.py): one textured GLB per image.

The same-input control: it lifts the very images our orbit model lifts, so a count it loses is lost in 3D lifting.

Settings follow the repo's example.py at the pinned commit: default pipeline type from the model's pipeline.json
(1024_cascade), seed 42, and the same GLB export (remesh, 1M-face decimation target, 4096 texture), except textures
are stored as PNG rather than WebP. TRELLIS.2 uses the input's alpha channel as given, so its own background remover
never runs.

Writes <out>/<plant>/mesh.glb and run.json (settings + seconds). Run in the trellis2 env on a >= 24 GB GPU:
    python baselines/text_to_3d/run_trellis2.py $ORBIT_OUTPUTS/fruit_counting/baselines/inputs \
        --out $ORBIT_OUTPUTS/fruit_counting/baselines/trellis2 [--only tomato_n5_s42 ...]
"""
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # as in example.py: saves GPU memory

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

TRELLIS2_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TRELLIS.2"
sys.path.insert(0, str(TRELLIS2_DIR))
import o_voxel  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402

MODEL = "microsoft/TRELLIS.2-4B"
SEED = 42
EXPORT = {"decimation_target": 1000000, "texture_size": 4096, "remesh": True, "remesh_band": 1, "remesh_project": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, help="directory of <plant>.png RGBA inputs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="plant names to run (default: all)")
    a = p.parse_args()

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL)
    pipeline.cuda()
    for path in sorted(a.inputs.glob("*.png")):
        plant = path.stem
        out = a.out / plant
        if (a.only and plant not in a.only) or (out / "mesh.glb").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        mesh = pipeline.run(Image.open(path), seed=SEED)[0]
        mesh.simplify(16777216)  # nvdiffrast limit (example.py)
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                                         coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                                         aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], verbose=False, **EXPORT)
        glb.export(out / "mesh.glb")  # PNG textures (example.py: WebP), so standard loaders read them
        secs = round(time.time() - start, 1)
        (out / "run.json").write_text(json.dumps({"model": MODEL, "seed": SEED, "input": str(path),
                                                  "pipeline_type": pipeline.default_pipeline_type, "export": EXPORT,
                                                  "seconds": secs}, indent=2))
        print(f"{plant}: {secs} s", flush=True)


if __name__ == "__main__":
    main()
