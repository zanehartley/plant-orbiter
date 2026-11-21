"""SEVA (Stable Virtual Camera, Zhou et al. ICCV 2025) as a video-model baseline: it replaces only our
MiniMax-H3 + orbit-LoRA step; everything downstream is our pipeline (mask_frames.py, known_poses.py with the given
cameras, make_splat_kp.sh, align_splat.py, 2DGS).

Setting: single-image orbit, SEVA's documented recipe (img2trajvid_s-prob, traj_prior orbit, cfg 4,2, guider 1,2,
two-pass procedural sampling, chunk interp), 72 frames = the input + 71 targets, i.e. 5 deg per frame as our orbit.
Input = the same cutout as ours (whole photo on the white square canvas, make_nvs_inputs.py full/), 576 x 576.

Writes per input <out>/<stem>[_seed<N>]/frames/00000..00071.png (00000 = input view) and cams.json (OpenCV c2w,
recentred on the orbit target and scaled to radius 1; focal in px at 576), plus SEVA's own transforms.json.

    python baselines/single_photo/run_seva.py inputs/nvs/full/bean_3_0009_train.png [...] --seed 42 --out outputs/nvs/seva/pd
Run in the "nvs" env (setup/setup_nvs.sh).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVA = ROOT / "third_party" / "stable-virtual-camera"


def lookat_target(c2w):
    """Point closest to every optical axis (OpenCV: camera looks along +z)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for m in c2w:
        c, f = m[:3, 3], m[:3, 2] / np.linalg.norm(m[:3, 2])
        P = np.eye(3) - np.outer(f, f)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


# SEVA's autoencoder is the SD 2.1-base VAE, but stabilityai/stable-diffusion-2-1-base has been removed from the Hub
# (checked October 2026). LanguageMachines/stable-diffusion-2-1-base carries the byte-identical file: sha256 a1d99348... is the
# same VAE as in stabilityai/stable-diffusion-2-1 and -2-depth. We redirect the download and verify the hash.
VAE_SHA = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"


def patch_vae():
    import hashlib
    from diffusers.models import AutoencoderKL
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("LanguageMachines/stable-diffusion-2-1-base", "vae/diffusion_pytorch_model.safetensors")
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert h == VAE_SHA, f"SD 2.1 VAE mirror hash mismatch: {h}"
    orig = AutoencoderKL.from_pretrained.__func__

    def from_pretrained(cls, name, *args, **kw):
        if name == "stabilityai/stable-diffusion-2-1-base":
            name = "LanguageMachines/stable-diffusion-2-1-base"
        return orig(cls, name, *args, **kw)
    AutoencoderKL.from_pretrained = classmethod(from_pretrained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-targets", type=int, default=71)
    ap.add_argument("--reuse-work", help="existing SEVA work dir from an interrupted run: convert, don't re-sample")
    a = ap.parse_args()
    work = Path(a.reuse_work) if a.reuse_work else Path(tempfile.mkdtemp(prefix="seva_", dir=Path(os.environ.get("ORBIT_OUTPUTS", ROOT / "data" / "outputs")) / "nvs"))
    data = work / "in"
    data.mkdir(exist_ok=True)
    todo = []
    for im in a.images:
        run = a.out / (im.stem + ("" if a.seed == 42 else f"_seed{a.seed}"))
        if (run / "cams.json").exists():
            continue
        if not (data / im.name).exists():
            os.symlink(im.resolve(), data / im.name)
        todo.append((im, run))
    if not todo:
        print("SEVA: nothing to do")
        return
    sys.path.insert(0, str(SEVA))
    cwd = os.getcwd()
    if not a.reuse_work:
        patch_vae()
        os.chdir(work)  # demo.py writes to ./work_dirs/demo/<task>/<save_subdir>/<scene>/
        import demo
        demo.main(data_path=str(data), task="img2trajvid_s-prob", replace_or_include_input=True, traj_prior="orbit",
              # documented recipe "--cfg 4.0,2.0 --guider 1,2"; the code reads cfg as a tuple and the guider
              # per pass as options["guider_types"] (a "guider" option is silently ignored), so pass both explicitly
              cfg=(4.0, 2.0), guider_types=(1, 2), num_targets=a.num_targets, L_short=576, use_traj_prior=True,
              chunk_strategy="interp", seed=a.seed, save_subdir=f"s{a.seed}", video_save_fps=24.0)
        os.chdir(cwd)
    for im, run in todo:
        sd = work / "work_dirs" / "demo" / "img2trajvid_s-prob" / f"s{a.seed}" / im.stem
        tj = json.loads((sd / "transforms.json").read_text())
        (run / "frames").mkdir(parents=True, exist_ok=True)
        c2w = []
        for k, fr in enumerate(tj["frames"]):
            shutil.copy2(sd / fr["file_path"], run / "frames" / f"{k:05d}.png")
            m = np.eye(4)
            m[:3, :4] = np.array(fr["transform_matrix"], float)[:3, :4]  # SEVA stores 3x4
            c2w.append(m)
        c2w = np.stack(c2w) @ np.diag([1.0, -1.0, -1.0, 1.0])  # transforms.json is OpenGL (camera looks down -z) -> OpenCV
        fwd0 = c2w[0, :3, 2]
        assert fwd0[2] > 0.99, f"camera 0 should look along +z (OpenCV) in SEVA's orbit, got {fwd0}"
        tgt = lookat_target(c2w)
        r = np.linalg.norm(c2w[0, :3, 3] - tgt)
        c2w[:, :3, 3] = (c2w[:, :3, 3] - tgt) / r
        fr0 = tj["frames"][0]
        # SEVA scales K by the ORIGINAL input size (e.g. 1063 / 1560 px), not the 576 px it generates: rescale by
        # the principal point (cx = W_input / 2) so the focal is in px at 576 (= 54 deg FOV, its default)
        f = fr0["fl_x"] * (fr0["w"] / 2) / fr0["cx"]
        (run / "cams.json").write_text(json.dumps({"c2w": c2w.tolist(), "focal": f, "source": "SEVA transforms.json",
                                                   "radius_raw": r, "target_raw": tgt.tolist()}, indent=1))
        shutil.copy2(sd / "transforms.json", run / "seva_transforms.json")
        print(f"SEVA {run.name}: {len(c2w)} frames, focal {f:.1f} px", flush=True)
    shutil.rmtree(work)  # this job's own temporary SEVA work dir


if __name__ == "__main__":
    main()
