"""SV3D_p (Voleti et al. ECCV 2024) as a video-model baseline: it replaces only our MiniMax-H3 + orbit
LoRA step; everything downstream is our pipeline (mask_frames.py, known_poses.py, make_splat_kp.sh, align_splat.py).

Sampling follows scripts/sampling/simple_video_sample.py (sv3d_p: 21 frames at 576 x 576, 50 steps, cond_aug 1e-5,
constant elevation 10 deg, default azimuths, the conditioning photo written back into the last frame), without the
watermark and NSFW filter (they alter frames / download extra models). Attention falls back to plain softmax /
vanilla (no xformers). Input = our plant-centred square crop (make_nvs_inputs.py crop/, white background), passed
as RGBA with full alpha so SV3D's own re-centring keeps it as is.

Frames are re-ordered so 00000 is the input view (SV3D's last frame, azimuth 0); frame k (k >= 1) is SV3D frame
k - 1 at azimuth 360 k / 21. cams.json holds the orbit (radius 1, elevation 10 deg) for BOTH turning directions
('c2w' and 'alt_c2w') plus a focal search range: known_poses.py keeps the direction and focal whose visual hull
reproduces the masks best (SV3D documents neither its field of view nor the sign of its azimuth).

    python baselines/single_photo/run_sv3d.py inputs/nvs/crop/bean_3_0009_train.png [...] --seed 42 --out outputs/nvs/sv3d/pd
Run in the "nvs" env (setup/setup_nvs.sh) with HF access to stabilityai/sv3d.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SV3D = ROOT / "third_party" / "generative-models"
N, ELEV = 21, 10.0


def orbit(sign):
    """OpenCV c2w for our frame order: frame 0 = input at azimuth 0, frame k at sign * 360 k / N, elevation ELEV,
    radius 1, target at the origin, world z up."""
    out = []
    e = math.radians(ELEV)
    for k in range(N):
        phi = sign * 2 * math.pi * k / N
        C = np.array([math.cos(e) * math.cos(phi), math.cos(e) * math.sin(phi), math.sin(e)])
        z = -C / np.linalg.norm(C)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, :3] = np.stack([x, y, z], 1)
        m[:3, 3] = C
        out.append(m.tolist())
    return out


def load(ckpt):
    sys.path.insert(0, str(SV3D))
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    cfg = OmegaConf.load(SV3D / "scripts" / "sampling" / "configs" / "sv3d_p.yaml")
    p = cfg.model.params
    p.ckpt_path = str(ckpt)
    p.network_config.params.spatial_transformer_attn_type = "softmax"
    p.conditioner_config.params.emb_models[1].params.encoder_config.params.ddconfig.attn_type = "vanilla"
    p.first_stage_config.params.decoder_config.params.attn_type = "vanilla"  # AutoencodingEngine: decoder_config, not ddconfig
    p.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = "cuda"
    p.sampler_config.params.num_steps = 50
    p.sampler_config.params.guider_config.params.num_frames = N
    p.sampler_config.params.verbose = False
    with torch.device("cuda"):
        model = instantiate_from_config(cfg.model).to("cuda").eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="sv3d_p.safetensors")
    a = ap.parse_args()
    todo = [(im, a.out / (im.stem + ("" if a.seed == 42 else f"_seed{a.seed}"))) for im in a.images]
    todo = [(im, r) for im, r in todo if not (r / "cams.json").exists()]
    if not todo:
        print("SV3D: nothing to do")
        return
    model = load(a.ckpt)
    sys.path.insert(0, str(SV3D / "scripts" / "sampling"))
    # simple_video_sample imports rembg (background removal) and the NSFW / watermark filters at module level; we use
    # neither (inputs are already cut out; no watermark / filter, see the docstring), so stub them before importing
    import types
    for name in ("rembg", "imwatermark", "scripts.util.detection.nsfw_and_watermark_dectection"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:  # noqa: BLE001
                class _Unused:  # instantiated at import by sgm.inference.helpers; never applied to our frames
                    def __init__(self, *a, **k): pass
                    def set_watermark(self, *a, **k): pass
                sys.modules[name] = types.SimpleNamespace(remove=None, DeepFloydDataFiltering=_Unused, WatermarkEncoder=_Unused)
    from simple_video_sample import get_batch, get_unique_embedder_keys_from_conditioner
    from einops import repeat, rearrange
    polars = [math.radians(90 - ELEV)] * N
    az = np.linspace(0, 360, N + 1)[1:] % 360
    azr = [math.radians((x - az[-1]) % 360) for x in az]
    azr[:-1] = sorted(azr[:-1])
    for im_path, run in todo:
        torch.manual_seed(a.seed)
        im = Image.open(im_path).convert("RGB").resize((576, 576), Image.LANCZOS)
        image = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].cuda() * 2 - 1
        vd = {"cond_frames_without_noise": image, "motion_bucket_id": 127, "fps_id": 6, "cond_aug": 1e-5,
              "cond_frames": image + 1e-5 * torch.randn_like(image), "polars_rad": polars, "azimuths_rad": azr}
        with torch.no_grad(), torch.autocast("cuda"):
            batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), vd, [1, N],
                                        T=N, device="cuda")
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, batch_uc=batch_uc, force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"])
            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=N)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=N)
                c[k] = repeat(c[k], "b ... -> b t ...", t=N)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=N)
            randn = torch.randn((N, 4, 72, 72), device="cuda")
            extra = {"image_only_indicator": torch.zeros(2, N, device="cuda"), "num_video_frames": batch["num_video_frames"]}
            den = lambda x, s, cc: model.denoiser(model.model, x, s, cc, **extra)
            z = model.sampler(den, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = 1  # 7 (the script's value) runs out of memory on a 24 GB card
            del c, uc, batch, batch_uc, randn
            torch.cuda.empty_cache()
            x = model.decode_first_stage(z)
            x[-1:] = image
        frames = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        (run / "frames").mkdir(parents=True, exist_ok=True)
        order = [N - 1] + list(range(N - 1))  # input view first
        for k, j in enumerate(order):
            Image.fromarray(frames[j]).save(run / "frames" / f"{k:05d}.png")
        cj = im_path.with_suffix(".json")
        if cj.exists():
            shutil.copy2(cj, run / "input.json")  # crop shifts for align_splat.py
        (run / "cams.json").write_text(json.dumps({
            "c2w": orbit(1), "alt_c2w": orbit(-1), "search": "200:1500:25", "source": "SV3D_p orbit, elevation 10",
            "elevation_deg": ELEV, "frames": N}, indent=1))
        print(f"SV3D {run.name}: {N} frames", flush=True)


if __name__ == "__main__":
    main()
