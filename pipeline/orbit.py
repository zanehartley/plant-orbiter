#!/usr/bin/env python3
"""Generate a 360° frozen-time orbit video from one image with MiniMax-H3 + the
pablodawson Orbit LoRA, via a running ComfyUI server.

Usage:
    python pipeline/orbit.py photo.jpg [photo2.png ...] [--seed N] [--steps 28] [--out $ORBIT_OUTPUTS]
    python orbit.py photo.jpg --remove-bg [--bg-color ffffff]
    python orbit.py front.jpg --rear back.jpg --remove-bg     # real rear view pinned halfway round
    python orbit.py front.jpg --rear back.jpg --guide left.jpg@18 --guide right.jpg@54 --remove-bg

--remove-bg cuts the subject out of the input (BiRefNet) onto a plain background before
generating, then masks every output frame. Writes $ORBIT_OUTPUTS/<name>_s<seed>/ with
orbit.mp4, cutout.png, frames/, masks/ (255 = subject) and rgba/ for 3DGS training.

--rear pins a real photo of the back of the subject at the middle frame (MiniMaxH3AddGuide), so
the far side is constrained by a real view instead of invented. --guide PHOTO@FRAME pins further
real views (e.g. the sides at frames 18 and 54). Output dir gets a _rear / _<N>view suffix.

Start ComfyUI first:
    conda run -n orbit_cam python $COMFYUI_DIR/main.py --listen 127.0.0.1 --port 8188 \
        --extra-model-paths-config config/comfyui_extra_model_paths.yaml
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

# Paths come from config/paths.sh (source it first); the fallbacks match its defaults.
REPO = Path(__file__).resolve().parent.parent
COMFY_OUTPUT = Path(os.environ.get("COMFY_OUTPUT", REPO / "third_party" / "ComfyUI" / "output"))  # cluster jobs: per-job --output-directory
ORBIT_OUTPUTS = Path(os.environ.get("ORBIT_OUTPUTS", REPO / "data" / "outputs"))

# Verbatim from the LoRA model card. For prompt tests, ORBIT_PROMPT (env) replaces it, ORBIT_SUBJECT appends a sentence.
PROMPT = (
    "One frozen instant. Only the camera moves. In a continuous 360 orbit. Preserve every person "
    "and object in exactly the same world position, orientation, shape and pose throughout the shot. "
    "Airborne objects remain suspended at the captured height and angle: no wobbling, shaking, "
    "spinning, drifting, falling or continued action. Keep faces, hands, clothing, liquids and the "
    "background motionless while retaining their natural appearance. Camera parallax is the only "
    "source of apparent movement. No cuts, zoom, morphing or added objects."
)


def build_workflow(image_name, seed, steps, size, length, lora_strength, prefix, remove_bg=False, bg_color=0xFFFFFF,
                   guides=None):
    """ComfyUI API-format graph, adapted from the official video_minimax_h3_i2v template."""
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_flf2v_lora_v1.safetensors",
                         "strength_model": lora_strength}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax"}},
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        # Same image as first AND last keyframe -> full loop back to the start.
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": os.environ.get("ORBIT_PROMPT", PROMPT) + (" " + os.environ["ORBIT_SUBJECT"] if os.environ.get("ORBIT_SUBJECT") else ""),
                         "width": size, "height": size, "length": length,
                         "first_frame": ["5", 0], "last_frame": ["5", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        # Guidance-distilled model: BasicGuider, no CFG / negative prompt.
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["6", 0]}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler",
               "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["8", 0], "sampler": ["9", 0],
                          "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        # Audio off: video frames only.
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0], "filename_prefix": f"{prefix}/orbit", "format": "auto", "codec": "auto"}},
    }
    if remove_bg:
        wf.update({
            "20": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
            # Crop to the square canvas first so the cut-out and the keyframe line up.
            "21": {"class_type": "ImageScale",
                   "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "width": size, "height": size, "crop": "center"}},
            "22": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["20", 0], "image": ["21", 0]}},
            "23": {"class_type": "EmptyImage", "inputs": {"width": size, "height": size, "batch_size": 1, "color": bg_color}},
            "24": {"class_type": "ImageCompositeMasked",
                   "inputs": {"destination": ["23", 0], "source": ["21", 0], "x": 0, "y": 0,
                              "resize_source": False, "mask": ["22", 0]}},
            "25": {"class_type": "SaveImage", "inputs": {"images": ["24", 0], "filename_prefix": f"{prefix}/cutout"}},
            # Per-frame foreground masks of the generated orbit.
            "26": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["20", 0], "image": ["12", 0]}},
            "27": {"class_type": "MaskToImage", "inputs": {"mask": ["26", 0]}},
            "28": {"class_type": "SaveImage", "inputs": {"images": ["27", 0], "filename_prefix": f"{prefix}/mask"}},
            "29": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"{prefix}/frame"}},
        })
        wf["6"]["inputs"]["first_frame"] = ["24", 0]
        wf["6"]["inputs"]["last_frame"] = ["24", 0]
    for g, (gname, gframe) in enumerate(guides or []):
        nid = 30 + 10 * g  # node ids 30.., 40.., ...
        wf[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": gname}}
        img = [str(nid), 0]
        if remove_bg:  # same cut-out treatment as the front photo
            wf.update({
                str(nid + 1): {"class_type": "ImageScale",
                               "inputs": {"image": img, "upscale_method": "lanczos", "width": size, "height": size, "crop": "center"}},
                str(nid + 2): {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["20", 0], "image": [str(nid + 1), 0]}},
                str(nid + 3): {"class_type": "ImageCompositeMasked",
                               "inputs": {"destination": ["23", 0], "source": [str(nid + 1), 0], "x": 0, "y": 0,
                                          "resize_source": False, "mask": [str(nid + 2), 0]}},
                str(nid + 5): {"class_type": "SaveImage", "inputs": {"images": [str(nid + 3), 0], "filename_prefix": f"{prefix}/guide{g}"}},
            })
            img = [str(nid + 3), 0]
        # guide image pinned at its frame; chained so every guide is kept
        wf[str(nid + 4)] = {"class_type": "MiniMaxH3AddGuide",
                            "inputs": {"positive": wf["8"]["inputs"]["conditioning"], "vae": ["4", 0], "latent": ["6", 1],
                                       "image": img, "frame_idx": gframe}}
        wf["8"]["inputs"]["conditioning"] = [str(nid + 4), 0]
    return wf


def api(server, path, data=None, headers=None):
    req = urllib.request.Request(f"http://{server}{path}", data=data, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def upload_image(server, path):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    res = api(server, "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return res["name"]


def run(server, workflow):
    client_id = uuid.uuid4().hex
    res = api(server, "/prompt", json.dumps({"prompt": workflow, "client_id": client_id}).encode(),
              {"Content-Type": "application/json"})
    if res.get("node_errors"):
        sys.exit(f"Workflow rejected: {json.dumps(res['node_errors'], indent=2)}")
    prompt_id = res["prompt_id"]
    start = time.time()
    while True:
        hist = api(server, f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                sys.exit(f"Generation failed: {json.dumps(msgs, indent=2)[:3000]}")
            if status.get("completed"):
                return entry["outputs"], time.time() - start
        time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--seed", type=int, default=None, help="random if omitted")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--size", type=int, default=768, help="square resolution (trained at 768)")
    p.add_argument("--length", type=int, default=73, help="frames, 17k+5 (trained at 73)")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--remove-bg", action="store_true", help="cut out subject, export frames/masks/rgba")
    p.add_argument("--bg-color", default="ffffff", help="hex background colour for --remove-bg")
    p.add_argument("--rear", type=Path, help="real photo of the back of the subject, pinned at the middle frame")
    p.add_argument("--guide", action="append", default=[], metavar="PHOTO@FRAME",
                   help="pin another real view at a frame, e.g. side.jpg@18 (repeatable; 73 frames = 360°, ~5° per frame)")
    p.add_argument("--tag", help="run-dir suffix instead of _rear/_<N>view (e.g. _A2)")
    p.add_argument("--out", type=Path, default=ORBIT_OUTPUTS)
    p.add_argument("--server", default=os.environ.get("COMFY", "127.0.0.1:8188"))
    args = p.parse_args()

    try:
        api(args.server, "/system_stats")
    except OSError:
        sys.exit(f"ComfyUI isn't reachable at {args.server}. Start it first (see --help).")

    args.out.mkdir(parents=True, exist_ok=True)
    for img in args.images:
        if not img.is_file():
            print(f"skip: {img} not found")
            continue
        seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
        print(f"{img.name}: seed {seed}, {args.steps} steps, {args.size}px, {args.length} frames ...", flush=True)
        name = upload_image(args.server, img)
        guide_specs = [(args.rear, args.length // 2)] if args.rear else []
        for spec in args.guide:
            path, frame = spec.rsplit("@", 1)
            guide_specs.append((Path(path), int(frame)))
        guides = [(upload_image(args.server, gp), gf) for gp, gf in guide_specs]
        prefix = f"orbit/{img.stem}_{uuid.uuid4().hex[:8]}"
        wf = build_workflow(name, seed, args.steps, args.size, args.length, args.lora_strength, prefix,
                            args.remove_bg, int(args.bg_color, 16), guides)
        outputs, secs = run(args.server, wf)

        def files(node_id):
            node = outputs.get(node_id, {})
            items = node.get("images", []) + node.get("videos", [])
            return sorted(COMFY_OUTPUT / i.get("subfolder", "") / i["filename"] for i in items)

        video = files("14")[0]
        if not args.remove_bg:
            dst = args.out / f"{img.stem}_orbit_s{seed}{video.suffix}"
            shutil.copy2(video, dst)
            print(f"  -> {dst}  ({secs / 60:.1f} min)")
            continue

        tag = args.tag if args.tag is not None else f"_{len(guides) + 1}view" if len(guides) > 1 else ("_rear" if guides else "")
        run_dir = args.out / f"{img.stem}{tag}_s{seed}"
        for sub in ("frames", "masks", "rgba"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, run_dir / f"orbit{video.suffix}")
        shutil.copy2(files("25")[0], run_dir / "cutout.png")
        if args.remove_bg:
            for g in range(len(guides)):
                shutil.copy2(files(str(35 + 10 * g))[0], run_dir / f"guide{g}_cutout.png")
        frames, masks = files("29"), files("28")
        for i, (f, m) in enumerate(zip(frames, masks)):
            shutil.copy2(f, run_dir / "frames" / f"{i:05d}.png")
            mask = Image.open(m).convert("L")
            mask.save(run_dir / "masks" / f"{i:05d}.png")
            rgba = Image.open(f).convert("RGB")
            rgba.putalpha(mask)
            rgba.save(run_dir / "rgba" / f"{i:05d}.png")
        print(f"  -> {run_dir}/  ({len(frames)} frames + masks, {secs / 60:.1f} min)")


if __name__ == "__main__":
    main()
