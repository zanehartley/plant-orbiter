"""Text -> input image for the orbit pipeline, with Qwen-Image-2.1 (diffusers).

Synthetic-plant prompts are PlantDreamer's own guidance prompt (geometry/plant_handler.py, commit d73c40a):
"A photo of {species} plants". The orbit needs a whole, centred subject on plain white, seen from eye level
(README, "Inputs"), so a fixed framing sentence is appended. Every species and seed gets the same framing.

Framings (--framing):
  studio  long-lens product shot; images come out near-orthographic (orbit focal ~3000 px at 768)
  rgba    transparent background instead of white (to compare with the BiRefNet cutout)

Fruit counting: with --counts, the prompt is prompts.py count_prompt() for each
species and count, e.g. "A photo of tomato plants with 12 tomatoes", followed by the same framing sentence;
--count-words spells the count ("twelve", "one tomato"), since most text-to-image models count better from number
words (GeckoNum, arXiv 2406.14774, Table A3; our own pilot agreed). The paper uses --count-words.

Output: <out>/<species>_s<seed>.png (2048x2048), or <out>/<species>_n<count>_s<seed>.png with --counts, plus
prompts.json recording the exact text used.

    python pipeline/t2i_qwen.py --framing studio --species bean kale mint --seeds 42 1 2 \
        --out $ORBIT_OUTPUTS/synthetic_plants/images
    python pipeline/t2i_qwen.py --species tomato --counts $(seq 1 7) --count-words --seeds 42 1 2 \
        --out $ORBIT_OUTPUTS/fruit_counting/images
"""
import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

from prompts import count_prompt

BASE_PROMPT = "A photo of {species} plants"
FRAMINGS = {
    "studio": ("Single potted plant, whole plant and pot fully in frame and centred with margin around it, "
               "eye-level side view, isolated on a plain white background, studio lighting."),
    "rgba": "This is an RGBA image with transparency. The image has alpha channel and the background is transparent.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species", nargs="+", default=["bean", "kale", "mint"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    p.add_argument("--counts", nargs="+", type=int, help="fruit counts (fruit counting): one prompt per species and count")
    p.add_argument("--count-words", action="store_true", help="write the count as a word (twelve) instead of digits")
    p.add_argument("--model", default=os.path.join(os.environ.get("ORBIT_WEIGHTS", "data/weights"),
                                                   "hf", "Qwen-Image-2.1"))
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--framing", choices=FRAMINGS, default="studio", help="framing sentence appended to the prompt")
    p.add_argument("--framing-text", help="custom framing sentence (overrides --framing; for prompt experiments)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, dtype=torch.bfloat16)
    # Text encoder + transformer + VAE together overflow a 48 GB A6000 when decoding at 2048^2;
    # offloading keeps only the component in use on the GPU (Qwen model card's recommendation).
    pipe.enable_model_cpu_offload()

    framing = args.framing_text or FRAMINGS[args.framing]
    if args.counts:
        jobs = [(f"{species}_n{count}", count_prompt(species, count, args.count_words))
                for species in args.species for count in args.counts]
    else:
        jobs = [(species, BASE_PROMPT.format(species=species)) for species in args.species]
    prompts = {}
    for name, base in jobs:
        prompt = f"{base}. {framing}"
        prompts[name] = prompt
        for seed in args.seeds:
            path = args.out / f"{name}_s{seed}.png"
            if path.exists():
                continue
            image = pipe(prompt=prompt, width=args.size, height=args.size, num_inference_steps=args.steps,
                         generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            if args.framing != "rgba":
                image = image.convert("RGB")  # the model returns RGBA even for opaque prompts (alpha >= 251)
            image.save(path)
            print("saved", path, image.mode, image.size, flush=True)

    record = args.out / "prompts.json"  # merged, so a run with more counts or species keeps the earlier entries
    if record.exists():
        prompts = {**json.loads(record.read_text())["prompts"], **prompts}
    record.write_text(json.dumps(
        {"model": args.model, "steps": args.steps, "size": args.size, "framing": args.framing, "prompts": prompts}, indent=2))


if __name__ == "__main__":
    main()
