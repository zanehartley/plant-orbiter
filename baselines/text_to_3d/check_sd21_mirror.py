"""Check that the sd2-community SD 2.1-base mirror's text encoder is the original one (MVDream's prompt processor).

stabilityai no longer hosts Stable Diffusion 2.x. SD 2.x's text encoder is LAION's OpenCLIP ViT-H/14
(laion2b_s32b_b79k), frozen during SD training and read at its penultimate layer + final layer norm; diffusers ships
it as a 23-layer CLIPTextModel. This encodes the PlantDreamer prompts with both the mirror and LAION's own weights
(laion/CLIP-ViT-H-14-laion2B-s32B-b79K, primary source) and reports token-id agreement and the largest difference
between the hidden states. Run in the mvdream env (open_clip 2.7.0):
    python baselines/text_to_3d/check_sd21_mirror.py
"""
import open_clip
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPTextModel

from text_prompts import PROMPT, SPECIES

TOLERANCE = 1e-3  # fp32; identical weights agree to ~1e-5, a different encoder differs by O(1)


@torch.no_grad()
def openclip_penultimate(model, tokens):
    """SD 2.x's FrozenOpenCLIPEmbedder(layer="penultimate"): all blocks but the last, then ln_final."""
    x = (model.token_embedding(tokens) + model.positional_embedding).permute(1, 0, 2)
    for block in model.transformer.resblocks[:-1]:
        x = block(x, attn_mask=model.attn_mask)
    return model.ln_final(x.permute(1, 0, 2))


@torch.no_grad()
def main():
    prompts = [PROMPT.format(species=s) for s in SPECIES] + [""]  # "" is the unconditional embedding
    mirror = "sd2-community/stable-diffusion-2-1-base"
    tokenizer = AutoTokenizer.from_pretrained(mirror, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(mirror, subfolder="text_encoder").eval()
    ids = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    hf = encoder(ids).last_hidden_state

    laion = hf_hub_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin")
    model = open_clip.create_model("ViT-H-14", pretrained=laion).eval()
    tokens = open_clip.tokenize(prompts)
    reference = openclip_penultimate(model, tokens)

    same_tokens = bool((ids == tokens).all())
    diff = float((hf - reference).abs().max())
    print(f"token ids identical: {same_tokens}; max |hidden-state difference| = {diff:.2e} (tolerance {TOLERANCE})")
    if not same_tokens or diff > TOLERANCE:
        raise SystemExit("MISMATCH: the mirror's text encoder is not LAION's OpenCLIP ViT-H/14")
    print("OK: the mirror's text encoder matches LAION's original weights")


if __name__ == "__main__":
    main()
