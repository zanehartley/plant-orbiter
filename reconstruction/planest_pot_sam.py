"""Pot masks for PLANesT-3D views: OWLv2 ("a black plant pot") box -> SAM 2.1 box prompt (run in the "orbit_cam" env:
transformers >= 4.56; GPU if visible, else CPU). Called by make_planest_masks.py.

Box choice: highest-scoring detection with a pot-like shape (height < 1.6 x width); if none, the best detection of
any shape (in some pepper views that is the whole potted plant, which SAM then segments without the stand: also fine).
No detection (close-ups without the pot) -> no file.

    python planest_pot_sam.py <plant> <view> [<view> ...]
Writes data/converted/plants/<plant>/images/pot/<view>.png (255 = pot) and pot_boxes.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
from po_paths import DATASETS

ROOT = Path(__file__).resolve().parent
DST = DATASETS / "converted" / "plants"


def main():
    plant, views = sys.argv[1], sys.argv[2:]
    D = DST / plant
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval().to(dev)
    sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").eval().to(dev)
    out = D / "images" / "pot"
    out.mkdir(parents=True, exist_ok=True)
    bp = D / "pot_boxes.json"
    boxes = json.loads(bp.read_text()) if bp.exists() else {}
    for n in views:
        im = Image.open(D / "images" / "undistorted" / f"{n}.png").convert("RGB")
        inp = op(text=[["a black plant pot"]], images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            det = om(**inp)
        r = op.post_process_grounded_object_detection(det, threshold=0.1, target_sizes=torch.tensor([[max(im.size)] * 2]))[0]
        (out / f"{n}.png").unlink(missing_ok=True)
        if not len(r["scores"]):
            boxes[n] = None
            continue
        b, s = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
        potlike = (b[:, 3] - b[:, 1]) < 1.6 * (b[:, 2] - b[:, 0])
        i = int(np.argmax(np.where(potlike, s, -1))) if potlike.any() else int(np.argmax(s))
        box = b[i].tolist()
        si = sp(images=im, input_boxes=[[box]], return_tensors="pt").to(dev)
        with torch.no_grad():
            so = sm(**si, multimask_output=False)
        m = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][0, 0].numpy() > 0
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{n}.png")
        boxes[n] = {"box": box, "score": float(s[i]), "potlike": bool(potlike[i])}
        print(n, f"score {s[i]:.2f}", "pot" if potlike[i] else "whole plant", flush=True)
    bp.write_text(json.dumps(boxes, indent=1))


if __name__ == "__main__":
    main()
