"""Pot masks for the traits pot cut. planest_pot_sam.py falls back to a whole-plant box when no pot-shaped
detection wins, and on rose01/02/06/07 that made the "pot" mask the whole plant (a pot cut with it would delete the plant).
Here only pot-shaped OWLv2 boxes (height < 1.6 x width, threshold 0.02) are accepted, the box must lie in the lower half of
the photo. Detection runs on the lower 55% of the photo (pots sit low in every PLANesT input view; on bushy roses a full-photo
detection returns plant+pot) and boxes taller than 0.4 x photo height are rejected; SAM 2.1 box prompt on the full photo. Input views only. Writes images/pot_traits/<front>.png + pot_traits_boxes.json,
and a check sheet.
    python traits_pot_masks.py <sheet.png> <plant> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor
from po_paths import DATASETS

DST = DATASETS / "converted" / "plants"


def main():
    sheet, plants = sys.argv[1], sys.argv[2:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval().to(dev)
    sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sm = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").eval().to(dev)
    panels = []
    for plant in plants:
        D = DST / plant
        n = (D / "front.txt").read_text().strip()
        im = Image.open(D / "images" / "undistorted" / f"{n}.png").convert("RGB")
        W, H = im.size
        y0 = int(0.45 * H)
        crop = im.crop((0, y0, W, H))
        inp = op(text=[["a black plant pot"]], images=crop, return_tensors="pt").to(dev)
        with torch.no_grad():
            det = om(**inp)
        r = op.post_process_grounded_object_detection(det, threshold=0.02, target_sizes=torch.tensor([[max(crop.size)] * 2]))[0]
        b, s = r["boxes"].cpu().numpy() + np.array([0, y0, 0, y0]), r["scores"].cpu().numpy()
        hgt = b[:, 3] - b[:, 1]
        ok = (hgt < 1.6 * (b[:, 2] - b[:, 0])) & (hgt < 0.4 * H)
        out = D / "images" / "pot_traits"
        out.mkdir(parents=True, exist_ok=True)
        rec = {}
        a = np.asarray(im).astype(float) / 255
        if ok.any():
            i = int(np.argmax(np.where(ok, s, -1)))
            box = b[i].tolist()
            si = sp(images=im, input_boxes=[[box]], return_tensors="pt").to(dev)
            with torch.no_grad():
                so = sm(**si, multimask_output=False)
            m = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][0, 0].numpy() > 0
            Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{n}.png")
            rec = {"box": box, "score": float(s[i]), "frac": float(m.mean())}
            a[m] = 0.45 * a[m] + 0.55 * np.array([1.0, 0.15, 0.15])
        else:
            (out / f"{n}.png").unlink(missing_ok=True)
        (D / "pot_traits_boxes.json").write_text(json.dumps({n: rec or None}, indent=1))
        print(plant, n, rec or "NO POT", flush=True)
        panels.append((plant, a, rec))
    from PIL import ImageDraw
    c, tw = 5, 400
    th = int(tw * panels[0][1].shape[0] / panels[0][1].shape[1])
    canvas = Image.new("RGB", (c * tw, ((len(panels) + c - 1) // c) * (th + 18)), "white")
    for k, (p, a, rec) in enumerate(panels):
        x, y = (k % c) * tw, (k // c) * (th + 18)
        canvas.paste(Image.fromarray((a * 255).astype(np.uint8)).resize((tw, th)), (x, y + 18))
        ImageDraw.Draw(canvas).text((x + 4, y + 3), f"{p[3:]}  " + (f"pot {rec['frac']:.3f}" if rec else "NO POT"), fill="black")
    canvas.save(sheet)


if __name__ == "__main__":
    main()
