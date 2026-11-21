"""Find a ring of real views around a plant (nerfstudio/instant-ngp transforms.json) and pick
front / side / rear photos (the front is the input view for PlantDreamer plants).

Prints, for the chosen ring, every view with its azimuth relative to the front view, and the
picks at 0 / 90 / 180 / 270 deg. Used by batch_single_photo.py.
"""

import json
from pathlib import Path

import numpy as np


def load_views(transforms):
    t = json.loads(Path(transforms).read_text())
    names, c2w = [], []
    for f in t["frames"]:
        names.append(Path(f["file_path"]).name)
        c2w.append(np.array(f["transform_matrix"], float))
    return names, np.stack(c2w)


def ring_geometry(c2w):
    """Plant centre (closest point to all optical axes), up axis, elevation/azimuth of each camera."""
    centres, fwd = c2w[:, :3, 3], -c2w[:, :3, 2]  # OpenGL: camera looks down -z
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, f in zip(centres, fwd):
        P = np.eye(3) - np.outer(f, f)
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)
    up = c2w[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    rel = centres - target
    elev = np.degrees(np.arcsin(rel @ up / np.linalg.norm(rel, axis=1)))
    x = rel[0] - (rel[0] @ up) * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    azim = np.degrees(np.arctan2(rel @ y, rel @ x)) % 360
    return target, up, elev, azim


def lora_views(plant):
    """Views of this plant used to train PlantDreamer's LoRA (lists name plants 'bean_1' or 'Mint1')."""
    plant = Path(plant)
    out = set()
    for txt in (plant.parents[1] / "lora_imgs").glob("*_Original_Imgs.txt"):
        for line in txt.read_text().splitlines():
            if "," not in line:
                continue
            p, n = [x.strip() for x in line.split(",")]
            p = p.lower()
            if "_" not in p:
                p = p.rstrip("0123456789") + "_" + p[len(p.rstrip("0123456789")):]
            if p == plant.name:
                out.add(int(n))
    return out


def pick_ring_auto(names, c2w, exclude=(), max_elev=40.0):
    """Tight ring first; captures on a spiral (mint) need a wider elevation band."""
    for band, tol, min_n in ((4.0, 6.0, 16), (8.0, 8.0, 10), (12.0, 10.0, 8)):
        try:
            r = pick_ring(names, c2w, exclude, tol=tol, band=band, min_views=min_n, max_elev=max_elev)
            r["band"] = band
            return r
        except RuntimeError:
            continue
    raise RuntimeError("no usable ring")


def pick_ring(names, c2w, exclude=(), tol=6.0, band=4.0, min_elev=5.0, min_views=16, max_elev=90.0):
    """Choose the lowest well-populated elevation band and a front view with partners at 90/180/270."""
    _, _, elev, azim = ring_geometry(c2w)
    best = None
    for e0 in np.arange(np.floor(elev.min()), np.ceil(elev.max()), 1.0):
        idx = np.where(np.abs(elev - e0) <= band / 2)[0]
        if len(idx) < min_views or e0 < min_elev or e0 > max_elev:
            continue
        for f in idx:
            if int(names[f][:4]) in exclude:
                continue
            rel = (azim[idx] - azim[f]) % 360
            picks = [f]
            for target in (90, 180, 270):
                d = np.abs((rel - target + 180) % 360 - 180)
                if d.min() > tol:
                    break
                picks.append(idx[d.argmin()])
            else:
                score = (len(idx), -e0)
                if best is None or score > best[0]:
                    best = (score, e0, idx, picks)
        if best is not None:
            break  # lowest usable band wins (closest to a side-on view)
    if best is None:
        raise RuntimeError("no ring with views near 0/90/180/270")
    _, e0, idx, picks = best
    ring = sorted(idx, key=lambda i: (azim[i] - azim[picks[0]]) % 360)
    rel = {names[i]: float((azim[i] - azim[picks[0]]) % 360) for i in ring}
    return {"elevation": float(e0), "ring": [names[i] for i in ring], "rel_azimuth": rel,
            "front": names[picks[0]], "side90": names[picks[1]], "rear": names[picks[2]], "side270": names[picks[3]]}


if __name__ == "__main__":
    import sys
    plant = Path(sys.argv[1])
    names, c2w = load_views(plant / "transforms" / "original" / "transforms.json")
    excl = lora_views(plant)
    r = pick_ring_auto(names, c2w, exclude=excl)
    print(json.dumps({k: v for k, v in r.items() if k not in ("rel_azimuth", "ring")}), "ring size", len(r["ring"]))
    print("excluded (LoRA training views):", sorted(excl))
    print({k: round(r["rel_azimuth"][r[k]], 1) for k in ("front", "side90", "rear", "side270")})
