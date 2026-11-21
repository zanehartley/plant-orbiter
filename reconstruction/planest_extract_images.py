"""PLANesT-3D images at 1/4 resolution, read straight from the authors' Dropbox archive (dropbox_full.zip, which nests
the 47 GB image zip uncompressed) without unpacking it.
Photos 6240x4160 -> 1560x1040 (JPEG q95) -> images_q4/<Species>/<FOLDER>/; camera XMLs, READMEs and scripts as they
are; plant point clouds extracted to pointclouds/. Raw scene point clouds (20 GB) are skipped.
Writes extract_q4.json (counts). Run by scripts/fetch_datasets.sh; the archive is not deleted.

    python planest_extract_images.py        (works in $ORBIT_DATASETS/PLANesT-3D)"""
import io, json, os, zipfile
from multiprocessing import Pool
from PIL import Image

from po_paths import DATASETS

D = str(DATASETS / "PLANesT-3D")
Z = os.path.join(D, "dropbox_full.zip")
IMG = "PLANesT-3D_Images.zip"


class SubFile(io.RawIOBase):
    """Byte range of the outer zip holding the STORED inner zip: real seeks (zipfile's ZipExtFile.seek re-reads)."""
    def __init__(self, path, start, size):
        self.f, self.start, self.size, self.pos = open(path, "rb"), start, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        self.f.seek(self.start + self.pos); data = self.f.read(n); b[:len(data)] = data; self.pos += len(data)
        return len(data)


def open_inner():
    z = zipfile.ZipFile(Z); info = z.getinfo(IMG)
    assert info.compress_type == zipfile.ZIP_STORED
    with open(Z, "rb") as f:  # data starts after the local file header (30 bytes + name + extra)
        f.seek(info.header_offset); h = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
    return zipfile.ZipFile(io.BufferedReader(SubFile(Z, start, info.file_size), 1 << 20))


def names():
    return open_inner().namelist()


def work(chunk):
    inner = open_inner(); done = 0
    for n in chunk:
        out = os.path.join(D, "images_q4", n.split("/", 1)[1])
        if os.path.exists(out):
            done += 1; continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        im = Image.open(io.BytesIO(inner.read(n)))
        exif = im.info.get("exif", b"")
        im.convert("RGB").reduce(4).save(out + ".tmp", "JPEG", quality=95, exif=exif)
        os.replace(out + ".tmp", out); done += 1
    return done


if __name__ == "__main__":
    all_n = names()
    imgs = [n for n in all_n if n.lower().endswith((".jpg", ".jpeg"))]
    other = [n for n in all_n if not n.endswith("/") and n not in imgs]
    z = zipfile.ZipFile(Z); inner = open_inner()
    for n in other:
        out = os.path.join(D, "images_q4", n.split("/", 1)[1]); os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(inner.read(n))
    for n in ("README.txt", "read_visualize_ply_files.py", "read_visualize_raw_point_clouds.py"):
        open(os.path.join(D, n), "wb").write(z.read(n))
    if not os.path.isdir(os.path.join(D, "pointclouds", "PLANesT_3D_PlantPointClouds")):
        with zipfile.ZipFile(z.open("PLANesT_3D_PlantPointClouds.zip")) as pc:
            pc.extractall(os.path.join(D, "pointclouds"))
    chunks = [imgs[i::6] for i in range(6)]
    with Pool(6) as p:
        n_done = sum(p.map(work, chunks))
    on_disk = sum(len([f for f in fs if f.lower().endswith((".jpg", ".jpeg"))]) for _, _, fs in os.walk(os.path.join(D, "images_q4")))
    json.dump({"images_in_zip": len(imgs), "images_on_disk": on_disk, "other_files": len(other)},
              open(os.path.join(D, "extract_q4.json"), "w"), indent=1)
    print("EXTRACT_Q4", len(imgs), on_disk, flush=True)
