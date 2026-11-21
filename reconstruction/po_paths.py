"""Where the reconstruction scripts read and write data. config/paths.sh sets these; the defaults below match it."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_DATA = Path(os.environ.get("ORBIT_DATA", REPO / "data"))
DATASETS = Path(os.environ.get("ORBIT_DATASETS", _DATA / "datasets"))  # downloaded and converted datasets, derived masks
INPUTS = Path(os.environ.get("ORBIT_INPUTS", _DATA / "inputs"))  # prepared input photos
OUTPUTS = Path(os.environ.get("ORBIT_OUTPUTS", _DATA / "outputs"))  # runs and results
PIPELINE = REPO / "pipeline"
# PlantDreamer hybrid masks (shipped in the repo; masks/README.md)
HYBRID_MASKS = Path(os.environ.get("ORBIT_HYBRID_MASKS", REPO / "masks" / "plantdreamer_hybrid"))  # orbit.py, known_poses.py, make_splat_kp.sh
