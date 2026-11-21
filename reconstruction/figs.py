"""Where QA sheets and check images go: $ORBIT_OUTPUTS/<section>/<date>_<topic>/.

out_dir("checks", "planest_masks") -> $ORBIT_OUTPUTS/checks/<today>_planest_masks/ (created).
"""
import datetime
from pathlib import Path

from po_paths import OUTPUTS

SECTIONS = ("results", "checks", "demos")


def out_dir(section, topic, date=None):
    assert section in SECTIONS, section
    d = OUTPUTS / section / f"{date or datetime.date.today().isoformat()}_{topic}"
    d.mkdir(parents=True, exist_ok=True)
    return d
