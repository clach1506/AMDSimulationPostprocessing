#!/usr/bin/env python3
"""Launch the simulation visualizer GUI.

    python3 Vizualiser.py [sim_dir]

Pick a simulation folder (containing grid.dat + density/levelset/velocity/...
subfolders), scrub frames with a live preview, and export GIFs / snapshot
grids for density, levelset, and velocity — including a zoomable subwindow
with one-click auto-detection of the bridge/neck pinch-off region.

The actual implementation lives in the `visualizer` package (data.py /
render.py / bridge.py / app.py) so it stays reusable from scripts too, not
just this GUI entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer.app import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", nargs="?", default=None, help="Simulation root folder to load on startup")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.sim_dir) if args.sim_dir else None)
