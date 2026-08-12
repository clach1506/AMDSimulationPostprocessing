"""Optional second data source: a folder of binary segmentation masks (one
per timepoint, e.g. '..._SEG004.png'), previewed and exported with the same
step-colored contour-evolution overlay used for the simulation's own
density/levelset boundary (see `render.Exporter.draw_contour_evolution`).

Segmentation images live in pixel space, not the simulation's physical grid
(and per this project's other scripts, the two aren't a known affine match),
so this draws on plain pixel-index axes rather than trying to align onto the
simulation preview.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from .render import EVOLUTION_CMAP

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_TRAILING_NUM_RE = re.compile(r"(\d+)$")

IndexedFile = Tuple[int, Path]


class SegmentationData:
    """A folder of binary segmentation masks, one per timepoint, ordered by
    the trailing number in each filename."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.files: List[IndexedFile] = self._discover(self.folder)
        if not self.files:
            raise FileNotFoundError(f"No numbered segmentation images found in {self.folder}")

    @staticmethod
    def _discover(folder: Path) -> List[IndexedFile]:
        found = []
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
                continue
            m = _TRAILING_NUM_RE.search(p.stem)
            if not m:
                continue
            found.append((int(m.group(1)), p))
        found.sort(key=lambda t: t[0])
        return found

    def indices(self) -> List[int]:
        return [i for i, _ in self.files]

    @staticmethod
    def load_mask(path: Path) -> np.ndarray:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        # Image row 0 is the top of the picture; flip once so the plot's
        # y-axis reads bottom-to-top like the rest of the app instead of
        # relying on contour()'s own axis-inversion (which, combined with a
        # flip, tends to double-flip).
        return np.flipud(arr)


def draw_segmentation_evolution(ax, seg: SegmentationData, indices: Optional[Sequence[int]] = None,
                                 threshold: float = 0.5, discrete: bool = True, grid: bool = True) -> None:
    """The boundary line for every selected mask, overlaid on one plot and
    colored by index — same style as `Exporter.draw_contour_evolution`."""
    selected = [(i, p) for i, p in seg.files if indices is None or i in indices]
    if not selected:
        raise ValueError("No segmentation frames selected")

    n = len(selected)
    if discrete:
        cmap = matplotlib.colormaps[EVOLUTION_CMAP].resampled(n)
        norm = mcolors.BoundaryNorm(np.arange(n + 1) - 0.5, n)
        colors = [cmap(k) for k in range(n)]
    else:
        cmap = matplotlib.colormaps[EVOLUTION_CMAP]
        norm = mcolors.Normalize(vmin=selected[0][0], vmax=selected[-1][0])
        colors = [cmap(norm(i)) for i, _ in selected]

    ax.set_facecolor("white")
    ref_shape = None
    for (idx, path), color in zip(selected, colors):
        mask = seg.load_mask(path)
        if ref_shape is None:
            ref_shape = mask.shape
        elif mask.shape != ref_shape:
            continue  # skip frames whose pixel size doesn't match the rest
        ax.contour(mask, levels=[threshold], colors=[color], linewidths=2.0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if discrete:
        cbar = ax.figure.colorbar(sm, ax=ax, label="Frame", fraction=0.046, ticks=range(n))
        cbar.set_ticklabels([str(i) for i, _ in selected])
    else:
        ax.figure.colorbar(sm, ax=ax, label="Frame", fraction=0.046)

    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if grid:
        ax.grid(alpha=0.3)
    ax.set_title(f"segmentation boundary evolution — {n} frame(s) ({selected[0][0]}→{selected[-1][0]})")


def export_segmentation_evolution(seg: SegmentationData, output_path: Path, indices: Optional[Sequence[int]] = None,
                                   threshold: float = 0.5, discrete: bool = True, grid: bool = True) -> Path:
    fig = Figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111)
    draw_segmentation_evolution(ax, seg, indices=indices, threshold=threshold, discrete=discrete, grid=grid)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    return output_path
