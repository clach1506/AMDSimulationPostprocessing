#!/usr/bin/env python3
"""Convert segmentation-image and simulation-domain lesion areas into real
physical units (mm^2), using a fixed pixel<->micrometre scale, and compare
them.

Standalone script for testing this in isolation — not wired into the GUI or
match_segmentation.py yet.

Physical scale (fixed, from the imaging system's calibration):
    12 pixels == 200 micrometres   (~16.667 um/pixel)

The simulation's dimensionless domain (grid.dat's [xmin,xmax] x [ymin,ymax])
is taken to span the *entire* segmentation image, at whatever native pixel
resolution that image has — the same assumption `load_segmentation_mask`
(in visualizer/seg_match.py) already makes when it resizes a segmentation
onto the simulation's (nx, ny) grid for the Dice comparison. So the physical
size of "the whole domain" is derived from each segmentation image's own
native pixel dimensions rather than a single global value, which matters if
different acquisitions have different pixel resolutions for the same
physical field of view.

Both areas are computed independently, each anchored to the same fixed
um/pixel scale:
  - segmentation area: count "inside" pixels at the image's *native*
    resolution (not resampled, so no interpolation error) x physical pixel
    area.
  - simulation area: the fraction of the (nx, ny) grid that's "inside" the
    lesion x the physical domain area (derived from the segmentation
    image's native pixel size, as above).

    python3 lesion_area.py <segmentation_image> <sim_dir> [--step N] [--field levelset]

Without --step, the best-matching simulation step is found first (Dice
coefficient, coarse-to-fine search — see visualizer/seg_match.py) and areas
are compared at that step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer.data import SimulationData
from visualizer.seg_match import (
    coarse_to_fine_search,
    default_output_dir,
    draw_overlap_panel,
    load_segmentation_mask,
    simulation_inside_mask,
)

# ── fixed imaging calibration ────────────────────────────────────────────
PIXELS_PER_REFERENCE = 12.0
MICROMETERS_PER_REFERENCE = 200.0
UM_PER_PIXEL = MICROMETERS_PER_REFERENCE / PIXELS_PER_REFERENCE  # ~16.667 um/px
MM_PER_PIXEL = UM_PER_PIXEL / 1000.0
PIXEL_AREA_MM2 = MM_PER_PIXEL ** 2


def segmentation_lesion_area_mm2(path: Path, threshold: float = 0.5) -> Tuple[float, int, int]:
    """(area_mm2, width_px, height_px) — computed at the image's *native*
    resolution, not resampled, so the pixel-count -> area conversion stays
    exact."""
    img = Image.open(path).convert("L")
    width, height = img.size
    arr = np.asarray(img, dtype=np.float64) / 255.0
    inside = int((arr > threshold).sum())
    return inside * PIXEL_AREA_MM2, width, height


def domain_area_mm2(width_px: int, height_px: int) -> float:
    """Physical area of 'the whole domain' — the segmentation image's full
    field of view, which the simulation's grid.dat domain is taken to span
    exactly (same assumption `load_segmentation_mask` makes when resizing
    a segmentation onto the simulation grid)."""
    return (width_px * MM_PER_PIXEL) * (height_px * MM_PER_PIXEL)


def simulation_lesion_area_mm2(data: SimulationData, field: str, step: int, width_px: int, height_px: int,
                                density_threshold: float = 0.5) -> float:
    """The simulation's 'inside the lesion' fraction of its domain, scaled
    by the physical domain area derived from a segmentation image's native
    pixel size (see `domain_area_mm2`) — resolution-independent on the
    simulation side, so the (nx, ny) grid resolution doesn't matter here."""
    mask = simulation_inside_mask(data, field, step, density_threshold)
    nx, ny = mask.shape
    fraction_inside = mask.sum() / (nx * ny)
    return float(fraction_inside * domain_area_mm2(width_px, height_px))


def compare(segmentation_path: Path, data: SimulationData, step: int, field: str = "levelset",
            seg_threshold: float = 0.5, density_threshold: float = 0.5) -> Dict[str, object]:
    seg_area, width_px, height_px = segmentation_lesion_area_mm2(segmentation_path, seg_threshold)
    sim_area = simulation_lesion_area_mm2(data, field, step, width_px, height_px, density_threshold)
    return {
        "segmentation_area_mm2": seg_area,
        "simulation_area_mm2": sim_area,
        "difference_mm2": sim_area - seg_area,
        "ratio_sim_over_seg": (sim_area / seg_area) if seg_area else float("nan"),
        "domain_area_mm2": domain_area_mm2(width_px, height_px),
        "image_size_px": (width_px, height_px),
    }


# ── figure: overlap map + area legend, all in real physical units (mm) ──────

def render_comparison_figure(segmentation_path: Path, data: SimulationData, step: int, result: Dict[str, object],
                              field: str = "levelset", seg_threshold: float = 0.5,
                              density_threshold: float = 0.5) -> Figure:
    """Segmentation-vs-simulation overlap (red = segmentation only, blue =
    simulation only, gray = both), axes and scale bar in mm (see
    `draw_overlap_panel`), with a legend spelling out what each color means
    and the actual mm^2 areas being compared."""
    nx, ny = int(data.grid["nx"]), int(data.grid["ny"])
    width_px, height_px = result["image_size_px"]
    mm_per_cell = ((width_px * MM_PER_PIXEL) / nx, (height_px * MM_PER_PIXEL) / ny)

    seg_mask = load_segmentation_mask(segmentation_path, nx, ny, seg_threshold)
    sim_mask = simulation_inside_mask(data, field, step, density_threshold)

    fig = Figure(figsize=(6, 6.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_overlap_panel(ax, seg_mask, sim_mask, mm_per_cell,
                        title=f"{field} t={step}  ·  domain {result['domain_area_mm2']:.1f} mm²")

    handles = [
        Patch(facecolor=(0.83, 0.18, 0.18), label="Segmentation only (red)"),
        Patch(facecolor=(0.16, 0.45, 0.78), label="Simulation only (blue)"),
        Patch(facecolor=(0.80, 0.83, 0.86), label="Both (gray)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)

    fig.suptitle(
        f"{segmentation_path.name}\n"
        f"Segmentation: {result['segmentation_area_mm2']:.3f} mm²   "
        f"Simulation: {result['simulation_area_mm2']:.3f} mm²   "
        f"Δ = {result['difference_mm2']:+.3f} mm²   ratio (sim/seg) = {result['ratio_sim_over_seg']:.3f}",
        fontsize=9.5)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segmentation", type=Path, help="Path to a binary segmentation image")
    parser.add_argument("sim_dir", type=Path, help="Simulation folder (grid.dat + field subfolders)")
    parser.add_argument("--step", type=int, default=None,
                         help="Simulation step to compare against (default: auto-find the best Dice match)")
    parser.add_argument("--field", choices=["levelset", "density"], default="levelset",
                         help="Which simulation field to use (default: levelset)")
    parser.add_argument("--seg-threshold", type=float, default=0.5, help="Segmentation binarization threshold")
    parser.add_argument("--density-threshold", type=float, default=0.5,
                         help="Inside/outside threshold when --field density")
    parser.add_argument("--subsample", type=int, default=10, help="Coarse search step (auto-match only)")
    parser.add_argument("--neighborhood", type=int, default=8, help="Fine search window (auto-match only)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Figure output path (default: <segmentation>/../../visualization/"
                              "lesion_area_<segmentation stem>.png)")
    parser.add_argument("--no-figure", action="store_true", help="Skip rendering the comparison figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = SimulationData(args.sim_dir)

    dice_score = None
    step = args.step
    if step is None:
        nx, ny = int(data.grid["nx"]), int(data.grid["ny"])
        seg_mask = load_segmentation_mask(args.segmentation, nx, ny, args.seg_threshold)
        step, dice_score, _ = coarse_to_fine_search(seg_mask, data, args.field, args.density_threshold,
                                                      args.subsample, args.neighborhood)

    result = compare(args.segmentation, data, step, args.field, args.seg_threshold, args.density_threshold)

    print(f"Calibration: {PIXELS_PER_REFERENCE:g} px = {MICROMETERS_PER_REFERENCE:g} um  "
          f"({UM_PER_PIXEL:.4f} um/px, {MM_PER_PIXEL:.6f} mm/px)")
    print(f"Segmentation image: {result['image_size_px'][0]} x {result['image_size_px'][1]} px")
    print(f"Domain area:        {result['domain_area_mm2']:.4f} mm^2")
    if dice_score is not None:
        print(f"Best-matching step: {step}  (Dice = {dice_score:.4f})")
    else:
        print(f"Simulation step:    {step}")
    print()
    print(f"Segmentation lesion area: {result['segmentation_area_mm2']:.4f} mm^2")
    print(f"Simulation lesion area:   {result['simulation_area_mm2']:.4f} mm^2")
    print(f"Difference (sim - seg):   {result['difference_mm2']:+.4f} mm^2")
    print(f"Ratio (sim / seg):        {result['ratio_sim_over_seg']:.4f}")

    if not args.no_figure:
        fig_path = args.output or (default_output_dir(args.segmentation.parent) /
                                    f"lesion_area_{args.segmentation.stem}.png")
        fig = render_comparison_figure(args.segmentation, data, step, result, args.field,
                                        args.seg_threshold, args.density_threshold)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure -> {fig_path}")


if __name__ == "__main__":
    main()
