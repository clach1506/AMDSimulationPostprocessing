"""Segmentation-vs-simulation matching: for a segmentation image (or a whole
folder of them), find the simulation step whose "inside the lesion" mask has
the highest Dice overlap, and visualize the match — either the matched
boundary drawn over a paired source photo, or a spatial-agreement map
(clinical-only / simulation-only / both) in real physical units.

Shared by the standalone `match_segmentation.py` CLI and the GUI's
Region Matching tab, so there's one implementation to keep in sync.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from .data import SimulationData

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_NUM_RE = re.compile(r"(\d+)")

# Fixed imaging calibration: this many pixels == this many micrometres, e.g.
# 12 px == 200 um. Used only for physical-unit displays (scale bars) — the
# Dice matching itself is resolution-independent and doesn't need it.
PIXELS_PER_REFERENCE = 12.0
MICROMETERS_PER_REFERENCE = 200.0
UM_PER_PIXEL = MICROMETERS_PER_REFERENCE / PIXELS_PER_REFERENCE  # ~16.667 um/px
MM_PER_PIXEL = UM_PER_PIXEL / 1000.0


def default_output_dir(segmentation_dir: Path) -> Path:
    """All exports land in one 'visualization' folder at the root of the
    series (the segmentation folder's parent — e.g. .../<patient>/segmentations
    -> .../<patient>/visualization), not scattered per-export-type."""
    return Path(segmentation_dir).parent / "visualization"


def native_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


# ── segmentation -> simulation grid ────────────────────────────────────────

def load_segmentation_mask(path: Path, nx: int, ny: int, threshold: float = 0.5) -> np.ndarray:
    """Load a binary segmentation image and resample it onto the
    simulation's (nx, ny) grid resolution.

    Returns a boolean array shaped (nx, ny), in the same orientation as
    `SimulationData.load_field` — mask[i, j] is the value at x-index i,
    y-index j (y increasing with j) — so it can be compared cell-for-cell
    against a simulation frame with no further conversion.
    """
    img = Image.open(path).convert("L")
    if img.size != (nx, ny):
        # BILINEAR (not NEAREST) so a high-resolution photo mask gets
        # properly area-averaged when downsampled onto the coarser
        # simulation grid, instead of aliasing.
        img = img.resize((nx, ny), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0  # (ny, nx), row 0 = top of the image
    arr = np.flipud(arr)  # row 0 = bottom, so y increases upward like the rest of this app
    return arr.T > threshold  # (nx, ny)


def simulation_inside_mask(data: SimulationData, field: str, step: int, density_threshold: float = 0.5) -> np.ndarray:
    """Boolean (nx, ny) 'inside the lesion' mask for one simulation frame."""
    values = data.load_field(field, step)
    if field == "levelset":
        # This project's levelset convention: 0 inside (flat), negative
        # outside (see visualizer/render.py's _prepare_levelset) — so
        # "not negative" is exactly the interior, no threshold needed.
        return values >= 0.0
    return values >= density_threshold


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


# ── search ──────────────────────────────────────────────────────────────────

def _score_step(seg_mask: np.ndarray, data: SimulationData, field: str, step: int, density_threshold: float,
                 cache: Dict[int, float]) -> float:
    if step not in cache:
        sim_mask = simulation_inside_mask(data, field, step, density_threshold)
        cache[step] = dice(seg_mask, sim_mask)
    return cache[step]


def exhaustive_search(seg_mask: np.ndarray, data: SimulationData, field: str,
                       density_threshold: float = 0.5) -> List[Tuple[int, float]]:
    """Every available step — the ground truth `coarse_to_fine_search` is
    checked against, and still useful directly for a small simulation or a
    one-off run where the extra cost doesn't matter."""
    cache: Dict[int, float] = {}
    steps = data.available_steps(field)
    return [(s, _score_step(seg_mask, data, field, s, density_threshold, cache)) for s in steps]


def coarse_to_fine_search(seg_mask: np.ndarray, data: SimulationData, field: str, density_threshold: float = 0.5,
                           subsample: int = 10, neighborhood: int = 8) -> Tuple[int, float, Dict[int, float]]:
    """Coarse sweep (every `subsample`-th step) to find an approximate best
    step, then a dense sweep over the `neighborhood`-index window around it
    (wide enough to fully cover the gap the coarse sweep skipped, plus
    slack). Returns (best_step, best_score, {step: score, ...} evaluated)."""
    steps = data.available_steps(field)
    if not steps:
        return -1, 0.0, {}
    cache: Dict[int, float] = {}

    coarse_idx = list(range(0, len(steps), max(1, subsample)))
    if coarse_idx[-1] != len(steps) - 1:
        coarse_idx.append(len(steps) - 1)
    for i in coarse_idx:
        _score_step(seg_mask, data, field, steps[i], density_threshold, cache)

    best_coarse_step = max(coarse_idx, key=lambda i: cache[steps[i]])
    window = max(neighborhood, subsample)
    lo, hi = max(0, best_coarse_step - window), min(len(steps) - 1, best_coarse_step + window)
    for i in range(lo, hi + 1):
        _score_step(seg_mask, data, field, steps[i], density_threshold, cache)

    best_step = max(cache, key=cache.get)
    return best_step, cache[best_step], cache


# ── file discovery ─────────────────────────────────────────────────────────

def _sort_key(path: Path):
    """Numeric-aware sort: 'frame_2' before 'frame_10'. Falls back to plain
    alphabetical for names without digits."""
    nums = [int(n) for n in _NUM_RE.findall(path.stem)]
    return (nums, path.stem)


def discover_images(folder: Path) -> List[Path]:
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS),
                  key=_sort_key)


# ── dates.csv (optional) ─────────────────────────────────────────────────

_DATE_FORMATS = ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d")


def _parse_date(text: str) -> Optional[date]:
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def load_dates_csv(path: Path) -> Dict[str, date]:
    """{frame_key: date} from a CSV with 'frame' and 'date' columns (any
    case), e.g. a row 'frame_00,2012/08/22'."""
    out: Dict[str, date] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = (row.get("frame") or row.get("Frame") or "").strip()
            date_str = (row.get("date") or row.get("Date") or "").strip()
            parsed = _parse_date(date_str) if date_str else None
            if frame and parsed is not None:
                out[frame] = parsed
    return out


def match_dates_to_files(files: Sequence[Path], dates_by_key: Dict[str, date]) -> List[Optional[date]]:
    """One date per file, matched by the dates.csv 'frame' key appearing in
    the filename (e.g. key 'frame_06' matches 'frame_06.png'). Falls back to
    pairing positionally by sorted order if no key matches any filename at
    all (dates.csv rows assumed to already be in the same order as `files`,
    the same convention used for pairing source photos)."""
    if not dates_by_key:
        return [None] * len(files)
    matched: List[Optional[date]] = []
    any_name_match = False
    for f in files:
        hit = None
        if f.stem in dates_by_key:
            hit = dates_by_key[f.stem]
        else:
            # Longest matching key wins, so e.g. 'frame_05bis' doesn't
            # accidentally grab the 'frame_05' entry meant for 'frame_05.png'.
            candidates = [key for key in dates_by_key if key in f.stem]
            if candidates:
                hit = dates_by_key[max(candidates, key=len)]
        if hit is not None:
            any_name_match = True
        matched.append(hit)
    if any_name_match:
        return matched
    ordered = [d for _, d in sorted(dates_by_key.items())]
    return [ordered[i] if i < len(ordered) else None for i in range(len(files))]


def day_offsets(dates: Sequence[Optional[date]]) -> List[Optional[int]]:
    """Days since the earliest date in the list (day 0 == first acquisition)."""
    valid = [d for d in dates if d is not None]
    if not valid:
        return [None] * len(dates)
    day0 = min(valid)
    return [(d - day0).days if d is not None else None for d in dates]


# ── overlay (matched boundary over a source photo) ──────────────────────────

def draw_overlay(ax, source_path: Path, data: SimulationData, field: str, step: int,
                  density_threshold: float = 0.5, title: Optional[str] = None) -> None:
    """The matched simulation step's boundary, drawn over the source photo.
    Both are plotted in the simulation's own physical coordinates: the
    photo via `imshow(..., extent=domain_bounds)`, the contour via the
    simulation's own (xs, ys) — so they register automatically without
    needing to resample the photo into grid space (only the segmentation
    mask needs that, for the pixel-for-pixel Dice comparison)."""
    img = np.asarray(Image.open(source_path).convert("L"))
    xs, ys, _, _ = data.mesh_data()
    x_bounds = (float(data.grid["xmin"]), float(data.grid["xmax"]))
    y_bounds = (float(data.grid["ymin"]), float(data.grid["ymax"]))
    values = data.load_field(field, step)
    level = 0.0 if field == "levelset" else density_threshold

    ax.clear()
    ax.imshow(img, cmap="gray", extent=(*x_bounds, *y_bounds), origin="upper")
    ax.contour(xs, ys, values.T, levels=[level], colors=["red"], linewidths=2.0)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or f"{source_path.name}  —  sim {field} t={step}", fontsize=9)


def render_overlay(source_path: Path, data: SimulationData, field: str, step: int,
                    density_threshold: float = 0.5, title: Optional[str] = None, figsize=(7, 7)) -> Image.Image:
    """Same as `draw_overlay`, rendered standalone to a PIL image (used for
    saving a single overlay and for building GIF frames)."""
    fig = Figure(figsize=figsize)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_overlay(ax, source_path, data, field, step, density_threshold, title)
    fig.tight_layout()
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(buf).convert("RGB")


def save_overlay(source_path: Path, data: SimulationData, field: str, step: int, output_path: Path,
                  density_threshold: float = 0.5, title: Optional[str] = None) -> Path:
    img = render_overlay(source_path, data, field, step, density_threshold, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def export_overlay_gif(frames: Sequence["FrameMatch"], data: SimulationData, field: str, output_path: Path,
                        density_threshold: float = 0.5, fps: float = 2.0) -> Path:
    """One animated GIF cycling through every frame's matched overlay, in
    the order given (normally the segmentation folder's sorted order)."""
    usable = [f for f in frames if f.source is not None]
    if not usable:
        raise ValueError("No frames with a paired source image to animate")
    images = [render_overlay(f.source, data, field, f.step, density_threshold,
                              title=f"{f.segmentation.name}  —  t={f.step}  (Dice={f.dice:.3f})")
              for f in usable]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:],
                    duration=int(1000 / max(0.1, fps)), loop=0)
    return output_path


# ── spatial-agreement map (clinical-only / simulation-only / both) ──────────

_BOTH_COLOR = (0.80, 0.83, 0.86)
_SEG_COLOR = (0.83, 0.18, 0.18)
_SIM_COLOR = (0.16, 0.45, 0.78)


def draw_overlap_panel(ax, seg_mask: np.ndarray, sim_mask: np.ndarray, mm_per_cell: Tuple[float, float],
                        title: str = "", seg_label: str = "Clinical", sim_label: str = "Simulation",
                        scale_bar_um: float = 1000.0) -> None:
    """A 3-color spatial-agreement map — `seg_label`-only, `sim_label`-only,
    and both — in real physical units (mm), with a scale bar. `seg_mask` and
    `sim_mask` must already be the same (nx, ny) shape (e.g. `seg_mask` from
    `load_segmentation_mask` on the simulation's own grid resolution)."""
    if seg_mask.shape != sim_mask.shape:
        raise ValueError(f"mask shape mismatch: {seg_mask.shape} vs {sim_mask.shape}")
    nx, ny = seg_mask.shape
    mm_x, mm_y = mm_per_cell

    rgb = np.ones((ny, nx, 3))
    rgb[(seg_mask & sim_mask).T] = _BOTH_COLOR
    rgb[(seg_mask & ~sim_mask).T] = _SEG_COLOR
    rgb[(~seg_mask & sim_mask).T] = _SIM_COLOR

    width_mm, height_mm = nx * mm_x, ny * mm_y
    ax.clear()
    ax.imshow(rgb, extent=(0, width_mm, 0, height_mm), origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#999999")
    ax.set_title(title, fontsize=10)

    bar_mm = scale_bar_um / 1000.0
    x0, y0 = width_mm * 0.06, height_mm * 0.08
    ax.plot([x0, x0 + bar_mm], [y0, y0], color="black", linewidth=3, solid_capstyle="butt")
    label = f"{scale_bar_um:g} µm" if scale_bar_um < 1000 else f"{scale_bar_um / 1000:g} mm"
    ax.text(x0, y0 + height_mm * 0.02, label, fontsize=8, va="bottom", ha="left")


def export_overlap_figure(matches: Sequence["FrameMatch"], data: SimulationData, field: str, output_path: Path,
                           seg_threshold: float = 0.5, density_threshold: float = 0.5,
                           suptitle: Optional[str] = None, seg_label: str = "Clinical",
                           sim_label: str = "Simulation", scale_bar_um: float = 1000.0) -> Path:
    """One figure, one panel per frame in `matches`, each a spatial-agreement
    map titled by day-since-first-frame (if the match has a `.day`) or
    filename otherwise, with a shared legend."""
    if not matches:
        raise ValueError("No frames to plot")
    nx, ny = int(data.grid["nx"]), int(data.grid["ny"])

    fig = Figure(figsize=(4 * len(matches), 4.6))
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(matches), squeeze=False)[0]
    for ax, m in zip(axes, matches):
        width_px, height_px = native_image_size(m.segmentation)
        mm_per_cell = ((width_px * MM_PER_PIXEL) / nx, (height_px * MM_PER_PIXEL) / ny)
        seg_mask = load_segmentation_mask(m.segmentation, nx, ny, seg_threshold)
        sim_mask = simulation_inside_mask(data, field, m.step, density_threshold)
        title = (f"Day {m.day} · overlap {m.dice * 100:.0f}%" if m.day is not None
                 else f"{m.segmentation.stem} · overlap {m.dice * 100:.0f}%")
        draw_overlap_panel(ax, seg_mask, sim_mask, mm_per_cell, title, seg_label, sim_label, scale_bar_um)

    handles = [
        Patch(facecolor=_SEG_COLOR, label=f"{seg_label} only (red)"),
        Patch(facecolor=_SIM_COLOR, label=f"{sim_label} only (blue)"),
        Patch(facecolor=_BOTH_COLOR, label="Both (gray)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95 if suptitle else 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.draw()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# ── whole-folder matching ──────────────────────────────────────────────────

@dataclass
class FrameMatch:
    segmentation: Path
    source: Optional[Path]
    step: int
    dice: float
    date: Optional[date] = None
    day: Optional[int] = None  # days since the series' first dated frame


ProgressCallback = Callable[[int, int, "FrameMatch"], None]


def process_folder(segmentation_dir: Path, sim: Union[Path, SimulationData], source_dir: Optional[Path] = None,
                    field: str = "levelset", density_threshold: float = 0.5,
                    subsample: int = 10, neighborhood: int = 8, dates_csv: Optional[Path] = None,
                    progress: Optional[ProgressCallback] = None) -> List[FrameMatch]:
    """One best-matching simulation step per segmentation image, sorted the
    same numeric-aware way as the segmentation folder. `source_dir` is
    optional — matching only needs the segmentation and the simulation.

    The only validation error raised here is a segmentation/source count
    mismatch when `source_dir` is given — segmentation and source images are
    paired strictly by sorted position, so a mismatched count means the
    pairing itself can't be trusted. Anything else (different image sizes,
    unusual filenames, ...) is handled leniently/best-effort.
    """
    data = sim if isinstance(sim, SimulationData) else SimulationData(sim)
    nx, ny = int(data.grid["nx"]), int(data.grid["ny"])

    seg_files = discover_images(segmentation_dir)
    if not seg_files:
        raise FileNotFoundError(f"No images found in {segmentation_dir}")

    source_files: List[Optional[Path]] = [None] * len(seg_files)
    if source_dir is not None:
        found = discover_images(source_dir)
        if len(found) != len(seg_files):
            raise ValueError(f"{len(seg_files)} segmentation image(s) but {len(found)} source image(s) — "
                              "they must match 1:1 in sorted order.")
        source_files = found

    dates_list: List[Optional[date]] = [None] * len(seg_files)
    if dates_csv is not None:
        dates_list = match_dates_to_files(seg_files, load_dates_csv(dates_csv))
    days_list = day_offsets(dates_list)

    results: List[FrameMatch] = []
    for i, (seg_path, source_path, frame_date, day) in enumerate(zip(seg_files, source_files, dates_list, days_list)):
        seg_mask = load_segmentation_mask(seg_path, nx, ny)
        step, score, _ = coarse_to_fine_search(seg_mask, data, field, density_threshold, subsample, neighborhood)
        match = FrameMatch(segmentation=seg_path, source=source_path, step=step, dice=score,
                            date=frame_date, day=day)
        results.append(match)
        if progress is not None:
            progress(i + 1, len(seg_files), match)
    return results
