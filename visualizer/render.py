"""Drawing and export logic, kept separate from `SimulationData` (I/O) and
from the GUI (widgets/state).

`FieldRenderer` knows how to draw one field's array onto a given `Axes`.
`Exporter` composes those drawings into previews, GIFs, and multi-panel
figures, and writes them to disk. Neither class touches Tkinter, and neither
uses `matplotlib.pyplot` — figures are plain `matplotlib.figure.Figure`
objects, never registered with pyplot's global figure manager. Doing preview
redraws through pyplot (`plt.subplots`/`plt.figure`) is what previously left
stray extra windows and piled-up colorbars behind on every frame change.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import numpy as np

from .data import SimulationData

ZoomBox = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

# White-at-zero speed colormap (white -> yellow -> orange -> red), matching
# the project's other velocity plots (SPEED_CMAP in the older scripts)
# instead of a colormap whose zero value is already a saturated color.
_ylorrd = matplotlib.colormaps["YlOrRd"]
SPEED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "speed_white0", ["#ffffff", *[_ylorrd(t) for t in np.linspace(0, 1, 255)]]
)

# Common colormaps offered in the GUI dropdown. "speed" is the custom
# white-at-zero map used by default for velocity.
COMMON_CMAPS = ["gray_r", "gray", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r", "speed"]

SCALAR_CMAPS = {"density": "gray_r"}
# Default contour threshold per field — the value the interface/lesion
# boundary is drawn at.
CONTOUR_LEVELS = {"density": 0.5, "levelset": 0.0}
CONTOUR_COLOR = "black"
CONTOUR_LINEWIDTH = 1.0
DEFAULT_SCALAR_CMAP = "inferno"
DEFAULT_VELOCITY_CMAP = "speed"
DEFAULT_BANDS = 9
EVOLUTION_CMAP = "plasma"


def resolve_cmap(name: Optional[str]):
    """Colormap name -> something pcolormesh/contourf accepts. "speed" is our
    custom white-at-zero map; everything else (including None) passes
    through."""
    return SPEED_CMAP if name == "speed" else name


def _prepare_levelset(values: np.ndarray) -> np.ndarray:
    """Raw levelset files store 0 everywhere inside the lesion and NEGATIVE
    values outside — the outward distance, stored negated. The interesting
    signal lives outside (the interior is just a flat 0), so displaying the
    raw values directly would show the background as negative and clip to
    black/off-scale on most colormaps. Negate so the background reads as
    positive distance (0 inside, growing positive outward), then clip away
    any stray positive noise from the interior so it stays a clean flat 0
    instead of a jagged near-zero contour."""
    return np.clip(-values, 0.0, None)


class FieldRenderer:
    """Stateless drawing helpers: array in, artists on an existing Axes out."""

    @staticmethod
    def scalar(ax, x_edges, y_edges, values: np.ndarray, cmap, title: str = "", colorbar: bool = True):
        mesh = ax.pcolormesh(x_edges, y_edges, values.T, cmap=resolve_cmap(cmap), shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        return mesh

    @staticmethod
    def contour(ax, xs, ys, x_edges, y_edges, backdrop_values: np.ndarray, contour_values: np.ndarray, level: float,
                backdrop_cmap: Optional[str] = None, banded: bool = True, n_bands: int = DEFAULT_BANDS,
                show_contour: bool = True, title: str = "", colorbar: bool = True):
        """`backdrop_values` (a heatmap, using `backdrop_cmap`) with the
        interface/boundary contour line for `contour_values` drawn on top in a
        thin black line. `contour_values` is deliberately a separate array
        from `backdrop_values` — for a diffuse field like density,
        thresholding the density itself is only an approximation of the real
        interface, so the boundary is more accurately drawn from the
        levelset's exact zero-crossing when one is available (see
        `Exporter.default_contour_field`), while the backdrop still shows the
        density heatmap you actually asked for.

        `banded=True` fills the backdrop with `n_bands` discrete levels
        (`contourf`) computed on the *same* coordinate grid as the contour
        line itself, instead of a continuous `pcolormesh` — the two are
        guaranteed pixel-consistent this way, and it reads more clearly than
        a smooth gradient. Pass `backdrop_cmap=None` for a plain white
        background instead. Pass `show_contour=False` to hide the boundary
        line and show just the field heatmap."""
        mesh = None
        if backdrop_cmap is not None:
            if banded:
                mesh = ax.contourf(xs, ys, backdrop_values.T, levels=n_bands, cmap=resolve_cmap(backdrop_cmap))
            else:
                mesh = ax.pcolormesh(x_edges, y_edges, backdrop_values.T, cmap=resolve_cmap(backdrop_cmap), shading="auto")
            if colorbar:
                ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_facecolor("white")
            ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
            ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
        cs = None
        if show_contour:
            cs = ax.contour(xs, ys, contour_values.T, levels=[level], colors=CONTOUR_COLOR, linewidths=CONTOUR_LINEWIDTH)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        return mesh, cs

    @staticmethod
    def velocity(ax, xs, ys, x_edges, y_edges, vector: np.ndarray, cmap=DEFAULT_VELOCITY_CMAP,
                 quiver: bool = True, quiver_stride: Optional[int] = None, title: str = "", colorbar: bool = True):
        vx = vector[:, :, 0].T
        vy = vector[:, :, 1].T
        speed = np.sqrt(vx ** 2 + vy ** 2)
        mesh = ax.pcolormesh(x_edges, y_edges, speed, cmap=resolve_cmap(cmap), shading="auto")
        if quiver:
            # Dense by default (roughly one arrow every couple of cells) —
            # a sparse quiver hides the flow pattern near the interface,
            # which is usually exactly what you want to see.
            stride = quiver_stride or max(1, vx.shape[1] // 45)
            X, Y = np.meshgrid(xs, ys)
            sl = (slice(None, None, stride), slice(None, None, stride))
            ax.quiver(X[sl], Y[sl], vx[sl], vy[sl], color="black", scale=None)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label="Speed", fraction=0.046, pad=0.04)
        return mesh

    @staticmethod
    def draw_zoom_box(ax, zoom_box: ZoomBox, color: str = "red") -> None:
        """Outline only — does not touch axis limits, so it never expands the
        view beyond the field's own extent (drawing via `ax.plot` would
        otherwise autoscale to include the box, squeezing the real data into
        the middle of the figure if the box falls outside the domain)."""
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xmin, xmax, ymin, ymax = zoom_box
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color=color, linewidth=1.5, linestyle="--",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


class Exporter:
    """High-level renders: one frame, an animated series, a contour-evolution
    overlay, or a velocity small-multiples grid, backed by a
    `SimulationData` instance."""

    def __init__(self, data: SimulationData) -> None:
        self.data = data
        self.xs, self.ys, self.x_edges, self.y_edges = data.mesh_data()
        self._x_bounds = (float(self.x_edges[0]), float(self.x_edges[-1]))
        self._y_bounds = (float(self.y_edges[0]), float(self.y_edges[-1]))

    def domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._x_bounds, self._y_bounds

    def default_cmap(self, field_name: str) -> str:
        if field_name == "velocity":
            return DEFAULT_VELOCITY_CMAP
        return SCALAR_CMAPS.get(field_name, DEFAULT_SCALAR_CMAP)

    def default_level(self, field_name: str) -> float:
        return CONTOUR_LEVELS.get(field_name, 0.5)

    def is_contour_field(self, field_name: str) -> bool:
        return field_name in CONTOUR_LEVELS

    def default_contour_field(self, field_name: str) -> str:
        """Which field's zero/threshold crossing should draw the boundary
        line for `field_name`. Density is a diffuse field — thresholding it
        directly is only an approximation of the true interface — so prefer
        the levelset's exact zero-crossing when one is on disk."""
        if field_name == "density" and "levelset" in self.data.available_fields():
            return "levelset"
        return field_name

    def _load_for_contour(self, field_name: str, step: int) -> np.ndarray:
        values = self.data.load_field(field_name, step)
        return _prepare_levelset(values) if field_name == "levelset" else values

    def _resolve_contour_values(self, contour_field: str, step: int, fallback: np.ndarray) -> np.ndarray:
        c_step = self.data.nearest_step(contour_field, step)
        if c_step is None:
            return fallback
        return self._load_for_contour(contour_field, c_step)

    # -- single-frame drawing (shared by preview + zoom preview) ------------

    def draw_field(self, ax, field_name: str, step: int, cmap: Optional[str] = None, level: Optional[float] = None,
                    backdrop: bool = True, banded: bool = True, show_contour: bool = True,
                    contour_field: Optional[str] = None,
                    quiver: bool = True, quiver_stride: Optional[int] = None, colorbar: bool = True) -> None:
        if field_name == "velocity":
            values = self.data.load_field(field_name, step)
            FieldRenderer.velocity(ax, self.xs, self.ys, self.x_edges, self.y_edges, values,
                                    cmap=cmap or DEFAULT_VELOCITY_CMAP, quiver=quiver, quiver_stride=quiver_stride,
                                    title=f"velocity  t={step}", colorbar=colorbar)
        elif self.is_contour_field(field_name):
            values = self._load_for_contour(field_name, step)
            c_field = contour_field if contour_field is not None else self.default_contour_field(field_name)
            contour_values = values if c_field == field_name else self._resolve_contour_values(c_field, step, values)
            eff_level = level if level is not None else self.default_level(c_field)
            FieldRenderer.contour(ax, self.xs, self.ys, self.x_edges, self.y_edges,
                                   backdrop_values=values, contour_values=contour_values, level=eff_level,
                                   backdrop_cmap=(cmap or DEFAULT_SCALAR_CMAP) if backdrop else None, banded=banded,
                                   show_contour=show_contour, title=f"{field_name}  t={step}", colorbar=colorbar)
        else:
            values = self.data.load_field(field_name, step)
            FieldRenderer.scalar(ax, self.x_edges, self.y_edges, values,
                                  cmap=cmap or DEFAULT_SCALAR_CMAP,
                                  title=f"{field_name}  t={step}", colorbar=colorbar)

    # -- GIF (animated single field over time) ---------------------------------

    def export_gif(self, field_name: str, steps: Sequence[int], output_path: Path, fps: int = 10,
                    cmap: Optional[str] = None, level: Optional[float] = None, backdrop: bool = True,
                    show_contour: bool = True, contour_field: Optional[str] = None,
                    zoom_box: Optional[ZoomBox] = None) -> Path:
        if field_name == "velocity":
            raise ValueError("GIF export is only supported for scalar fields (density, levelset, strain*). "
                              "Use the velocity snapshot grid export instead.")
        if not steps:
            raise ValueError("No steps selected")
        c_field = contour_field if contour_field is not None else self.default_contour_field(field_name)
        eff_level = level if level is not None else self.default_level(c_field)
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        cmap_obj = resolve_cmap(cmap or DEFAULT_SCALAR_CMAP)
        if not backdrop:
            ax.set_facecolor("white")
        artists = []
        for step in steps:
            values = self._load_for_contour(field_name, step)
            contour_values = values if c_field == field_name else self._resolve_contour_values(c_field, step, values)
            frame_artists = []
            if backdrop:
                mesh = ax.pcolormesh(self.x_edges, self.y_edges, values.T, cmap=cmap_obj, shading="auto")
                frame_artists.append(mesh)
            if show_contour:
                cs = ax.contour(self.xs, self.ys, contour_values.T, levels=[eff_level],
                                 colors=CONTOUR_COLOR, linewidths=CONTOUR_LINEWIDTH)
                # matplotlib >=3.8 makes QuadContourSet itself a single Artist;
                # older versions expose the line segments via .collections.
                frame_artists.extend(getattr(cs, "collections", [cs]))
            artists.append(frame_artists)
        if zoom_box is not None:
            xmin, xmax, ymin, ymax = zoom_box
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(*self._x_bounds)
            ax.set_ylim(*self._y_bounds)
        ax.set_aspect("equal")
        ax.set_title(field_name)
        writer = animation.PillowWriter(fps=fps)
        ani = animation.ArtistAnimation(fig, artists, interval=1000 // fps, blit=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(output_path), writer=writer)
        return output_path

    # -- contour evolution (all selected steps' boundary, overlaid on one plot) -

    def draw_contour_evolution(self, ax, field_name: str, steps: Sequence[int],
                                level: Optional[float] = None, contour_field: Optional[str] = None,
                                discrete: bool = True, grid: bool = True,
                                zoom_box: Optional[ZoomBox] = None) -> None:
        """The boundary line at every selected step, overlaid on a single
        white-background plot and colored by step — the interface-evolution
        style used throughout this project's other scripts
        (segmentation_contour_evolution.py, velocity_snapshots_cli.py's
        `--mode contour`, bridge_zoom_study.py's `--export-contours`) instead
        of a separate heatmap panel per step. Draws directly onto `ax`, so
        it doubles as both the live preview and (via `export_contour_evolution`)
        the saved figure."""
        if not self.is_contour_field(field_name):
            raise ValueError(f"{field_name} has no boundary to trace — pick density or levelset")
        if not steps:
            raise ValueError("No steps selected")
        c_field = contour_field if contour_field is not None else self.default_contour_field(field_name)
        eff_level = level if level is not None else self.default_level(c_field)

        available = set(self.data.available_steps(c_field))
        selected = [s for s in steps if s in available] if c_field == field_name else list(steps)
        if not selected:
            raise ValueError(f"None of the selected steps have a {c_field} frame on disk")

        n = len(selected)
        if discrete:
            cmap = matplotlib.colormaps[EVOLUTION_CMAP].resampled(n)
            norm = mcolors.BoundaryNorm(np.arange(n + 1) - 0.5, n)
            colors = [cmap(i) for i in range(n)]
        else:
            cmap = matplotlib.colormaps[EVOLUTION_CMAP]
            norm = mcolors.Normalize(vmin=selected[0], vmax=selected[-1])
            colors = [cmap(norm(s)) for s in selected]

        ax.set_facecolor("white")
        plotted = False
        for step, color in zip(selected, colors):
            contour_values = self._resolve_contour_values(c_field, step, None) if c_field != field_name \
                else self._load_for_contour(c_field, step)
            if contour_values is None or not np.all(np.isfinite(contour_values)):
                continue
            ax.contour(self.xs, self.ys, contour_values.T, levels=[eff_level], colors=[color], linewidths=2.0)
            plotted = True
        if not plotted:
            raise ValueError("No finite contour frames available to plot")

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if discrete:
            cbar = ax.figure.colorbar(sm, ax=ax, label="Step", fraction=0.046, ticks=range(n))
            cbar.set_ticklabels([str(s) for s in selected])
        else:
            ax.figure.colorbar(sm, ax=ax, label="Step", fraction=0.046)

        if zoom_box is not None:
            xmin, xmax, ymin, ymax = zoom_box
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(*self._x_bounds)
            ax.set_ylim(*self._y_bounds)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if grid:
            ax.grid(alpha=0.3)
        ax.set_title(f"{field_name} boundary evolution — {n} step(s) ({selected[0]}→{selected[-1]})")

    def export_contour_evolution(self, field_name: str, steps: Sequence[int], output_path: Path,
                                  level: Optional[float] = None, contour_field: Optional[str] = None,
                                  discrete: bool = True, grid: bool = True,
                                  zoom_box: Optional[ZoomBox] = None) -> Path:
        fig = Figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111)
        self.draw_contour_evolution(ax, field_name, steps, level=level, contour_field=contour_field,
                                     discrete=discrete, grid=grid, zoom_box=zoom_box)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        return output_path

    # -- velocity small-multiples grid -----------------------------------------

    def export_velocity_grid(self, steps: Sequence[int], output_path: Path, cmap: Optional[str] = None,
                              quiver: bool = True, quiver_stride: Optional[int] = None,
                              zoom_box: Optional[ZoomBox] = None) -> Path:
        if not steps:
            raise ValueError("No steps selected")
        ncols = min(6, max(1, len(steps)))
        nrows = -(-len(steps) // ncols)
        fig = Figure(figsize=(5 * ncols, 4.5 * nrows))
        axes = fig.subplots(nrows, ncols, squeeze=False)
        flat = list(axes.flat)
        for ax in flat[len(steps):]:
            ax.set_visible(False)
        for ax, step in zip(flat, steps):
            values = self.data.load_field("velocity", step)
            FieldRenderer.velocity(ax, self.xs, self.ys, self.x_edges, self.y_edges, values,
                                    cmap=cmap or DEFAULT_VELOCITY_CMAP, quiver=quiver,
                                    quiver_stride=quiver_stride, title=f"t={step}")
            if zoom_box is not None:
                xmin, xmax, ymin, ymax = zoom_box
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        return output_path
