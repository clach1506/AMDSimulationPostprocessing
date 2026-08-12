"""Tkinter GUI shell: the top-level window, the Simulation/Segmentation
folder loaders shared by both tabs, and the Notebook that hosts them.

The two tabs' actual behavior lives in dedicated mixins so this file stays a
short "wiring" layer:
  - `viz_tab.VisualizerTabMixin` — field preview, zoom, and exports (GIF,
    contour evolution, velocity grid, segmentation evolution).
  - `match_tab.MatchTabMixin` — segmentation-vs-simulation Dice matching,
    with a results table, live preview, and overlay/GIF/overlap exports
    (see `visualizer.seg_match`).

Launch with `python3 Vizualiser.py [sim_dir]`.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from matplotlib.widgets import RectangleSelector

from .data import SimulationData
from .gui_style import apply_style
from .match_tab import MatchTabMixin
from .render import Exporter
from .segmentation import SegmentationData
from .viz_tab import VisualizerTabMixin


class VisualizerApp(VisualizerTabMixin, MatchTabMixin, tk.Tk):
    def __init__(self, sim_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.title("Simulation Visualizer")
        self.geometry("1360x840")
        apply_style(self)

        self.data: Optional[SimulationData] = None
        self.exporter: Optional[Exporter] = None
        self.segmentation: Optional[SegmentationData] = None
        self.match_results = []
        self._selector: Optional[RectangleSelector] = None

        self._build_widgets()
        if sim_dir is not None:
            self.sim_path_var.set(str(sim_dir))
            self.load_simulation()

    # -- layout ---------------------------------------------------------------

    def _build_widgets(self) -> None:
        header = ttk.Frame(self, padding=(16, 14, 16, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Simulation Visualizer", style="Header.TLabel").pack(anchor="w")

        folder_row = ttk.Frame(self, padding=(16, 0, 16, 6))
        folder_row.pack(fill="x")
        ttk.Label(folder_row, text="Simulation:", width=13, style="Muted.TLabel").pack(side="left")
        self.sim_path_var = tk.StringVar()
        ttk.Entry(folder_row, textvariable=self.sim_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(folder_row, text="Browse", command=self._browse).pack(side="left", padx=(8, 0))
        ttk.Button(folder_row, text="Load", style="Accent.TButton", command=self.load_simulation).pack(
            side="left", padx=(8, 0))

        seg_folder_row = ttk.Frame(self, padding=(16, 0, 16, 10))
        seg_folder_row.pack(fill="x")
        ttk.Label(seg_folder_row, text="Segmentation:", width=13, style="Muted.TLabel").pack(side="left")
        self.segmentation_path_var = tk.StringVar()
        ttk.Entry(seg_folder_row, textvariable=self.segmentation_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(seg_folder_row, text="Browse", command=self._browse_segmentation).pack(side="left", padx=(8, 0))
        ttk.Button(seg_folder_row, text="Load", command=self.load_segmentation).pack(side="left", padx=(8, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=(12, 0))

        viz_tab = ttk.Frame(notebook)
        match_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Visualizer")
        notebook.add(match_tab, text="Region Matching")

        # Vertical split: parameters on the left, preview + player on the
        # right (the player is the from/until/stride + frame slider, kept
        # right next to the thing it's driving).
        left = ttk.Frame(viz_tab, width=330)
        left.pack(side="left", fill="y", pady=(0, 12))
        left.pack_propagate(False)

        right = ttk.Frame(viz_tab)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=(0, 12))

        self._build_left_panel(left)
        self._build_right_panel(right)
        self._build_matching_tab(match_tab)

        status_frame = ttk.Frame(self, padding=(16, 6, 16, 10))
        status_frame.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")

    # -- loading (shared by both tabs) -------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Select simulation folder")
        if path:
            self.sim_path_var.set(path)

    def load_simulation(self) -> None:
        try:
            sim_path = Path(self.sim_path_var.get()).expanduser()
            if not sim_path.exists():
                raise FileNotFoundError(sim_path)
            self.data = SimulationData(sim_path)
            self.exporter = Exporter(self.data)
            fields = self.data.available_fields()
            self.field_menu["values"] = fields
            if fields:
                self.field_menu.set(fields[0])
            x_bounds, y_bounds = self.exporter.domain_bounds()
            self.zoom_control.set_box((*x_bounds, *y_bounds))
            self._on_field_change()
            self.status_var.set(f"Loaded {sim_path}  —  fields: {', '.join(fields)}")
        except Exception as exc:
            self.status_var.set(f"Load failed: {exc}")
            messagebox.showerror("Load error", str(exc))

    def _browse_segmentation(self) -> None:
        path = filedialog.askdirectory(title="Select segmentation folder")
        if path:
            self.segmentation_path_var.set(path)

    def load_segmentation(self) -> None:
        try:
            seg_path = Path(self.segmentation_path_var.get()).expanduser()
            if not seg_path.exists():
                raise FileNotFoundError(seg_path)
            self.segmentation = SegmentationData(seg_path)
            n = len(self.segmentation.files)
            self.segmentation_visible_var.set(True)
            self.status_var.set(f"Loaded segmentation folder {seg_path}  —  {n} frame(s)")
            self._refresh_preview()
        except Exception as exc:
            self.status_var.set(f"Segmentation load failed: {exc}")
            messagebox.showerror("Load error", str(exc))


def main(sim_dir: Optional[Path] = None) -> None:
    app = VisualizerApp(sim_dir)
    app.mainloop()
