"""The "Visualizer" tab: field/colormap/contour controls, the zoom
subwindow, the live preview (with an optional side-by-side segmentation
evolution panel), and the field-level exports (GIF, contour evolution,
velocity grid, segmentation evolution).

`VisualizerTabMixin` is mixed into `VisualizerApp` (see app.py) rather than
used standalone — it expects `self.data`, `self.exporter`, `self.segmentation`,
`self.sim_path_var`, `self.segmentation_path_var` and `self.status_var` to
already exist on the composed instance.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from .gui_style import PANEL_BG
from .gui_widgets import StepRangeControl, ZoomBox, ZoomControl
from .render import COMMON_CMAPS, FieldRenderer
from .segmentation import draw_segmentation_evolution, export_segmentation_evolution


class VisualizerTabMixin:
    """Builds and drives the Visualizer tab. See module docstring for the
    attributes this expects the composed class to already provide."""

    # -- layout ---------------------------------------------------------------

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        display = ttk.LabelFrame(parent, text="Field", padding=10)
        display.pack(fill="x")

        ttk.Label(display, text="Field").grid(row=0, column=0, sticky="w")
        self.field_var = tk.StringVar()
        self.field_menu = ttk.Combobox(display, textvariable=self.field_var, state="readonly", width=18)
        self.field_menu.grid(row=0, column=1, sticky="ew", pady=2)
        self.field_menu.bind("<<ComboboxSelected>>", lambda _e: self._on_field_change())

        ttk.Label(display, text="Colormap").grid(row=1, column=0, sticky="w")
        self.cmap_var = tk.StringVar(value=COMMON_CMAPS[0])
        self.cmap_menu = ttk.Combobox(display, textvariable=self.cmap_var, state="readonly",
                                       width=18, values=COMMON_CMAPS)
        self.cmap_menu.grid(row=1, column=1, sticky="ew", pady=2)
        self.cmap_menu.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        display.columnconfigure(1, weight=1)

        contour = ttk.LabelFrame(parent, text="Contour style", padding=10)
        contour.pack(fill="x", pady=(12, 0))

        self.contour_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(contour, text="show contour", variable=self.contour_var,
                         command=self._refresh_preview).grid(row=0, column=0, sticky="w")

        self.quiver_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(contour, text="quiver arrows (velocity)", variable=self.quiver_var,
                         command=self._refresh_preview).grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.banded_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(contour, text="banded (discrete levels)", variable=self.banded_var,
                         command=self._refresh_preview).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.evolution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(contour, text="evolution preview (all selected steps, not just this frame)",
                         variable=self.evolution_var, command=self._refresh_preview).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.zoom_control = ZoomControl(parent, on_change=self._refresh_preview)
        self.zoom_control.pack(fill="x", pady=(12, 0))

        export = ttk.LabelFrame(parent, text="Export  (uses the zoom box below if enabled)", padding=10)
        export.pack(fill="x", pady=(12, 0))
        for text, command in (
            ("Export GIF", self.export_gif),
            ("Export contour evolution", self.export_contour_evolution),
            ("Export velocity grid", self.export_velocity_grid),
        ):
            ttk.Button(export, text=text, command=command).pack(fill="x", pady=3)

        seg = ttk.LabelFrame(parent, text="Segmentation (optional)", padding=10)
        seg.pack(fill="x", pady=(12, 0))

        self.segmentation_visible_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(seg, text="Show side by side", variable=self.segmentation_visible_var,
                         command=self._refresh_preview).grid(row=0, column=0, sticky="w")

        ttk.Label(seg, text="threshold").grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.segmentation_threshold_var = tk.StringVar(value="0.5")
        threshold_entry = ttk.Entry(seg, textvariable=self.segmentation_threshold_var, width=6)
        threshold_entry.grid(row=0, column=2, sticky="w", padx=(4, 0))
        threshold_entry.bind("<Return>", lambda _e: self._refresh_preview())
        threshold_entry.bind("<FocusOut>", lambda _e: self._refresh_preview())

        ttk.Button(seg, text="Export segmentation evolution",
                   command=self.export_segmentation_evolution_action).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.Frame(parent, style="Panel.TFrame")
        preview_frame.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(8, 6))
        self.figure.set_facecolor(PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=1, pady=1)

        player_frame = ttk.LabelFrame(parent, text="Player", padding=10)
        player_frame.pack(fill="x", pady=(12, 0))
        self.range_control = StepRangeControl(player_frame, on_change=self._refresh_preview)
        self.range_control.pack(fill="x")

    # -- loading hook (called by VisualizerApp.load_simulation) -----------------

    def _on_field_change(self) -> None:
        if self.data is None:
            return
        field_name = self.field_var.get()
        self.cmap_var.set(self.exporter.default_cmap(field_name))
        self.range_control.reset_for_field(self.data.available_steps(field_name))

    # -- zoom / preview ---------------------------------------------------------

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        self.zoom_control.set_box((x0, x1, y0, y1))
        self.zoom_control.outline_var.set(True)
        self._refresh_preview()

    def _attach_selector(self) -> None:
        # Re-created on every redraw since redrawing clears the Axes the
        # previous selector was attached to. Kept as self._selector so it
        # isn't garbage-collected mid-use (RectangleSelector goes silently
        # unresponsive otherwise).
        self._selector = RectangleSelector(
            self.ax, self._on_rectangle_select, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords="pixels", interactive=False,
        )

    def _show_segmentation(self) -> bool:
        return self.segmentation is not None and self.segmentation_visible_var.get()

    def _refresh_preview(self) -> None:
        if self.exporter is None:
            return
        field_name = self.field_var.get()
        step = self.range_control.current_step()
        if not field_name or step is None:
            return
        try:
            self.figure.clf()
            show_seg = self._show_segmentation()
            if show_seg:
                self.ax, self.seg_ax = self.figure.subplots(1, 2)
            else:
                self.ax = self.figure.add_subplot(111)
                self.seg_ax = None

            crop = self.zoom_control.box() if self.zoom_control.enabled_var.get() else None
            if self.evolution_var.get():
                steps = self.range_control.selected_steps
                self.exporter.draw_contour_evolution(self.ax, field_name, steps, zoom_box=crop)
                status = f"Previewing {field_name} evolution  —  {len(steps)} step(s)"
            else:
                self.exporter.draw_field(self.ax, field_name, step, cmap=self.cmap_var.get(),
                                          backdrop=True, banded=self.banded_var.get(),
                                          show_contour=self.contour_var.get(), quiver=self.quiver_var.get())
                if crop is not None:
                    self.ax.set_xlim(crop[0], crop[1])
                    self.ax.set_ylim(crop[2], crop[3])
                elif self.zoom_control.outline_var.get():
                    try:
                        FieldRenderer.draw_zoom_box(self.ax, self.zoom_control.box())
                    except ValueError:
                        pass
                status = f"Previewing {field_name}  step {step}"

            if show_seg:
                try:
                    draw_segmentation_evolution(self.seg_ax, self.segmentation,
                                                 threshold=self._segmentation_threshold())
                except Exception as exc:
                    self.seg_ax.set_axis_off()
                    self.seg_ax.text(0.5, 0.5, f"segmentation preview failed:\n{exc}",
                                      ha="center", va="center", wrap=True, transform=self.seg_ax.transAxes)

            self.figure.tight_layout()
            self._attach_selector()
            self.canvas.draw_idle()
            self.status_var.set(status)
        except Exception as exc:
            self.status_var.set(f"Preview failed: {exc}")

    # -- exports ----------------------------------------------------------------

    def _selected_steps(self) -> List[int]:
        steps = self.range_control.selected_steps
        if not steps:
            raise ValueError("No steps selected — check the field and range")
        return steps

    def _output_path(self, name: str) -> Path:
        return Path(self.sim_path_var.get()) / "viz" / name

    def _active_zoom_box(self) -> Optional[ZoomBox]:
        return self.zoom_control.box() if self.zoom_control.enabled_var.get() else None

    def export_gif(self) -> None:
        try:
            assert self.exporter is not None
            steps = self._selected_steps()
            field_name = self.field_var.get()
            out = self._output_path(f"{field_name}_anim_{steps[0]}_{steps[-1]}_s{self.range_control.stride_var.get()}.gif")
            path = self.exporter.export_gif(field_name, steps, out, cmap=self.cmap_var.get(), backdrop=True,
                                             show_contour=self.contour_var.get(), zoom_box=self._active_zoom_box())
            self.status_var.set(f"GIF exported to {path}")
            messagebox.showinfo("Export done", f"GIF saved to {path}")
        except Exception as exc:
            self.status_var.set(f"GIF export failed: {exc}")
            messagebox.showerror("Export error", str(exc))

    def export_contour_evolution(self) -> None:
        try:
            assert self.exporter is not None
            steps = self._selected_steps()
            field_name = self.field_var.get()
            out = self._output_path(f"{field_name}_evolution_{steps[0]}_{steps[-1]}_s{self.range_control.stride_var.get()}.png")
            path = self.exporter.export_contour_evolution(field_name, steps, out, zoom_box=self._active_zoom_box())
            self.status_var.set(f"Contour evolution exported to {path}")
            messagebox.showinfo("Export done", f"Contour evolution saved to {path}")
        except Exception as exc:
            self.status_var.set(f"Contour evolution export failed: {exc}")
            messagebox.showerror("Export error", str(exc))

    def _segmentation_threshold(self) -> float:
        try:
            return float(self.segmentation_threshold_var.get())
        except ValueError:
            return 0.5

    def export_segmentation_evolution_action(self) -> None:
        if self.segmentation is None:
            messagebox.showerror("Export error", "Load a segmentation folder first")
            return
        try:
            indices = self.segmentation.indices()
            out = Path(self.segmentation_path_var.get()) / "viz" / \
                f"segmentation_evolution_{indices[0]}_{indices[-1]}.png"
            path = export_segmentation_evolution(self.segmentation, out, threshold=self._segmentation_threshold())
            self.status_var.set(f"Segmentation evolution exported to {path}")
            messagebox.showinfo("Export done", f"Segmentation evolution saved to {path}")
        except Exception as exc:
            self.status_var.set(f"Segmentation evolution export failed: {exc}")
            messagebox.showerror("Export error", str(exc))

    def export_velocity_grid(self) -> None:
        try:
            assert self.exporter is not None
            steps = self._selected_steps()
            out = self._output_path(f"velocity_snapshots_{steps[0]}_{steps[-1]}_s{self.range_control.stride_var.get()}.png")
            path = self.exporter.export_velocity_grid(steps, out, cmap=self.cmap_var.get(),
                                                        quiver=self.quiver_var.get(), zoom_box=self._active_zoom_box())
            self.status_var.set(f"Velocity grid exported to {path}")
            messagebox.showinfo("Export done", f"Velocity grid saved to {path}")
        except Exception as exc:
            self.status_var.set(f"Velocity export failed: {exc}")
            messagebox.showerror("Export error", str(exc))
