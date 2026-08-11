"""Tkinter GUI: load a simulation folder, scrub through frames with a single
live preview, and export GIFs / snapshot grids for the selected field and
step range. Drag a rectangle directly on the preview to pick a zoom
subwindow, then export that cropped view separately.

Layout is a vertical split: parameters on the left, the preview and its
player (frame slider + from/until/stride) on the right.

Launch with `python3 Vizualiser.py [sim_dir]`.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from .data import SimulationData
from .render import COMMON_CMAPS, Exporter, FieldRenderer

ZoomBox = Tuple[float, float, float, float]

_BG = "#f4f5f7"
_PANEL_BG = "#ffffff"
_ACCENT = "#2f6feb"
_TEXT = "#1f2328"
_MUTED = "#57606a"


def _apply_style(root: tk.Tk) -> None:
    """A light, flat, modern-ish theme built on ttk's "clam" base — no
    external theming dependency, just consistent spacing/colors/fonts."""
    root.configure(bg=_BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    base_font = ("Helvetica Neue", 11)
    header_font = ("Helvetica Neue", 15, "bold")
    label_font = ("Helvetica Neue", 11, "bold")

    style.configure(".", background=_BG, foreground=_TEXT, font=base_font)
    style.configure("TFrame", background=_BG)
    style.configure("Panel.TFrame", background=_PANEL_BG)
    style.configure("TLabel", background=_BG, foreground=_TEXT)
    style.configure("Panel.TLabel", background=_PANEL_BG, foreground=_TEXT)
    style.configure("Muted.TLabel", background=_BG, foreground=_MUTED)
    style.configure("Header.TLabel", background=_BG, foreground=_TEXT, font=header_font)
    style.configure("TLabelframe", background=_BG, borderwidth=0)
    style.configure("TLabelframe.Label", background=_BG, foreground=_TEXT, font=label_font)
    style.configure("TCheckbutton", background=_BG, foreground=_TEXT)
    style.configure("TButton", padding=(10, 6))
    style.configure("Accent.TButton", padding=(10, 6), foreground="white", background=_ACCENT)
    style.map("Accent.TButton", background=[("active", _ACCENT), ("pressed", _ACCENT)])
    style.configure("TEntry", padding=4)
    style.configure("TCombobox", padding=4)
    style.configure("TScale", background=_BG)
    style.configure("Horizontal.TSeparator", background=_MUTED)


class StepRangeControl(ttk.Frame):
    """From / until / stride entries + a frame slider ("the player"), all
    driven off one field's available steps. Calls `on_change` whenever the
    selected frame changes so the owner can redraw a preview."""

    def __init__(self, master, on_change) -> None:
        super().__init__(master)
        self._on_change = on_change
        self.steps: List[int] = []
        self.selected_steps: List[int] = []

        range_row = ttk.Frame(self)
        range_row.pack(fill="x")
        ttk.Label(range_row, text="From").pack(side="left")
        self.from_var = tk.StringVar()
        ttk.Entry(range_row, textvariable=self.from_var, width=9).pack(side="left", padx=(4, 12))

        ttk.Label(range_row, text="Until").pack(side="left")
        self.until_var = tk.StringVar()
        ttk.Entry(range_row, textvariable=self.until_var, width=9).pack(side="left", padx=(4, 12))

        ttk.Label(range_row, text="Stride").pack(side="left")
        self.stride_var = tk.StringVar(value="1")
        ttk.Entry(range_row, textvariable=self.stride_var, width=7).pack(side="left", padx=(4, 12))

        ttk.Button(range_row, text="Apply", command=self._apply_range).pack(side="left")

        player_row = ttk.Frame(self)
        player_row.pack(fill="x", pady=(10, 0))
        self.frame_index_var = tk.DoubleVar(value=0)
        self.frame_slider = ttk.Scale(player_row, variable=self.frame_index_var, from_=0, to=0,
                                       orient="horizontal", command=self._on_slider)
        self.frame_slider.pack(side="left", fill="x", expand=True)
        self.frame_label = ttk.Label(player_row, text="step: -", width=14, anchor="e")
        self.frame_label.pack(side="left", padx=(10, 0))

    def reset_for_field(self, all_steps: List[int]) -> None:
        self.steps = all_steps
        if all_steps:
            self.from_var.set(str(all_steps[0]))
            self.until_var.set(str(all_steps[-1]))
        self._apply_range()

    def _parse_int(self, value: str) -> Optional[int]:
        value = value.strip()
        return int(value) if value else None

    def _apply_range(self) -> None:
        if not self.steps:
            self.selected_steps = []
        else:
            start = self._parse_int(self.from_var.get())
            end = self._parse_int(self.until_var.get())
            stride = max(1, self._parse_int(self.stride_var.get()) or 1)
            start = self.steps[0] if start is None else start
            end = self.steps[-1] if end is None else end
            self.selected_steps = [s for s in self.steps if start <= s <= end][::stride]
        n = len(self.selected_steps)
        self.frame_slider.config(from_=0, to=max(0, n - 1))
        self.frame_index_var.set(0)
        self._update_label()
        self._on_change()

    def _on_slider(self, _value: str) -> None:
        self._update_label()
        self._on_change()

    def _update_label(self) -> None:
        step = self.current_step()
        self.frame_label.config(text=f"step: {step}" if step is not None else "step: -")

    def current_step(self) -> Optional[int]:
        if not self.selected_steps:
            return None
        index = int(round(self.frame_index_var.get()))
        index = max(0, min(index, len(self.selected_steps) - 1))
        return self.selected_steps[index]


class ZoomControl(ttk.LabelFrame):
    """Manual x/y bounds for the zoom subwindow. Bounds are normally set by
    dragging a rectangle on the preview (see VisualizerApp's RectangleSelector)
    but can also be typed in directly."""

    def __init__(self, master, on_change) -> None:
        super().__init__(master, text="Zoom subwindow")
        self._on_change = on_change

        ttk.Label(self, text="Drag a rectangle on the preview to set it", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Crop preview to zoom box", variable=self.enabled_var,
                         command=self._on_change).grid(row=1, column=0, columnspan=2, sticky="w")

        # Off by default: an unset box defaults to the full domain, and
        # drawing that box unconditionally is what used to show a confusing
        # frame around the preview (and, since it may not match the real
        # domain bounds, could force the axes to autoscale and shrink the
        # actual field into the middle of the figure).
        self.outline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Show outline on full preview", variable=self.outline_var,
                         command=self._on_change).grid(row=2, column=0, columnspan=2, sticky="w")

        bounds = ttk.Frame(self)
        bounds.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.xmin_var = tk.StringVar(value="0.0")
        self.xmax_var = tk.StringVar(value="1.0")
        self.ymin_var = tk.StringVar(value="0.0")
        self.ymax_var = tk.StringVar(value="1.0")
        for label, var in (("xmin", self.xmin_var), ("xmax", self.xmax_var),
                            ("ymin", self.ymin_var), ("ymax", self.ymax_var)):
            cell = ttk.Frame(bounds)
            cell.pack(side="left", padx=(0, 8))
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=var, width=9)
            entry.pack()
            entry.bind("<Return>", lambda _e: self._on_change())
            entry.bind("<FocusOut>", lambda _e: self._on_change())

    def box(self) -> ZoomBox:
        return (float(self.xmin_var.get()), float(self.xmax_var.get()),
                float(self.ymin_var.get()), float(self.ymax_var.get()))

    def set_box(self, box: ZoomBox) -> None:
        self.xmin_var.set(f"{box[0]:.6g}")
        self.xmax_var.set(f"{box[1]:.6g}")
        self.ymin_var.set(f"{box[2]:.6g}")
        self.ymax_var.set(f"{box[3]:.6g}")


class VisualizerApp(tk.Tk):
    def __init__(self, sim_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.title("Simulation Visualizer")
        self.geometry("1360x840")
        _apply_style(self)

        self.data: Optional[SimulationData] = None
        self.exporter: Optional[Exporter] = None
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

        folder_row = ttk.Frame(self, padding=(16, 0, 16, 10))
        folder_row.pack(fill="x")
        self.sim_path_var = tk.StringVar()
        ttk.Entry(folder_row, textvariable=self.sim_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(folder_row, text="Browse", command=self._browse).pack(side="left", padx=(8, 0))
        ttk.Button(folder_row, text="Load", style="Accent.TButton", command=self.load_simulation).pack(
            side="left", padx=(8, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Vertical split: parameters on the left, preview + player on the
        # right (the player is the from/until/stride + frame slider, kept
        # right next to the thing it's driving).
        body = ttk.Frame(self, padding=(16, 12, 16, 12))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=330)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self._build_left_panel(left)
        self._build_right_panel(right)

        status_frame = ttk.Frame(self, padding=(16, 0, 16, 10))
        status_frame.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")

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

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.Frame(parent, style="Panel.TFrame")
        preview_frame.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(8, 6))
        self.figure.set_facecolor(_PANEL_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=1, pady=1)

        player_frame = ttk.LabelFrame(parent, text="Player", padding=10)
        player_frame.pack(fill="x", pady=(12, 0))
        self.range_control = StepRangeControl(player_frame, on_change=self._refresh_preview)
        self.range_control.pack(fill="x")

    # -- loading --------------------------------------------------------------

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

    def _refresh_preview(self) -> None:
        if self.exporter is None:
            return
        field_name = self.field_var.get()
        step = self.range_control.current_step()
        if not field_name or step is None:
            return
        try:
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            crop = self.zoom_control.box() if self.zoom_control.enabled_var.get() else None
            if self.evolution_var.get():
                steps = self.range_control.selected_steps
                self.exporter.draw_contour_evolution(self.ax, field_name, steps, zoom_box=crop)
                self.figure.tight_layout()
                self._attach_selector()
                self.canvas.draw_idle()
                self.status_var.set(f"Previewing {field_name} evolution  —  {len(steps)} step(s)")
                return
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
            self.figure.tight_layout()
            self._attach_selector()
            self.canvas.draw_idle()
            self.status_var.set(f"Previewing {field_name}  step {step}")
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


def main(sim_dir: Optional[Path] = None) -> None:
    app = VisualizerApp(sim_dir)
    app.mainloop()
