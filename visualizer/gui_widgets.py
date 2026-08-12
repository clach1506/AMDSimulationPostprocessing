"""Small reusable ttk composite widgets shared by the GUI's tabs — no
app-specific state, just self-contained controls that call back via
`on_change`."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Tuple

ZoomBox = Tuple[float, float, float, float]


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
