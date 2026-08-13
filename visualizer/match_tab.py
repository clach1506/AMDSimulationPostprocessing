"""The "Region Matching" tab: segmentation-vs-simulation matching by Dice
coefficient (see `visualizer.seg_match`), with a results table, a live
preview (overlay-on-photo or spatial-agreement map), and exports (overlays,
overlay GIF, overlap figure).

`MatchTabMixin` is mixed into `VisualizerApp` (see app.py) rather than used
standalone — it expects `self.data`, `self.segmentation` and `self.status_var`
to already exist on the composed instance.
"""

from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .gui_style import PANEL_BG
from .seg_match import (
    MM_PER_PIXEL,
    FrameMatch,
    draw_overlap_panel,
    draw_overlay,
    export_overlap_figure,
    export_overlay_gif,
    load_segmentation_mask,
    native_image_size,
    process_folder,
    save_overlay,
    simulation_inside_mask,
)


class MatchTabMixin:
    """Builds and drives the Region Matching tab. See module docstring for
    the attributes this expects the composed class to already provide."""

    # -- layout ---------------------------------------------------------------

    def _build_matching_tab(self, parent: ttk.Frame) -> None:
        """Segmentation-vs-simulation matching: for every image in the
        loaded segmentation folder, find the simulation step whose "inside
        the lesion" mask has the highest Dice overlap (coarse-to-fine search
        — see `visualizer.seg_match`), then preview/export that boundary
        drawn over a paired source photo."""
        controls = ttk.LabelFrame(parent, text="Segmentation vs simulation (Dice)", padding=10)
        controls.pack(fill="x", pady=(12, 0))

        ttk.Label(controls, text="Sources (optional)").grid(row=0, column=0, sticky="w")
        self.match_sources_path_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.match_sources_path_var).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=4)
        ttk.Button(controls, text="Browse", command=self._browse_match_sources).grid(row=0, column=4, padx=(4, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Dates.csv (optional)").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.match_dates_path_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.match_dates_path_var).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(controls, text="Browse", command=self._browse_match_dates).grid(
            row=1, column=4, padx=(4, 0), pady=(4, 0))

        ttk.Label(controls, text="Field").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.match_field_var = tk.StringVar(value="levelset")
        ttk.Combobox(controls, textvariable=self.match_field_var, state="readonly", width=10,
                     values=["levelset", "density"]).grid(row=2, column=1, sticky="w", padx=4, pady=(8, 0))

        ttk.Label(controls, text="Density threshold").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.match_density_threshold_var = tk.StringVar(value="0.5")
        ttk.Entry(controls, textvariable=self.match_density_threshold_var, width=6).grid(
            row=2, column=3, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="Subsample").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.match_subsample_var = tk.StringVar(value="10")
        ttk.Entry(controls, textvariable=self.match_subsample_var, width=6).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 0))

        ttk.Label(controls, text="Neighborhood").grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        self.match_neighborhood_var = tk.StringVar(value="8")
        ttk.Entry(controls, textvariable=self.match_neighborhood_var, width=6).grid(
            row=3, column=3, sticky="w", pady=(4, 0))

        ttk.Label(controls, text="View").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.match_view_var = tk.StringVar(value="Overlay on source")
        view_menu = ttk.Combobox(controls, textvariable=self.match_view_var, state="readonly", width=16,
                                  values=["Overlay on source", "Overlap map"])
        view_menu.grid(row=4, column=1, sticky="w", padx=4, pady=(8, 0))
        view_menu.bind("<<ComboboxSelected>>", self._on_match_row_select)

        btn_row = ttk.Frame(controls)
        btn_row.grid(row=5, column=0, columnspan=5, sticky="w", pady=(10, 0))
        ttk.Button(btn_row, text="Run matching", style="Accent.TButton",
                   command=self.run_segmentation_matching).pack(side="left")
        ttk.Button(btn_row, text="Export overlays", command=self.export_match_overlays).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Export overlay GIF", command=self.export_match_overlay_gif).pack(
            side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Export overlap figure", command=self.export_match_overlap_figure).pack(
            side="left", padx=(8, 0))

        results = ttk.Frame(parent)
        results.pack(fill="both", expand=True, pady=(12, 0))

        plot_frame = ttk.Frame(results, style="Panel.TFrame")
        plot_frame.pack(side="left", fill="both", expand=True)
        self.match_figure = Figure(figsize=(6, 6))
        self.match_figure.set_facecolor(PANEL_BG)
        self.match_canvas = FigureCanvasTkAgg(self.match_figure, master=plot_frame)
        self.match_canvas.get_tk_widget().pack(fill="both", expand=True, padx=1, pady=1)

        table_frame = ttk.Frame(results, width=380)
        table_frame.pack(side="left", fill="y", padx=(12, 0))
        table_frame.pack_propagate(False)

        columns = ("segmentation", "day", "source", "step", "dice")
        self.match_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)
        for col, label, width in (("segmentation", "Segmentation", 120), ("day", "Day", 45),
                                   ("source", "Source", 90), ("step", "Step", 55), ("dice", "Dice", 55)):
            self.match_tree.heading(col, text=label)
            self.match_tree.column(col, width=width, anchor="center")
        self.match_tree.pack(fill="both", expand=True)
        self.match_tree.bind("<<TreeviewSelect>>", self._on_match_row_select)

        self.match_summary_var = tk.StringVar(
            value="Load a simulation + segmentation folder above, then run matching. "
                  "Sources and dates.csv are both optional.")
        ttk.Label(table_frame, textvariable=self.match_summary_var, style="Muted.TLabel",
                  wraplength=360, justify="left").pack(fill="x", pady=(8, 0))

    # -- browsing ---------------------------------------------------------------

    def _browse_match_sources(self) -> None:
        path = filedialog.askdirectory(title="Select source photo folder (same order as segmentations)")
        if path:
            self.match_sources_path_var.set(path)

    def _browse_match_dates(self) -> None:
        path = filedialog.askopenfilename(title="Select dates.csv", filetypes=[("CSV", "*.csv"), ("All files", "*")])
        if path:
            self.match_dates_path_var.set(path)

    def _match_density_threshold(self) -> float:
        try:
            return float(self.match_density_threshold_var.get())
        except ValueError:
            return 0.5

    def _match_output_dir(self) -> Path:
        # Shared with the Visualizer tab's exports (see VisualizerTabMixin.
        # _viz_output_dir) so everything lands in one 'visualization' folder.
        return self._viz_output_dir()

    def _fail_match_export(self, message: str) -> None:
        self.status_var.set(message)
        messagebox.showerror("Export error", message)

    # -- matching -----------------------------------------------------------

    def run_segmentation_matching(self) -> None:
        if self.data is None:
            messagebox.showerror("Matching error", "Load a simulation folder first")
            return
        if self.segmentation is None:
            messagebox.showerror("Matching error", "Load a segmentation folder first")
            return
        # Sources and dates.csv are both optional — only used where relevant
        # (overlays/GIF need sources; dates.csv just adds day numbers).
        sources_text = self.match_sources_path_var.get().strip()
        sources_path = Path(sources_text).expanduser() if sources_text else None
        if sources_path is not None and not sources_path.is_dir():
            messagebox.showerror("Matching error", "Sources folder doesn't exist — clear the field to skip it")
            return
        dates_text = self.match_dates_path_var.get().strip()
        dates_path = Path(dates_text).expanduser() if dates_text else None
        if dates_path is not None and not dates_path.is_file():
            messagebox.showerror("Matching error", "dates.csv doesn't exist — clear the field to skip it")
            return
        try:
            field = self.match_field_var.get()
            density_threshold = self._match_density_threshold()
            subsample = max(1, int(self.match_subsample_var.get()))
            neighborhood = max(0, int(self.match_neighborhood_var.get()))

            def progress(i: int, n: int, match: FrameMatch) -> None:
                self.status_var.set(f"Matching {i}/{n}: {match.segmentation.name}  ->  "
                                     f"step {match.step}  (Dice {match.dice:.3f})")
                self.update_idletasks()

            # process_folder raises only on a segmentation/source count
            # mismatch — that's the one thing that makes the frame-by-frame
            # pairing untrustworthy; everything else (image size, naming) is
            # handled leniently.
            self.match_results = process_folder(
                self.segmentation.folder, self.data, sources_path, field=field,
                density_threshold=density_threshold, subsample=subsample, neighborhood=neighborhood,
                dates_csv=dates_path, progress=progress)
            self._populate_match_table()
            if self.match_results:
                self._preview_match(self.match_results[0])
                self.match_tree.selection_set(self.match_tree.get_children()[0])
            self.status_var.set(f"Matched {len(self.match_results)} frame(s)")
        except Exception as exc:
            self.status_var.set(f"Matching failed: {exc}")
            messagebox.showerror("Matching error", str(exc))

    def _populate_match_table(self) -> None:
        self.match_tree.delete(*self.match_tree.get_children())
        for i, m in enumerate(self.match_results):
            self.match_tree.insert("", "end", iid=str(i), values=(
                m.segmentation.name, m.day if m.day is not None else "-",
                m.source.name if m.source else "-", m.step, f"{m.dice:.4f}"))
        if self.match_results:
            mean_dice = sum(m.dice for m in self.match_results) / len(self.match_results)
            self.match_summary_var.set(f"{len(self.match_results)} frame(s) matched  —  mean Dice {mean_dice:.3f}\n"
                                        f"Click a row to preview it.")
        else:
            self.match_summary_var.set("No matches yet.")

    def _on_match_row_select(self, _event=None) -> None:
        selection = self.match_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.match_results):
            self._preview_match(self.match_results[index])

    def _preview_match(self, match: FrameMatch) -> None:
        self.match_figure.clf()
        ax = self.match_figure.add_subplot(111)
        field = self.match_field_var.get()
        density_threshold = self._match_density_threshold()
        label = f"{match.segmentation.name}" + (f"  (day {match.day})" if match.day is not None else "")

        if self.match_view_var.get() == "Overlap map":
            nx, ny = int(self.data.grid["nx"]), int(self.data.grid["ny"])
            width_px, height_px = native_image_size(match.segmentation)
            mm_per_cell = ((width_px * MM_PER_PIXEL) / nx, (height_px * MM_PER_PIXEL) / ny)
            seg_mask = load_segmentation_mask(match.segmentation, nx, ny)
            sim_mask = simulation_inside_mask(self.data, field, match.step, density_threshold)
            draw_overlap_panel(ax, seg_mask, sim_mask, mm_per_cell,
                                title=f"{label} · overlap {match.dice * 100:.0f}%")
        elif match.source is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No paired source image for this frame\n(switch View to \"Overlap map\")",
                    ha="center", va="center")
        else:
            draw_overlay(ax, match.source, self.data, field, match.step, density_threshold,
                         title=f"{label}  —  t={match.step}  (Dice={match.dice:.3f})")
        self.match_figure.tight_layout()
        self.match_canvas.draw_idle()

    # -- exports ----------------------------------------------------------------

    def export_match_overlays(self) -> None:
        if not self.match_results:
            self._fail_match_export("Run matching first")
            return
        if not any(m.source is not None for m in self.match_results):
            self._fail_match_export("No source images loaded — set a Sources folder and re-run matching")
            return
        try:
            out_dir = self._match_output_dir()
            field = self.match_field_var.get()
            density_threshold = self._match_density_threshold()
            n = 0
            for m in self.match_results:
                if m.source is None:
                    continue
                save_overlay(m.source, self.data, field, m.step, out_dir / f"overlay_{m.segmentation.stem}.png",
                             density_threshold)
                n += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "match_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["segmentation", "source", "date", "day", "best_step", "dice"])
                for m in self.match_results:
                    writer.writerow([m.segmentation.name, m.source.name if m.source else "",
                                      m.date.isoformat() if m.date else "", m.day if m.day is not None else "",
                                      m.step, f"{m.dice:.4f}"])
            self.status_var.set(f"Exported {n} overlay(s) + CSV to {out_dir}")
            messagebox.showinfo("Export done", f"Exported {n} overlay(s) + CSV to {out_dir}")
        except Exception as exc:
            self.status_var.set(f"Overlay export failed: {exc}")
            messagebox.showerror("Export error", str(exc))

    def export_match_overlay_gif(self) -> None:
        if not self.match_results:
            self._fail_match_export("Run matching first")
            return
        if not any(m.source is not None for m in self.match_results):
            self._fail_match_export("No source images loaded — set a Sources folder and re-run matching")
            return
        try:
            out_dir = self._match_output_dir()
            path = export_overlay_gif(self.match_results, self.data, self.match_field_var.get(),
                                       out_dir / "overlay.gif", self._match_density_threshold(), fps=2.0)
            self.status_var.set(f"Overlay GIF exported to {path}")
            messagebox.showinfo("Export done", f"Overlay GIF saved to {path}")
        except Exception as exc:
            self.status_var.set(f"Overlay GIF export failed: {exc}")
            messagebox.showerror("Export error", str(exc))

    def export_match_overlap_figure(self) -> None:
        if not self.match_results:
            self._fail_match_export("Run matching first")
            return
        try:
            out_dir = self._match_output_dir()
            suptitle = f"{self.segmentation.folder.parent.name} — spatial disagreement"
            path = export_overlap_figure(self.match_results, self.data, self.match_field_var.get(),
                                          out_dir / "overlap.png", density_threshold=self._match_density_threshold(),
                                          suptitle=suptitle)
            self.status_var.set(f"Overlap figure exported to {path}")
            messagebox.showinfo("Export done", f"Overlap figure saved to {path}")
        except Exception as exc:
            self.status_var.set(f"Overlap figure export failed: {exc}")
            messagebox.showerror("Export error", str(exc))
