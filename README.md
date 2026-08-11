# Postprocessing Visualizer

A small Tkinter GUI for visualizing and exporting simulation field data
(density, levelset, velocity, ...) from a simulation output folder
(`grid.dat` + one subfolder per field, CSF1 `.bin` frames).

## Run

Install dependencies:

```bash
python3 -m pip install -U matplotlib numpy Pillow
```

Launch the GUI (needs a Python with both `tkinter` and `matplotlib` — on
this machine that's `python3.13`, not the default Homebrew `python3`):

```bash
python3.13 Vizualiser.py [sim_dir]
```

## Layout

- `Vizualiser.py` — launcher entry point.
- `visualizer/` — the actual implementation, split by concern:
  - `data.py` — `SimulationData`: discovers and loads a sim folder's fields.
  - `render.py` — `FieldRenderer` (draws one field on an `Axes`) and
    `Exporter` (previews, GIFs, contour-evolution overlays, velocity grids).
  - `app.py` — the Tkinter GUI (`VisualizerApp`).
- `geometry/` — CSF1 grid/field file format I/O (`read_grid`,
  `read_field_compact`), used by `visualizer/data.py`.

Everything else that used to live here (standalone one-off scripts, and
older CLI tools that imported a `src.core.viz.simulation_visualizer` module
not present in this repo) has been removed — this package is just the GUI.

## Features

- Load a simulation folder; pick a field, colormap, step range and stride.
- Live preview with a frame slider, always showing the current field.
- For density/levelset: a banded heatmap with the boundary contour on top —
  by default traced from the levelset's exact zero-crossing rather than an
  approximate density threshold, since density is a diffuse field.
- Drag a rectangle directly on the preview to define a zoom subwindow, then
  optionally crop the preview and any export to it.
- Exports (all zoom-aware): animated GIF, a contour-evolution overlay (every
  selected step's boundary on one plot, colored by step), and a velocity
  snapshot grid (quiver + speed heatmap, one panel per step).
