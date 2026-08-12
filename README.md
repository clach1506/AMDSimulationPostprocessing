# Postprocessing Visualizer

A small Tkinter GUI (plus a couple of standalone CLI scripts) for
visualizing simulation field data, and for matching/comparing it against
clinical segmentation images — density, levelset, velocity, ... from a
simulation output folder (`grid.dat` + one subfolder per field, CSF1 `.bin`
frames).

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

The two CLI scripts (`match_segmentation.py`, `lesion_area.py`) don't need
`tkinter`, so plain `python3` works for those.

## Layout

- `Vizualiser.py` — GUI launcher entry point.
- `match_segmentation.py` — CLI: find the simulation step(s) that best match
  a segmentation image/folder by Dice coefficient (coarse-to-fine search),
  with overlay/GIF/spatial-agreement-figure export.
- `lesion_area.py` — CLI: convert segmentation and simulation lesion areas
  into real mm², using a fixed pixel<->micrometre calibration, and compare
  them (with a figure).
- `visualizer/` — the shared implementation, split by concern:
  - `data.py` — `SimulationData`: discovers and loads a sim folder's fields.
  - `render.py` — `FieldRenderer` (draws one field on an `Axes`) and
    `Exporter` (previews, GIFs, contour-evolution overlays, velocity grids).
  - `segmentation.py` — `SegmentationData`: a folder of segmentation-mask
    images, plus its own contour-evolution preview/export.
  - `seg_match.py` — segmentation-vs-simulation matching: Dice search,
    physical-unit calibration, dates.csv handling, overlay/overlap-map
    rendering. Shared by `match_segmentation.py`, `lesion_area.py`, and the
    GUI's Region Matching tab.
  - `app.py` — the GUI shell (`VisualizerApp`): top-level window, the
    Simulation/Segmentation loaders, and the tab `Notebook`.
  - `viz_tab.py` / `match_tab.py` — the two tabs' actual widgets and
    behavior, mixed into `VisualizerApp`.
  - `gui_style.py` / `gui_widgets.py` — the shared ttk theme and small
    reusable composite widgets (step range + zoom controls).
- `geometry/` — CSF1 grid/field file format I/O (`read_grid`,
  `read_field_compact`), used by `visualizer/data.py`.

## Features

### Visualizer tab
- Load a simulation folder; pick a field, colormap, step range and stride.
- Live preview with a frame slider, always showing the current field.
- For density/levelset: a banded heatmap with the boundary contour on top —
  by default traced from the levelset's exact zero-crossing rather than an
  approximate density threshold, since density is a diffuse field.
- Drag a rectangle directly on the preview to define a zoom subwindow, then
  optionally crop the preview and any export to it.
- Optionally load a segmentation folder to preview its own contour evolution
  side by side with the simulation.
- Exports (all zoom-aware): animated GIF, a contour-evolution overlay (every
  selected step's boundary on one plot, colored by step), and a velocity
  snapshot grid (quiver + speed heatmap, one panel per step).

### Region Matching tab
- For every image in the loaded segmentation folder, finds the simulation
  step whose field best matches it by Dice coefficient.
- Source photos and a `dates.csv` (acquisition dates -> days since the first
  frame) are both optional extras, not required to run matching.
- Preview as either the matched boundary overlaid on the source photo, or a
  spatial-agreement map (segmentation-only / simulation-only / both, in real
  mm, with a scale bar).
- Exports: per-frame overlays, an overlay GIF, and a multi-panel
  spatial-agreement figure — all written to a single `visualization/` folder
  at the root of the series (next to `segmentations/`, `sources/`, etc.).
