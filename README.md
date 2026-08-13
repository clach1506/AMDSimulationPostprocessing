# Postprocessing Visualizer

A small user interface using Tkinter for visualizing simulation field data, and for matching/comparing it against
medical images 

The interface takes as a minimal a simulation output folder, directly compatible with actual AMD_tools simulation 
export logic :
- `grid.dat` 
- one subfolder per field: density, velocity, levelset, strain (`field_xxx.bin` files)

## Run
Install dependencies:
```bash
python3 -m pip install -U matplotlib numpy Pillow
or 
pip install -e .
```

To launch the interface : 
```bash
python3.13 SimulationPostprocessing.py [sim_dir]
```

## Features

### Visualizer tab
- Load a simulation folder; pick a field, colormap, step range and stride.
- Live preview with a frame slider
- TO ZOOM on a specific area (bridges per example) : Draw a rectangle directly on the preview to define a zoom subwindow, then
  optionally crop the preview
- Optionally load a segmentation folder to preview its own contour evolution
  side by side with the simulation
- Exports (all zoom-aware): animated GIF, a contour-evolution overlay (every
  selected step's boundary on one plot, colored by step), and a velocity
  snapshot grid (quiver + speed heatmap, one panel per step)

### Region Matching tab
- For every image in the loaded segmentation folder, finds the simulation
  step whose field best matches it by using Dice coefficient
- Sources IR images and a `dates.csv` (acquisition dates -> days since the first
  frame) are both optional extras, not required to run matching
- Preview as either the matched boundary overlaid on the source photo, or a
  spatial-agreement map 
- Exports: per-frame or gif overlays, and a multi-panel overlap figure all written to a single `visualization/` folder
  at the root of the series (next to `segmentations/`, `sources/`, etc.)

  ## Layout

- `SimulationPostprocessing.py` — GUI launcher entry point.
- `match_segmentation.py` : find the simulation step(s) that best match
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
- `geometry/` — grid/field file format I/O (`read_grid`,
  `read_field_compact`), used by `visualizer/data.py`.
