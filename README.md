# Postprocessing Visualizer

The expected format to be loaded is matching actual `AMD_tools` simulation 
export logic :
- `grid.dat` 
- one subfolder per field: density, velocity, levelset, strain (`field_xxx.bin` files)
- Optional nematic visualization is derived from scalar `density`, `strain_xx`,
  `strain_yy`, and `strain_xy` folders with common timesteps. It reconstructs
  `nematic_xx = (strain_xx - strain_yy) / 2` and `nematic_xy = strain_xy`;
  their principal strain difference is
  `sqrt((strain_xx - strain_yy)^2 + 4 strain_xy^2)`. The director is derived
  while plotting rather than exported separately. Director opacity represents
  this dimensionless small-shape strain; it is not a physical segment length.

## Run
Install dependencies:
```bash
python3 -m pip install -U matplotlib numpy Pillow
or 
pip install -e .
```

To launch the interface : 
```bash
python SimulationPostprocessing.py [sim_dir] 
python3. SimulationPostprocessing.py [sim_dir] #for python3
python3.xx SimulationPostprocessing.py [sim_dir] #for a specific version
```
You can also launch it without a prescribed simulation directory, there is everything needed to browse your files in the interface.
