"""Filesystem access to a simulation output folder.

A simulation folder holds a `grid.dat` plus one subfolder per field
(density, levelset, velocity, strain, ...), each containing `<field>_<step>.bin`
CSF1 files. `SimulationData` discovers what's on disk and loads arrays from it;
it has no opinion about how those arrays get drawn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry.format import read_field_compact, read_grid

STEP_PATTERN = re.compile(r"_(\d+)\.bin$")

VECTOR_FIELDS = {"velocity"}


@dataclass
class SimulationData:
    sim_dir: Path
    field_root: Path = field(init=False)
    grid: Dict[str, float] = field(init=False)
    fields: Dict[str, List[int]] = field(init=False)
    metadata: Dict = field(init=False)

    def __post_init__(self) -> None:
        self.sim_dir = Path(self.sim_dir)
        self.field_root = self._find_field_root(self.sim_dir)
        self.grid = read_grid(self._find_grid_path(self.sim_dir))
        self.fields = self._discover_fields(self.field_root)
        self.metadata = self._read_metadata(self.sim_dir, self.field_root)

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def _find_grid_path(sim_dir: Path) -> Path:
        if (sim_dir / "grid.dat").exists():
            return sim_dir / "grid.dat"
        candidate = sim_dir / "simulation" / "grid.dat"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find grid.dat in {sim_dir} or {sim_dir / 'simulation'}")

    @staticmethod
    def _find_field_root(sim_dir: Path) -> Path:
        if any((sim_dir / name).is_dir() for name in ("density", "levelset", "velocity")):
            return sim_dir
        candidate = sim_dir / "simulation"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find a simulation field root in {sim_dir}")

    @staticmethod
    def _read_metadata(sim_dir: Path, field_root: Path) -> Dict:
        """Optional `metadata.json` (physics/numerics/... written alongside
        grid.dat by the solver) — used for e.g. `dt`, so exported GIFs can
        label frames with real simulation time instead of just the raw step
        index. Silently empty if not present or unreadable, everything that
        uses it is best-effort."""
        for candidate in (sim_dir / "metadata.json", field_root / "metadata.json"):
            if candidate.is_file():
                try:
                    return json.loads(candidate.read_text())
                except (json.JSONDecodeError, OSError):
                    return {}
        return {}

    @staticmethod
    def _discover_fields(root: Path) -> Dict[str, List[int]]:
        fields: Dict[str, List[int]] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            steps = []
            for path in sorted(child.glob("*.bin")):
                match = STEP_PATTERN.search(path.name)
                if match:
                    steps.append(int(match.group(1)))
            if steps:
                fields[child.name] = sorted(set(steps))
        return fields

    # -- queries -------------------------------------------------------------

    def available_fields(self) -> List[str]:
        return sorted(self.fields.keys())

    def available_steps(self, field_name: str) -> List[int]:
        return self.fields.get(field_name, [])

    def nearest_step(self, field_name: str, step: int) -> Optional[int]:
        steps = self.available_steps(field_name)
        if not steps:
            return None
        return min(steps, key=lambda s: abs(s - step))

    def field_steps(self, field_name: str, start: Optional[int], end: Optional[int], stride: int) -> List[int]:
        steps = self.available_steps(field_name)
        if not steps:
            return []
        if start is None:
            start = steps[0]
        if end is None:
            end = steps[-1]
        selected = [s for s in steps if start <= s <= end]
        return selected[:: max(1, stride)]

    def is_vector_field(self, field_name: str) -> bool:
        return field_name in VECTOR_FIELDS

    def dt(self) -> Optional[float]:
        return self.metadata.get("numerics", {}).get("dt")

    def simulation_time(self, step: int) -> Optional[float]:
        """Physical simulation time at `step` (step * dt), or None if the
        solver's metadata.json (or its `dt`) isn't available — callers
        should fall back to displaying the raw step index in that case."""
        dt = self.dt()
        return step * dt if dt is not None else None

    def time_label(self, step: int) -> str:
        t = self.simulation_time(step)
        return f"t = {t:.4g}" if t is not None else f"step {step}"

    # -- loading ---------------------------------------------------------------

    def get_field_path(self, field_name: str, step: int) -> Path:
        path = self.field_root / field_name / f"{field_name}_{step:04d}.bin"
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing field file: {path}")

    def load_field(self, field_name: str, step: int) -> np.ndarray:
        path = self.get_field_path(field_name, step)
        arr = read_field_compact(path)
        if self.is_vector_field(field_name):
            if arr.ndim != 3:
                raise ValueError(f"Expected vector field for {field_name}@{step}, got shape {arr.shape}")
            return arr
        if arr.ndim != 2:
            raise ValueError(f"Expected scalar field for {field_name}@{step}, got shape {arr.shape}")
        return arr

    # -- geometry -----------------------------------------------------------

    def mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (xs, ys, x_edges, y_edges): cell centers and pcolormesh edges."""
        nx = int(self.grid["nx"])
        ny = int(self.grid["ny"])
        dx = float(self.grid["dx"])
        dy = float(self.grid["dy"])
        xmin = float(self.grid["xmin"])
        xmax = float(self.grid["xmax"])
        ymin = float(self.grid["ymin"])
        ymax = float(self.grid["ymax"])
        xs = xmin + np.arange(nx) * dx
        ys = ymin + np.arange(ny) * dy
        x_edges = np.linspace(xmin - dx / 2, xmax + dx / 2, nx + 1)
        y_edges = np.linspace(ymin - dy / 2, ymax + dy / 2, ny + 1)
        return xs, ys, x_edges, y_edges

    def index_bounds(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> Tuple[slice, slice]:
        """World-space (xmin,xmax)/(ymin,ymax) -> index slices into the field arrays."""
        xs, ys, _, _ = self.mesh_data()
        xmin, xmax = sorted(x_bounds)
        ymin, ymax = sorted(y_bounds)
        ix0 = int(np.searchsorted(xs, xmin, side="left"))
        ix1 = int(np.searchsorted(xs, xmax, side="right"))
        iy0 = int(np.searchsorted(ys, ymin, side="left"))
        iy1 = int(np.searchsorted(ys, ymax, side="right"))
        ix0 = max(0, min(ix0, len(xs) - 1))
        ix1 = max(ix0 + 1, min(ix1, len(xs)))
        iy0 = max(0, min(iy0, len(ys) - 1))
        iy1 = max(iy0 + 1, min(iy1, len(ys)))
        return slice(ix0, ix1), slice(iy0, iy1)
