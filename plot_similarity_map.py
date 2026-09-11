#!/usr/bin/env python3
"""Plot matched simulation time against clinical days with a slope fit."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from fit_slope import match_times, read_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matches", type=Path)
    parser.add_argument("dates", type=Path)
    parser.add_argument("slope", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with open(args.matches, newline="") as f:
        rows = list(csv.DictReader(f))
    dates = read_dates(args.dates)
    if len(rows) != len(dates):
        raise ValueError("matches and dates must have the same number of rows")
    if not rows:
        raise ValueError("Need at least one matched frame")
    slope = json.loads(args.slope.read_text())
    metadata_dt = json.loads(args.metadata.read_text())["numerics"]["dt"]
    dt = slope["dt"]
    if dt != metadata_dt:
        raise ValueError("Slope and metadata dt do not match")
    day0 = min(dates)
    days = [(item - day0).days for item in dates]
    times = match_times(rows, dt)
    neighborhood_counts = [int(row.get("neighborhood_width_count", 1) or 1) for row in rows]
    x0 = slope["origin_day"]
    y0 = slope["origin_sim_time"]
    x_line = [min(days), max(days)]
    y_line = [y0 + slope["slope"] * (x - x0) for x in x_line]

    scores = [float(row["dice"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 7))
    marker_radii = [6.0 + 3.0 * count for count in neighborhood_counts]
    points = ax.scatter(days, times, s=[radius ** 2 for radius in marker_radii], c=scores,
                        cmap="Greys", vmin=0.5, vmax=1.0,
                        marker="o", zorder=2)
    ax.plot(x_line, y_line, color="black", linewidth=1.5,
            label=f"Slope fit (R²={slope['r2']:.3f})", zorder=1)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Score")
    ax.set_xlabel("Days Since First Visit")
    ax.set_ylabel("Simulation Time")
    ax.set_title("Temporal Matching: MI_C_OG")
    ax.set_xticks(days, [row["segmentation"].replace("MI_C_IR OG_", "") for row in rows], rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_box_aspect(0.7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
