#!/usr/bin/env python3
"""Render slope-predicted simulation contours over source images."""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image

from fit_slope import read_dates
from visualizer.data import SimulationData
from visualizer.seg_match import discover_images, render_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matches", type=Path)
    parser.add_argument("dates", type=Path)
    parser.add_argument("slope", type=Path)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with open(args.matches, newline="") as f:
        matches = list(csv.DictReader(f))
    dates = read_dates(args.dates)
    with open(args.slope) as f:
        slope = json.load(f)

    if len(matches) != len(dates):
        raise ValueError("matches and dates must have the same number of rows")
    if not matches:
        raise ValueError("Need at least one matched frame")
    sources = discover_images(args.sources)
    if len(sources) != len(matches):
        raise ValueError("sources and matches must have the same number of rows")

    data = SimulationData(args.sim_dir)
    available = data.available_steps("levelset")
    if not available:
        raise ValueError("No levelset simulation steps available")
    dt = slope.get("dt")
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Slope dt must be a finite positive number")
    day0 = min(dates)
    output_images = []
    args.output.mkdir(parents=True, exist_ok=True)

    for row, frame_date, source in zip(matches, dates, sources):
        days = (frame_date - day0).days
        predicted_time = slope["origin_sim_time"] + slope["slope"] * (days - slope["origin_day"])
        predicted_step = predicted_time / dt
        if predicted_step < available[0] or predicted_step > available[-1]:
            raise ValueError(f"Predicted step {predicted_step:.1f} is outside available simulation steps")
        step = min(available, key=lambda candidate: abs(candidate - predicted_step))
        image = render_overlay(source, data, "levelset", step,
                               title=f"{source.name} - predicted step {step} (day {days})")
        path = args.output / f"overlay_{source.stem}.png"
        image.save(path)
        output_images.append(image)
        print(f"{source.name}: day={days}, predicted_step={predicted_step:.1f}, using={step}")

    output_images[0].save(args.output / "slope_overlay.gif", save_all=True,
                          append_images=output_images[1:], duration=500, loop=0)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
