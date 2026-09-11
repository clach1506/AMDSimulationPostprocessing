#!/usr/bin/env python3
"""Fit simulation time against clinical days from match_segmentation output."""

import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path

from visualizer.seg_match import _parse_date


def read_dates(path):
    with open(path, newline="") as f:
        dates = [_parse_date(row["date"]) for row in csv.DictReader(f)]
    if any(item is None for item in dates):
        raise ValueError("Invalid or missing clinical date")
    return dates


def match_times(rows, dt):
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("metadata numerics.dt must be a finite positive number")
    times = []
    for row in rows:
        value = row.get("simulation_time")
        value = float(value) if value not in (None, "") else float(row["best_step"]) * dt
        if not math.isfinite(value):
            raise ValueError("Simulation times must be finite")
        times.append(value)
    return times


def fit_slope(matches_path, dates_path, metadata_path):
    with open(matches_path, newline="") as f:
        rows = list(csv.DictReader(f))
    dates = read_dates(dates_path)
    if len(rows) != len(dates):
        raise ValueError(f"{len(rows)} matches but {len(dates)} dates")
    if len(rows) < 2:
        raise ValueError("Need at least 2 matched frames")

    with open(metadata_path) as f:
        dt = json.load(f)["numerics"]["dt"]

    day0 = min(dates)
    days = [(d - day0).days for d in dates]
    times = match_times(rows, dt)
    x0, y0 = days[0], times[0]
    dx = [x - x0 for x in days]
    dy = [y - y0 for y in times]
    denominator = sum(x * x for x in dx)
    if denominator == 0:
        raise ValueError("Clinical dates do not span time")
    slope = sum(x * y for x, y in zip(dx, dy)) / denominator
    predicted = [y0 + slope * x for x in dx]
    mean = sum(times) / len(times)
    ss_res = sum((y - p) ** 2 for y, p in zip(times, predicted))
    ss_tot = sum((y - mean) ** 2 for y in times)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else (1.0 if ss_res == 0 else 0.0)

    return {
        "slope": slope,
        "origin_day": float(x0),
        "origin_sim_time": y0,
        "r2": r2,
        "dt": dt,
        "matches": str(matches_path),
        "computed": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matches", type=Path)
    parser.add_argument("dates", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    result = fit_slope(args.matches, args.dates, args.metadata)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Slope: {result['slope']:.6e} simulation_time/day")
    print(f"Origin: day={result['origin_day']:.0f}, sim_time={result['origin_sim_time']:.6f}")
    print(f"R2: {result['r2']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
