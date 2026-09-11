#!/usr/bin/env python3
"""Plot series density, observation span, and slope-fit quality."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def series_metrics(name, rows, slope, r2):
    days = [float(row["days_since_first"]) for row in rows if row.get("days_since_first")]
    if len(days) < 2:
        raise ValueError(f"{name}: need at least two dated observations")
    span = max(days) - min(days)
    return {
        "name": name,
        "span": span,
        "gap": span / (len(days) - 1),
        "points": len(days),
        "slope": slope,
        "r2": r2,
    }


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_existing(root):
    slopes = json.loads((root / "slopes.json").read_text())
    output = []
    for key, entry in slopes.items():
        csv_ref = Path(entry["csv"])
        csv_path = root / csv_ref.parent.name / csv_ref.name
        if not csv_path.exists():
            csv_path = root / csv_ref
        if not csv_path.exists():
            matches = list(root.glob(f"results_*/{csv_ref.name}"))
            if not matches:
                continue
            csv_path = matches[0]
        name = key.removeprefix("results_")
        parts = name.split("_")
        if len(parts) >= 3:
            name = "_".join([part.upper() for part in parts[:2]] + parts[2:])
        output.append(series_metrics(name, read_csv(csv_path), entry["slope"], entry["r2"]))
    return output


def load_mi(root):
    rows = read_csv(root / "visualizations" / "match_results.csv")
    slope = json.loads((root / "visualizations" / "slope.json").read_text())
    if rows and "day" in rows[0]:
        dated = [{"days_since_first": row["day"]} for row in rows]
    else:
        dates_path = root / "dates.csv"
        if not dates_path.exists():
            raise ValueError("match_results.csv has no day column and dates.csv is missing")
        from fit_slope import read_dates
        dates = read_dates(dates_path)
        if len(rows) != len(dates):
            raise ValueError("matches and dates must have the same number of rows")
        day0 = min(dates)
        dated = [{"days_since_first": (item - day0).days} for item in dates]
    return series_metrics("MI_C_OG", dated, slope["slope"], slope["r2"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("existing_root", type=Path)
    parser.add_argument("mi_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    series = load_existing(args.existing_root) + [load_mi(args.mi_root)]
    fig, ax = plt.subplots(figsize=(12, 8))
    r2_values = [item["r2"] for item in series]
    r2_min = min(r2_values)
    r2_range = 1.0 - r2_min
    base_radius, r2_radius = 7.0, 18.0
    radii = [base_radius + r2_radius * (item["r2"] - r2_min) / r2_range
             if r2_range else base_radius + r2_radius for item in series]
    sizes = [radius ** 2 for radius in radii]

    for index, (item, size) in enumerate(zip(series, sizes)):
        ax.scatter(item["gap"], item["span"], s=size, color="#2563eb", alpha=0.72,
                   marker="o", edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(item["name"], (item["gap"], item["span"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Average gap between observations (days; lower = denser)")
    ax.set_ylabel("Observation span (days)")
    ax.set_title("Clinical Series Distribution: Density, Span, and Slope Fit")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.1, y=0.08)
    ax.text(0.01, 0.01, "Bigger circle = higher R² = better slope fit",
            transform=ax.transAxes, fontsize=9, alpha=0.7)
    reference_x = [0.82, 0.94]
    reference_y = [0.08, 0.08]
    reference_radii = [base_radius, base_radius + r2_radius]
    ax.scatter(reference_x, reference_y, s=[radius ** 2 for radius in reference_radii],
               color="#64748b", alpha=0.72, edgecolor="white", linewidth=1.2,
               transform=ax.transAxes, zorder=4)
    ax.text(0.82, 0.03, f"R² min\n({r2_min:.3f})", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
    ax.text(0.94, 0.03, "R² = 1\n(perfect)", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
