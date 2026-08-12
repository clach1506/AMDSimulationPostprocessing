#!/usr/bin/env python3
"""Find the simulation timestep whose field best matches a given
segmentation mask (or, given a whole folder, one best-matching step per
frame), scored by Dice coefficient. Optionally overlays the matched
simulation boundary on a paired source photo, or animates all of them into
a GIF.

CLI wrapper around `visualizer.seg_match` — the same module the GUI's
Region Matching tab uses, so behavior stays identical between the two.

Single image:
    python3 match_segmentation.py <segmentation_image> <sim_dir>

Whole folder (one best match per frame, sorted the same way as the
segmentation folder; overlaid on paired source images, optionally animated):
    python3 match_segmentation.py <segmentation_dir> <sim_dir> --sources <source_dir> --output <out_dir> --gif

Step 1 is turning each segmentation image into a mask on the simulation's
own grid: segmentation images are almost never the same pixel size as the
simulation grid (nx, ny) from grid.dat, so each one is resampled onto
(nx, ny) before comparing anything.

Step 2 is finding the best-matching step without scanning every one:
`coarse_to_fine_search` samples every `--subsample`-th step first, then
densely re-scans a `--neighborhood`-wide window around the best coarse hit.
Dice-vs-step is a smooth, essentially unimodal curve for a growing lesion,
so this reliably finds the same optimum as a full scan for a fraction of the
field loads. Pass --exhaustive (single-image mode only) to scan every step
and print the full curve instead.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer.data import SimulationData
from visualizer.seg_match import (
    coarse_to_fine_search,
    exhaustive_search,
    export_overlay_gif,
    load_segmentation_mask,
    process_folder,
    save_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segmentation", type=Path, help="A segmentation image, or a folder of them")
    parser.add_argument("sim_dir", type=Path, help="Simulation folder (grid.dat + field subfolders)")
    parser.add_argument("--sources", type=Path, default=None,
                         help="Folder of source photos, same count and sort order as the segmentation folder "
                              "(folder mode only) — draws the matched simulation boundary over each one")
    parser.add_argument("--output", type=Path, default=None,
                         help="Output folder for overlays/CSV/GIF (folder mode only; "
                              "default: <segmentation>/match_results)")
    parser.add_argument("--gif", action="store_true",
                         help="Folder mode with --sources: also animate all matched overlays into one GIF")
    parser.add_argument("--fps", type=float, default=2.0, help="GIF playback speed (default: 2 fps)")
    parser.add_argument("--field", choices=["levelset", "density"], default="levelset",
                         help="Which simulation field to compare against (default: levelset)")
    parser.add_argument("--density-threshold", type=float, default=0.5,
                         help="Inside/outside threshold when --field density (default: 0.5)")
    parser.add_argument("--subsample", type=int, default=10,
                         help="Coarse search: check every Nth step first (default: 10)")
    parser.add_argument("--neighborhood", type=int, default=8,
                         help="Fine search: dense re-scan +/- N steps (by index) around the coarse best (default: 8)")
    parser.add_argument("--exhaustive", action="store_true",
                         help="Single-image mode only: scan every step instead of coarse-to-fine, "
                              "and print the full curve")
    return parser.parse_args()


def run_folder(args: argparse.Namespace) -> None:
    output_dir = args.output or (args.segmentation / "match_results")
    data = SimulationData(args.sim_dir)

    def progress(i, n, match):
        print(f"[{i}/{n}] {match.segmentation.name:>28}  ->  step {match.step:<8} dice {match.dice:.4f}")

    results = process_folder(args.segmentation, data, args.sources, field=args.field,
                              density_threshold=args.density_threshold, subsample=args.subsample,
                              neighborhood=args.neighborhood, progress=progress)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "match_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segmentation", "source", "best_step", "dice"])
        for m in results:
            writer.writerow([m.segmentation.name, m.source.name if m.source else "", m.step, f"{m.dice:.4f}"])
    print(f"\nCSV -> {csv_path}")

    if args.sources is not None:
        for m in results:
            save_overlay(m.source, data, args.field, m.step, output_dir / f"overlay_{m.segmentation.stem}.png",
                          args.density_threshold)
        print(f"Overlays -> {output_dir}/")
        if args.gif:
            gif_path = export_overlay_gif(results, data, args.field, output_dir / "overlay.gif",
                                           args.density_threshold, args.fps)
            print(f"GIF -> {gif_path}")


def run_single(args: argparse.Namespace) -> None:
    data = SimulationData(args.sim_dir)
    nx, ny = int(data.grid["nx"]), int(data.grid["ny"])
    seg_mask = load_segmentation_mask(args.segmentation, nx, ny)

    if args.exhaustive:
        scores = exhaustive_search(seg_mask, data, args.field, args.density_threshold)
        best_step, best_score = max(scores, key=lambda t: t[1])
        print(f"{'step':>8}  {'dice':>8}")
        for step, score in scores:
            marker = "  <-- best" if step == best_step else ""
            print(f"{step:>8}  {score:>8.4f}{marker}")
        print()
    else:
        best_step, best_score, evaluated = coarse_to_fine_search(
            seg_mask, data, args.field, args.density_threshold, args.subsample, args.neighborhood)
        print(f"Evaluated {len(evaluated)} of {len(data.available_steps(args.field))} step(s) "
              f"(coarse-to-fine; pass --exhaustive to scan every one and print the full curve)")
    print(f"Best match: step {best_step}  (Dice = {best_score:.4f})")


def main() -> None:
    args = parse_args()
    if args.segmentation.is_dir():
        run_folder(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()

