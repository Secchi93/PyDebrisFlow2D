from __future__ import annotations

"""Temporal CFL robustness study for the native 5 m Marsicano case.

This driver runs the same spatial model at a sequence of CFL values while
holding the grid, physical parameters, end time, and numerical order fixed.
It then writes one comparison table that separates timestep sensitivity from
grid sensitivity.

Typical publication run on CUDA::

    python run_cfl_study.py --backend cuda

The default sequence is CFL = 0.25, 0.15, 0.10, 0.075. Existing complete
runs are resumed unless --force is supplied.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parent

QOI_KEYS = (
    "h_max_m",
    "h_p99_wet_m",
    "speed_max_ms",
    "speed_p95_wet_ms",
    "speed_p99_wet_ms",
    "wet_area_m2",
    "max_distance_from_release_centroid_m",
)


def _slug(cfl: float) -> str:
    return f"cfl{cfl:.4f}".replace(".", "p")


def _read_single_row(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one 5 m row in {path}, found {len(rows)}")
    row: Dict[str, object] = dict(rows[0])
    numeric = {
        "cfl", "dx_m", "h_max_m", "h_p99_wet_m", "speed_max_ms",
        "speed_p95_wet_ms", "speed_p99_wet_ms", "wet_area_m2",
        "max_distance_from_release_centroid_m", "steps", "elapsed_wall_s",
        "hllc_face_fallback_count", "global_first_order_fallback_steps",
        "time_step_retries", "global_first_order_fallback_fraction",
        "hllc_face_fallbacks_per_step",
    }
    for key in numeric:
        if key in row and row[key] not in (None, ""):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    return row


def _rel_percent(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference)):
        return math.nan
    if abs(reference) <= 1.0e-30:
        return 0.0 if abs(value) <= 1.0e-30 else math.nan
    return 100.0 * (value - reference) / abs(reference)


def _write_summary(output_root: Path, rows: List[Dict[str, object]]) -> Path:
    ordered = sorted(rows, key=lambda r: float(r["cfl"]), reverse=True)
    ref = min(ordered, key=lambda r: float(r["cfl"]))
    for i, row in enumerate(ordered):
        for key in QOI_KEYS:
            if key in row and key in ref:
                row[f"{key}_rel_to_smallest_cfl_percent"] = _rel_percent(
                    float(row[key]), float(ref[key])
                )
        if i + 1 < len(ordered):
            finer = ordered[i + 1]
            for key in QOI_KEYS:
                if key in row and key in finer:
                    row[f"{key}_adjacent_change_percent"] = _rel_percent(
                        float(finer[key]), float(row[key])
                    )
        else:
            for key in QOI_KEYS:
                row[f"{key}_adjacent_change_percent"] = 0.0

    fields: List[str] = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "marsicano_cfl_stability_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    with (output_root / "marsicano_cfl_stability_results.json").open("w", encoding="utf-8") as stream:
        json.dump(ordered, stream, indent=2, allow_nan=True)

    print("\n" + "=" * 111)
    print("MARSICANO 5 m TEMPORAL CFL ROBUSTNESS")
    print("=" * 111)
    print("CFL     hmax[m]  vmax[m/s]   v99[m/s]   wet_area[m2]  runout[m]  retries  FO-fallback/step")
    print("-" * 111)
    for row in ordered:
        print(
            f"{float(row['cfl']):<7.3g} "
            f"{float(row.get('h_max_m', math.nan)):>8.4f} "
            f"{float(row.get('speed_max_ms', math.nan)):>10.4f} "
            f"{float(row.get('speed_p99_wet_ms', math.nan)):>10.4f} "
            f"{float(row.get('wet_area_m2', math.nan)):>13.1f} "
            f"{float(row.get('max_distance_from_release_centroid_m', math.nan)):>10.1f} "
            f"{int(float(row.get('time_step_retries', 0))):>8d} "
            f"{float(row.get('global_first_order_fallback_fraction', math.nan)):>16.5f}"
        )
    print(f"[CFL] wrote {csv_path}")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native-5 m Marsicano temporal CFL robustness study.")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config", type=Path, default=Path("configs/marsicano_multiphysics.yaml"))
    parser.add_argument("--cfl-values", nargs="+", type=float, default=[0.25, 0.15, 0.10, 0.075])
    parser.add_argument("--dx", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=275.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--force", action="store_true", help="Recompute even when a matching run already exists.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dx < 5.0:
        raise ValueError("Sub-5 m runs are intentionally unsupported because the native DEM is 5 m")
    cfl_values = [float(v) for v in args.cfl_values]
    if not cfl_values or any(not (0.0 < v <= 1.0) for v in cfl_values):
        raise ValueError("All CFL values must satisfy 0 < CFL <= 1")

    output_root = args.output_root.expanduser().resolve()
    rows: List[Dict[str, object]] = []
    for cfl in cfl_values:
        case_root = output_root / _slug(cfl)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "run_marsicano_studies.py"),
            "--config", str(args.config),
            "--grid-sensitivity-only",
            "--backend", args.backend,
            "--grid-spacings", f"{args.dx:g}",
            "--grid-cfl", f"{cfl:.12g}",
            "--grid-output-root", str(case_root),
            "--t-end", f"{args.t_end:.12g}",
            "--no-grid-warmup",
        ]
        if args.force:
            cmd.append("--no-resume")
        print("\n[CFL] " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            row = _read_single_row(case_root / "marsicano_grid_sensitivity_results.csv")
            row["cfl"] = cfl
            rows.append(row)

    if args.dry_run:
        print("[CFL][DRY RUN] no simulation executed")
        return 0
    _write_summary(output_root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
