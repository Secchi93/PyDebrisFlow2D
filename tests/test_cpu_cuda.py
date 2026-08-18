from __future__ import annotations

"""Standalone CPU-vs-CUDA performance benchmark for PyDebrisFlow2D.

This is intentionally independent from ``test.py`` (Ritter verification) and
``marsicano_ablation.py`` (Marsicano first/second-order and multiphysics study).
It changes only execution backend and target DEM spacing while keeping the same
scientific configuration, and reports speedup, time per step, and estimated
working-array memory.

Run from the repository root, for example::

    python tests/test_cpu_cuda.py

or::

    python tests/test_cpu_cuda.py --dx 20 10 5 --t-end 20 --order 2
"""

# Boundary-condition kernels intentionally use small CUDA launch grids. Hide the
# low-occupancy warning because it is expected for these lightweight kernels.
import os
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Allow direct execution from tests/ without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydebrisflow.config import load_config, validate_config
from pydebrisflow.simulation import run_solver


def _fmt_dx(dx: float) -> str:
    return f"{dx:g}".replace("-", "m").replace(".", "p")


def _safe_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0.0:
        return math.nan
    return num / den


def _prepare_cfg(base_cfg, *, backend: str, dx: float, t_end: float, order: int, root: Path, tag_prefix: str = ""):
    cfg = copy.deepcopy(base_cfg)
    cfg.compute.backend = backend
    if backend == "cuda":
        # Never silently time CPU when CUDA is requested.
        cfg.compute.cuda_allow_fallback = False
    cfg.numerics.space_order = int(order)
    cfg.numerics.time_order = int(order)
    cfg.grid.target_dx = float(dx)
    tag = f"{tag_prefix}dx{_fmt_dx(dx)}_{backend}_O{order}"
    cfg.grid.cache_path = str((root / "cache" / f"dem_{tag}.npz").resolve())
    cfg.output.out_dir = str((root / "runs" / tag).resolve())
    cfg.numerics.t_end = float(t_end)
    cfg.numerics.output_dt = float(t_end)
    cfg.output.save_snapshots = False
    cfg.output.make_png = False
    cfg.output.make_gif = False
    cfg.output.save_setup_preview = False
    cfg.output.progress_every_steps = 0
    validate_config(cfg)
    return cfg


def _run_case(base_cfg, *, backend: str, dx: float, t_end: float, order: int, root: Path) -> dict:
    cfg = _prepare_cfg(base_cfg, backend=backend, dx=dx, t_end=t_end, order=order, root=root)
    try:
        metrics = run_solver(cfg)
        shape = metrics.get("shape", [None, None])
        ny, nx = shape[0], shape[1]
        elapsed = float(metrics.get("elapsed_wall_s", math.nan))
        steps = int(metrics.get("steps", 0))
        return {
            "requested_backend": backend,
            "selected_backend": metrics.get("backend", "unknown"),
            "order": int(order),
            "dx_m": metrics.get("dx_m"),
            "dy_m": metrics.get("dy_m"),
            "ny": ny,
            "nx": nx,
            "cells": int(ny * nx) if ny is not None and nx is not None else "",
            "t_end_s": metrics.get("t_final_s"),
            "steps": steps,
            "elapsed_wall_s": elapsed,
            "time_per_step_s": elapsed / steps if steps > 0 and math.isfinite(elapsed) else math.nan,
            "memory_mib_estimate": metrics.get(
                "cuda_allocated_mib_estimate",
                metrics.get("cpu_array_memory_mib_estimate"),
            ),
            "hllc_face_fallback_count": metrics.get("hllc_face_fallback_count", ""),
            "global_first_order_fallback_steps": metrics.get("global_first_order_fallback_steps", ""),
            "time_step_retries": metrics.get("time_step_retries", ""),
            "device": metrics.get("cuda_device") or (
                f"CPU threads={metrics.get('cpu_threads')}" if metrics.get("backend") == "cpu" else ""
            ),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "requested_backend": backend,
            "selected_backend": "",
            "order": int(order),
            "dx_m": dx,
            "dy_m": dx,
            "ny": "",
            "nx": "",
            "cells": "",
            "t_end_s": t_end,
            "steps": "",
            "elapsed_wall_s": "",
            "time_per_step_s": "",
            "memory_mib_estimate": "",
            "hllc_face_fallback_count": "",
            "global_first_order_fallback_steps": "",
            "time_step_retries": "",
            "device": "",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(rows: Sequence[dict], dx_values: Sequence[float], order: int) -> list[dict]:
    paired: list[dict] = []
    for dx in dx_values:
        cpu = next((r for r in rows if r["requested_backend"] == "cpu" and float(r["dx_m"]) == float(dx) and r["status"] == "ok"), None)
        gpu = next((r for r in rows if r["requested_backend"] == "cuda" and float(r["dx_m"]) == float(dx) and r["status"] == "ok"), None)
        if cpu is None or gpu is None:
            continue
        cpu_t = float(cpu["elapsed_wall_s"])
        gpu_t = float(gpu["elapsed_wall_s"])
        cpu_step = float(cpu["time_per_step_s"])
        gpu_step = float(gpu["time_per_step_s"])
        paired.append({
            "order": int(order),
            "dx_m": float(dx),
            "cells": cpu.get("cells", ""),
            "cpu_elapsed_wall_s": cpu_t,
            "cuda_elapsed_wall_s": gpu_t,
            "speedup_cpu_over_cuda": _safe_ratio(cpu_t, gpu_t),
            "cpu_steps": cpu.get("steps", ""),
            "cuda_steps": gpu.get("steps", ""),
            "cpu_time_per_step_s": cpu_step,
            "cuda_time_per_step_s": gpu_step,
            "time_per_step_speedup_cpu_over_cuda": _safe_ratio(cpu_step, gpu_step),
            "cpu_memory_mib_estimate": cpu.get("memory_mib_estimate", ""),
            "cuda_memory_mib_estimate": gpu.get("memory_mib_estimate", ""),
            "cpu_device": cpu.get("device", ""),
            "cuda_device": gpu.get("device", ""),
        })
    return paired


def _plot_products(root: Path, paired: Sequence[dict]) -> None:
    if not paired:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13.0,
        "axes.titlesize": 14.0,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 11.5,
        "lines.markersize": 6.5,
    })

    cells = np.asarray([float(row["cells"]) for row in paired], dtype=float)
    speedup = np.asarray([float(row["speedup_cpu_over_cuda"]) for row in paired], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(cells, speedup, marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of active grid cells")
    ax.set_ylabel(r"CPU/CUDA speedup $t_{CPU}/t_{CUDA}$")
    ax.set_title("PyDebrisFlow2D CPU/CUDA speedup")
    ax.grid(True, which="both", alpha=0.25)
    for row, x, y in zip(paired, cells, speedup):
        ax.annotate(f"dx={float(row['dx_m']):g} m", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(root / "cpu_cuda_speedup_vs_cells.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "cpu_cuda_speedup_vs_cells.pdf", bbox_inches="tight")
    plt.close(fig)

    cpu_mem = np.asarray([float(row["cpu_memory_mib_estimate"]) for row in paired], dtype=float)
    cuda_mem = np.asarray([float(row["cuda_memory_mib_estimate"]) for row in paired], dtype=float)
    x = np.arange(len(paired), dtype=float)
    labels = [f"{float(row['dx_m']):g} m" for row in paired]
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(x - width / 2, cpu_mem, width=width, label="CPU")
    ax.bar(x + width / 2, cuda_mem, width=width, label="CUDA")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Target grid spacing")
    ax.set_ylabel("Estimated working-array memory [MiB]")
    ax.set_title("CPU/CUDA memory scaling")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "cpu_cuda_memory_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "cpu_cuda_memory_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone PyDebrisFlow2D CPU/CUDA timing and memory benchmark.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "config_marsicano.yaml"))
    parser.add_argument("--dx", nargs="+", type=float, default=[20.0, 10.0, 5.0], help="Target DEM spacings [m].")
    parser.add_argument("--t-end", type=float, default=20.0, help="Common benchmark final time [s].")
    parser.add_argument("--order", type=int, choices=[1, 2], default=2, help="Numerical order used for the hardware benchmark.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "CPU_CUDA_Benchmark"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = Path(args.config).expanduser().resolve()
    base_cfg = load_config(str(config_path))
    root = Path(args.out_dir).expanduser().resolve()
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)

    # Warm each execution backend before measured cases so the first published
    # timing does not include Numba/CUDA first-call compilation overhead.
    warmup_dx = max(100.0, max(float(value) for value in args.dx))
    warmup_t_end = min(float(args.t_end), 0.05)
    warm_root = root / "warmup"
    (warm_root / "cache").mkdir(parents=True, exist_ok=True)
    (warm_root / "runs").mkdir(parents=True, exist_ok=True)
    for backend in ("cpu", "cuda"):
        print(f"[CPU/CUDA][WARMUP] backend={backend} dx={warmup_dx:g} m order=O{args.order} t_end={warmup_t_end:g} s")
        warm = _run_case(base_cfg, backend=backend, dx=warmup_dx, t_end=warmup_t_end, order=args.order, root=warm_root)
        if warm.get("status") != "ok":
            print(f"[CPU/CUDA][WARMUP][WARN] {backend}: {warm.get('error', 'unknown error')}")

    rows: list[dict] = []
    for dx in args.dx:
        for backend in ("cpu", "cuda"):
            print(f"[CPU/CUDA] backend={backend} dx={dx:g} m order=O{args.order} t_end={args.t_end:g} s")
            row = _run_case(base_cfg, backend=backend, dx=dx, t_end=args.t_end, order=args.order, root=root)
            rows.append(row)
            print(json.dumps(row, indent=2, allow_nan=True))

    _write_csv(root / "cpu_cuda_raw_results.csv", rows)
    with (root / "cpu_cuda_raw_results.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=True)

    paired = _paired_rows(rows, args.dx, args.order)
    _write_csv(root / "cpu_cuda_speedup.csv", paired)
    with (root / "cpu_cuda_speedup.json").open("w", encoding="utf-8") as stream:
        json.dump(paired, stream, indent=2, allow_nan=True)
    _plot_products(root, paired)

    print("\n" + "=" * 94)
    print("CPU/CUDA PERFORMANCE SUMMARY")
    print("=" * 94)
    if paired:
        print("dx[m]       cells      CPU[s]     CUDA[s]    speedup    CPU MiB    CUDA MiB")
        print("-" * 94)
        for row in paired:
            print(
                f"{float(row['dx_m']):<10g} {int(row['cells']):>10d} "
                f"{float(row['cpu_elapsed_wall_s']):>11.5g} {float(row['cuda_elapsed_wall_s']):>11.5g} "
                f"{float(row['speedup_cpu_over_cuda']):>10.4g} "
                f"{float(row['cpu_memory_mib_estimate']):>10.3f} {float(row['cuda_memory_mib_estimate']):>11.3f}"
            )
    else:
        print("No complete CPU/CUDA pairs were produced. Check failed rows in cpu_cuda_raw_results.csv.")

    print(f"[CPU/CUDA] outputs: {root}")
    failures = [row for row in rows if row.get("status") != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
