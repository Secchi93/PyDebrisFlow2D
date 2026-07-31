from __future__ import annotations

"""Command-line interface for PyDebrisFlow2D."""

import os

# Boundary-condition kernels intentionally use small CUDA launch grids. Hide the
# low-occupancy warning because it is expected for these lightweight kernels.
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

import argparse
from pathlib import Path
from typing import Optional, Sequence

from pydebrisflow.compute import cuda_runtime_info
from pydebrisflow.config import load_config, validate_config
from pydebrisflow.simulation import run_solver


def default_config_path() -> str:
    """Return the Marsicano configuration distributed with the repository."""
    path = Path(__file__).resolve().parent / "configs" / "config_marsicano.yaml"
    return str(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line options and run the requested repository workflow."""
    parser = argparse.ArgumentParser(
        description="Run the debris-flow solver or prepare the Marsicano DEM."
    )
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="YAML configuration file used by the solver.",
    )
    parser.add_argument(
        "--prepare-dem",
        action="store_true",
        help="Prepare the Marsicano GeoTIFF and ESRI ASCII DEM files, then exit.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override compute.backend from the YAML configuration.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override compute.cpu_threads; zero uses the Numba default.",
    )
    parser.add_argument(
        "--cuda-info",
        action="store_true",
        help="Print CUDA runtime and device information, then exit.",
    )
    parser.add_argument(
        "--interactive-setup",
        action="store_true",
        help="Draw two friction polygons followed by the release polygon.",
    )
    parser.add_argument(
        "--interactive-friction",
        action="store_true",
        help="Draw the two friction polygons.",
    )
    parser.add_argument(
        "--interactive-release-polygon",
        action="store_true",
        help="Draw the initial release polygon.",
    )
    parser.add_argument(
        "--no-hillshade",
        action="store_true",
        help="Disable hillshade in previews, maps, and animations.",
    )
    args = parser.parse_args(argv)

    if args.prepare_dem:
        try:
            from utils.prepare_dem import main as prepare_dem
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "utils/prepare_dem.py was not included among the supplied source files."
            ) from exc
        return prepare_dem()

    if args.cuda_info:
        available, name, detail = cuda_runtime_info()
        print(f"CUDA available: {available}")
        print(f"CUDA device: {name or 'none'}")
        print(f"Detail: {detail}")
        return 0 if available else 1

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg.compute.backend = args.backend
    if args.cpu_threads is not None:
        cfg.compute.cpu_threads = args.cpu_threads
    if args.interactive_setup or args.interactive_friction:
        cfg.friction.interactive = True
        cfg.friction.regions = []
    if args.interactive_setup or args.interactive_release_polygon:
        cfg.release.enabled = True
        cfg.release.mode = "polygon"
        cfg.release.interactive_polygon = True
        cfg.release.polygon = []
    if args.no_hillshade:
        cfg.output.use_hillshade = False
        cfg.output.hillshade_alpha = 0.0

    validate_config(cfg)
    run_solver(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
