from __future__ import annotations

"""Short end-to-end integration check for the O'Brien-Julien closure.

This is intentionally not a Marsicano calibration/validation experiment. It verifies
that the concentration-aware resistance can run through the full solver on the real
DEM while preserving finite states and material budgets.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydebrisflow.config import load_config, validate_config
from pydebrisflow.simulation import run_solver


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the short O'Brien-Julien end-to-end integration check.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "obrien_julien_quick.yaml")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--t-end", type=float, default=None, help="Optional shorter final time for smoke testing.")
    args = parser.parse_args()

    out = args.output_root.expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config.expanduser().resolve()))
    cfg.compute.backend = args.backend
    cfg.output.out_dir = str(out)
    if args.t_end is not None:
        cfg.numerics.t_end = float(args.t_end)
        cfg.numerics.output_dt = float(args.t_end)
    validate_config(cfg)

    metrics = run_solver(cfg)

    checks = {
        "rheology_is_obrien_julien": metrics.get("rheology_model") == "obrien_julien",
        "finite_hmax": math.isfinite(float(metrics.get("h_max_m", float("nan")))) and float(metrics.get("h_max_m", 0.0)) >= 0.0,
        "finite_vmax": math.isfinite(float(metrics.get("speed_max_ms", float("nan")))) and float(metrics.get("speed_max_ms", 0.0)) >= 0.0,
        "finite_density": math.isfinite(float(metrics.get("rho_min_wet_kgm3", float("nan")))) and math.isfinite(float(metrics.get("rho_max_wet_kgm3", float("nan")))),
        "fluid_budget": abs(float(metrics.get("fluid_material_residual_m3", float("inf")))) <= 1.0e-6,
        "fine_budget": abs(float(metrics.get("fine_material_residual_m3", float("inf")))) <= 1.0e-6,
        "coarse_budget": abs(float(metrics.get("coarse_material_residual_m3", float("inf")))) <= 1.0e-6,
        "completed_requested_time": abs(float(metrics.get("t_final_s", -1.0)) - float(cfg.numerics.t_end)) <= 1.0e-9,
    }
    summary = {
        "purpose": "integration_check_not_field_calibration",
        "config": str(args.config.expanduser().resolve()),
        "backend_requested": args.backend,
        "checks": checks,
        "pass": all(checks.values()),
        "metrics": metrics,
    }
    (out / "oj_end_to_end_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("O'Brien-Julien end-to-end integration checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"O'Brien-Julien end-to-end integration: {'PASS' if summary['pass'] else 'FAIL'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
