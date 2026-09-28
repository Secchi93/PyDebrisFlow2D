from __future__ import annotations

"""Dense concentration sweep for the O'Brien-Julien closure.

The test compares the legacy hard cs=0.20 switch with the recommended
continuous-asymptotic formulation.  It is a constitutive diagnostic, not a
Marsicano calibration or field validation.

Run directly:
    python tests/test_concentration_sweep.py

Or through the one-stage orchestrator:
    python run_full_suite.py --only-concentration-sweep
"""

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pydebrisflow as sm
from pydebrisflow.config import load_config, validate_config
from pydebrisflow.constants import NV
from pydebrisflow.physics import composition_fields


def _state(cfg, cs: float, fc: float, h: float, speed: float) -> np.ndarray:
    U = np.zeros((NV, 1, 1), dtype=float)
    U[:, 0, 0] = sm.primitive_to_state(
        h, speed, 0.0, cs, 0.5, fc, fc,
        cfg.material.rho_fluid, cfg.material.rho_solid, cfg.material.max_solid_fraction,
    )
    return U


def _evaluate(cfg, cs: float, fc: float, h: float, speed: float, dt: float) -> dict:
    U = _state(cfg, cs, fc, h, speed)
    f = composition_fields(U, cfg)
    props = sm.obrien_julien_properties(f, cfg)
    tau = float(sm.basal_shear(f, np.ones((1, 1), dtype=float), cfg)[0, 0])
    mass = float(f["mass"][0, 0])
    s0 = float(f["speed"][0, 0])
    sm.apply_basal_resistance(U, np.ones((1, 1), dtype=float), dt, cfg)
    s1 = float(composition_fields(U, cfg)["speed"][0, 0])
    return {
        "cs": float(cs),
        "cv_rheology": float(props["cv_rheology"][0, 0]),
        "rho_kgm3": float(f["rho"][0, 0]),
        "yield_stress_pa": float(props["yield_stress_pa"][0, 0]),
        "dynamic_viscosity_pas": float(props["dynamic_viscosity_pas"][0, 0]),
        "n_td": float(props["n_td"][0, 0]),
        "basal_shear_pa": tau,
        "instantaneous_deceleration_tau_over_mass_ms2": tau / max(mass, 1.0e-30),
        "one_step_deceleration_ms2": (s0 - s1) / dt,
    }


def _is_nondecreasing(values: list[float], atol: float = 1.0e-10) -> bool:
    return all(b >= a - atol for a, b in zip(values[:-1], values[1:]))


def _nearest(rows: list[dict], cs: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["cs"]) - cs))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(path: Path, piecewise: list[dict], continuous: list[dict], threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["cs"] for r in piecewise], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    specs = [
        ("yield_stress_pa", "Yield stress [Pa]"),
        ("dynamic_viscosity_pas", "Dynamic viscosity [Pa s]"),
        ("n_td", "Turbulent/dispersive Manning n"),
        ("one_step_deceleration_ms2", "Resistance deceleration [m/s2]"),
    ]
    for ax, (key, ylabel) in zip(axes.ravel(), specs):
        ax.plot(x, [r[key] for r in piecewise], label="legacy piecewise")
        ax.plot(x, [r[key] for r in continuous], label="continuous asymptotic", linestyle="--")
        ax.axvline(threshold, linewidth=1.0, linestyle=":", label="legacy cs=0.20" if key == specs[0][0] else None)
        ax.set_xlabel("Total solid volume fraction cs [-]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("O'Brien-Julien concentration sweep: hard switch vs continuous asymptotic anchoring")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense cs sweep for concentration-rheology continuity.")
    parser.add_argument("--config", default="configs/obrien_julien_quick.yaml")
    parser.add_argument("--output-root", default="outputs/Concentration_Rheology_Sweep")
    parser.add_argument("--fc", type=float, default=0.40, help="Coarse fraction within solids.")
    parser.add_argument("--h", type=float, default=1.0, help="Flow depth [m].")
    parser.add_argument("--speed", type=float, default=5.0, help="Initial speed [m/s].")
    parser.add_argument("--dt", type=float, default=1.0e-5, help="Small source-step dt used for numerical deceleration [s].")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.rheology.model = "obrien_julien"
    validate_config(cfg)

    max_cs = min(0.70, float(cfg.material.max_solid_fraction))
    threshold = float(cfg.rheology.oj_clear_water_cv_threshold)
    cs_values = set(float(v) for v in np.arange(0.0, max_cs + 0.005, 0.01))
    # Dense probe around the old regime boundary. It is no longer a branch
    # point for the recommended continuous law.
    for value in (threshold - 0.001, threshold, threshold + 0.001, max_cs):
        if 0.0 <= value <= max_cs:
            cs_values.add(float(value))
    cs_values = sorted(cs_values)

    piece_cfg = copy.deepcopy(cfg)
    piece_cfg.rheology.oj_transition_mode = "literature_piecewise"
    continuous_cfg = copy.deepcopy(cfg)
    continuous_cfg.rheology.oj_transition_mode = "continuous_asymptotic"
    validate_config(piece_cfg)
    validate_config(continuous_cfg)

    piecewise = [_evaluate(piece_cfg, cs, args.fc, args.h, args.speed, args.dt) for cs in cs_values]
    continuous = [_evaluate(continuous_cfg, cs, args.fc, args.h, args.speed, args.dt) for cs in cs_values]

    p_below = _nearest(piecewise, threshold - 0.001)
    p_at = _nearest(piecewise, threshold)
    c_below = _nearest(continuous, threshold - 0.001)
    c_at = _nearest(continuous, threshold)
    c_above = _nearest(continuous, threshold + 0.001)

    piece_drop = max(0.0, (p_below["one_step_deceleration_ms2"] - p_at["one_step_deceleration_ms2"]) / max(abs(p_below["one_step_deceleration_ms2"]), 1e-30))
    c_local_change = max(
        abs(c_at["one_step_deceleration_ms2"] - c_below["one_step_deceleration_ms2"]),
        abs(c_above["one_step_deceleration_ms2"] - c_at["one_step_deceleration_ms2"]),
    ) / max(abs(c_at["one_step_deceleration_ms2"]), 1e-30)

    checks = {
        "continuous_clear_water_limit_pass": (
            abs(continuous[0]["yield_stress_pa"]) < 1e-14
            and abs(continuous[0]["dynamic_viscosity_pas"] - cfg.rheology.oj_water_dynamic_viscosity_pas) < 1e-14
            and abs(continuous[0]["n_td"] - cfg.rheology.oj_water_manning_n) < 1e-14
        ),
        "continuous_all_outputs_finite": all(math.isfinite(float(v)) for r in continuous for v in r.values()),
        "continuous_yield_stress_monotone": _is_nondecreasing([r["yield_stress_pa"] for r in continuous]),
        "continuous_viscosity_monotone": _is_nondecreasing([r["dynamic_viscosity_pas"] for r in continuous]),
        "continuous_n_td_monotone": _is_nondecreasing([r["n_td"] for r in continuous]),
        "continuous_deceleration_monotone": _is_nondecreasing([r["one_step_deceleration_ms2"] for r in continuous], atol=1e-8),
        "continuous_local_change_near_0p20_lt_2pct": c_local_change < 0.02,
        "legacy_piecewise_global_deceleration_monotone": _is_nondecreasing([r["one_step_deceleration_ms2"] for r in piecewise], atol=1e-8),
    }
    implementation_ok = all(v for k, v in checks.items() if not k.startswith("legacy_"))

    out = Path(args.output_root)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    merged = []
    for p, c in zip(piecewise, continuous):
        row = {f"piecewise_{k}": v for k, v in p.items()}
        for k, v in c.items():
            row[f"continuous_{k}"] = v
        merged.append(row)
    _write_csv(out / "concentration_rheology_sweep.csv", merged)
    _make_figure(out / "concentration_rheology_sweep.png", piecewise, continuous, threshold)

    summary = {
        "experiment": "dense_concentration_rheology_sweep",
        "status": "pass" if implementation_ok else "fail",
        "interpretation": (
            "Constitutive diagnostic only; not a Marsicano field calibration. The recommended continuous-asymptotic law preserves "
            "the O'Brien-Julien concentration slopes while enforcing the carrier-fluid limit supported by continuous suspension "
            "rheology. The legacy hard threshold is retained only as a reproducibility comparator."
        ),
        "literature_basis": [
            "O'Brien & Julien (1988), J. Hydraul. Eng. 114(8): exponential concentration dependence of mud-matrix viscosity and yield stress.",
            "Boyer, Guazzelli & Pouliquen (2011), Phys. Rev. Lett. 107, 188301: continuous suspension rheology recovering the carrier-fluid/Einstein limit at low particle volume fraction and frictional resistance near jamming.",
            "Pellegrino & Schippa (2018), Water 10, 21: application of continuous frictional suspension rheology to debris/hyperconcentrated granular-fluid mixtures and irregular grains.",
        ],
        "regularization_disclosure": (
            "The shifted excess-over-water equations are a PyDebrisFlow2D asymptotic regularization constrained by those literature limits; "
            "they are not claimed to be a verbatim O'Brien-Julien equation."
        ),
        "cs_range": [0.0, max_cs],
        "nominal_step": 0.01,
        "legacy_reference_threshold_cs": threshold,
        "fixed_state": {"h_m": args.h, "speed_ms": args.speed, "coarse_fraction_within_solids": args.fc, "source_dt_s": args.dt},
        "legacy_piecewise_relative_drop_at_0p20": piece_drop,
        "continuous_relative_local_change_near_0p20": c_local_change,
        "checks": checks,
        "products": {
            "csv": str(out / "concentration_rheology_sweep.csv"),
            "figure": str(out / "concentration_rheology_sweep.png"),
        },
    }
    with (out / "concentration_rheology_sweep_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("\nO'BRIEN-JULIEN CONTINUOUS CONCENTRATION SWEEP")
    print(f"cs range: 0.00 -> {max_cs:.2f}; legacy threshold={threshold:.3f}; fc={args.fc:.2f}; h={args.h:g} m; speed={args.speed:g} m/s")
    print(f"legacy piecewise drop at 0.20:             {100.0 * piece_drop:.3f} %")
    print(f"continuous local change near 0.20:         {100.0 * c_local_change:.3f} %")
    print(f"continuous yield monotonic:                {checks['continuous_yield_stress_monotone']}")
    print(f"continuous viscosity monotonic:            {checks['continuous_viscosity_monotone']}")
    print(f"continuous roughness monotonic:            {checks['continuous_n_td_monotone']}")
    print(f"continuous resistance monotonic:           {checks['continuous_deceleration_monotone']}")
    print(f"diagnostic status:                         {summary['status'].upper()}")
    print(f"CSV:    {summary['products']['csv']}")
    print(f"Figure: {summary['products']['figure']}")
    print(f"JSON:   {out / 'concentration_rheology_sweep_summary.json'}")

    if not implementation_ok:
        print("concentration rheology dense sweep: FAIL")
        return 1
    print("concentration rheology dense sweep: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
