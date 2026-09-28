from __future__ import annotations

"""Reviewer-requested physical/numerical diagnostics for PyDebrisFlow2D.

This module complements the core verification suite with experiments requested
in the Applied Sciences review of manuscript applsci-4512403:

1. analytical verification of the two-layer segregation closure in the
   no-remixing (sharp-segregation) limit;
2. a kinematic phase-slip envelope documenting the limitation of the common
   mixture-velocity assumption;
3. a fluid-content / clear-water resistance diagnostic;
4. flow-based hydrostatic-validity diagnostics based on |kappa| u^2 / g over
   wet cells, together with vertical-versus-bed-normal depth differences;
5. one-at-a-time sensitivity of mu, xi, solid fraction, and erosion parameters.

The real-topography Grid Convergence Index (GCI) and bottleneck profiles are
implemented in ``run_marsicano_studies.py --grid-sensitivity-only`` because they
reuse the existing 20/15/10/8/6.5/5 m workflow.
"""

import argparse
import copy
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

import pydebrisflow as sm
from pydebrisflow.config import SolverConfig, load_config, validate_config
from pydebrisflow.constants import HCL, HCU, HF, HSL, HSU, MX, MY, NV
from pydebrisflow.physics import composition_fields, terrain_geometry
from pydebrisflow.simulation import run_solver


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "marsicano_multiphysics.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(payload), f, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _safe_rel_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


# -----------------------------------------------------------------------------
# 1. Two-layer segregation analytical benchmark
# -----------------------------------------------------------------------------

def _segregation_exact_hcu(
    t: np.ndarray | float,
    *,
    hcu0: float,
    hcl0: float,
    hsu: float,
    hsl: float,
    wseg: float,
) -> np.ndarray:
    """Exact coarse-upper depth for J = w phi_l (1-phi_u), Dr=0.

    With H_u and H_l fixed and C=h_c,u+h_c,l conserved,

      dx/dt = w/(H_l H_u) (C-x)(H_u-x).

    This is the analytical ODE associated with the implemented two-compartment
    segregation operator before availability caps activate.
    """
    tt = np.asarray(t, dtype=float)
    C = float(hcu0 + hcl0)
    Hu = float(hsu)
    Hl = float(hsl)
    if Hu <= 0.0 or Hl <= 0.0 or wseg < 0.0:
        raise ValueError("Layer depths must be positive and wseg non-negative")
    A = float(wseg) / (Hl * Hu)
    x0 = float(hcu0)
    if abs(C - Hu) < 1.0e-14:
        denom = 1.0 / max(C - x0, 1.0e-30) + A * tt
        return C - 1.0 / denom
    R0 = (Hu - x0) / (C - x0)
    R = R0 * np.exp(A * (Hu - C) * tt)
    return (R * C - Hu) / (R - 1.0)


def segregation_analytical_benchmark(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SolverConfig()
    cfg.numerics.h_dry = 1.0e-9
    cfg.material.initial_solid_fraction = 0.60
    cfg.material.initial_upper_solid_fraction = 0.50
    cfg.segregation.enabled = True
    cfg.segregation.segregation_length_m = 0.012
    cfg.segregation.max_segregation_speed_ms = 1.0
    cfg.segregation.remix_diffusivity_m2s = 0.0  # sharp-segregation / no-remixing limit
    cfg.segregation.min_layer_thickness_m = 1.0e-6

    h = 1.20
    speed = 4.0
    cs = 0.60
    lam = 0.50
    fc_u0 = 0.10
    fc_l0 = 0.70
    state0 = sm.primitive_to_state(
        h, speed, 0.0, cs, lam, fc_u0, fc_l0,
        cfg.material.rho_fluid, cfg.material.rho_solid, cfg.material.max_solid_fraction,
    )
    hsu = float(state0[HSU]); hsl = float(state0[HSL])
    hcu0 = float(state0[HCU]); hcl0 = float(state0[HCL])
    shear_rate = speed / h
    wseg = min(cfg.segregation.segregation_length_m * shear_rate, cfg.segregation.max_segregation_speed_ms)
    T = 2.0
    dt_values = [0.20, 0.10, 0.05, 0.025, 0.0125]
    exact_T = float(_segregation_exact_hcu(T, hcu0=hcu0, hcl0=hcl0, hsu=hsu, hsl=hsl, wseg=wseg))

    rows: List[Dict[str, Any]] = []
    histories: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for dt in dt_values:
        U = np.zeros((NV, 1, 1), dtype=float)
        U[:, 0, 0] = state0
        n = int(math.ceil(T / dt))
        times = [0.0]
        values = [hcu0]
        coarse0 = hcu0 + hcl0
        fine0 = (hsu - hcu0) + (hsl - hcl0)
        max_cons = 0.0
        t = 0.0
        for _ in range(n):
            dti = min(dt, T - t)
            if dti <= 0.0:
                break
            sm.apply_segregation(U, dti, cfg)
            t += dti
            x = float(U[HCU, 0, 0])
            c = float(U[HCU, 0, 0] + U[HCL, 0, 0])
            f = float((U[HSU, 0, 0] - U[HCU, 0, 0]) + (U[HSL, 0, 0] - U[HCL, 0, 0]))
            max_cons = max(max_cons, abs(c - coarse0), abs(f - fine0))
            times.append(t); values.append(x)
        val = values[-1]
        err = abs(val - exact_T)
        rows.append({
            "dt_s": dt,
            "hcu_numerical_m": val,
            "hcu_exact_m": exact_T,
            "absolute_error_m": err,
            "relative_error": _safe_rel_error(val, exact_T),
            "max_constituent_conservation_residual_m": max_cons,
        })
        histories[dt] = (np.asarray(times), np.asarray(values))

    for i in range(len(rows) - 1):
        e0 = float(rows[i]["absolute_error_m"])
        e1 = float(rows[i + 1]["absolute_error_m"])
        r = float(rows[i]["dt_s"]) / float(rows[i + 1]["dt_s"])
        rows[i + 1]["observed_temporal_order"] = math.log(e0 / e1) / math.log(r) if e0 > 0 and e1 > 0 else math.nan
    rows[0]["observed_temporal_order"] = math.nan

    finite_orders = np.array([r["observed_temporal_order"] for r in rows[1:] if np.isfinite(r["observed_temporal_order"])])
    median_order = float(np.median(finite_orders)) if finite_orders.size else math.nan
    finest_rel = float(rows[-1]["relative_error"])
    max_cons = max(float(r["max_constituent_conservation_residual_m"]) for r in rows)
    passed = bool(0.85 <= median_order <= 1.15 and finest_rel < 2.0e-3 and max_cons < 1.0e-12)

    _write_csv(out_dir / "segregation_analytical_convergence.csv", rows)
    t_dense = np.linspace(0.0, T, 500)
    exact = _segregation_exact_hcu(t_dense, hcu0=hcu0, hcl0=hcl0, hsu=hsu, hsl=hsl, wseg=wseg)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    ax.plot(t_dense, exact, linewidth=2.0, label="Analytical two-compartment solution")
    for dt in (dt_values[0], dt_values[2], dt_values[-1]):
        tnum, xnum = histories[dt]
        ax.plot(tnum, xnum, marker="o", markevery=max(1, len(tnum)//10), linewidth=1.1, label=f"Operator, dt={dt:g} s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Upper-layer coarse depth $h_{c,u}$ [m]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "segregation_analytical_solution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "segregation_analytical_solution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.loglog([r["dt_s"] for r in rows], [r["absolute_error_m"] for r in rows], marker="o", label="Final-time error")
    ref = rows[-1]["absolute_error_m"] * np.asarray(dt_values) / dt_values[-1]
    ax.loglog(dt_values, ref, linestyle="--", label="First-order reference")
    ax.set_xlabel("Time step [s]")
    ax.set_ylabel("Absolute error [m]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "segregation_analytical_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "experiment": "two_layer_segregation_analytical_verification",
        "interpretation": "No-remixing sharp-segregation limit of the implemented two-compartment closure; constant layer depths and constant shear rate.",
        "parameters": {
            "h_m": h, "speed_ms": speed, "solid_fraction": cs,
            "upper_layer_fraction": lam, "initial_fc_upper": fc_u0,
            "initial_fc_lower": fc_l0, "wseg_ms": wseg, "T_s": T,
        },
        "median_observed_temporal_order": median_order,
        "finest_relative_error": finest_rel,
        "max_constituent_conservation_residual_m": max_cons,
        "pass": passed,
    }
    _save_json(out_dir / "segregation_analytical_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# 2. Single-velocity limitation: kinematic phase-slip envelope
# -----------------------------------------------------------------------------

def phase_slip_envelope(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ubar = 5.0
    T = 30.0
    slip_ratios = [0.0, 0.02, 0.05, 0.10, 0.20]
    rows = []
    for eps in slip_ratios:
        us = ubar * (1.0 + eps)
        uf = ubar * (1.0 - eps)
        xs = us * T
        xf = uf * T
        xm = ubar * T
        rows.append({
            "relative_half_slip": eps,
            "solid_velocity_ms": us,
            "fluid_velocity_ms": uf,
            "common_velocity_ms": ubar,
            "solid_front_m": xs,
            "fluid_front_m": xf,
            "common_model_front_m": xm,
            "solid_front_bias_m": xm - xs,
            "fluid_front_bias_m": xm - xf,
            "phase_front_separation_m": xs - xf,
        })
    _write_csv(out_dir / "single_velocity_phase_slip_envelope.csv", rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = [100.0 * r["relative_half_slip"] for r in rows]
    ax.plot(x, [r["phase_front_separation_m"] for r in rows], marker="o")
    ax.set_xlabel("Imposed half-slip relative to common velocity [%]")
    ax.set_ylabel("Analytical fluid-solid front separation after 30 s [m]")
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "single_velocity_phase_slip_envelope.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "experiment": "single_velocity_kinematic_limit",
        "status": "diagnostic_not_validation",
        "message": (
            "The common-velocity model exactly represents the zero-slip limit. It cannot create horizontal solid-fluid phase-front separation; "
            "the table quantifies the kinematic bias that would arise if real phase slip were present. Vertical coarse/fine sorting remains possible through the segregation operator."
        ),
        "ubar_ms": ubar,
        "duration_s": T,
        "maximum_phase_front_separation_m": max(r["phase_front_separation_m"] for r in rows),
    }
    _save_json(out_dir / "single_velocity_phase_slip_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# 3. Fluid-content / clear-water resistance diagnostic
# -----------------------------------------------------------------------------

def fluid_content_resistance_diagnostic(out_dir: Path, cfg: SolverConfig) -> Dict[str, Any]:
    """Compare the legacy Voellmy closure with the concentration-aware option.

    The second branch implements the O'Brien-Julien quadratic mudflow rheology:
    concentration-dependent yield stress and viscosity (O'Brien & Julien,
    1988) embedded in the depth-integrated yield/viscous/turbulent resistance
    used by O'Brien, Julien & Fullerton (1993). The recommended transition is
    continuous-asymptotic: the O'Brien-Julien exponential concentration slopes
    are retained as an excess above the carrier-fluid limit, consistent with
    the continuous dilute-to-dense suspension framework of Boyer et al. (2011)
    and its debris/hyperconcentrated-flow application by Pellegrino & Schippa
    (2018). This shifted anchoring is a PyDebrisFlow2D regularization, not a
    verbatim equation from those papers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = copy.deepcopy(cfg)
    base.erosion.enabled = False
    base.deposition.enabled = False
    base.segregation.enabled = False
    base.material.yield_N0_pa = 0.0

    legacy = copy.deepcopy(base)
    legacy.rheology.model = "voellmy"

    ojcfg = copy.deepcopy(base)
    ojcfg.rheology.model = "obrien_julien"
    ojcfg.rheology.oj_transition_mode = "continuous_asymptotic"
    ojcfg.rheology.oj_clear_water_cv_threshold = 0.20
    validate_config(ojcfg)

    h = 1.0
    speed = 5.0
    fc = 0.40
    dt = 1.0e-4
    cs_values = np.linspace(0.0, min(base.material.max_solid_fraction, 0.68), 18)
    rows = []
    for cs in cs_values:
        U0 = np.zeros((NV, 1, 1), dtype=float)
        U0[:, 0, 0] = sm.primitive_to_state(
            h, speed, 0.0, float(cs), 0.5, fc, fc,
            base.material.rho_fluid, base.material.rho_solid, base.material.max_solid_fraction,
        )

        # Legacy composition-dependent Voellmy branch.
        Ul = U0.copy()
        fl = composition_fields(Ul, legacy)
        s0 = float(fl["speed"][0, 0])
        tau_l = float(sm.basal_shear(fl, np.ones((1, 1)), legacy)[0, 0])
        sm.apply_voellmy_friction(Ul, np.ones((1, 1)), dt, legacy)
        s1_l = float(composition_fields(Ul, legacy)["speed"][0, 0])

        # O'Brien-Julien concentration-aware branch.
        Uo = U0.copy()
        fo = composition_fields(Uo, ojcfg)
        oj = sm.obrien_julien_properties(fo, ojcfg)
        tau_o = float(sm.basal_shear(fo, np.ones((1, 1)), ojcfg)[0, 0])
        sm.apply_voellmy_friction(Uo, np.ones((1, 1)), dt, ojcfg)
        s1_o = float(composition_fields(Uo, ojcfg)["speed"][0, 0])

        rows.append({
            "solid_fraction": float(cs),
            "fluid_fraction": float(1.0 - cs),
            "coarse_fraction_of_solids": float(fo["fc"][0, 0]),
            "rho_kgm3": float(fo["rho"][0, 0]),
            "legacy_mu": float(fl["mu"][0, 0]),
            "legacy_xi_ms2": float(fl["xi"][0, 0]),
            "legacy_basal_shear_pa": tau_l,
            "legacy_friction_deceleration_ms2": (s0 - s1_l) / dt,
            "oj_cv_rheology": float(oj["cv_rheology"][0, 0]),
            "oj_yield_stress_pa": float(oj["yield_stress_pa"][0, 0]),
            "oj_dynamic_viscosity_pas": float(oj["dynamic_viscosity_pas"][0, 0]),
            "oj_n_td": float(oj["n_td"][0, 0]),
            "oj_basal_shear_pa": tau_o,
            "oj_friction_deceleration_ms2": (s0 - s1_o) / dt,
        })
    _write_csv(out_dir / "fluid_content_resistance_diagnostic.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    x = [r["solid_fraction"] for r in rows]
    axes[0].plot(x, [r["legacy_friction_deceleration_ms2"] for r in rows], marker="o", label="Legacy Voellmy")
    axes[0].plot(x, [r["oj_friction_deceleration_ms2"] for r in rows], marker="s", label="O'Brien-Julien")
    axes[0].set_xlabel("Total solid volume fraction $c_s$")
    axes[0].set_ylabel("Instantaneous basal deceleration [m/s²]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].semilogy(x, [max(r["oj_dynamic_viscosity_pas"], 1e-12) for r in rows], marker="o", label=r"$\eta$")
    axes[1].semilogy(x, [max(r["oj_yield_stress_pa"], 1e-12) for r in rows], marker="s", label=r"$\tau_y$")
    axes[1].set_xlabel("Total solid volume fraction $c_s$")
    axes[1].set_ylabel("O'Brien-Julien property [SI units; log scale]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "fluid_content_resistance_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    clear = rows[0]
    concentrated = rows[-1]
    oj_decel = np.asarray([r["oj_friction_deceleration_ms2"] for r in rows], dtype=float)
    monotonic = bool(np.all(np.diff(oj_decel) >= -1.0e-9))
    clear_limit_pass = (
        abs(clear["oj_yield_stress_pa"]) < 1.0e-12
        and abs(clear["oj_dynamic_viscosity_pas"] - ojcfg.rheology.oj_water_dynamic_viscosity_pas) < 1.0e-12
        and abs(clear["oj_n_td"] - ojcfg.rheology.oj_water_manning_n) < 1.0e-12
    )
    summary = {
        "experiment": "fluid_content_and_clear_water_resistance",
        "fixed_coarse_fraction_of_solids": fc,
        "legacy_closure": (
            "Voellmy mu and xi depend on the coarse fraction within solids, not total solid concentration. "
            "Therefore the pure-water limit inherits the fine-endmember debris-flow coefficients."
        ),
        "implemented_concentration_closure": (
            "Optional O'Brien-Julien quadratic resistance with concentration-dependent yield stress and viscosity; "
            "turbulent/dispersive resistance also varies with concentration. The recommended continuous-asymptotic "
            "mode anchors the empirical concentration dependence to zero yield stress and carrier-fluid viscosity/roughness at cs=0, "
            "avoiding a hard clear-water/debris-flow threshold."
        ),
        "literature": [
            "O'Brien, J.S. & Julien, P.Y. (1988), Journal of Hydraulic Engineering 114(8), 877-887, DOI 10.1061/(ASCE)0733-9429(1988)114:8(877)",
            "O'Brien, J.S., Julien, P.Y. & Fullerton, W.T. (1993), Journal of Hydraulic Engineering 119(2), 244-261, DOI 10.1061/(ASCE)0733-9429(1993)119:2(244)",
            "Boyer, F., Guazzelli, E. & Pouliquen, O. (2011), Physical Review Letters 107, 188301, DOI 10.1103/PhysRevLett.107.188301: continuous dilute-to-dense suspension rheology and carrier-fluid limit.",
            "Pellegrino, A.M. & Schippa, L. (2018), Water 10, 21, DOI 10.3390/w10010021: concentration-dependent frictional rheology for debris/hyperconcentrated granular-fluid mixtures.",
            "EDDA 2.0 (GMD, 2018) is retained only as provenance for the legacy hard Cv=0.20 piecewise comparator.",
        ],
        "coefficient_set": "Aspen Pit 2 demonstration coefficients from O'Brien-Julien/FLO-2D tabulation; not a Marsicano calibration",
        # Backward-compatible legacy keys.
        "pure_water_mu": clear["legacy_mu"],
        "pure_water_xi_ms2": clear["legacy_xi_ms2"],
        "pure_water_friction_deceleration_ms2": clear["legacy_friction_deceleration_ms2"],
        "highest_cs_friction_deceleration_ms2": concentrated["legacy_friction_deceleration_ms2"],
        # New concentration-aware evidence.
        "oj_clear_water_yield_stress_pa": clear["oj_yield_stress_pa"],
        "oj_clear_water_dynamic_viscosity_pas": clear["oj_dynamic_viscosity_pas"],
        "oj_clear_water_n_td": clear["oj_n_td"],
        "oj_clear_water_friction_deceleration_ms2": clear["oj_friction_deceleration_ms2"],
        "oj_highest_cs_friction_deceleration_ms2": concentrated["oj_friction_deceleration_ms2"],
        "oj_clear_water_limit_pass": clear_limit_pass,
        "oj_deceleration_monotonic_with_cs_in_test": monotonic,
        "oj_resistance_changes_with_cs_in_test": bool(np.ptp(oj_decel) > 1.0e-6),
        "oj_transition_mode": ojcfg.rheology.oj_transition_mode,
        "oj_legacy_piecewise_reference_threshold_cs": ojcfg.rheology.oj_clear_water_cv_threshold,
        "applicability_note": (
            "The O'Brien-Julien empirical coefficients are material-specific, especially for fine-rich mud matrices. "
            "The continuous anchoring resolves the structural concentration-dependence and hard-threshold discontinuity problems, "
            "but predictive Marsicano use still requires rheological calibration or an explicitly justified coefficient set."
        ),
    }
    _save_json(out_dir / "fluid_content_resistance_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# 4. Hydrostatic validity / vertical acceleration diagnostics
# -----------------------------------------------------------------------------

def _load_final_state(path: Path, cfg: SolverConfig) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        zb = np.asarray(d["zb"], dtype=float)
        U = np.asarray(d["U"], dtype=float)
    f = composition_fields(U, cfg)
    return {"x": x, "y": y, "zb": zb, "U": U, **f}


def _percentiles(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_p50": math.nan, f"{prefix}_p90": math.nan, f"{prefix}_p95": math.nan, f"{prefix}_p99": math.nan, f"{prefix}_max": math.nan}
    q = np.percentile(values, [50, 90, 95, 99, 100])
    return {f"{prefix}_p50": float(q[0]), f"{prefix}_p90": float(q[1]), f"{prefix}_p95": float(q[2]), f"{prefix}_p99": float(q[3]), f"{prefix}_max": float(q[4])}


def hydrostatic_validity_diagnostic(
    state_path: Path,
    cfg: SolverConfig,
    out_dir: Path,
    *,
    wet_threshold: float = 0.01,
    curvature_smoothing_sigma_cells: float = 1.0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    st = _load_final_state(state_path, cfg)
    x, y, zb = st["x"], st["y"], st["zb"]
    h, u, v, speed = st["h"], st["u"], st["v"], st["speed"]
    dx = float(abs(x[1] - x[0])); dy = float(abs(y[1] - y[0]))

    zwork = np.asarray(zb, dtype=float)
    if curvature_smoothing_sigma_cells > 0.0:
        # The source Marsicano DEM is complete; nearest filling guards general use.
        validz = np.isfinite(zwork)
        if not np.all(validz):
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~validz, return_distances=False, return_indices=True)
            zwork = zwork[tuple(idx)]
        zwork = gaussian_filter(zwork, sigma=float(curvature_smoothing_sigma_cells), mode="nearest")

    dzdy, dzdx = np.gradient(zwork, dy, dx)
    zxy, zxx = np.gradient(dzdx, dy, dx)
    zyy, _ = np.gradient(dzdy, dy, dx)
    cosbeta = 1.0 / np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    moving = speed > 1.0e-8
    ex = np.zeros_like(speed); ey = np.zeros_like(speed)
    ex[moving] = u[moving] / speed[moving]
    ey[moving] = v[moving] / speed[moving]
    directional_slope = dzdx * ex + dzdy * ey
    second = zxx * ex * ex + 2.0 * zxy * ex * ey + zyy * ey * ey
    kappa = second / np.power(1.0 + directional_slope * directional_slope, 1.5)
    # The prognostic u,v components are map-plane velocities. For the
    # curvature acceleration we convert to the speed along the local vertical
    # terrain profile in the velocity direction.
    speed_surface = speed * np.sqrt(1.0 + directional_slope * directional_slope)
    ratio = np.abs(kappa) * speed_surface * speed_surface / cfg.numerics.g

    wet = np.isfinite(zb) & (h > wet_threshold)
    wet_moving = wet & moving
    depth_normal = h * cosbeta
    depth_rel_diff = np.zeros_like(h)
    depth_rel_diff[wet] = np.abs(h[wet] - depth_normal[wet]) / np.maximum(h[wet], 1.0e-30)

    vals_ratio = ratio[wet_moving & np.isfinite(ratio)]
    vals_slope = slope_deg[wet & np.isfinite(slope_deg)]
    vals_depthdiff = depth_rel_diff[wet & np.isfinite(depth_rel_diff)]
    cell_area = dx * dy
    summary: Dict[str, Any] = {
        "experiment": "hydrostatic_validity_curvature_acceleration",
        "state_file": str(state_path),
        "wet_threshold_m": wet_threshold,
        "curvature_smoothing_sigma_cells": curvature_smoothing_sigma_cells,
        "depth_definition": "h is vertical depth / constituent volume per unit horizontal plan area; approximate bed-normal depth is h*cos(beta).",
        "curvature_speed_definition": "kappa uses vertical-profile curvature in the horizontal velocity direction; u_surface = u_map*sqrt(1+(grad(z).e_u)^2).",
        "centrifugal_term_in_governing_momentum": False,
        "wet_cells": int(np.sum(wet)),
        "wet_moving_cells": int(np.sum(wet_moving)),
        "wet_area_m2": float(np.sum(wet) * cell_area),
        "fraction_wet_moving_kappa_u2_over_g_gt_0p05": float(np.mean(vals_ratio > 0.05)) if vals_ratio.size else 0.0,
        "fraction_wet_moving_kappa_u2_over_g_gt_0p10": float(np.mean(vals_ratio > 0.10)) if vals_ratio.size else 0.0,
        "fraction_wet_moving_kappa_u2_over_g_gt_0p20": float(np.mean(vals_ratio > 0.20)) if vals_ratio.size else 0.0,
        **_percentiles(vals_ratio, "kappa_u2_over_g"),
        **_percentiles(vals_slope, "wet_slope_deg"),
        **_percentiles(vals_depthdiff, "vertical_to_normal_depth_relative_difference"),
    }
    _save_json(out_dir / "hydrostatic_validity_summary.json", summary)
    np.savez_compressed(
        out_dir / "hydrostatic_validity_fields.npz",
        x=x, y=y, wet=wet.astype(np.uint8), slope_deg=slope_deg.astype(np.float32),
        curvature_1pm=kappa.astype(np.float32), kappa_u2_over_g=ratio.astype(np.float32),
        h_vertical_m=h.astype(np.float32), h_normal_approx_m=depth_normal.astype(np.float32),
    )

    extent = [x[0]-0.5*dx, x[-1]+0.5*dx, y[0]-0.5*dy, y[-1]+0.5*dy]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    im0 = axes[0].imshow(np.ma.masked_where(~wet_moving, ratio), origin="lower", extent=extent, interpolation="nearest")
    axes[0].set_title(r"Wet-cell hydrostatic diagnostic $|\kappa|u^2/g$")
    axes[0].set_xlabel("Easting [m]"); axes[0].set_ylabel("Northing [m]")
    fig.colorbar(im0, ax=axes[0], label=r"$|\kappa|u^2/g$")
    if vals_ratio.size:
        axes[1].hist(vals_ratio, bins=60, density=True)
    axes[1].axvline(0.05, linestyle="--", label="0.05")
    axes[1].axvline(0.10, linestyle="--", label="0.10")
    axes[1].axvline(0.20, linestyle="--", label="0.20")
    axes[1].set_xlabel(r"$|\kappa|u^2/g$")
    axes[1].set_ylabel("Probability density over wet moving cells")
    axes[1].grid(alpha=0.25); axes[1].legend(frameon=False)
    fig.savefig(out_dir / "hydrostatic_validity_curvature.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


# -----------------------------------------------------------------------------
# 5. One-at-a-time parameter sensitivity
# -----------------------------------------------------------------------------

def _runout_metrics(state_path: Path, cfg: SolverConfig, wet_threshold: float) -> Dict[str, float]:
    st = _load_final_state(state_path, cfg)
    x, y, h, speed = st["x"], st["y"], st["h"], st["speed"]
    dx = float(abs(x[1] - x[0])); dy = float(abs(y[1] - y[0])); area = dx * dy
    wet = h > wet_threshold
    X, Y = np.meshgrid(x, y)
    # Reconstruct release centroid from configured polygon/ellipse by using the final-state stored release mask when present.
    with np.load(state_path) as d:
        release_mask = np.asarray(d["release_mask"], dtype=bool) if "release_mask" in d else np.zeros_like(h, dtype=bool)
    if np.any(wet) and np.any(release_mask):
        cx = float(np.mean(X[release_mask])); cy = float(np.mean(Y[release_mask]))
        runout = float(np.max(np.hypot(X[wet] - cx, Y[wet] - cy)))
    else:
        runout = 0.0
    return {
        "h_max_m": float(np.max(h)),
        "speed_max_ms": float(np.max(speed)),
        "wet_area_m2": float(np.sum(wet) * area),
        "runout_from_release_centroid_m": runout,
        "mobile_volume_m3": float(np.sum(h, dtype=np.float64) * area),
    }


def _apply_sensitivity_case(cfg: SolverConfig, parameter: str, value: float) -> None:
    if parameter == "mu":
        cfg.material.mu_fine = value; cfg.material.mu_coarse = value
    elif parameter == "xi":
        cfg.material.xi_fine = value; cfg.material.xi_coarse = value
    elif parameter == "solid_fraction":
        cfg.material.initial_solid_fraction = value; cfg.release.solid_fraction = value
    elif parameter == "erosion_critical_shear_pa":
        cfg.erosion.critical_shear_pa = value
    elif parameter == "erosion_rate_ms":
        cfg.erosion.excess_shear_rate_ms = value
    else:
        raise KeyError(parameter)


def parameter_sensitivity(
    base_cfg: SolverConfig,
    out_dir: Path,
    *,
    backend: Optional[str] = None,
    grid_spacing_m: float = 20.0,
    t_end_s: float = 275.0,
    wet_threshold_m: float = 0.01,
    resume: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = copy.deepcopy(base_cfg)
    base.grid.target_dx = float(grid_spacing_m)
    base.grid.cache_path = str((out_dir / "cache" / f"dem_dx{grid_spacing_m:g}.npz").resolve())
    base.numerics.space_order = 2; base.numerics.time_order = 2
    base.numerics.t_end = float(t_end_s); base.numerics.output_dt = float(t_end_s)
    base.output.save_snapshots = False; base.output.make_png = False; base.output.make_gif = False
    base.output.save_setup_preview = False; base.output.progress_every_steps = 0
    if backend is not None:
        base.compute.backend = backend

    # Reviewer sensitivity concerns the complete multiphysics formulation.
    base.erosion.enabled = True
    base.deposition.enabled = True
    base.segregation.enabled = True
    if base.release.solid_fraction is None or base.release.solid_fraction <= 0.0:
        base.release.solid_fraction = base.material.initial_solid_fraction
    if base.release.coarse_fraction is None:
        base.release.coarse_fraction = base.material.initial_coarse_fraction

    baseline = {
        "mu": float(base.material.mu_fine),
        "xi": float(base.material.xi_fine),
        "solid_fraction": float(base.release.solid_fraction),
        "erosion_critical_shear_pa": float(base.erosion.critical_shear_pa),
        "erosion_rate_ms": float(base.erosion.excess_shear_rate_ms),
    }
    levels = {
        "mu": [0.8*baseline["mu"], baseline["mu"], 1.2*baseline["mu"]],
        "xi": [0.8*baseline["xi"], baseline["xi"], 1.2*baseline["xi"]],
        "solid_fraction": [max(0.05, baseline["solid_fraction"]-0.10), baseline["solid_fraction"], min(base.material.max_solid_fraction-0.01, baseline["solid_fraction"]+0.10)],
        "erosion_critical_shear_pa": [0.75*baseline["erosion_critical_shear_pa"], baseline["erosion_critical_shear_pa"], 1.25*baseline["erosion_critical_shear_pa"]],
        "erosion_rate_ms": [0.5*baseline["erosion_rate_ms"], baseline["erosion_rate_ms"], 2.0*baseline["erosion_rate_ms"]],
    }

    # One shared baseline plus low/high cases for each factor.
    cases: List[Tuple[str, str, float]] = [("baseline", "baseline", 1.0)]
    for par, vals in levels.items():
        cases.append((f"{par}_low", par, float(vals[0])))
        cases.append((f"{par}_high", par, float(vals[2])))

    rows: List[Dict[str, Any]] = []
    for case_id, par, val in cases:
        cfg = copy.deepcopy(base)
        if par != "baseline":
            _apply_sensitivity_case(cfg, par, val)
        case_dir = out_dir / "runs" / case_id
        cfg.output.out_dir = str(case_dir.resolve())
        validate_config(cfg)
        metrics_path = case_dir / "metrics.json"
        state_path = case_dir / "final_state.npz"
        try:
            if resume and metrics_path.is_file() and state_path.is_file():
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                reused = True
            else:
                case_dir.mkdir(parents=True, exist_ok=True)
                metrics = run_solver(cfg)
                reused = False
            q = _runout_metrics(state_path, cfg, wet_threshold_m)
            rows.append({
                "case": case_id,
                "parameter": par,
                "parameter_value": val,
                "status": "success",
                "reused": reused,
                "grid_spacing_m": grid_spacing_m,
                "t_end_s": t_end_s,
                "elapsed_wall_s": float(metrics.get("elapsed_wall_s", math.nan)),
                "eroded_bulk_m3": float(metrics.get("eroded_bulk_m3", math.nan)),
                "deposited_bulk_m3": float(metrics.get("deposited_bulk_m3", math.nan)),
                "upward_coarse_m3": float(metrics.get("upward_coarse_m3", math.nan)),
                **q,
            })
        except Exception as exc:
            rows.append({"case": case_id, "parameter": par, "parameter_value": val, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    _write_csv(out_dir / "parameter_sensitivity_results.csv", rows)
    _save_json(out_dir / "parameter_sensitivity_results.json", rows)
    success = {str(r["case"]): r for r in rows if r.get("status") == "success"}
    sensitivity_rows: List[Dict[str, Any]] = []
    qois = ["runout_from_release_centroid_m", "wet_area_m2", "h_max_m", "speed_max_ms", "mobile_volume_m3"]
    if "baseline" in success:
        rb = success["baseline"]
        for par in levels:
            lo = success.get(f"{par}_low"); hi = success.get(f"{par}_high")
            if lo is None or hi is None:
                continue
            p0 = baseline[par]; plo = float(lo["parameter_value"]); phi = float(hi["parameter_value"])
            for qoi in qois:
                qb = float(rb[qoi]); qlo = float(lo[qoi]); qhi = float(hi[qoi])
                denom_p = (phi - plo) / max(abs(p0), 1.0e-30)
                elasticity = ((qhi - qlo) / max(abs(qb), 1.0e-30)) / denom_p if abs(denom_p) > 0 else math.nan
                sensitivity_rows.append({
                    "parameter": par, "qoi": qoi,
                    "baseline_parameter": p0, "low_parameter": plo, "high_parameter": phi,
                    "baseline_qoi": qb, "low_qoi": qlo, "high_qoi": qhi,
                    "central_normalized_sensitivity": elasticity,
                    "low_relative_change": (qlo-qb)/max(abs(qb),1e-30),
                    "high_relative_change": (qhi-qb)/max(abs(qb),1e-30),
                })
    _write_csv(out_dir / "parameter_sensitivity_indices.csv", sensitivity_rows)

    try:
        if sensitivity_rows:
            pars = list(levels.keys())
            fig, axes = plt.subplots(len(qois), 1, figsize=(9.0, 2.8*len(qois)), constrained_layout=True)
            axes = np.atleast_1d(axes)
            for ax, qoi in zip(axes, qois):
                subset = [r for r in sensitivity_rows if r["qoi"] == qoi]
                values = [next((r["central_normalized_sensitivity"] for r in subset if r["parameter"] == p), math.nan) for p in pars]
                ax.barh(pars, values)
                ax.axvline(0.0, linewidth=0.8)
                ax.set_xlabel("Central normalized sensitivity")
                ax.set_title(qoi)
                ax.grid(alpha=0.2, axis="x")
            fig.savefig(out_dir / "parameter_sensitivity_indices.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"[SENSITIVITY][PLOT][WARN] {type(exc).__name__}: {exc}")

    summary = {
        "experiment": "Marsicano_one_at_a_time_parameter_sensitivity",
        "grid_spacing_m": grid_spacing_m,
        "t_end_s": t_end_s,
        "number_of_cases": len(cases),
        "successful_cases": sum(1 for r in rows if r.get("status") == "success"),
        "baseline_parameters": baseline,
        "levels": levels,
    }
    _save_json(out_dir / "parameter_sensitivity_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_synthetic_suite(config_path: Path, output_root: Path) -> Dict[str, Any]:
    cfg = load_config(str(config_path))
    out = output_root / "synthetic"
    summaries = {
        "segregation": segregation_analytical_benchmark(out / "segregation_analytical"),
        "single_velocity": phase_slip_envelope(out / "single_velocity_limit"),
        "fluid_content": fluid_content_resistance_diagnostic(out / "fluid_content_resistance", cfg),
    }
    _save_json(out / "synthetic_reviewer_experiments_summary.json", summaries)
    return summaries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run reviewer-requested PyDebrisFlow2D experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base YAML configuration.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--synthetic", action="store_true", help="Run segregation analytical, phase-slip, and fluid-content diagnostics.")
    parser.add_argument("--hydrostatic-state", type=Path, default=None, help="final_state.npz for |kappa|u^2/g diagnostics.")
    parser.add_argument("--wet-threshold", type=float, default=0.01)
    parser.add_argument("--curvature-sigma", type=float, default=1.0, help="Gaussian DEM smoothing sigma in grid cells for second derivatives.")
    parser.add_argument("--parameter-sensitivity", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--grid-spacing", type=float, default=None, help="Sensitivity grid spacing; YAML default if omitted.")
    parser.add_argument("--t-end", type=float, default=None, help="Sensitivity final time; YAML default if omitted.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run synthetic suite and parameter sensitivity. Hydrostatic analysis still needs --hydrostatic-state.")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    raw = _load_yaml(config_path)
    reviewer = raw.get("reviewer_experiments", {}) if isinstance(raw.get("reviewer_experiments", {}), Mapping) else {}
    output_root = _resolve(str(reviewer.get("output_root", str(output_root))), config_path.parent) if args.output_root == DEFAULT_OUTPUT_ROOT else output_root
    output_root.mkdir(parents=True, exist_ok=True)

    selected = args.synthetic or args.parameter_sensitivity or args.hydrostatic_state is not None or args.all
    if not selected:
        args.synthetic = True

    summaries: Dict[str, Any] = {}
    if args.synthetic or args.all:
        summaries["synthetic"] = run_synthetic_suite(config_path, output_root)

    if args.hydrostatic_state is not None:
        cfg = load_config(str(config_path))
        summaries["hydrostatic"] = hydrostatic_validity_diagnostic(
            args.hydrostatic_state.expanduser().resolve(), cfg,
            output_root / "hydrostatic_validity",
            wet_threshold=args.wet_threshold,
            curvature_smoothing_sigma_cells=args.curvature_sigma,
        )

    if args.parameter_sensitivity or args.all:
        cfg = load_config(str(config_path))
        sens = reviewer.get("parameter_sensitivity", {}) if isinstance(reviewer.get("parameter_sensitivity", {}), Mapping) else {}
        grid = float(args.grid_spacing if args.grid_spacing is not None else sens.get("grid_spacing_m", 20.0))
        tend = float(args.t_end if args.t_end is not None else sens.get("t_end_s", 275.0))
        wet = float(sens.get("wet_threshold_m", args.wet_threshold))
        summaries["parameter_sensitivity"] = parameter_sensitivity(
            cfg, output_root / "parameter_sensitivity",
            backend=args.backend, grid_spacing_m=grid, t_end_s=tend,
            wet_threshold_m=wet, resume=not args.no_resume,
        )

    _save_json(output_root / "reviewer_experiments_run_summary.json", summaries)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
