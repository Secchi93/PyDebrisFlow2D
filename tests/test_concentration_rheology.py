from __future__ import annotations

"""Regression checks for the concentration-aware O'Brien-Julien resistance.

Run directly with:
    python tests/test_concentration_rheology.py
"""

import copy
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


def _state(cfg, cs: float, fc: float = 0.40, h: float = 1.0, speed: float = 5.0) -> np.ndarray:
    U = np.zeros((NV, 1, 1), dtype=float)
    U[:, 0, 0] = sm.primitive_to_state(
        h, speed, 0.0, cs, 0.5, fc, fc,
        cfg.material.rho_fluid, cfg.material.rho_solid, cfg.material.max_solid_fraction,
    )
    return U


def _deceleration(cfg, cs: float, fc: float = 0.40, dt: float = 1.0e-5) -> float:
    U = _state(cfg, cs, fc=fc)
    f = composition_fields(U, cfg)
    s0 = float(f["speed"][0, 0])
    sm.apply_basal_resistance(U, np.ones((1, 1)), dt, cfg)
    s1 = float(composition_fields(U, cfg)["speed"][0, 0])
    return (s0 - s1) / dt


def main() -> int:
    cfg = load_config("configs/marsicano_multiphysics.yaml")
    assert cfg.rheology.model == "voellmy"  # backward-compatible default

    oj = copy.deepcopy(cfg)
    oj.rheology.model = "obrien_julien"
    oj.rheology.oj_transition_mode = "continuous_asymptotic"
    oj.rheology.oj_concentration_basis = "fine_matrix"
    validate_config(oj)

    # 1. Exact carrier-fluid limit: zero yield stress, configured water
    # viscosity and clear-water roughness.
    U0 = _state(oj, 0.0)
    f0 = composition_fields(U0, oj)
    p0 = sm.obrien_julien_properties(f0, oj)
    assert abs(float(p0["yield_stress_pa"][0, 0])) < 1e-14
    assert abs(float(p0["dynamic_viscosity_pas"][0, 0]) - oj.rheology.oj_water_dynamic_viscosity_pas) < 1e-14
    assert abs(float(p0["n_td"][0, 0]) - oj.rheology.oj_water_manning_n) < 1e-14

    # 2. The continuous-asymptotic branch must vary smoothly and monotonically
    # with total solid concentration. O'Brien-Julien yield/viscosity use the
    # fine-matrix proxy cs*(1-fc) by default; the turbulent/dispersive term uses
    # total cs.
    csv = np.linspace(0.0, min(0.68, oj.material.max_solid_fraction), 69)
    yields, viscosities, roughness, decels = [], [], [], []
    for cs in csv:
        U = _state(oj, float(cs))
        f = composition_fields(U, oj)
        props = sm.obrien_julien_properties(f, oj)
        assert abs(float(props["cv_rheology"][0, 0]) - float(cs) * 0.60) < 1e-12
        yields.append(float(props["yield_stress_pa"][0, 0]))
        viscosities.append(float(props["dynamic_viscosity_pas"][0, 0]))
        roughness.append(float(props["n_td"][0, 0]))
        decels.append(_deceleration(oj, float(cs)))

    def nondecreasing(values, atol=1e-10):
        return all(b >= a - atol for a, b in zip(values[:-1], values[1:]))

    assert nondecreasing(yields)
    assert nondecreasing(viscosities)
    assert nondecreasing(roughness)
    assert nondecreasing(decels, atol=1e-8)
    assert decels[-1] > decels[0]

    # 3. Explicit continuity probe around the old cs=0.20 regime boundary.
    # This concentration no longer controls the recommended law; it is retained
    # only as a diagnostic reference and for the legacy piecewise option.
    d_lo = _deceleration(oj, 0.199)
    d_mid = _deceleration(oj, 0.200)
    d_hi = _deceleration(oj, 0.201)
    assert d_lo <= d_mid <= d_hi
    local_rel_change = max(abs(d_mid - d_lo), abs(d_hi - d_mid)) / max(abs(d_mid), 1e-30)
    assert local_rel_change < 0.02

    # 4. Verify the one-step implicit update against the scalar formula used by
    # the implementation for one representative concentrated state.
    U = _state(oj, 0.55)
    f = composition_fields(U, oj)
    props = sm.obrien_julien_properties(f, oj)
    h = float(f["h"][0, 0])
    rho = float(f["rho"][0, 0])
    mass = float(f["mass"][0, 0])
    s0 = float(f["speed"][0, 0])
    tau_y = float(props["yield_stress_pa"][0, 0])
    eta = float(props["dynamic_viscosity_pas"][0, 0])
    ntd = float(props["n_td"][0, 0])
    dt = 2.5e-4
    a = tau_y / mass
    c = oj.rheology.oj_laminar_K * eta / (8.0 * rho * h * h)
    b = oj.numerics.g * ntd * ntd / (h ** (4.0 / 3.0))
    R = max(s0 - dt * a, 0.0)
    A = dt * b
    B = 1.0 + dt * c
    expected = 0.0 if R == 0.0 else (2.0 * R / (B + math.sqrt(B * B + 4.0 * A * R)) if A > 1e-14 else R / B)
    sm.apply_basal_resistance(U, np.ones((1, 1)), dt, oj)
    actual = float(composition_fields(U, oj)["speed"][0, 0])
    assert abs(actual - expected) < 1e-12

    # 5. Basal shear remains finite and concentration-sensitive.
    tau0 = float(sm.basal_shear(composition_fields(_state(oj, 0.0), oj), np.ones((1, 1)), oj)[0, 0])
    tau55 = float(sm.basal_shear(composition_fields(_state(oj, 0.55), oj), np.ones((1, 1)), oj)[0, 0])
    assert math.isfinite(tau0) and math.isfinite(tau55)
    assert tau0 > 0.0 and tau55 > tau0

    print("continuous concentration-aware O'Brien-Julien rheology regression checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
