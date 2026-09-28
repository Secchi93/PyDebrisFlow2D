from __future__ import annotations

"""Lightweight regression checks for the post-review experiment package.

Run directly with:
    python tests/test_run_reviewer_diagnostics.py
"""

import math
import tempfile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydebrisflow.config import load_config
from pydebrisflow.dem import load_or_build_dem

# Repository root is already importable when launched from the project root.
import run_marsicano_studies as ma
import run_reviewer_diagnostics as rexp


def main() -> int:
    cfg = load_config("configs/marsicano_multiphysics.yaml")
    raw = ma._load_raw_yaml(Path("configs/marsicano_multiphysics.yaml").resolve())
    assert abs(float(raw["grid_sensitivity"]["cfl"]) - 0.10) < 1.0e-12
    x, y, z, valid = load_or_build_dem(cfg.grid)
    assert z.shape == (800, 1200)
    assert x.size == 1200 and y.size == 800
    assert valid.all()
    assert abs((x[1] - x[0]) - 5.0) < 1.0e-12

    # Synthetic p=2 sequence: q(h)=1 + 1e-4 h^2.
    g = ma._gci_triplet(1.0 + 1e-4 * 5.0**2, 1.0 + 1e-4 * 10.0**2, 1.0 + 1e-4 * 20.0**2)
    assert g["convergence_type"] == "monotonic_asymptotic_candidate"
    assert abs(float(g["apparent_order"]) - 2.0) < 1.0e-12
    assert math.isfinite(float(g["gci_fine_percent"]))


    # Flow-aligned bottleneck geometry: one fixed physical transect sampled on a regular grid.
    import numpy as np
    xg = np.linspace(0.0, 100.0, 21)
    yg = np.linspace(0.0, 100.0, 21)
    Xg, Yg = np.meshgrid(xg, yg)
    hg = np.exp(-((Xg - 50.0) ** 2 + (Yg - 50.0) ** 2) / 500.0)
    ug = np.full_like(hg, 2.0)
    vg = np.full_like(hg, 4.0)
    state = {
        "x": xg, "y": yg, "h": hg, "u": ug, "v": vg,
        "speed": np.hypot(ug, vg), "zb": 100.0 - Yg,
    }
    geom = ma._aligned_bottleneck_geometry(
        state,
        {"y_m": 50.0, "xmin_m": 20.0, "xmax_m": 80.0, "search_half_height_m": 30.0,
         "half_length_m": 40.0, "sample_spacing_m": 2.0},
        0.01,
    )
    bm = ma._bottleneck_metrics_for_state_aligned(state, geom, wet_threshold=0.01)
    assert bm["valid_sample_fraction"] == 1.0
    assert bm["wet_width_m"] > 0.0
    assert math.isfinite(bm["normal_discharge_m3s"])
    assert 0.0 <= bm["speed_p95_wet_ms"] <= bm["speed_p99_wet_ms"] <= bm["speed_max_ms"]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        s = rexp.segregation_analytical_benchmark(root / "seg")
        assert s["pass"] is True
        assert s["max_constituent_conservation_residual_m"] < 1.0e-12
        f = rexp.fluid_content_resistance_diagnostic(root / "fluid", cfg)
        assert f["pure_water_mu"] == cfg.material.mu_fine
        assert f["pure_water_xi_ms2"] == cfg.material.xi_fine
        assert f["oj_clear_water_limit_pass"] is True
        assert f["oj_resistance_changes_with_cs_in_test"] is True
        assert f["oj_deceleration_monotonic_with_cs_in_test"] is True
        assert f["oj_transition_mode"] == "continuous_asymptotic"
        assert abs(f["oj_highest_cs_friction_deceleration_ms2"] - f["oj_clear_water_friction_deceleration_ms2"]) > 1.0e-6

    print("reviewer experiment regression checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
