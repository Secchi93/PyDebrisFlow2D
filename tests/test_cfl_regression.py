from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_cfl_study as cfls


def test_slug() -> None:
    assert cfls._slug(0.10) == "cfl0p1000"
    assert cfls._slug(0.075) == "cfl0p0750"


def test_summary_relative_changes() -> None:
    rows = [
        {
            "cfl": 0.10,
            "dx_m": 5.0,
            "h_max_m": 17.48,
            "h_p99_wet_m": 4.0,
            "speed_max_ms": 17.13,
            "speed_p95_wet_ms": 8.0,
            "speed_p99_wet_ms": 12.0,
            "wet_area_m2": 566350.0,
            "max_distance_from_release_centroid_m": 1998.6,
            "time_step_retries": 0.0,
            "global_first_order_fallback_fraction": 0.333,
        },
        {
            "cfl": 0.075,
            "dx_m": 5.0,
            "h_max_m": 17.506,
            "h_p99_wet_m": 4.01,
            "speed_max_ms": 16.67,
            "speed_p95_wet_ms": 7.98,
            "speed_p99_wet_ms": 11.9,
            "wet_area_m2": 569850.0,
            "max_distance_from_release_centroid_m": 1998.6,
            "time_step_retries": 0.0,
            "global_first_order_fallback_fraction": 0.3445,
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        path = cfls._write_summary(Path(td), rows)
        assert path.is_file()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            out = list(csv.DictReader(stream))
        assert len(out) == 2
        fine = min(out, key=lambda r: float(r["cfl"]))
        assert abs(float(fine["max_distance_from_release_centroid_m_rel_to_smallest_cfl_percent"])) < 1e-12
        coarse = max(out, key=lambda r: float(r["cfl"]))
        assert math.isfinite(float(coarse["speed_max_ms_adjacent_change_percent"]))


def main() -> int:
    test_slug()
    test_summary_relative_changes()
    print("cfl stability regression checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
