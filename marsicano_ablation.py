from __future__ import annotations

"""Run the 16-case Marsicano multiphysics ablation study.

The script executes the complete 2^3 erosion/deposition/segregation matrix with
first-order and second-order discretizations. It writes one crash-safe CSV after
every completed simulation, prints live comparisons against the propagation-only
baseline, generates the same six final PNG maps as a normal simulation, and
prints a final first-vs-second-order comparison. PNG files can also be generated
from resumed ``final_state.npz`` results without rerunning the solver.
"""
import os

# Boundary-condition kernels intentionally use small CUDA launch grids. Hide the
# low-occupancy warning because it is expected for these lightweight kernels.
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from pydebrisflow.config import SolverConfig, load_config, validate_config
from pydebrisflow.constants import HF
from pydebrisflow.outputs import save_maps
from pydebrisflow.physics import composition_fields
from pydebrisflow.simulation import run_solver


@dataclass(frozen=True)
class OrderSpec:
    id: str
    name: str
    space_order: int
    time_order: int


@dataclass(frozen=True)
class CaseSpec:
    id: str
    name: str
    erosion: bool
    deposition: bool
    segregation: bool


CSV_FIELDS: Tuple[str, ...] = (
    "run_index",
    "run_id",
    "order_id",
    "order_name",
    "space_order",
    "time_order",
    "case_id",
    "case_name",
    "erosion_enabled",
    "deposition_enabled",
    "segregation_enabled",
    "status",
    "reused_existing_result",
    "elapsed_s",
    "backend",
    "device",
    "t_final_s",
    "steps",
    "dx_m",
    "dy_m",
    "wet_threshold_m",
    "wet_cells",
    "wet_area_m2",
    "max_distance_from_release_centroid_m",
    "h_max_m",
    "speed_max_ms",
    "mobile_volume_m3",
    "mobile_fluid_m3",
    "mobile_solid_m3",
    "mobile_fine_m3",
    "mobile_coarse_m3",
    "mean_solid_fraction_wet",
    "mean_coarse_fraction_solid_weighted",
    "segregation_index_weighted",
    "eroded_bulk_m3",
    "deposited_bulk_m3",
    "net_bed_exchange_m3",
    "upward_coarse_m3",
    "bed_elevation_change_min_m",
    "bed_elevation_change_max_m",
    "fluid_material_residual_m3",
    "fine_material_residual_m3",
    "coarse_material_residual_m3",
    "max_abs_material_residual_m3",
    "relative_material_residual",
    "hllc_face_fallback_count",
    "global_first_order_fallback_steps",
    "time_step_retries",
    "baseline_delta_h_max_m",
    "baseline_delta_speed_max_ms",
    "baseline_delta_wet_area_m2",
    "baseline_delta_mobile_volume_m3",
    "baseline_depth_rmse_all_m",
    "baseline_depth_rmse_union_m",
    "baseline_depth_mae_union_m",
    "baseline_depth_max_abs_diff_m",
    "baseline_speed_rmse_union_ms",
    "baseline_wet_iou",
    "second_minus_first_h_max_m",
    "second_minus_first_speed_max_ms",
    "second_minus_first_wet_area_m2",
    "second_minus_first_mobile_volume_m3",
    "order_pair_depth_rmse_union_m",
    "order_pair_speed_rmse_union_ms",
    "order_pair_wet_iou",
    "output_dir",
    "error",
)


FINAL_MAP_NAMES: Tuple[str, ...] = (
    "h_final.png",
    "speed_final.png",
    "rho_final.png",
    "solid_fraction_final.png",
    "coarse_fraction_final.png",
    "segregation_final.png",
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "config_marsicano_ablation.yaml"


def _load_raw_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("The YAML root must be a mapping")
    return data


def _resolve_relative(path_value: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _parse_matrix(raw: Mapping[str, Any]) -> Tuple[List[OrderSpec], List[CaseSpec], Mapping[str, Any]]:
    section = raw.get("ablation")
    if not isinstance(section, Mapping):
        raise ValueError("Missing top-level 'ablation' section")

    orders: List[OrderSpec] = []
    for item in section.get("orders", []):
        orders.append(
            OrderSpec(
                id=str(item["id"]),
                name=str(item["name"]),
                space_order=int(item["space_order"]),
                time_order=int(item["time_order"]),
            )
        )

    cases: List[CaseSpec] = []
    for item in section.get("cases", []):
        cases.append(
            CaseSpec(
                id=str(item["id"]),
                name=str(item["name"]),
                erosion=bool(item["erosion"]),
                deposition=bool(item["deposition"]),
                segregation=bool(item["segregation"]),
            )
        )

    if len(orders) != 2:
        raise ValueError(f"Expected exactly 2 numerical orders, found {len(orders)}")
    if len(cases) != 8:
        raise ValueError(f"Expected exactly 8 ablation cases, found {len(cases)}")
    combinations = {(c.erosion, c.deposition, c.segregation) for c in cases}
    expected = {
        (e, d, s)
        for e in (False, True)
        for d in (False, True)
        for s in (False, True)
    }
    if combinations != expected:
        missing = expected - combinations
        duplicate_or_extra = combinations - expected
        raise ValueError(
            "The case matrix is not a complete 2^3 factorial design; "
            f"missing={sorted(missing)}, extra={sorted(duplicate_or_extra)}"
        )
    if not any(not c.erosion and not c.deposition and not c.segregation for c in cases):
        raise ValueError("A propagation-only P000 baseline is required")
    for order in orders:
        if order.space_order not in (1, 2) or order.time_order not in (1, 2):
            raise ValueError(f"Unsupported order specification: {order}")
    return orders, cases, section


def _build_case_config(
    base_cfg: SolverConfig,
    order: OrderSpec,
    case: CaseSpec,
    output_dir: Path,
    backend_override: Optional[str],
    t_end_override: Optional[float],
) -> SolverConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.numerics.space_order = order.space_order
    cfg.numerics.time_order = order.time_order
    cfg.erosion.enabled = case.erosion
    cfg.deposition.enabled = case.deposition
    cfg.segregation.enabled = case.segregation
    cfg.output.out_dir = str(output_dir)
    if backend_override is not None:
        cfg.compute.backend = backend_override
    if t_end_override is not None:
        if t_end_override <= 0.0:
            raise ValueError("--t-end must be positive")
        cfg.numerics.t_end = float(t_end_override)
        cfg.numerics.output_dt = float(t_end_override)
    validate_config(cfg)
    return cfg


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _config_payload(cfg: SolverConfig) -> Dict[str, Any]:
    """Convert the nested dataclass configuration to a stable JSON mapping."""
    return json.loads(json.dumps(cfg, default=lambda obj: obj.__dict__))


def _config_fingerprint(cfg: SolverConfig) -> str:
    payload = json.dumps(_config_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_run_manifest(output_dir: Path, cfg: SolverConfig, order: OrderSpec, case: CaseSpec) -> None:
    manifest = {
        "config_sha256": _config_fingerprint(cfg),
        "order_id": order.id,
        "case_id": case.id,
        "space_order": order.space_order,
        "time_order": order.time_order,
        "erosion": case.erosion,
        "deposition": case.deposition,
        "segregation": case.segregation,
    }
    temporary = output_dir / "ablation_run.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(temporary, output_dir / "ablation_run.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_state(path: Path, cfg: SolverConfig) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"x", "y", "zb", "U", "release_mask"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")
        U = np.asarray(data["U"], dtype=np.float64)
        fields = composition_fields(U, cfg)
        return {
            "x": np.asarray(data["x"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
            "zb": np.asarray(data["zb"], dtype=np.float64),
            "U": U,
            "release_mask": np.asarray(data["release_mask"], dtype=bool),
            "h": np.asarray(fields["h"], dtype=np.float64),
            "speed": np.asarray(fields["speed"], dtype=np.float64),
            "cs": np.asarray(fields["cs"], dtype=np.float64),
            "fc": np.asarray(fields["fc"], dtype=np.float64),
            "hsu": np.asarray(fields["hsu"], dtype=np.float64),
            "hsl": np.asarray(fields["hsl"], dtype=np.float64),
            "hcu": np.asarray(fields["hcu"], dtype=np.float64),
            "hcl": np.asarray(fields["hcl"], dtype=np.float64),
        }


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    local_weights = np.where(mask, np.maximum(weights, 0.0), 0.0)
    denom = float(np.sum(local_weights, dtype=np.float64))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.where(mask, values, 0.0) * local_weights, dtype=np.float64) / denom)


def _state_metrics(state: Mapping[str, np.ndarray], cfg: SolverConfig, wet_threshold: float) -> Dict[str, Any]:
    x = state["x"]
    y = state["y"]
    U = state["U"]
    h = state["h"]
    speed = state["speed"]
    cs = state["cs"]
    fc = state["fc"]
    hsu = state["hsu"]
    hsl = state["hsl"]
    hcu = state["hcu"]
    hcl = state["hcl"]
    release_mask = state["release_mask"]

    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    cell_area = dx * dy
    wet = h > wet_threshold
    solid_depth = np.maximum(hsu + hsl, 0.0)
    fine_depth = np.maximum((hsu - hcu) + (hsl - hcl), 0.0)
    coarse_depth = np.maximum(hcu + hcl, 0.0)

    phi_u = np.where(hsu > 0.0, hcu / np.maximum(hsu, 1.0e-30), 0.0)
    phi_l = np.where(hsl > 0.0, hcl / np.maximum(hsl, 1.0e-30), 0.0)
    segregation_index = _weighted_mean(
        np.abs(phi_u - phi_l), solid_depth, solid_depth > cfg.numerics.h_dry
    )

    runout = 0.0
    if np.any(wet) and np.any(release_mask):
        X, Y = np.meshgrid(x, y)
        cx = float(np.mean(X[release_mask]))
        cy = float(np.mean(Y[release_mask]))
        runout = float(np.max(np.hypot(X[wet] - cx, Y[wet] - cy)))

    return {
        "dx_m": dx,
        "dy_m": dy,
        "wet_threshold_m": wet_threshold,
        "wet_cells": int(np.sum(wet)),
        "wet_area_m2": float(np.sum(wet) * cell_area),
        "max_distance_from_release_centroid_m": runout,
        "h_max_m": float(np.max(h)),
        "speed_max_ms": float(np.max(speed)),
        "mobile_volume_m3": float(np.sum(h, dtype=np.float64) * cell_area),
        "mobile_fluid_m3": float(np.sum(U[HF], dtype=np.float64) * cell_area),
        "mobile_solid_m3": float(np.sum(solid_depth, dtype=np.float64) * cell_area),
        "mobile_fine_m3": float(np.sum(fine_depth, dtype=np.float64) * cell_area),
        "mobile_coarse_m3": float(np.sum(coarse_depth, dtype=np.float64) * cell_area),
        "mean_solid_fraction_wet": float(np.mean(cs[wet])) if np.any(wet) else 0.0,
        "mean_coarse_fraction_solid_weighted": _weighted_mean(fc, solid_depth, solid_depth > cfg.numerics.h_dry),
        "segregation_index_weighted": segregation_index,
    }


def _compact_comparison_state(state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Retain only fields needed for comparisons, using float32 to limit RAM."""
    return {
        "h": np.asarray(state["h"], dtype=np.float32).copy(),
        "speed": np.asarray(state["speed"], dtype=np.float32).copy(),
    }


def _ensure_final_maps(
    output_dir: Path,
    state: Mapping[str, np.ndarray],
    cfg: SolverConfig,
    t_final: float,
    overwrite: bool = False,
) -> List[Path]:
    """Create the same six final PNG maps produced by a normal simulation.

    The maps are generated from ``final_state.npz`` after each ablation run.
    This also works for resumed simulations, so missing figures can be created
    without repeating the expensive numerical calculation.
    """
    expected = [output_dir / name for name in FINAL_MAP_NAMES]
    if not overwrite and all(path.is_file() for path in expected):
        return expected

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = save_maps(
        str(output_dir),
        float(t_final),
        np.asarray(state["U"]),
        np.asarray(state["zb"]),
        np.asarray(state["x"]),
        np.asarray(state["y"]),
        cfg,
        "final",
    )
    return [Path(path) for path in generated]


def _field_comparison(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], wet_threshold: float) -> Dict[str, float]:
    h_ref = reference["h"]
    h_cur = candidate["h"]
    v_ref = reference["speed"]
    v_cur = candidate["speed"]
    if h_ref.shape != h_cur.shape:
        raise ValueError(f"Cannot compare different shapes: {h_ref.shape} and {h_cur.shape}")

    wet_ref = h_ref > wet_threshold
    wet_cur = h_cur > wet_threshold
    union = wet_ref | wet_cur
    intersection = wet_ref & wet_cur
    diff_h = h_cur - h_ref
    diff_v = v_cur - v_ref
    union_count = int(np.sum(union))
    union_denom = max(union_count, 1)
    union_bool_denom = int(np.sum(union))
    iou_denom = int(np.sum(union))

    return {
        "depth_rmse_all_m": float(np.sqrt(np.mean(diff_h * diff_h))),
        "depth_rmse_union_m": float(np.sqrt(np.sum((diff_h[union]) ** 2) / union_denom)) if union_count else 0.0,
        "depth_mae_union_m": float(np.mean(np.abs(diff_h[union]))) if union_count else 0.0,
        "depth_max_abs_diff_m": float(np.max(np.abs(diff_h))),
        "speed_rmse_union_ms": float(np.sqrt(np.sum((diff_v[union]) ** 2) / union_denom)) if union_bool_denom else 0.0,
        "wet_iou": float(np.sum(intersection) / iou_denom) if iou_denom else 1.0,
    }


def _metrics_row(
    run_index: int,
    order: OrderSpec,
    case: CaseSpec,
    cfg: SolverConfig,
    output_dir: Path,
    metrics: Mapping[str, Any],
    state_metrics: Mapping[str, Any],
    elapsed_s: float,
    reused: bool,
) -> Dict[str, Any]:
    residuals = [
        abs(_safe_float(metrics.get("fluid_material_residual_m3"), 0.0)),
        abs(_safe_float(metrics.get("fine_material_residual_m3"), 0.0)),
        abs(_safe_float(metrics.get("coarse_material_residual_m3"), 0.0)),
    ]
    initial_budget = metrics.get("initial_budget", {}) or {}
    total_inventory = (
        abs(_safe_float(initial_budget.get("total_fluid_m3"), 0.0))
        + abs(_safe_float(initial_budget.get("total_fine_m3"), 0.0))
        + abs(_safe_float(initial_budget.get("total_coarse_m3"), 0.0))
    )
    max_residual = max(residuals)
    device = metrics.get("cuda_device") or (
        f"CPU threads={metrics.get('cpu_threads')}" if metrics.get("backend") == "cpu" else ""
    )
    eroded = _safe_float(metrics.get("eroded_bulk_m3"), 0.0)
    deposited = _safe_float(metrics.get("deposited_bulk_m3"), 0.0)

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "run_index": run_index,
            "run_id": f"{order.id}_{case.id}",
            "order_id": order.id,
            "order_name": order.name,
            "space_order": order.space_order,
            "time_order": order.time_order,
            "case_id": case.id,
            "case_name": case.name,
            "erosion_enabled": case.erosion,
            "deposition_enabled": case.deposition,
            "segregation_enabled": case.segregation,
            "status": "success",
            "reused_existing_result": reused,
            "elapsed_s": elapsed_s,
            "backend": metrics.get("backend", ""),
            "device": device,
            "t_final_s": metrics.get("t_final_s", ""),
            "steps": metrics.get("steps", ""),
            **state_metrics,
            "eroded_bulk_m3": eroded,
            "deposited_bulk_m3": deposited,
            "net_bed_exchange_m3": deposited - eroded,
            "upward_coarse_m3": metrics.get("upward_coarse_m3", 0.0),
            "bed_elevation_change_min_m": metrics.get("bed_elevation_change_min_m", ""),
            "bed_elevation_change_max_m": metrics.get("bed_elevation_change_max_m", ""),
            "fluid_material_residual_m3": metrics.get("fluid_material_residual_m3", ""),
            "fine_material_residual_m3": metrics.get("fine_material_residual_m3", ""),
            "coarse_material_residual_m3": metrics.get("coarse_material_residual_m3", ""),
            "max_abs_material_residual_m3": max_residual,
            "relative_material_residual": max_residual / max(total_inventory, 1.0e-30),
            "hllc_face_fallback_count": metrics.get("hllc_face_fallback_count", ""),
            "global_first_order_fallback_steps": metrics.get("global_first_order_fallback_steps", ""),
            "time_step_retries": metrics.get("time_step_retries", ""),
            "output_dir": str(output_dir),
        }
    )
    return row


def _error_row(run_index: int, order: OrderSpec, case: CaseSpec, output_dir: Path, elapsed_s: float, exc: BaseException) -> Dict[str, Any]:
    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "run_index": run_index,
            "run_id": f"{order.id}_{case.id}",
            "order_id": order.id,
            "order_name": order.name,
            "space_order": order.space_order,
            "time_order": order.time_order,
            "case_id": case.id,
            "case_name": case.name,
            "erosion_enabled": case.erosion,
            "deposition_enabled": case.deposition,
            "segregation_enabled": case.segregation,
            "status": "error",
            "elapsed_s": elapsed_s,
            "output_dir": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item.get("run_index", 0) or 0)):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(temporary, path)


def _fmt(value: Any, width: int = 10, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)[:width].rjust(width)
    if not math.isfinite(number):
        return "nan".rjust(width)
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1.0e5 or magnitude < 1.0e-3):
        text = f"{number:.3e}"
    else:
        text = f"{number:.{precision}f}"
    return text[:width].rjust(width)


def _print_matrix(orders: Sequence[OrderSpec], cases: Sequence[CaseSpec], output_root: Path, csv_path: Path) -> None:
    print("\n[MATRIX] Marsicano multiphysics ablation")
    print(f"[MATRIX] output_root={output_root}")
    print(f"[MATRIX] live_csv={csv_path}")
    print("[MATRIX] 16 runs = 2 numerical orders x 8 E/D/S combinations\n")
    print(" #  run_id       order          E D S  case")
    print("--  -----------  -------------  -----  ------------------------")
    index = 1
    for order in orders:
        for case in cases:
            print(
                f"{index:2d}  {order.id}_{case.id:<7}  {order.name:<13}  "
                f"{int(case.erosion)} {int(case.deposition)} {int(case.segregation)}  {case.name}"
            )
            index += 1


def _print_live_row(row: Mapping[str, Any]) -> None:
    print(
        "[COMPARE] "
        f"{row.get('run_id', ''):<10} "
        f"hmax={_fmt(row.get('h_max_m'), 10)} m  "
        f"vmax={_fmt(row.get('speed_max_ms'), 10)} m/s  "
        f"wet={_fmt(row.get('wet_area_m2'), 11, 1)} m2  "
        f"Vmob={_fmt(row.get('mobile_volume_m3'), 11, 1)} m3  "
        f"E={_fmt(row.get('eroded_bulk_m3'), 11, 1)} m3  "
        f"D={_fmt(row.get('deposited_bulk_m3'), 11, 1)} m3  "
        f"S={_fmt(row.get('upward_coarse_m3'), 11, 1)} m3"
    )
    if row.get("baseline_wet_iou", "") != "":
        print(
            "[VS BASE] "
            f"RMSE_h={_fmt(row.get('baseline_depth_rmse_union_m'), 10)} m  "
            f"IoU={_fmt(row.get('baseline_wet_iou'), 8)}  "
            f"delta_wet={_fmt(row.get('baseline_delta_wet_area_m2'), 11, 1)} m2  "
            f"delta_Vmob={_fmt(row.get('baseline_delta_mobile_volume_m3'), 11, 1)} m3"
        )
    if row.get("order_pair_wet_iou", "") != "":
        print(
            "[O2 VS O1] "
            f"RMSE_h={_fmt(row.get('order_pair_depth_rmse_union_m'), 10)} m  "
            f"IoU={_fmt(row.get('order_pair_wet_iou'), 8)}  "
            f"delta_hmax={_fmt(row.get('second_minus_first_h_max_m'), 10)} m"
        )


def _apply_baseline_comparison(
    row: MutableMapping[str, Any],
    state: Mapping[str, np.ndarray],
    baseline_row: Mapping[str, Any],
    baseline_state: Mapping[str, np.ndarray],
    wet_threshold: float,
) -> None:
    comparison = _field_comparison(baseline_state, state, wet_threshold)
    row.update(
        {
            "baseline_delta_h_max_m": _safe_float(row.get("h_max_m"), 0.0) - _safe_float(baseline_row.get("h_max_m"), 0.0),
            "baseline_delta_speed_max_ms": _safe_float(row.get("speed_max_ms"), 0.0) - _safe_float(baseline_row.get("speed_max_ms"), 0.0),
            "baseline_delta_wet_area_m2": _safe_float(row.get("wet_area_m2"), 0.0) - _safe_float(baseline_row.get("wet_area_m2"), 0.0),
            "baseline_delta_mobile_volume_m3": _safe_float(row.get("mobile_volume_m3"), 0.0) - _safe_float(baseline_row.get("mobile_volume_m3"), 0.0),
            "baseline_depth_rmse_all_m": comparison["depth_rmse_all_m"],
            "baseline_depth_rmse_union_m": comparison["depth_rmse_union_m"],
            "baseline_depth_mae_union_m": comparison["depth_mae_union_m"],
            "baseline_depth_max_abs_diff_m": comparison["depth_max_abs_diff_m"],
            "baseline_speed_rmse_union_ms": comparison["speed_rmse_union_ms"],
            "baseline_wet_iou": comparison["wet_iou"],
        }
    )


def _apply_order_pair_comparison(
    first_row: MutableMapping[str, Any],
    second_row: MutableMapping[str, Any],
    first_state: Mapping[str, np.ndarray],
    second_state: Mapping[str, np.ndarray],
    wet_threshold: float,
) -> None:
    comparison = _field_comparison(first_state, second_state, wet_threshold)
    shared = {
        "second_minus_first_h_max_m": _safe_float(second_row.get("h_max_m"), 0.0) - _safe_float(first_row.get("h_max_m"), 0.0),
        "second_minus_first_speed_max_ms": _safe_float(second_row.get("speed_max_ms"), 0.0) - _safe_float(first_row.get("speed_max_ms"), 0.0),
        "second_minus_first_wet_area_m2": _safe_float(second_row.get("wet_area_m2"), 0.0) - _safe_float(first_row.get("wet_area_m2"), 0.0),
        "second_minus_first_mobile_volume_m3": _safe_float(second_row.get("mobile_volume_m3"), 0.0) - _safe_float(first_row.get("mobile_volume_m3"), 0.0),
        "order_pair_depth_rmse_union_m": comparison["depth_rmse_union_m"],
        "order_pair_speed_rmse_union_ms": comparison["speed_rmse_union_ms"],
        "order_pair_wet_iou": comparison["wet_iou"],
    }
    first_row.update(shared)
    second_row.update(shared)


def _print_final_tables(rows: Sequence[Mapping[str, Any]], cases: Sequence[CaseSpec]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    print("\n" + "=" * 122)
    print("FINAL ABLATION COMPARISON AGAINST THE PROPAGATION-ONLY BASELINE")
    print("=" * 122)
    print(
        "run_id     hmax[m]   vmax[m/s]   wet_area[m2]   mobile_V[m3]   "
        "eroded[m3]  deposited[m3]  seg_up[m3]   RMSE_h[m]   IoU"
    )
    print("-" * 122)
    for row in sorted(successful, key=lambda item: int(item["run_index"])):
        print(
            f"{str(row['run_id']):<10} "
            f"{_fmt(row.get('h_max_m'), 9)} "
            f"{_fmt(row.get('speed_max_ms'), 11)} "
            f"{_fmt(row.get('wet_area_m2'), 14, 1)} "
            f"{_fmt(row.get('mobile_volume_m3'), 14, 1)} "
            f"{_fmt(row.get('eroded_bulk_m3'), 12, 1)} "
            f"{_fmt(row.get('deposited_bulk_m3'), 14, 1)} "
            f"{_fmt(row.get('upward_coarse_m3'), 11, 1)} "
            f"{_fmt(row.get('baseline_depth_rmse_union_m', 0.0), 11)} "
            f"{_fmt(row.get('baseline_wet_iou', 1.0), 7)}"
        )

    by_run = {(str(row.get("order_id")), str(row.get("case_id"))): row for row in successful}
    print("\n" + "=" * 102)
    print("SECOND ORDER MINUS FIRST ORDER, MATCHED BY PHYSICAL CASE")
    print("=" * 102)
    print("case       delta_hmax[m]  delta_vmax[m/s]  delta_wet[m2]  RMSE_h[m]  speed_RMSE  wet_IoU")
    print("-" * 102)
    for case in cases:
        second = by_run.get(("O2", case.id))
        first = by_run.get(("O1", case.id))
        row = second or first
        if row is None:
            continue
        print(
            f"{case.id:<10} "
            f"{_fmt(row.get('second_minus_first_h_max_m'), 14)} "
            f"{_fmt(row.get('second_minus_first_speed_max_ms'), 17)} "
            f"{_fmt(row.get('second_minus_first_wet_area_m2'), 14, 1)} "
            f"{_fmt(row.get('order_pair_depth_rmse_union_m'), 10)} "
            f"{_fmt(row.get('order_pair_speed_rmse_union_ms'), 11)} "
            f"{_fmt(row.get('order_pair_wet_iou'), 8)}"
        )


def _existing_result_is_complete(output_dir: Path, cfg: SolverConfig) -> bool:
    metrics_path = output_dir / "metrics.json"
    state_path = output_dir / "final_state.npz"
    manifest_path = output_dir / "ablation_run.json"
    if not (metrics_path.is_file() and state_path.is_file() and manifest_path.is_file()):
        return False
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return False
    return manifest.get("config_sha256") == _config_fingerprint(cfg)


def _select(items: Sequence[Any], selected: Optional[Sequence[str]], attribute: str) -> List[Any]:
    if not selected:
        return list(items)
    allowed = {value.strip() for value in selected if value.strip()}
    result = [item for item in items if str(getattr(item, attribute)) in allowed]
    found = {str(getattr(item, attribute)) for item in result}
    missing = allowed - found
    if missing:
        raise ValueError(f"Unknown selections for {attribute}: {sorted(missing)}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 8 first-order and 8 second-order Marsicano multiphysics ablation tests."
    )
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Base YAML configuration.")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default=None, help="Override compute.backend.")
    parser.add_argument("--restart", action="store_true", help="Delete the ablation output directory before running.")
    parser.add_argument("--no-resume", action="store_true", help="Re-run cases even when metrics.json and final_state.npz exist.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately after the first failed simulation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the 16-case matrix without solving.")
    parser.add_argument("--orders", nargs="*", metavar="ORDER_ID", help="Optional subset, e.g. --orders O1 or --orders O2.")
    parser.add_argument("--cases", nargs="*", metavar="CASE_ID", help="Optional subset, e.g. --cases P000 E100 EDS111.")
    parser.add_argument("--t-end", type=float, default=None, help="Optional short final time for smoke testing.")
    parser.add_argument(
        "--make-png",
        dest="make_png",
        action="store_true",
        default=None,
        help="Generate the six normal final maps for every selected ablation run.",
    )
    parser.add_argument(
        "--no-make-png",
        dest="make_png",
        action="store_false",
        help="Do not generate final maps, overriding the YAML ablation setting.",
    )
    parser.add_argument(
        "--overwrite-png",
        action="store_true",
        help="Regenerate final maps even when all six PNG files already exist.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    raw = _load_raw_yaml(config_path)
    orders_all, cases_all, ablation_cfg = _parse_matrix(raw)
    orders = _select(orders_all, args.orders, "id")
    cases = _select(cases_all, args.cases, "id")

    output_root = _resolve_relative(
        str(ablation_cfg.get("output_root", "../outputs/Marsicano_Ablation")),
        config_path.parent,
    )
    csv_path = output_root / str(ablation_cfg.get("csv_name", "marsicano_ablation_results.csv"))
    wet_threshold = float(ablation_cfg.get("wet_threshold_m", 0.01))
    resume = bool(ablation_cfg.get("resume_completed", True)) and not args.no_resume
    make_final_png = (
        bool(ablation_cfg.get("make_final_png", True))
        if args.make_png is None
        else bool(args.make_png)
    )
    overwrite_final_png = bool(ablation_cfg.get("overwrite_final_png", False)) or args.overwrite_png

    if args.restart and output_root.exists():
        print(f"[RESET] removing {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(str(config_path))
    if base_cfg.material.initial_solid_fraction <= 0.0:
        raise ValueError("The ablation release must contain solids; initial_solid_fraction is zero")
    if not (0.0 < base_cfg.release.solid_fraction <= base_cfg.material.max_solid_fraction):
        raise ValueError("release.solid_fraction must be positive for deposition and segregation tests")
    if base_cfg.erosion.erodible_depth_m <= 0.0:
        raise ValueError("erosion.erodible_depth_m must be positive for erosion tests")

    _print_matrix(orders, cases, output_root, csv_path)
    if args.dry_run:
        print("\n[DRY RUN] Configuration and factorial matrix are valid. No simulation was executed.")
        return 0

    rows: List[Dict[str, Any]] = []
    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    states_by_key: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    baseline_by_order: Dict[str, Tuple[Dict[str, Any], Dict[str, np.ndarray]]] = {}

    total_runs = len(orders) * len(cases)
    run_index = 0
    for order in orders:
        for case in cases:
            run_index += 1
            run_id = f"{order.id}_{case.id}"
            output_dir = output_root / order.name / f"{case.id}_{case.name}"
            cfg = _build_case_config(base_cfg, order, case, output_dir, args.backend, args.t_end)

            print("\n" + "#" * 122)
            print(
                f"[ABLATION {run_index:02d}/{total_runs:02d}] {run_id}: {order.name}, "
                f"erosion={case.erosion}, deposition={case.deposition}, segregation={case.segregation}"
            )
            print(f"[ABLATION] output={output_dir}")
            print("#" * 122)

            start = time.perf_counter()
            reused = resume and _existing_result_is_complete(output_dir, cfg)
            try:
                if reused:
                    print("[RESUME] Reusing complete metrics.json and final_state.npz")
                    metrics = _load_json(output_dir / "metrics.json")
                else:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    metrics = run_solver(cfg)
                    _write_run_manifest(output_dir, cfg, order, case)
                elapsed = time.perf_counter() - start
                state_full = _load_state(output_dir / "final_state.npz", cfg)
                if make_final_png:
                    map_paths = _ensure_final_maps(
                        output_dir,
                        state_full,
                        cfg,
                        _safe_float(metrics.get("t_final_s"), cfg.numerics.t_end),
                        overwrite=overwrite_final_png,
                    )
                    print(f"[PNG] final comparison maps ready ({len(map_paths)}): {output_dir}")
                diagnostics = _state_metrics(state_full, cfg, wet_threshold)
                state = _compact_comparison_state(state_full)
                del state_full
                row = _metrics_row(
                    run_index, order, case, cfg, output_dir, metrics, diagnostics, elapsed, reused
                )

                if case.id == "P000":
                    baseline_by_order[order.id] = (row, state)
                else:
                    baseline = baseline_by_order.get(order.id)
                    if baseline is None:
                        raise RuntimeError(
                            f"The propagation-only baseline for {order.id} must run before {case.id}"
                        )
                    _apply_baseline_comparison(row, state, baseline[0], baseline[1], wet_threshold)

                # The baseline compares with itself exactly.
                if case.id == "P000":
                    _apply_baseline_comparison(row, state, row, state, wet_threshold)

                rows.append(row)
                rows_by_key[(order.id, case.id)] = row
                states_by_key[(order.id, case.id)] = state

                first_row = rows_by_key.get(("O1", case.id))
                second_row = rows_by_key.get(("O2", case.id))
                first_state = states_by_key.get(("O1", case.id))
                second_state = states_by_key.get(("O2", case.id))
                if first_row is not None and second_row is not None and first_state is not None and second_state is not None:
                    _apply_order_pair_comparison(
                        first_row, second_row, first_state, second_state, wet_threshold
                    )

                _write_csv(csv_path, rows)
                _print_live_row(row)
                print(f"[CSV] updated after run {run_index}: {csv_path}")
            except Exception as exc:
                elapsed = time.perf_counter() - start
                row = _error_row(run_index, order, case, output_dir, elapsed, exc)
                rows.append(row)
                rows_by_key[(order.id, case.id)] = row
                _write_csv(csv_path, rows)
                print(f"[ERROR] {row['error']}", file=sys.stderr)
                print(f"[CSV] failure recorded in {csv_path}", file=sys.stderr)
                if args.fail_fast:
                    raise

    # Recompute all pair comparisons at the end in case some cases were reused
    # or the user selected a nonstandard execution subset/order.
    for case in cases_all:
        first_row = rows_by_key.get(("O1", case.id))
        second_row = rows_by_key.get(("O2", case.id))
        first_state = states_by_key.get(("O1", case.id))
        second_state = states_by_key.get(("O2", case.id))
        if (
            first_row is not None
            and second_row is not None
            and first_state is not None
            and second_state is not None
            and first_row.get("status") == "success"
            and second_row.get("status") == "success"
        ):
            _apply_order_pair_comparison(first_row, second_row, first_state, second_state, wet_threshold)

    _write_csv(csv_path, rows)
    _print_final_tables(rows, cases_all)
    failures = [row for row in rows if row.get("status") != "success"]
    print(f"\n[DONE] successful={len(rows) - len(failures)} failed={len(failures)}")
    print(f"[DONE] final CSV: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
