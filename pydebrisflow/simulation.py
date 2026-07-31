from __future__ import annotations

"""High-level simulation orchestration."""

import json
import os
from typing import Any, Dict, List

import numpy as np
import yaml

from .config import SolverConfig
from .compute import configure_cpu_threads, resolve_backend
from .constants import HCL, HCU, HF, HSL, HSU, NV
from .dem import load_or_build_dem
from .geometry import prepare_interactive_geometry, save_setup_preview
from .numerics import (
    FluxWorkspace,
    apply_boundary,
    check_admissible,
    compute_dt,
    conservative_roundoff_repair,
    transport_rhs,
    transport_step_ssprk2,
)
from .outputs import make_depth_gif, save_maps, save_snapshot
from .physics import (
    apply_erosion_deposition,
    apply_segregation,
    apply_speed_cap,
    apply_voellmy_friction,
    build_release,
    composition_fields,
    material_budget,
    terrain_geometry,
)

def run_solver_cpu(cfg: SolverConfig) -> Dict[str, Any]:
    cpu_threads = configure_cpu_threads(cfg)
    print(f"[BACKEND] CPU / Numba parallel, threads={cpu_threads}")
    x, y, zb0, valid = load_or_build_dem(cfg.grid)
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    ny, nx = zb0.shape
    os.makedirs(cfg.output.out_dir, exist_ok=True)
    mu_override, xi_override = prepare_interactive_geometry(cfg, x, y, zb0, valid, dx, dy)
    ng = 2
    shape = (ny + 2 * ng, nx + 2 * ng)
    U = np.zeros((NV, *shape), dtype=np.float64)
    zb = np.zeros(shape, dtype=np.float64)
    active = np.zeros(shape, dtype=np.uint8)
    core = (slice(ng, ng + ny), slice(ng, ng + nx))
    # Invalid DEM cells are high barriers but marked inactive for reflective face treatment.
    zmax = float(np.nanmax(zb0[valid]))
    zfill = np.where(valid, zb0, zmax + cfg.grid.nodata_barrier_m)
    zb[core] = zfill
    active[core] = valid.astype(np.uint8)
    Uc = U[:, core[0], core[1]]
    release_mask = build_release(Uc, x, y, valid, cfg)
    if cfg.output.save_setup_preview:
        save_setup_preview(cfg.output.out_dir, cfg, x, y, zb0, valid, dx, dy, release_mask)

    e = cfg.erosion
    bed_total_solid = (1.0 - e.bed_porosity) * e.erodible_depth_m * valid.astype(np.float64)
    bed_coarse = bed_total_solid * e.bed_coarse_fraction
    bed_fine = bed_total_solid - bed_coarse
    peak_tau = np.zeros((ny, nx), dtype=np.float64)

    with open(os.path.join(cfg.output.out_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=lambda o: o.__dict__)), f, sort_keys=False)

    ws = FluxWorkspace(shape)
    apply_boundary(U, zb, active, cfg.numerics, ng)
    # JIT warmup on real shapes.
    transport_rhs(U, zb, active, cfg, ws, dx, dy, order=1, flux_name="hll")

    cell_area = dx * dy
    budget0 = material_budget(Uc, bed_fine, bed_coarse, cell_area, cfg)
    t = 0.0
    step = 0
    output_index = 0
    next_output = 0.0
    snapshots: List[str] = []
    fallback_faces = 0
    global_fallback_steps = 0
    retries_total = 0
    source_totals = {"eroded_bulk_m3": 0.0, "deposited_bulk_m3": 0.0, "upward_coarse_m3": 0.0}
    boundary_component_change = np.zeros(5, dtype=np.float64)
    z_initial = zb[core].copy()

    while t < cfg.numerics.t_end - 1.0e-12:
        if t + 1.0e-12 >= next_output:
            if cfg.output.save_snapshots:
                snapshots.append(save_snapshot(cfg.output.out_dir, output_index, t, Uc, zb[core], x, y, cfg))
            output_index += 1
            next_output += cfg.numerics.output_dt

        dt = min(compute_dt(U, core, dx, dy, cfg), cfg.numerics.t_end - t)
        accepted = False
        for retry in range(cfg.numerics.max_step_retries + 1):
            Utrial = U.copy()
            zb_trial = zb.copy()
            bed_f_trial = bed_fine.copy()
            bed_c_trial = bed_coarse.copy()
            peak_trial = peak_tau.copy()
            zcore = zb_trial[core]
            Ucore = Utrial[:, core[0], core[1]]
            _, _, cosbeta = terrain_geometry(zcore, dx, dy)
            # Strang source half-step.
            apply_voellmy_friction(Ucore, cosbeta, 0.5 * dt, cfg, mu_override, xi_override)
            b1 = apply_erosion_deposition(Ucore, zcore, bed_f_trial, bed_c_trial, peak_trial, cosbeta, 0.5 * dt, cfg, mu_override, xi_override)
            s1 = apply_segregation(Ucore, 0.5 * dt, cfg)
            apply_speed_cap(Ucore, cfg)
            apply_boundary(Utrial, zb_trial, active, cfg.numerics, ng)
            Utrans, fb, global_fb, boundary_rate = transport_step_ssprk2(Utrial, dt, zb_trial, active, cfg, ws, dx, dy, core)
            if Utrans is None:
                dt *= 0.5
                retries_total += 1
                continue
            Utrial = Utrans
            Ucore = Utrial[:, core[0], core[1]]
            zcore = zb_trial[core]
            _, _, cosbeta = terrain_geometry(zcore, dx, dy)
            b2 = apply_erosion_deposition(Ucore, zcore, bed_f_trial, bed_c_trial, peak_trial, cosbeta, 0.5 * dt, cfg, mu_override, xi_override)
            s2 = apply_segregation(Ucore, 0.5 * dt, cfg)
            apply_voellmy_friction(Ucore, cosbeta, 0.5 * dt, cfg, mu_override, xi_override)
            apply_speed_cap(Ucore, cfg)
            if not check_admissible(Utrial, core, cfg.numerics.positivity_tolerance):
                dt *= 0.5
                retries_total += 1
                continue
            conservative_roundoff_repair(Utrial, core, cfg.numerics.h_dry, cfg.numerics.positivity_tolerance)
            U = Utrial
            zb = zb_trial
            bed_fine = bed_f_trial
            bed_coarse = bed_c_trial
            peak_tau = peak_trial
            fallback_faces += fb
            boundary_component_change += dt * boundary_rate
            global_fallback_steps += int(global_fb)
            source_totals["eroded_bulk_m3"] += (b1["eroded_bulk"] + b2["eroded_bulk"]) * cell_area
            source_totals["deposited_bulk_m3"] += (b1["deposited_bulk"] + b2["deposited_bulk"]) * cell_area
            source_totals["upward_coarse_m3"] += (s1["upward_coarse"] + s2["upward_coarse"]) * cell_area
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Step failed after {cfg.numerics.max_step_retries} retries at t={t:.6g}")
        t += dt
        step += 1
        if cfg.output.progress_every_steps > 0 and step % cfg.output.progress_every_steps == 0:
            f = composition_fields(U[:, core[0], core[1]], cfg)
            print(f"[RUN] step={step} t={t:.3f}/{cfg.numerics.t_end:.3f} dt={dt:.3g} "
                  f"hmax={np.max(f['h']):.3g} vmax={np.max(f['speed']):.3g} "
                  f"rho=[{np.min(f['rho']):.1f},{np.max(f['rho']):.1f}] retries={retries_total}")

    Uc = U[:, core[0], core[1]]
    if cfg.output.save_snapshots:
        snapshots.append(save_snapshot(cfg.output.out_dir, output_index, t, Uc, zb[core], x, y, cfg))
    if cfg.output.make_png:
        save_maps(cfg.output.out_dir, t, Uc, zb[core], x, y, cfg, "final")
    if cfg.output.make_gif:
        if snapshots:
            make_depth_gif(snapshots, os.path.join(cfg.output.out_dir, "depth_evolution.gif"), cfg)
        else:
            print("[GIF][WARN] make_gif=true requires save_snapshots=true")

    budget1 = material_budget(Uc, bed_fine, bed_coarse, cell_area, cfg)
    boundary_fluid = float(boundary_component_change[HF])
    boundary_fine = float(boundary_component_change[HSU] - boundary_component_change[HCU] + boundary_component_change[HSL] - boundary_component_change[HCL])
    boundary_coarse = float(boundary_component_change[HCU] + boundary_component_change[HCL])
    fluid_residual = budget1["total_fluid_m3"] - budget0["total_fluid_m3"] - boundary_fluid
    fine_residual = budget1["total_fine_m3"] - budget0["total_fine_m3"] - boundary_fine
    coarse_residual = budget1["total_coarse_m3"] - budget0["total_coarse_m3"] - boundary_coarse
    f = composition_fields(Uc, cfg)
    metrics = {
        "backend": "cpu",
        "cpu_threads": cpu_threads,
        "t_final_s": t,
        "steps": step,
        "dx_m": dx,
        "dy_m": dy,
        "shape": [ny, nx],
        "release_cells": int(np.sum(release_mask)),
        "friction_override_cells": int(np.sum(np.isfinite(mu_override))),
        "friction_regions": cfg.friction.regions,
        "release_volume_m3": float(cfg.release.volume_m3),
        "h_max_m": float(np.max(f["h"])),
        "speed_max_ms": float(np.max(f["speed"])),
        "rho_min_wet_kgm3": float(np.min(f["rho"][f["wet"]])) if np.any(f["wet"]) else cfg.material.rho_fluid,
        "rho_max_wet_kgm3": float(np.max(f["rho"][f["wet"]])) if np.any(f["wet"]) else cfg.material.rho_fluid,
        "bed_elevation_change_min_m": float(np.min(zb[core] - z_initial)),
        "bed_elevation_change_max_m": float(np.max(zb[core] - z_initial)),
        "hllc_face_fallback_count": int(fallback_faces),
        "global_first_order_fallback_steps": int(global_fallback_steps),
        "time_step_retries": int(retries_total),
        "initial_budget": budget0,
        "final_budget": budget1,
        "boundary_fluid_change_m3": boundary_fluid,
        "boundary_fine_change_m3": boundary_fine,
        "boundary_coarse_change_m3": boundary_coarse,
        "fluid_material_residual_m3": fluid_residual,
        "fine_material_residual_m3": fine_residual,
        "coarse_material_residual_m3": coarse_residual,
        **source_totals,
    }
    with open(os.path.join(cfg.output.out_dir, "metrics.json"), "w", encoding="utf-8") as fjson:
        json.dump(metrics, fjson, indent=2)
    np.savez_compressed(os.path.join(cfg.output.out_dir, "final_state.npz"), x=x, y=y, zb=zb[core], U=Uc,
                        bed_fine_solid=bed_fine, bed_coarse_solid=bed_coarse, peak_tau=peak_tau,
                        mu_override=mu_override, xi_override=xi_override, release_mask=release_mask.astype(np.uint8))
    print(json.dumps(metrics, indent=2))
    return metrics


def run_solver(cfg: SolverConfig) -> Dict[str, Any]:
    """Run the solver using the selected CPU or CUDA backend."""
    info = resolve_backend(cfg)
    print(f"[BACKEND] requested={info.requested} selected={info.selected} reason={info.reason}")
    if info.selected == "cuda":
        from .cuda_backend import run_solver_cuda
        return run_solver_cuda(cfg, device_name=info.cuda_device_name)
    return run_solver_cpu(cfg)
