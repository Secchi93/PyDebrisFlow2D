from __future__ import annotations

"""Unified verification entry point for PyDebrisFlow2D.

"""

# Boundary-condition kernels intentionally use small CUDA launch grids. Hide the
# low-occupancy warning because it is expected for these lightweight kernels.
import os 
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from numba import cuda
from pydebrisflow.config import SolverConfig, validate_config
from pydebrisflow.constants import HF, HSL, HSU, MX, NV
from pydebrisflow.cuda_backend import CudaWorkspace, _compute_dt_device, _grid2, _transport_step_device
from pydebrisflow.numerics import FluxWorkspace, compute_dt, transport_step_ssprk2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydebrisflow as sm
import yaml

from pydebrisflow.compute import configure_cpu_threads, resolve_backend
from pydebrisflow.config import load_config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_CONFIG = REPO_ROOT / "configs" / "config_tests.yaml"
DEFAULT_MARSICANO_CONFIG = REPO_ROOT / "configs" / "config_marsicano.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "verification_figures"
LIMITER_CHOICES = ("minmod", "mc", "vanleer", "superbee")


def savefig(fig: plt.Figure, stem: str) -> list[str]:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labelled_axes: list[tuple[str, plt.Axes]] = []
    for ax in fig.axes:
        for artist in ax.texts:
            label = artist.get_text().strip().lower()
            if label in {"(a)", "(b)", "(c)", "(d)"}:
                labelled_axes.append((label[1], ax))
                break

    saved: list[str] = []
    if not labelled_axes:
        path = OUT_DIR / f"{stem}.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        saved.append(str(path))
    else:
        original_size = fig.get_size_inches().copy()
        original_visibility = {ax: ax.get_visible() for ax in fig.axes}
        original_positions = {ax: ax.get_position().frozen() for ax in fig.axes}
        try:
            fig.set_layout_engine(None)
            for suffix, primary_ax in labelled_axes:
                primary_bounds = np.asarray(original_positions[primary_ax].bounds)
                panel_axes = [
                    ax for ax in fig.axes
                    if np.allclose(
                        np.asarray(original_positions[ax].bounds),
                        primary_bounds,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                ]
                for ax in fig.axes:
                    ax.set_visible(ax in panel_axes)
                fig.set_size_inches(6.4, 4.8, forward=True)
                width = 0.70 if len(panel_axes) > 1 else 0.80
                target = [0.14, 0.15, width, 0.74]
                for ax in panel_axes:
                    ax.set_position(target)
                path = OUT_DIR / f"{stem}_{suffix}.png"
                fig.savefig(path, dpi=240, bbox_inches="tight")
                saved.append(str(path))
        finally:
            fig.set_size_inches(original_size, forward=True)
            for ax in fig.axes:
                ax.set_visible(original_visibility[ax])
                ax.set_position(original_positions[ax])

    plt.close(fig)
    return saved
def _artist_points_in_axes(
    source_ax: plt.Axes,
    reference_ax: plt.Axes,
    max_points_per_artist: int = 800,
) -> np.ndarray:
  
    chunks: list[np.ndarray] = []
    to_axes = reference_ax.transAxes.inverted()

    for line in source_ax.lines:
        try:
            x = np.ma.asarray(line.get_xdata(orig=False)).filled(np.nan).ravel()
            y = np.ma.asarray(line.get_ydata(orig=False)).filled(np.nan).ravel()
            n = min(x.size, y.size)
            if n == 0:
                continue
            x, y = x[:n], y[:n]
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            if x.size == 0:
                continue
            if x.size > max_points_per_artist:
                take = np.linspace(0, x.size - 1, max_points_per_artist, dtype=int)
                x, y = x[take], y[take]
            display = line.get_transform().transform(np.column_stack((x, y)))
            chunks.append(to_axes.transform(display))
        except Exception:
            continue

    for collection in source_ax.collections:
        try:
            offsets = np.ma.asarray(collection.get_offsets()).filled(np.nan)
            if offsets.ndim != 2 or offsets.shape[1] < 2 or offsets.size == 0:
                continue
            offsets = offsets[:, :2]
            finite = np.all(np.isfinite(offsets), axis=1)
            offsets = offsets[finite]
            if offsets.shape[0] > max_points_per_artist:
                take = np.linspace(0, offsets.shape[0] - 1, max_points_per_artist, dtype=int)
                offsets = offsets[take]
            display = collection.get_offset_transform().transform(offsets)
            chunks.append(to_axes.transform(display))
        except Exception:
            continue

    # Bars and other patches are represented by their display-space corners.
    try:
        renderer = reference_ax.figure.canvas.get_renderer()
        for patch in source_ax.patches:
            if not patch.get_visible():
                continue
            bbox = patch.get_window_extent(renderer=renderer)
            if bbox.width <= 0.0 or bbox.height <= 0.0:
                continue
            corners = np.array([
                [bbox.x0, bbox.y0], [bbox.x0, bbox.y1],
                [bbox.x1, bbox.y0], [bbox.x1, bbox.y1],
                [(bbox.x0 + bbox.x1) * 0.5, (bbox.y0 + bbox.y1) * 0.5],
            ])
            chunks.append(to_axes.transform(corners))
    except Exception:
        pass

    if not chunks:
        return np.empty((0, 2), dtype=float)
    points = np.vstack(chunks)
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def _bbox_overlap_area(a, b) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def smart_legend(
    ax: plt.Axes,
    handles=None,
    labels=None,
    *,
    extra_axes: tuple[plt.Axes, ...] = (),
    ncol: int = 1,
    fontsize=None,
    frameon: bool = True,
    allow_outside: bool = True,
    loc=None,
    bbox_to_anchor=None,
    **kwargs,
):

    if handles is None or labels is None:
        auto_handles, auto_labels = ax.get_legend_handles_labels()
        if handles is None:
            handles = auto_handles
        if labels is None:
            labels = auto_labels
    handles, labels = list(handles), list(labels)
    if not handles:
        return None

    if loc is not None:
        fixed_kwargs = dict(kwargs)
        if bbox_to_anchor is not None:
            fixed_kwargs["bbox_to_anchor"] = bbox_to_anchor
            fixed_kwargs.setdefault("bbox_transform", ax.transAxes)
        fixed_kwargs.setdefault("borderaxespad", 0.0)
        return ax.legend(
            handles, labels, loc=loc, ncol=ncol, fontsize=fontsize,
            frameon=frameon, **fixed_kwargs,
        )

    fig = ax.figure
    fig.canvas.draw()
    points = [_artist_points_in_axes(ax, ax)]
    points.extend(_artist_points_in_axes(other, ax) for other in extra_axes)
    points = np.vstack([p for p in points if p.size]) if any(p.size for p in points) else np.empty((0, 2))

    candidates = (
        "upper right", "lower right", "lower center",
    )
    preference_penalty = {
        "upper right": 0.000,
        "lower right": 0.004,
        "lower center": 0.008,
    }

    best_loc, best_score = candidates[0], float("inf")
    for loc in candidates:
        legend = ax.legend(
            handles, labels, loc=loc, ncol=ncol, fontsize=fontsize,
            frameon=frameon, **kwargs,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        expanded = bbox.expanded(1.05, 1.10)

        if points.size:
            inside = (
                (points[:, 0] >= expanded.x0) & (points[:, 0] <= expanded.x1) &
                (points[:, 1] >= expanded.y0) & (points[:, 1] <= expanded.y1)
            )
            score = float(np.count_nonzero(inside)) / max(1.0, math.sqrt(float(points.shape[0])))
        else:
            score = 0.0

        # Avoid pre-existing result annotations and other text boxes.
        for artist in ax.texts:
            if not artist.get_visible():
                continue
            try:
                text_bbox = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
                area = max(expanded.width * expanded.height, 1.0e-12)
                score += 2.0 * _bbox_overlap_area(expanded, text_bbox) / area
            except Exception:
                pass

        score += preference_penalty[loc]
        legend.remove()
        if score < best_score:
            best_loc, best_score = loc, score


    if allow_outside and best_score > 0.025:
        return ax.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            borderaxespad=0.0, ncol=ncol, fontsize=fontsize,
            frameon=frameon, **kwargs,
        )

    return ax.legend(
        handles, labels, loc=best_loc, ncol=ncol,
        fontsize=fontsize, frameon=frameon, **kwargs,
    )


def panel(ax: plt.Axes, txt: str) -> None:
    ax.annotate(
        txt,
        xy=(-0.015, 1.02),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        annotation_clip=False,
        clip_on=False,
    )


def status(ax: plt.Axes, txt: str, passed: bool = True, loc=None, y_offset: float = 0.0) -> None:
    display_text = txt if passed else "FAIL\n" + txt
    fig = ax.figure
    fig.canvas.draw()
    points = _artist_points_in_axes(ax, ax)
    renderer = fig.canvas.get_renderer()

    lines = max(1, display_text.count("\n") + 1)
    width = min(0.42, 0.20 + 0.008 * max(len(s) for s in display_text.splitlines()))
    height = min(0.34, 0.055 + 0.050 * lines)
    candidates = {
        "lower right": (1.0 - width - 0.015, 0.015, "left", "bottom"),
        "lower left": (0.015, 0.015, "left", "bottom"),
        "upper right": (1.0 - width - 0.015, 1.0 - height - 0.015, "left", "bottom"),
        "upper left": (0.015, 1.0 - height - 0.015, "left", "bottom"),
    }
    if loc is not None:
        if loc not in candidates:
            raise ValueError(f"Unknown status location: {loc}")
        best_name = loc
    else:
        best_name, best_score = "lower right", float("inf")
        for name, (x0, y0, _, _) in candidates.items():
            from matplotlib.transforms import Bbox
            bbox = Bbox.from_bounds(x0, y0, width, height)
            if points.size:
                inside = (
                    (points[:, 0] >= bbox.x0) & (points[:, 0] <= bbox.x1) &
                    (points[:, 1] >= bbox.y0) & (points[:, 1] <= bbox.y1)
                )
                score = float(np.count_nonzero(inside)) / max(1.0, math.sqrt(float(points.shape[0])))
            else:
                score = 0.0
            legend = ax.get_legend()
            if legend is not None:
                lb = legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
                score += 3.0 * _bbox_overlap_area(bbox, lb) / max(bbox.width * bbox.height, 1.0e-12)
            if score < best_score:
                best_name, best_score = name, score

    x0, y0, ha, va = candidates[best_name]
    y0 += y_offset
    ax.text(
        x0, y0, display_text,
        transform=ax.transAxes,
        ha=ha, va=va, fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.4", alpha=0.95),
        zorder=20,
    )

def depth(U: np.ndarray) -> np.ndarray:
    return np.maximum(U[sm.HF], 0.0) + np.maximum(U[sm.HSU], 0.0) + np.maximum(U[sm.HSL], 0.0)


def coarse_fraction(U: np.ndarray, cfg: sm.SolverConfig) -> np.ndarray:
    f = sm.composition_fields(U, cfg)
    return f["fc"]


def solid_inventory(U: np.ndarray, bed_f: np.ndarray, bed_c: np.ndarray) -> Dict[str, float]:
    mobile_fine = float(np.sum(np.maximum(U[sm.HSU] - U[sm.HCU], 0.0) + np.maximum(U[sm.HSL] - U[sm.HCL], 0.0)))
    mobile_coarse = float(np.sum(np.maximum(U[sm.HCU], 0.0) + np.maximum(U[sm.HCL], 0.0)))
    return {
        "mobile_fine": mobile_fine,
        "mobile_coarse": mobile_coarse,
        "bed_fine": float(np.sum(bed_f)),
        "bed_coarse": float(np.sum(bed_c)),
    }


# -----------------------------------------------------------------------------
# Lightweight built-in numerical verification suite.
# -----------------------------------------------------------------------------
def _flat_test_config() -> sm.SolverConfig:
    """Return the common flat-grid configuration used by verification tests."""
    cfg = sm.SolverConfig()
    cfg.numerics.bc_left = "reflective"
    cfg.numerics.bc_right = "reflective"
    cfg.numerics.bc_bottom = "reflective"
    cfg.numerics.bc_top = "reflective"
    cfg.numerics.space_order = 2
    cfg.numerics.time_order = 2
    cfg.numerics.flux = "hllc"
    cfg.erosion.enabled = False
    cfg.deposition.enabled = False
    cfg.segregation.enabled = False
    return cfg


def self_test(verbose: bool = True) -> Dict[str, float]:
    results: Dict[str, float] = {}
    cfg = _flat_test_config()
    m, n = cfg.material, cfg.numerics

    # 1) Flux consistency.
    U0 = sm.primitive_to_state(
        1.2, 2.0, -0.5, 0.50, 0.5, 0.3, 0.3,
        m.rho_fluid, m.rho_solid, m.max_solid_fraction,
    )
    Fp = sm.physical_flux(
        U0, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid, n.h_dry,
    )
    Fh, _ = sm.hllc_flux_state(
        U0, U0, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid,
        n.h_dry, n.hllc_dry_factor, n.hllc_max_froude,
    )
    results["hllc_consistency"] = float(np.max(np.abs(Fp - Fh)))
    assert results["hllc_consistency"] < 1.0e-10

    # 2) Lake at rest over a smooth bump, constant composition.
    nx, ny, ng = 44, 36, 2
    shape = (ny + 2 * ng, nx + 2 * ng)
    x = np.linspace(-20.0, 20.0, nx)
    y = np.linspace(-15.0, 15.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    zc = 0.5 * np.exp(-(X * X + Y * Y) / 50.0)
    eta0 = 2.0
    U = np.zeros((sm.NV, *shape))
    zb = np.zeros(shape)
    active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng + ny), slice(ng, ng + nx))
    zb[core] = zc
    h = eta0 - zc
    state = np.zeros((sm.NV, ny, nx))
    for j in range(ny):
        for i in range(nx):
            state[:, j, i] = sm.primitive_to_state(
                h[j, i], 0.0, 0.0, 0.5, 0.5, 0.4, 0.4,
                m.rho_fluid, m.rho_solid, m.max_solid_fraction,
            )
    U[:, core[0], core[1]] = state
    sm.apply_boundary(U, zb, active, n)
    ws = sm.FluxWorkspace(shape)
    rhs, _, _ = sm.transport_rhs(U, zb, active, cfg, ws, dx, dy)
    results["lake_at_rest_rhs"] = float(
        np.max(np.abs(rhs[:, core[0], core[1]]))
    )
    assert results["lake_at_rest_rhs"] < 5.0e-8

    # 3) Closed-box segregation conserves fine/coarse and enriches the upper layer.
    Uc = np.zeros((sm.NV, 1, 1))
    Uc[:, 0, 0] = sm.primitive_to_state(
        1.0, 4.0, 0.0, 0.6, 0.5, 0.2, 0.8,
        m.rho_fluid, m.rho_solid, m.max_solid_fraction,
    )
    fine0 = float(
        (Uc[sm.HSU] - Uc[sm.HCU] + Uc[sm.HSL] - Uc[sm.HCL])[0, 0]
    )
    coarse0 = float((Uc[sm.HCU] + Uc[sm.HCL])[0, 0])
    cfg.segregation.enabled = True
    sm.apply_segregation(Uc, 0.1, cfg)
    fine1 = float(
        (Uc[sm.HSU] - Uc[sm.HCU] + Uc[sm.HSL] - Uc[sm.HCL])[0, 0]
    )
    coarse1 = float((Uc[sm.HCU] + Uc[sm.HCL])[0, 0])
    results["segregation_fine_residual"] = fine1 - fine0
    results["segregation_coarse_residual"] = coarse1 - coarse0
    assert abs(results["segregation_fine_residual"]) < 1.0e-12
    assert abs(results["segregation_coarse_residual"]) < 1.0e-12
    assert Uc[sm.HCU, 0, 0] > 0.2 * Uc[sm.HSU, 0, 0]

    # 4) Erosion conserves each solid class together with the bed inventory.
    cfg.erosion.enabled = True
    cfg.erosion.model = "excess_shear"
    cfg.erosion.critical_shear_pa = 1.0
    cfg.erosion.excess_shear_rate_ms = 0.02
    cfg.deposition.enabled = False
    Uc = np.zeros((sm.NV, 2, 2))
    for j in range(2):
        for i in range(2):
            Uc[:, j, i] = sm.primitive_to_state(
                1.0, 5.0, 0.0, 0.5, 0.5, 0.4, 0.4,
                m.rho_fluid, m.rho_solid, m.max_solid_fraction,
            )
    bed_f = np.full((2, 2), 0.3)
    bed_c = np.full((2, 2), 0.3)
    z = np.zeros((2, 2))
    peak = np.zeros((2, 2))
    cosb = np.ones((2, 2))
    fine_before = float(
        np.sum((Uc[sm.HSU] - Uc[sm.HCU]) + (Uc[sm.HSL] - Uc[sm.HCL]) + bed_f)
    )
    coarse_before = float(np.sum(Uc[sm.HCU] + Uc[sm.HCL] + bed_c))
    sm.apply_erosion_deposition(Uc, z, bed_f, bed_c, peak, cosb, 0.1, cfg)
    fine_after = float(
        np.sum((Uc[sm.HSU] - Uc[sm.HCU]) + (Uc[sm.HSL] - Uc[sm.HCL]) + bed_f)
    )
    coarse_after = float(np.sum(Uc[sm.HCU] + Uc[sm.HCL] + bed_c))
    results["erosion_fine_residual"] = fine_after - fine_before
    results["erosion_coarse_residual"] = coarse_after - coarse_before
    assert abs(results["erosion_fine_residual"]) < 1.0e-12
    assert abs(results["erosion_coarse_residual"]) < 1.0e-12

    # 5) Variable-density mixture relation.
    Ud = sm.primitive_to_state(
        1.0, 0.0, 0.0, 0.50, 0.5, 0.2, 0.2,
        m.rho_fluid, m.rho_solid, m.max_solid_fraction,
    )
    rhod = sm.state_derived(
        Ud, m.rho_fluid, m.rho_solid, n.h_dry,
    )[-1]
    results["density_mixture_error"] = float(
        rhod - 0.5 * (m.rho_fluid + m.rho_solid)
    )
    assert abs(results["density_mixture_error"]) < 1.0e-12

    # 6) Deposition conserves saturated fluid/fine/coarse inventories.
    cfgd = _flat_test_config()
    cfgd.erosion.enabled = False
    cfgd.deposition.enabled = True
    cfgd.deposition.critical_shear_pa = 1.0e9
    Ucd = np.zeros((sm.NV, 2, 2))
    for j in range(2):
        for i in range(2):
            Ucd[:, j, i] = sm.primitive_to_state(
                1.0, 0.0, 0.0, 0.5, 0.5, 0.4, 0.4,
                m.rho_fluid, m.rho_solid, m.max_solid_fraction,
            )
    bed_fd = np.zeros((2, 2))
    bed_cd = np.zeros((2, 2))
    zd = np.zeros((2, 2))
    pkd = np.zeros((2, 2))
    cbd = np.ones((2, 2))
    bd0 = sm.material_budget(Ucd, bed_fd, bed_cd, 1.0, cfgd)
    sm.apply_erosion_deposition(Ucd, zd, bed_fd, bed_cd, pkd, cbd, 0.5, cfgd)
    bd1 = sm.material_budget(Ucd, bed_fd, bed_cd, 1.0, cfgd)
    results["deposition_fluid_residual"] = (
        bd1["total_fluid_m3"] - bd0["total_fluid_m3"]
    )
    results["deposition_fine_residual"] = (
        bd1["total_fine_m3"] - bd0["total_fine_m3"]
    )
    results["deposition_coarse_residual"] = (
        bd1["total_coarse_m3"] - bd0["total_coarse_m3"]
    )
    assert abs(results["deposition_fluid_residual"]) < 1.0e-12
    assert abs(results["deposition_fine_residual"]) < 1.0e-12
    assert abs(results["deposition_coarse_residual"]) < 1.0e-12

    # 7) HLLC automatically falls back at a wet-dry face.
    Ud0 = np.zeros(sm.NV, dtype=np.float64)
    _, dry_fb = sm.hllc_flux_state(
        U0, Ud0, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid,
        n.h_dry, n.hllc_dry_factor, n.hllc_max_froude,
    )
    results["hllc_dry_fallback"] = float(dry_fb)
    assert dry_fb == 1


    nyf, nxf = 5, 6
    Uf = np.zeros((sm.NV, nyf, nxf), dtype=np.float64)
    zbf = np.zeros((nyf, nxf), dtype=np.float64)
    activef = np.ones((nyf, nxf), dtype=np.uint8)
    Pf = np.zeros((7, nyf, nxf), dtype=np.float64)
    sxf = np.zeros_like(Pf)
    syf = np.zeros_like(Pf)
    fxf = np.zeros((sm.NV, nyf, nxf + 1), dtype=np.float64)
    fyf = np.zeros((sm.NV, nyf + 1, nxf), dtype=np.float64)
    cxl = np.zeros((nyf, nxf + 1), dtype=np.float64)
    cxr = np.zeros_like(cxl)
    cyl = np.zeros((nyf + 1, nxf), dtype=np.float64)
    cyr = np.zeros_like(cyl)
    counts = np.zeros(nyf + nxf + 4, dtype=np.int64)
    jf, iface = 2, 3
    Pf[0, jf, iface - 1] = 1.0
    Pf[0, jf, iface] = 5.0e-3
    sxf[0, jf, iface - 1] = 0.4
    sm.compute_face_fluxes(
        Uf, zbf, activef, Pf, sxf, syf, fxf, fyf,
        cxl, cxr, cyl, cyr, 2, sm.FLUX_HLLC,
        n.g, m.rho_fluid, m.rho_solid, m.max_solid_fraction,
        n.h_dry, n.hllc_dry_factor, n.hllc_max_froude,
        n.hybrid_depth_rel_jump, counts,
    )
    UL0, _ = sm.build_face_state(
        1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, m.rho_fluid, m.rho_solid,
        m.max_solid_fraction, n.g,
    )
    UR0, _ = sm.build_face_state(
        5.0e-3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, m.rho_fluid, m.rho_solid,
        m.max_solid_fraction, n.g,
    )
    expected_local_hll = sm.hll_flux_state(
        UL0, UR0, 1.0, 0.0, n.g,
        m.rho_fluid, m.rho_solid, n.h_dry,
    )
    local_fallback_error = float(
        np.max(np.abs(fxf[:, jf, iface] - expected_local_hll))
    )
    assert counts[jf] >= 1
    assert local_fallback_error < 1.0e-12

    # 8) Short dam-break positivity and material conservation on a flat grid.
    cfg = _flat_test_config()
    nx, ny, ng = 60, 12, 2
    shape = (ny + 2 * ng, nx + 2 * ng)
    U = np.zeros((sm.NV, *shape))
    zb = np.zeros(shape)
    active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng + ny), slice(ng, ng + nx))
    for j in range(ny):
        for i in range(nx):
            hh = 1.0 if i < nx // 3 else 0.02
            U[:, ng + j, ng + i] = sm.primitive_to_state(
                hh, 0.0, 0.0, 0.5, 0.5, 0.35, 0.35,
                m.rho_fluid, m.rho_solid, m.max_solid_fraction,
            )
    ws = sm.FluxWorkspace(shape)
    vol0 = np.sum(
        U[[sm.HF, sm.HSU, sm.HSL], core[0], core[1]], dtype=np.float64,
    )
    for _ in range(20):
        dt = min(sm.compute_dt(U, core, 1.0, 1.0, cfg), 0.02)
        Un, _, _, _ = sm.transport_step_ssprk2(
            U, dt, zb, active, cfg, ws, 1.0, 1.0, core,
        )
        assert Un is not None
        U = Un
    vol1 = np.sum(
        U[[sm.HF, sm.HSU, sm.HSL], core[0], core[1]], dtype=np.float64,
    )
    results["dam_break_volume_rel"] = float((vol1 - vol0) / vol0)
    results["dam_break_min_component"] = float(
        np.min(U[[sm.HF, sm.HSU, sm.HCU, sm.HSL, sm.HCL], core[0], core[1]])
    )
    assert abs(results["dam_break_volume_rel"]) < 5.0e-9
    assert results["dam_break_min_component"] >= -1.0e-12

    # 9) One-step open-boundary component budget matches integrated face flux.
    cfgb = _flat_test_config()
    cfgb.numerics.space_order = 1
    cfgb.numerics.time_order = 1
    cfgb.numerics.flux = "hll"
    cfgb.numerics.bc_left = "reflective"
    cfgb.numerics.bc_right = "outflow"
    cfgb.numerics.bc_bottom = "reflective"
    cfgb.numerics.bc_top = "reflective"
    nx, ny, ng = 20, 6, 2
    shape = (ny + 2 * ng, nx + 2 * ng)
    Ub = np.zeros((sm.NV, *shape))
    zbb = np.zeros(shape)
    ab = np.ones(shape, dtype=np.uint8)
    coreb = (slice(ng, ng + ny), slice(ng, ng + nx))
    for j in range(ny):
        for i in range(nx):
            Ub[:, ng + j, ng + i] = sm.primitive_to_state(
                0.5, 1.0, 0.0, 0.4, 0.5, 0.3, 0.3,
                m.rho_fluid, m.rho_solid, m.max_solid_fraction,
            )
    wsb = sm.FluxWorkspace(shape)
    before = np.sum(Ub[:5, coreb[0], coreb[1]], axis=(1, 2))
    dtb = 0.01
    Ubn, _, _, brb = sm.transport_step_ssprk2(
        Ub, dtb, zbb, ab, cfgb, wsb, 1.0, 1.0, coreb,
    )
    assert Ubn is not None
    after = np.sum(Ubn[:5, coreb[0], coreb[1]], axis=(1, 2))
    results["open_boundary_budget_error"] = float(
        np.max(np.abs((after - before) - dtb * brb))
    )
    assert results["open_boundary_budget_error"] < 1.0e-10

    if verbose:
        print(json.dumps(results, indent=2))
    return results


def flat_cfg() -> sm.SolverConfig:
    return _flat_test_config()


def fig01_hllc_consistency() -> Dict[str, float]:
    cfg = flat_cfg(); m, n = cfg.material, cfg.numerics
    alpha = np.linspace(0.05, 1.0, 40)
    errors = []
    rep_phys = rep_hllc = None
    for k, a in enumerate(alpha):
        U = sm.primitive_to_state(
            0.5 + 1.0 * a,
            1.4 * np.cos(0.7 + a),
            0.5 * np.sin(1.2 * a),
            0.10 + 0.55 * a,
            0.35 + 0.25 * a,
            0.15 + 0.20 * a,
            0.12 + 0.22 * a,
            m.rho_fluid, m.rho_solid, m.max_solid_fraction,
        )
        Fp = sm.physical_flux(U, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid, n.h_dry)
        Fh, _ = sm.hllc_flux_state(U, U, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid,
                                   n.h_dry, n.hllc_dry_factor, n.hllc_max_froude)
        errors.append(float(np.max(np.abs(Fp - Fh))))
        if k == len(alpha)//2:
            rep_phys = Fp.copy(); rep_hllc = Fh.copy()
    metric = float(np.max(errors))
    x = np.arange(sm.NV)
    labels = [r"$h_f$", r"$h_{s,u}$", r"$h_{c,u}$", r"$h_{s,l}$", r"$h_{c,l}$", r"$M_x$", r"$M_y$"]

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(x, rep_phys, marker="o", linewidth=1.8, label="Physical flux")
    axes[0].plot(x, rep_hllc, marker="s", linestyle="--", linewidth=1.4, label="HLLC flux")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Flux component")
    axes[0].grid(alpha=0.25)
    smart_legend(axes[0], loc="upper left", bbox_to_anchor=(0.02, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].semilogy(alpha, errors, marker="o", markersize=3, linewidth=1.8)
    axes[1].axhline(1.0e-10, linestyle="--", label="Tolerance")
    axes[1].set_xlabel(r"State-family parameter $\alpha$")
    axes[1].set_ylabel(r"$\max |F_{phys}-F_{HLLC}|$")
    axes[1].grid(alpha=0.25)
    smart_legend(axes[1], loc="upper right", bbox_to_anchor=(0.98, 0.86), allow_outside=False)
    status(axes[1], f"max error = {metric:.3e}", metric < 1.0e-10, loc="lower right")
    panel(axes[1], "(b)")
    savefig(fig, "test_01_hllc_consistency")
    return {"metric": metric, "threshold": 1.0e-10}


def lake_setup():
    cfg = flat_cfg(); m = cfg.material
    nx, ny, ng = 44, 36, 2
    shape = (ny + 2*ng, nx + 2*ng)
    x = np.linspace(-20.0, 20.0, nx)
    y = np.linspace(-15.0, 15.0, ny)
    dx, dy = x[1]-x[0], y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    zb_core = 0.5 * np.exp(-(X*X + Y*Y)/50.0)
    eta0 = 2.0
    U = np.zeros((sm.NV, *shape))
    zb = np.zeros(shape)
    active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng+ny), slice(ng, ng+nx))
    zb[core] = zb_core
    h = eta0 - zb_core
    for j in range(ny):
        for i in range(nx):
            U[:, ng+j, ng+i] = sm.primitive_to_state(h[j, i], 0.0, 0.0, 0.5, 0.5, 0.4, 0.4,
                                                     m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    sm.apply_boundary(U, zb, active, cfg.numerics)
    ws = sm.FluxWorkspace(shape)
    rhs, fallback, _ = sm.transport_rhs(U, zb, active, cfg, ws, dx, dy)
    return x, y, zb_core, h, eta0, rhs[:, core[0], core[1]], fallback


def fig02_lake_at_rest() -> Dict[str, float]:
    x, y, zb_core, h, eta0, rhs, fallback = lake_setup()
    resid = np.max(np.abs(rhs), axis=0)
    maxres = float(np.max(resid))
    mid = len(y)//2
    centerline = resid[mid, :]
    max_over_y = np.max(resid, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(x, zb_core[mid], linewidth=2.0, label=r"Bed elevation $z_b$")
    axes[0].plot(x, zb_core[mid] + h[mid], linewidth=1.8, label=r"Free surface $\eta=h+z_b$")
    axes[0].plot(x, eta0*np.ones_like(x), linestyle="--", linewidth=1.2, label=r"Exact $\eta_0$")
    axes[0].fill_between(x, 0.0, zb_core[mid], alpha=0.2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("Elevation [m]")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(-0.35, 2.10)
    smart_legend(axes[0], loc="upper right", bbox_to_anchor=(0.98, 0.88), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].semilogy(x, np.maximum(centerline, 1e-30), linewidth=1.8, label="Centerline residual")
    axes[1].semilogy(x, np.maximum(max_over_y, 1e-30), linestyle="--", linewidth=1.5, label="Maximum residual over y")
    axes[1].axhline(5.0e-8, linestyle="--", label="Tolerance")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel(r"$\max_k |\partial_t U_k|$")
    axes[1].grid(alpha=0.25)
    smart_legend(axes[1], loc="lower right", bbox_to_anchor=(0.98, 0.05), allow_outside=False)
    status(axes[1], f"max residual = {maxres:.3e}\nlocal fallbacks = {fallback}", maxres < 5.0e-8, loc="lower left")
    panel(axes[1], "(b)")
    savefig(fig, "test_02_lake_at_rest")
    return {"metric": maxres, "threshold": 5.0e-8}


def fig03_segregation() -> Dict[str, float]:
    cfg = flat_cfg(); m = cfg.material
    cfg.segregation.enabled = True
    U = np.zeros((sm.NV, 1, 1))
    U[:, 0, 0] = sm.primitive_to_state(1.0, 4.0, 0.0, 0.6, 0.5, 0.2, 0.8,
                                       m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    dt = 0.05
    t = [0.0]
    upper_c = [float(U[sm.HCU, 0, 0])]
    lower_c = [float(U[sm.HCL, 0, 0])]
    upper_f = [float((U[sm.HSU] - U[sm.HCU])[0, 0])]
    lower_f = [float((U[sm.HSL] - U[sm.HCL])[0, 0])]
    c0 = upper_c[0] + lower_c[0]
    f0 = upper_f[0] + lower_f[0]
    cres = [0.0]; fres = [0.0]
    for k in range(1, 31):
        sm.apply_segregation(U, dt, cfg)
        t.append(k*dt)
        upper_c.append(float(U[sm.HCU, 0, 0]))
        lower_c.append(float(U[sm.HCL, 0, 0]))
        upper_f.append(float((U[sm.HSU] - U[sm.HCU])[0, 0]))
        lower_f.append(float((U[sm.HSL] - U[sm.HCL])[0, 0]))
        cres.append((upper_c[-1]+lower_c[-1]) - c0)
        fres.append((upper_f[-1]+lower_f[-1]) - f0)
    metric = float(max(np.max(np.abs(cres)), np.max(np.abs(fres))))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(t, upper_c, linewidth=1.8, label="Coarse in upper layer")
    axes[0].plot(t, lower_c, linewidth=1.8, label="Coarse in lower layer")
    axes[0].plot(t, upper_f, linestyle="--", linewidth=1.5, label="Fine in upper layer")
    axes[0].plot(t, lower_f, linestyle="--", linewidth=1.5, label="Fine in lower layer")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Volumetric depth [m]")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(0.05, 0.36)
    smart_legend(axes[0], ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(t, cres, linewidth=1.8, label="Coarse residual")
    axes[1].plot(t, fres, linewidth=1.8, label="Fine residual")
    axes[1].axhline(1.0e-12, linestyle="--", label="Tolerance")
    axes[1].axhline(-1.0e-12, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Conservation residual [m]")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-1.40e-12, 1.45e-12)
    smart_legend(axes[1], loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    status(axes[1], f"max residual = {metric:.1e}", metric < 1.0e-12, loc="lower left")
    panel(axes[1], "(b)")
    savefig(fig, "test_03_segregation")
    return {"metric": metric, "threshold": 1.0e-12}


def fig04_erosion() -> Dict[str, float]:
    cfg = flat_cfg(); m = cfg.material
    cfg.erosion.enabled = True
    cfg.erosion.model = "excess_shear"
    cfg.erosion.critical_shear_pa = 1.0
    cfg.erosion.excess_shear_rate_ms = 0.02
    cfg.deposition.enabled = False
    U = np.zeros((sm.NV, 2, 2))
    for j in range(2):
        for i in range(2):
            U[:, j, i] = sm.primitive_to_state(1.0, 5.0, 0.0, 0.5, 0.5, 0.4, 0.4,
                                               m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    bed_f = np.full((2, 2), 0.30)
    bed_c = np.full((2, 2), 0.30)
    z = np.zeros((2, 2)); peak = np.zeros((2, 2)); cosb = np.ones((2, 2))
    inv0 = solid_inventory(U, bed_f, bed_c)
    t = [0.0]; mf = [inv0['mobile_fine']]; mc = [inv0['mobile_coarse']]; bf = [inv0['bed_fine']]; bc = [inv0['bed_coarse']]; zbm=[0.0]
    f0 = inv0['mobile_fine'] + inv0['bed_fine']; c0 = inv0['mobile_coarse'] + inv0['bed_coarse']
    fres=[0.0]; cres=[0.0]
    dt = 0.05
    for k in range(1, 21):
        sm.apply_erosion_deposition(U, z, bed_f, bed_c, peak, cosb, dt, cfg)
        inv = solid_inventory(U, bed_f, bed_c)
        t.append(k*dt); mf.append(inv['mobile_fine']); mc.append(inv['mobile_coarse']); bf.append(inv['bed_fine']); bc.append(inv['bed_coarse']); zbm.append(float(np.mean(z)))
        fres.append(inv['mobile_fine'] + inv['bed_fine'] - f0)
        cres.append(inv['mobile_coarse'] + inv['bed_coarse'] - c0)
    metric = float(max(np.max(np.abs(fres)), np.max(np.abs(cres))))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(t, mf, linewidth=1.8, label="Mobile fine")
    axes[0].plot(t, bf, linewidth=1.8, linestyle="--", label="Bed fine")
    axes[0].plot(t, mc, linewidth=1.8, label="Mobile coarse")
    axes[0].plot(t, bc, linewidth=1.8, linestyle="--", label="Bed coarse")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Solid inventory [m over unit area]")
    axes[0].grid(alpha=0.25)
    axr = axes[0].twinx()
    axr.plot(t, zbm, linestyle=":", linewidth=1.5, label=r"Mean bed elevation $\overline{z_b}$")
    axr.set_ylabel(r"$\overline{z_b}$ [m]")
    axes[0].set_ylim(0.75, 1.82)
    axr.set_ylim(-0.055, 0.085)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    smart_legend(axes[0], handles=lines1 + lines2, labels=labels1 + labels2,
                 extra_axes=(axr,), ncol=2, loc="upper right",
                 bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(t, fres, linewidth=1.8, label="Fine residual")
    axes[1].plot(t, cres, linewidth=1.8, label="Coarse residual")
    axes[1].axhline(1.0e-12, linestyle="--", label="Tolerance")
    axes[1].axhline(-1.0e-12, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Conservation residual")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-1.35e-12, 1.50e-12)
    smart_legend(axes[1], loc="upper left", bbox_to_anchor=(0.03, 0.93), allow_outside=False)
    status(axes[1], f"max residual = {metric:.2e}\nfinal mean bed change = {zbm[-1]:.3e} m", metric < 1.0e-12, loc="lower right")
    panel(axes[1], "(b)")
    savefig(fig, "test_04_erosion")
    return {"metric": metric, "threshold": 1.0e-12}


def fig05_density_closure() -> Dict[str, float]:
    cfg = flat_cfg(); m, n = cfg.material, cfg.numerics
    cs = np.linspace(0.0, m.max_solid_fraction, 150)
    rho_exact = (1.0 - cs) * m.rho_fluid + cs * m.rho_solid
    rho_num = []
    errors = []
    for c in cs:
        U = sm.primitive_to_state(1.0, 0.0, 0.0, c, 0.5, 0.2, 0.2, m.rho_fluid, m.rho_solid, m.max_solid_fraction)
        rho = sm.state_derived(U, m.rho_fluid, m.rho_solid, n.h_dry)[-1]
        rho_num.append(float(rho)); errors.append(float(rho - ((1.0-c)*m.rho_fluid + c*m.rho_solid)))
    rho_num = np.array(rho_num); errors = np.array(errors)
    metric = float(np.max(np.abs(errors)))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(cs, rho_exact, linewidth=1.8, label="Exact closure")
    axes[0].plot(cs, rho_num, linestyle="--", linewidth=1.5, label="Numerical value")
    axes[0].set_xlabel(r"Solid volume fraction $c_s$")
    axes[0].set_ylabel(r"Mixture density $\rho$ [kg m$^{-3}$]")
    axes[0].grid(alpha=0.25)
    smart_legend(axes[0], loc="lower right", bbox_to_anchor=(0.98, 0.05), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(cs, errors, linewidth=1.8)
    axes[1].axhline(1.0e-12, linestyle="--", label="Tolerance")
    axes[1].axhline(-1.0e-12, linestyle="--")
    axes[1].set_xlabel(r"Solid volume fraction $c_s$")
    axes[1].set_ylabel(r"$\rho_{num}-\rho_{exact}$")
    axes[1].grid(alpha=0.25)
    smart_legend(axes[1], loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    status(axes[1], f"max error = {metric:.1e}", metric < 1.0e-12, loc="lower right")
    panel(axes[1], "(b)")
    savefig(fig, "test_05_density_closure")
    return {"metric": metric, "threshold": 1.0e-12}


def fig06_active_deposition() -> Dict[str, float]:
    cfg = flat_cfg(); m = cfg.material
    cfg.erosion.enabled = False
    cfg.deposition.enabled = True
    cfg.deposition.critical_shear_pa = 1.0e9
    cfg.material.grain_diameter_fine_m = 0.01
    cfg.material.grain_diameter_coarse_m = 0.10
    U = np.zeros((sm.NV, 2, 2))
    for j in range(2):
        for i in range(2):
            U[:, j, i] = sm.primitive_to_state(1.0, 0.0, 0.0, 0.55, 0.5, 0.4, 0.4,
                                               cfg.material.rho_fluid, cfg.material.rho_solid, cfg.material.max_solid_fraction)
    bed_f = np.zeros((2, 2)); bed_c = np.zeros((2, 2)); z = np.zeros((2, 2)); peak = np.zeros((2, 2)); cosb = np.ones((2, 2))
    budget0 = sm.material_budget(U, bed_f, bed_c, 1.0, cfg)
    t = [0.0]
    mobile_f = [budget0['mobile_fine_m3']]; mobile_c=[budget0['mobile_coarse_m3']]
    bed_fine=[budget0['bed_fine_solid_m3']]; bed_coarse=[budget0['bed_coarse_solid_m3']]
    fluid_res=[0.0]; fine_res=[0.0]; coarse_res=[0.0]; zmean=[0.0]
    dt = 0.5
    for k in range(1, 21):
        sm.apply_erosion_deposition(U, z, bed_f, bed_c, peak, cosb, dt, cfg)
        bud = sm.material_budget(U, bed_f, bed_c, 1.0, cfg)
        t.append(k*dt)
        mobile_f.append(bud['mobile_fine_m3']); mobile_c.append(bud['mobile_coarse_m3'])
        bed_fine.append(bud['bed_fine_solid_m3']); bed_coarse.append(bud['bed_coarse_solid_m3'])
        fluid_res.append(bud['total_fluid_m3'] - budget0['total_fluid_m3'])
        fine_res.append(bud['total_fine_m3'] - budget0['total_fine_m3'])
        coarse_res.append(bud['total_coarse_m3'] - budget0['total_coarse_m3'])
        zmean.append(float(np.mean(z)))
    metric = float(max(np.max(np.abs(fluid_res)), np.max(np.abs(fine_res)), np.max(np.abs(coarse_res))))

    d_mobile_f = np.array(mobile_f) - mobile_f[0]
    d_mobile_c = np.array(mobile_c) - mobile_c[0]
    d_bed_f = np.array(bed_fine) - bed_fine[0]
    d_bed_c = np.array(bed_coarse) - bed_coarse[0]

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(t, d_mobile_f, linewidth=1.8, label="Δ mobile fine")
    axes[0].plot(t, d_bed_f, linewidth=1.8, linestyle="--", label="Δ bed fine")
    axes[0].plot(t, d_mobile_c, linewidth=1.8, label="Δ mobile coarse")
    axes[0].plot(t, d_bed_c, linewidth=1.8, linestyle="--", label="Δ bed coarse")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Inventory change [m$^3$]")
    axes[0].grid(alpha=0.25)
    axr = axes[0].twinx(); axr.plot(t, zmean, linestyle=":", linewidth=1.5, label=r"Mean bed elevation $\overline{z_b}$")
    axr.set_ylabel(r"$\overline{z_b}$ [m]")
    axes[0].set_ylim(-0.007, 0.015)
    axr.set_ylim(-0.0002, 0.0090)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    smart_legend(axes[0], handles=lines1 + lines2, labels=labels1 + labels2,
                 extra_axes=(axr,), ncol=2, loc="upper right",
                 bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(t, fluid_res, linewidth=1.8, label="Fluid residual")
    axes[1].plot(t, fine_res, linewidth=1.8, label="Fine residual")
    axes[1].plot(t, coarse_res, linewidth=1.8, label="Coarse residual")
    axes[1].axhline(1.0e-12, linestyle="--", label="Tolerance")
    axes[1].axhline(-1.0e-12, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Conservation residual [m$^3$]")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-1.50e-12, 1.60e-12)
    smart_legend(axes[1], ncol=2, loc="upper left", bbox_to_anchor=(0.02, 0.96), allow_outside=False)
    status(axes[1], f"max residual = {metric:.2e}\nfinal mean bed rise = {zmean[-1]:.3e} m", metric < 1.0e-12, loc="lower right")
    panel(axes[1], "(b)")
    savefig(fig, "test_06_active_deposition")
    return {"metric": metric, "threshold": 1.0e-12}


def fig07_wet_dry_fallback() -> Dict[str, float]:
    cfg = flat_cfg(); m, n = cfg.material, cfg.numerics
    wet = sm.primitive_to_state(1.2, 2.0, -0.5, 0.50, 0.5, 0.3, 0.3,
                                m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    dry_depths = np.geomspace(1.0e-12, 2.0e-1, 60)
    diffs = []; flags = []
    for hd in dry_depths:
        right = sm.primitive_to_state(hd, 0.0, 0.0, 0.20, 0.5, 0.1, 0.1,
                                      m.rho_fluid, m.rho_solid, m.max_solid_fraction)
        returned, fallback = sm.hllc_flux_state(wet, right, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid,
                                                n.h_dry, n.hllc_dry_factor, n.hllc_max_froude)
        href = sm.hll_flux_state(wet, right, 1.0, 0.0, n.g, m.rho_fluid, m.rho_solid, n.h_dry)
        diffs.append(float(np.max(np.abs(returned - href)))); flags.append(float(fallback))
    diffs = np.array(diffs); flags = np.array(flags)
    metric = float(np.max(np.where(flags > 0.5, diffs, 0.0)))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].loglog(dry_depths, np.maximum(diffs, 1.0e-30), marker="o", markersize=3, linewidth=1.8)
    axes[0].axhline(1.0e-12, linestyle="--", label="Tolerance")
    axes[0].set_xlabel(r"Right depth $h_R$ [m]")
    axes[0].set_ylabel(r"$\max|F_{ret}-F_{HLL}|$")
    axes[0].grid(alpha=0.25)
    smart_legend(axes[0], loc="lower right", bbox_to_anchor=(0.98, 0.05), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].semilogx(dry_depths, flags, linewidth=1.8, drawstyle="steps-post", label="Fallback flag")
    axes[1].axvline(n.h_dry * n.hllc_dry_factor, linestyle=":", label="Dry-threshold switch")
    axes[1].set_xlabel(r"Right depth $h_R$ [m]")
    axes[1].set_ylabel("HLL fallback")
    axes[1].set_ylim(-0.05, 1.35)
    axes[1].grid(alpha=0.25)
    smart_legend(axes[1], loc="upper left", bbox_to_anchor=(0.02, 0.96), allow_outside=False)
    status(axes[1], f"max difference under fallback = {metric:.1e}\ntrigger count = {int(np.sum(flags))}", metric < 1.0e-12, loc="lower left")
    panel(axes[1], "(b)")
    savefig(fig, "test_07_wet_dry_fallback")
    return {"metric": metric, "threshold": 1.0e-12}


def run_dambreak(nx: int, dx: float, T: float = 1.0):
    cfg = flat_cfg(); m = cfg.material
    cfg.numerics.bc_left = "outflow"
    cfg.numerics.bc_right = "outflow"
    cfg.numerics.bc_bottom = "reflective"
    cfg.numerics.bc_top = "reflective"
    cfg.erosion.enabled = False; cfg.deposition.enabled = False; cfg.segregation.enabled = False
    ny, ng = 3, 2
    shape = (ny + 2*ng, nx + 2*ng)
    U = np.zeros((sm.NV, *shape)); zb = np.zeros(shape); active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng+ny), slice(ng, ng+nx))
    xdam = nx // 3
    h0 = 1.0
    for j in range(ny):
        for i in range(nx):
            hh = h0 if i < xdam else 0.0
            U[:, ng+j, ng+i] = sm.primitive_to_state(hh, 0.0, 0.0, 0.5, 0.5, 0.3, 0.3,
                                                      m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    sm.apply_boundary(U, zb, active, cfg.numerics)
    ws = sm.FluxWorkspace(shape)
    t = 0.0
    vol0 = float(np.sum(U[[sm.HF, sm.HSU, sm.HSL], core[0], core[1]], dtype=np.float64))
    times=[0.0]; rel=[0.0]; minc=[float(np.min(U[[sm.HF, sm.HSU, sm.HCU, sm.HSL, sm.HCL], core[0], core[1]]))]
    profiles=[(0.0, np.mean(depth(U[:, core[0], core[1]]), axis=0).copy())]
    while t < T - 1.0e-12:
        dt = min(sm.compute_dt(U, core, dx, dx, cfg), 0.02, T-t)
        Un, _, _, _ = sm.transport_step_ssprk2(U, dt, zb, active, cfg, ws, dx, dx, core)
        if Un is None:
            raise RuntimeError("Dam-break run failed")
        U = Un; t += dt
        vol = float(np.sum(U[[sm.HF, sm.HSU, sm.HSL], core[0], core[1]], dtype=np.float64))
        times.append(t); rel.append((vol - vol0)/vol0)
        minc.append(float(np.min(U[[sm.HF, sm.HSU, sm.HCU, sm.HSL, sm.HCL], core[0], core[1]])))
        if abs(t - 0.25) < 0.03 or abs(t - 0.50) < 0.03 or abs(t - T) < 1e-10:
            profiles.append((t, np.mean(depth(U[:, core[0], core[1]]), axis=0).copy()))
    x = (np.arange(nx) - xdam) * dx
    h = np.mean(depth(U[:, core[0], core[1]]), axis=0)
    return cfg, x, h, np.array(times), np.array(rel), np.array(minc), profiles, t


def ritter_solution(x: np.ndarray, t: float, h0: float = 1.0, g: float = 9.81) -> np.ndarray:
    c0 = math.sqrt(g * h0)
    h = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= -c0 * t:
            h[i] = h0
        elif xi <= 2.0 * c0 * t:
            h[i] = ((2.0 * c0 - xi / t) ** 2) / (9.0 * g)
        else:
            h[i] = 0.0
    return h


def fig08_dambreak_profiles() -> Dict[str, float]:
    cfg, x, h, times, rel, minc, profiles, tf = run_dambreak(300, 0.2, T=1.0)
    metric = float(abs(rel[-1]))
    minv = float(np.min(minc))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    for t, prof in profiles[:4]:
        axes[0].plot(x, prof, linewidth=1.6, label=f"t = {t:.2f} s")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel(r"Depth $h(x,t)$ [m]")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(-0.05, 1.22)
    smart_legend(axes[0], ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(times, rel, linewidth=1.8, marker="o", markersize=3, label="Relative volume error")
    axes[1].axhline(5.0e-9, linestyle="--", label="Volume tolerance")
    axes[1].axhline(-5.0e-9, linestyle="--")
    axr = axes[1].twinx(); axr.plot(times, minc, linewidth=1.4, linestyle=":", label="Minimum state component")
    axr.axhline(-1.0e-12, linestyle="-.", label="Positivity threshold")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$(V(t)-V_0)/V_0$")
    axr.set_ylabel("Minimum transported component")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    smart_legend(axes[1], handles=lines1 + lines2, labels=labels1 + labels2,
                 extra_axes=(axr,), ncol=2, loc="upper left",
                 bbox_to_anchor=(0.00, 0.96), allow_outside=False)
    axes[1].grid(alpha=0.25)
    status(axes[1], f"final volume error = {metric:.3e}\nminimum component = {minv:.3e}", metric < 5.0e-9 and minv >= -1.0e-12, loc="lower right")
    panel(axes[1], "(b)")
    savefig(fig, "test_08_dambreak_profiles")
    return {"metric": metric, "threshold": 5.0e-9}


def fig09_composition_advection() -> Dict[str, float]:
    cfg = flat_cfg(); m = cfg.material
    cfg.numerics.bc_left = "outflow"; cfg.numerics.bc_right = "outflow"; cfg.numerics.bc_bottom = "reflective"; cfg.numerics.bc_top = "reflective"
    cfg.segregation.enabled = False; cfg.erosion.enabled = False; cfg.deposition.enabled = False
    nx, ny, ng = 120, 3, 2
    dx = 1.0
    shape = (ny + 2*ng, nx + 2*ng)
    U = np.zeros((sm.NV, *shape)); zb = np.zeros(shape); active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng+ny), slice(ng, ng+nx))
    x = np.arange(nx) * dx
    for j in range(ny):
        for i in range(nx):
            fc = 0.20 + 0.30 * math.exp(-((i - 30.0)/8.0)**2)
            U[:, ng+j, ng+i] = sm.primitive_to_state(1.0, 1.0, 0.0, 0.5, 0.5, fc, fc,
                                                      m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    sm.apply_boundary(U, zb, active, cfg.numerics)
    ws = sm.FluxWorkspace(shape)
    fc0 = sm.composition_fields(U[:, core[0], core[1]], cfg)["fc"][1].copy()
    T = 5.0; t = 0.0
    times=[0.0]; errs=[0.0]
    while t < T - 1.0e-12:
        dt = min(sm.compute_dt(U, core, dx, dx, cfg), 0.05, T-t)
        Un, _, _, _ = sm.transport_step_ssprk2(U, dt, zb, active, cfg, ws, dx, dx, core)
        if Un is None:
            raise RuntimeError("Composition-advection test failed")
        U = Un; t += dt
        fct = sm.composition_fields(U[:, core[0], core[1]], cfg)["fc"][1].copy()
        fcref = np.interp(np.arange(nx) - 1.0 * t / dx, np.arange(nx), fc0, left=fc0[0], right=fc0[-1])
        errs.append(float(np.sqrt(np.mean((fct - fcref)**2))))
        times.append(t)
    fcf = sm.composition_fields(U[:, core[0], core[1]], cfg)["fc"][1].copy()
    fcref = np.interp(np.arange(nx) - 1.0 * t / dx, np.arange(nx), fc0, left=fc0[0], right=fc0[-1])
    metric = float(np.sqrt(np.mean((fcf - fcref)**2)))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(x, fc0, linewidth=1.8, label="Initial coarse fraction")
    axes[0].plot(x, fcf, linewidth=1.6, linestyle="--", label=f"Numerical at t = {t:.1f} s")
    axes[0].plot(x, fcref, linewidth=1.4, linestyle=":", label="Translated reference")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel(r"Coarse fraction of solids $f_c$")
    axes[0].grid(alpha=0.25)
    smart_legend(axes[0], loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(times, errs, linewidth=1.8, marker="o", markersize=3, label=r"RMS error in $f_c$")
    axes[1].axhline(1.0e-2, linestyle="--", label="Acceptance threshold")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("RMS error")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-5.0e-4, 1.35e-2)
    smart_legend(axes[1], loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    status(axes[1], f"final RMS error = {metric:.3e}", metric < 1.0e-2, loc="lower left", y_offset=0.16)
    panel(axes[1], "(b)")
    savefig(fig, "test_09_composition_advection")
    return {"metric": metric, "threshold": 1.0e-2}


def fig10_open_boundary_budget() -> Dict[str, float]:
    cfg = flat_cfg(); m = cfg.material
    cfg.numerics.space_order = 1; cfg.numerics.time_order = 1; cfg.numerics.flux = "hll"
    cfg.numerics.bc_left = "reflective"; cfg.numerics.bc_right = "outflow"; cfg.numerics.bc_bottom = "reflective"; cfg.numerics.bc_top = "reflective"
    nx, ny, ng = 20, 6, 2
    shape = (ny+2*ng, nx+2*ng)
    U = np.zeros((sm.NV, *shape)); zb = np.zeros(shape); active = np.ones(shape, dtype=np.uint8)
    core = (slice(ng, ng+ny), slice(ng, ng+nx))
    for j in range(ny):
        for i in range(nx):
            U[:, ng+j, ng+i] = sm.primitive_to_state(0.5, 1.0, 0.0, 0.4, 0.5, 0.3, 0.3,
                                                      m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    ws = sm.FluxWorkspace(shape)
    dt = 0.01
    base = np.sum(U[:5, core[0], core[1]], axis=(1,2))
    times=[0.0]
    dstore = {"fluid":[0.0], "fine":[0.0], "coarse":[0.0]}
    dbound = {"fluid":[0.0], "fine":[0.0], "coarse":[0.0]}
    dres   = {"fluid":[0.0], "fine":[0.0], "coarse":[0.0]}
    bfluid = bfine = bcoarse = 0.0
    for k in range(1, 21):
        Un, _, _, br = sm.transport_step_ssprk2(U, dt, zb, active, cfg, ws, 1.0, 1.0, core)
        if Un is None:
            raise RuntimeError("Open-boundary test failed")
        U = Un
        after = np.sum(U[:5, core[0], core[1]], axis=(1,2))
        sfluid = float(after[sm.HF] - base[sm.HF])
        sfine = float((after[sm.HSU]-after[sm.HCU]+after[sm.HSL]-after[sm.HCL]) - (base[sm.HSU]-base[sm.HCU]+base[sm.HSL]-base[sm.HCL]))
        scoarse = float((after[sm.HCU]+after[sm.HCL]) - (base[sm.HCU]+base[sm.HCL]))
        bfluid += float(br[sm.HF] * dt)
        bfine += float((br[sm.HSU]-br[sm.HCU]+br[sm.HSL]-br[sm.HCL]) * dt)
        bcoarse += float((br[sm.HCU]+br[sm.HCL]) * dt)
        times.append(k*dt)
        dstore['fluid'].append(sfluid); dstore['fine'].append(sfine); dstore['coarse'].append(scoarse)
        dbound['fluid'].append(bfluid); dbound['fine'].append(bfine); dbound['coarse'].append(bcoarse)
        dres['fluid'].append(sfluid - bfluid); dres['fine'].append(sfine - bfine); dres['coarse'].append(scoarse - bcoarse)
    metric = float(max(np.max(np.abs(dres['fluid'])), np.max(np.abs(dres['fine'])), np.max(np.abs(dres['coarse']))))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].plot(times, dstore['fluid'], linewidth=1.8, label="Domain storage change — fluid")
    axes[0].plot(times, dbound['fluid'], linewidth=1.5, linestyle="--", label="Integrated boundary flux — fluid")
    axes[0].plot(times, dstore['fine'], linewidth=1.8, label="Domain storage change — fine")
    axes[0].plot(times, dbound['fine'], linewidth=1.5, linestyle="--", label="Integrated boundary flux — fine")
    axes[0].plot(times, dstore['coarse'], linewidth=1.8, label="Domain storage change — coarse")
    axes[0].plot(times, dbound['coarse'], linewidth=1.5, linestyle="--", label="Integrated boundary flux — coarse")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Cumulative change")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(-0.38, 0.25)
    smart_legend(axes[0], ncol=2, fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(times, dres['fluid'], linewidth=1.8, label="Fluid residual")
    axes[1].plot(times, dres['fine'], linewidth=1.8, label="Fine residual")
    axes[1].plot(times, dres['coarse'], linewidth=1.8, label="Coarse residual")
    axes[1].axhline(1.0e-10, linestyle="--", label="Tolerance")
    axes[1].axhline(-1.0e-10, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-1.80e-10, 1.35e-10)
    smart_legend(axes[1], ncol=2, loc="lower left", bbox_to_anchor=(0.02, 0.05), allow_outside=False)
    status(axes[1], f"max residual = {metric:.3e}", metric < 1.0e-10, loc="upper right", y_offset=0.04)
    panel(axes[1], "(b)")
    savefig(fig, "test_10_open_boundary_budget")
    return {"metric": metric, "threshold": 1.0e-10}


def fig11_grid_convergence() -> Dict[str, float]:
    grid_cases = [(150, 0.4), (300, 0.2), (600, 0.1)]
    dx = []; l1 = []; l2 = []; linf = []
    for nx, ddx in grid_cases:
        cfg, x, h, *_ = run_dambreak(nx, ddx, T=1.0)
        href = ritter_solution(x, 1.0, h0=1.0, g=cfg.numerics.g)
        mask = (x >= -math.sqrt(cfg.numerics.g)*1.0 - 1.0) & (x <= 2.0*math.sqrt(cfg.numerics.g)*1.0 + 1.0)
        err = h - href
        dx.append(ddx)
        l1.append(float(np.mean(np.abs(err[mask]))))
        l2.append(float(np.sqrt(np.mean(err[mask]**2))))
        linf.append(float(np.max(np.abs(err[mask]))))
    dx = np.array(dx); l1=np.array(l1); l2=np.array(l2); linf=np.array(linf)
    p12 = float(np.log(l2[0]/l2[1]) / np.log(dx[0]/dx[1]))
    p23 = float(np.log(l2[1]/l2[2]) / np.log(dx[1]/dx[2]))
    metric = float(max(abs(p12-1.0), abs(p23-1.0)))

    fig, axes = plt.subplots(2, 1, figsize=(9.3, 7.2), constrained_layout=True)
    axes[0].loglog(dx, l1, marker="o", linewidth=1.8, label=r"$L_1$")
    axes[0].loglog(dx, l2, marker="s", linewidth=1.8, label=r"$L_2$")
    axes[0].loglog(dx, linf, marker="^", linewidth=1.8, label=r"$L_\infty$")
    axes[0].loglog(dx, (l2[-1]/dx[-1]) * dx, linestyle="--", linewidth=1.2, label="First-order reference")
    axes[0].set_xlabel(r"Grid spacing $\Delta x$ [m]")
    axes[0].set_ylabel("Error norm")
    axes[0].grid(alpha=0.25)
    smart_legend(axes[0], loc="lower right", bbox_to_anchor=(0.98, 0.05), allow_outside=False)
    panel(axes[0], "(a)")

    axes[1].plot(dx[1:], [p12, p23], marker="o", linewidth=1.8, label="Observed order from consecutive grids")
    axes[1].axhline(1.0, linestyle="--", label="Expected first-order trend")
    axes[1].set_xlabel(r"Representative grid spacing $\Delta x$ [m]")
    axes[1].set_ylabel("Observed order")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(0.94, 1.07)
    smart_legend(axes[1], loc="upper right", bbox_to_anchor=(0.98, 0.96), allow_outside=False)
    status(axes[1], f"p12 = {p12:.2f}\np23 = {p23:.2f}", metric < 0.35, loc="lower left")
    panel(axes[1], "(b)")
    savefig(fig, "test_11_grid_convergence")
    return {"metric": metric, "threshold": 0.35}


def fig00_summary(results: Dict[str, Dict[str, float]]) -> None:
    labels = [
        "1 HLLC", "2 Lake-at-rest", "3 Segregation", "4 Erosion", "5 Density",
        "6 Deposition", "7 Wet-dry", "8 Dam-break", "9 Advection",
        "10 Open BC", "11 Convergence",
    ]
    keys = list(results.keys())
    ratios = np.maximum(np.array([abs(results[k]['metric']) / results[k]['threshold'] for k in keys]), 1e-16)
    idx = np.arange(1, len(labels)+1)
    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    ax.semilogy(idx, ratios, marker="o", linewidth=1.8)
    ax.axhline(1.0, linestyle="--", label="Acceptance limit")
    ax.set_xticks(idx, labels, rotation=20)
    ax.set_ylabel("metric / tolerance")
    ax.grid(alpha=0.25)
    smart_legend(ax, loc="lower right", bbox_to_anchor=(0.98, 0.05), allow_outside=False)
    for i, r in zip(idx, ratios):
        ax.annotate(f"{r:.1e}", xy=(i, r), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    status(ax, f"{int(np.sum(ratios < 1.0))}/{len(ratios)} tests within tolerance", bool(np.all(ratios < 1.0)), loc="upper left")
    savefig(fig, "test_00_summary")


def generate_all(verbose: bool = True) -> Dict[str, Dict[str, float]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "hllc_consistency": fig01_hllc_consistency(),
        "lake_at_rest": fig02_lake_at_rest(),
        "segregation": fig03_segregation(),
        "erosion": fig04_erosion(),
        "density_closure": fig05_density_closure(),
        "active_deposition": fig06_active_deposition(),
        "wet_dry_fallback": fig07_wet_dry_fallback(),
        "dam_break_profiles": fig08_dambreak_profiles(),
        "composition_advection": fig09_composition_advection(),
        "open_boundary_budget": fig10_open_boundary_budget(),
        "grid_convergence": fig11_grid_convergence(),
    }
    fig00_summary(results)
    with open(OUT_DIR / "figure_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(json.dumps(results, indent=2))
        print(f"[FIGURES] Saved in: {OUT_DIR}")
    return results
def _read_test_settings(path: Path) -> Dict[str, object]:
    """Read the optional tests section from the general test configuration."""
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    settings = data.get("tests", {})
    if not isinstance(settings, dict):
        raise TypeError("The tests section in config_tests.yaml must be a mapping.")
    return settings


def _resolve_from_config(config_path: Path, value: str) -> Path:
    """Resolve a path relative to the directory containing a YAML file."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def validate_configuration_files(test_config: Path, marsicano_config: Path) -> Dict[str, str]:
    """Load and validate the two repository configuration files."""
    test_cfg = load_config(str(test_config))
    marsicano_cfg = load_config(str(marsicano_config))
    if test_cfg.numerics.flux.lower() != "hllc":
        raise AssertionError("The general test configuration must use the HLLC flux.")
    if marsicano_cfg.numerics.flux.lower() != "hllc":
        raise AssertionError("The Marsicano configuration must use the HLLC flux.")
    for label, cfg in (("general test", test_cfg), ("Marsicano", marsicano_cfg)):
        if cfg.compute.backend.lower() != "cuda":
            raise AssertionError(
                f"The {label} configuration must request CUDA as the preferred backend."
            )
        if not cfg.compute.cuda_allow_fallback:
            raise AssertionError(
                f"The {label} configuration must enable automatic CPU fallback."
            )
    return {
        "test_config": str(test_config.resolve()),
        "marsicano_config": str(marsicano_config.resolve()),
    }


@dataclass
class _backend_RitterResult:
    nx: int
    order: int
    limiter: str
    x: np.ndarray
    h: np.ndarray
    u: np.ndarray
    qx: np.ndarray
    h_exact: np.ndarray
    u_exact: np.ndarray
    qx_exact: np.ndarray
    dt_mode: str
    requested_dt: Optional[float]
    min_dt: float
    max_dt: float
    mean_dt: float
    steps: int
    fallback_faces: int
    first_order_fallback_steps: int
    retries: int

def _backend_ritter_exact(x: np.ndarray, t: float, h0: float=10.0, g: float=9.81, dam_x: float=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x = np.asarray(x, dtype=np.float64)
    h = np.zeros_like(x)
    u = np.zeros_like(x)
    if t <= 0.0:
        h[x < dam_x] = h0
        return (h, u, h * u)
    c0 = math.sqrt(g * h0)
    similarity = (x - dam_x) / t
    reservoir = similarity <= -c0
    fan = (similarity > -c0) & (similarity < 2.0 * c0)
    h[reservoir] = h0
    h[fan] = np.square(2.0 * c0 - similarity[fan]) / (9.0 * g)
    u[fan] = 2.0 / 3.0 * (c0 + similarity[fan])
    return (h, u, h * u)

def _backend__cuda_device_label(allow_cudasim: bool) -> str:
    if not cuda.is_available():
        raise RuntimeError('A working CUDA device is required for these verification tests. Install a compatible NVIDIA driver and CUDA-enabled Numba environment.')
    is_sim = bool(getattr(cuda.config, 'ENABLE_CUDASIM', False))
    if is_sim:
        if not allow_cudasim:
            raise RuntimeError('NUMBA_ENABLE_CUDASIM is active. These tests are configured to require a real GPU; enable allow_cudasim=True only when CUDA simulation is intentionally required.')
        return 'Numba CUDA simulator'
    devices = list(cuda.gpus)
    if not devices:
        raise RuntimeError('CUDA is available but no CUDA device was enumerated')
    with devices[0]:
        raw = getattr(cuda.current_context().device, 'name', 'CUDA device 0')
        return raw.decode() if isinstance(raw, bytes) else str(raw)

def _backend__ritter_config(order: int, limiter: str, t_end: float, cfl: float, h_dry: float, backend: str) -> SolverConfig:
    cfg = SolverConfig()
    cfg.compute.backend = backend
    cfg.compute.cuda_block_x = 16
    cfg.compute.cuda_block_y = 8
    cfg.compute.cuda_threads_1d = 256
    cfg.numerics.g = 9.81
    cfg.numerics.cfl = float(cfl)
    cfg.numerics.dt_max = 1.0
    cfg.numerics.t_end = float(t_end)
    cfg.numerics.output_dt = float(t_end)
    cfg.numerics.h_dry = float(h_dry)
    cfg.numerics.space_order = int(order)
    cfg.numerics.time_order = int(order)
    cfg.numerics.limiter = str(limiter)
    cfg.numerics.flux = 'hllc'
    cfg.numerics.hllc_dry_factor = 5.0
    cfg.numerics.hllc_max_froude = 50.0
    cfg.numerics.hybrid_depth_rel_jump = 0.75
    cfg.numerics.max_step_retries = 10
    cfg.numerics.positivity_tolerance = 1e-10
    cfg.numerics.speed_cap_ms = 0.0
    cfg.numerics.bc_left = 'outflow'
    cfg.numerics.bc_right = 'outflow'
    cfg.numerics.bc_bottom = 'reflective'
    cfg.numerics.bc_top = 'reflective'
    cfg.material.rho_fluid = 1000.0
    cfg.material.rho_solid = 1000.0
    cfg.material.initial_solid_fraction = 0.0
    cfg.material.initial_coarse_fraction = 0.0
    cfg.material.initial_upper_solid_fraction = 0.0
    cfg.material.yield_N0_pa = 0.0
    cfg.erosion.enabled = False
    cfg.deposition.enabled = False
    cfg.segregation.enabled = False
    validate_config(cfg)
    return cfg

def _backend__allocate_workspace(shape: Tuple[int, int], nx_core: int, ny_core: int, threads_1d: int) -> CudaWorkspace:
    ny, nx = shape
    cells = nx_core * ny_core
    reduction_blocks = max(1, (cells + threads_1d - 1) // threads_1d)
    P = cuda.device_array((7, ny, nx), dtype=np.float64)
    return CudaWorkspace(P=P, sx=cuda.device_array_like(P), sy=cuda.device_array_like(P), fx=cuda.device_array((NV, ny, nx + 1), dtype=np.float64), fy=cuda.device_array((NV, ny + 1, nx), dtype=np.float64), cxl=cuda.device_array((ny, nx + 1), dtype=np.float64), cxr=cuda.device_array((ny, nx + 1), dtype=np.float64), cyl=cuda.device_array((ny + 1, nx), dtype=np.float64), cyr=cuda.device_array((ny + 1, nx), dtype=np.float64), rhs=cuda.device_array((NV, ny, nx), dtype=np.float64), fallback=cuda.device_array(1, dtype=np.int64), boundary_rate=cuda.device_array(5, dtype=np.float64), invalid=cuda.device_array(1, dtype=np.int32), block_sx=cuda.device_array(reduction_blocks, dtype=np.float64), block_sy=cuda.device_array(reduction_blocks, dtype=np.float64), block_h=cuda.device_array(reduction_blocks, dtype=np.float64), block_speed=cuda.device_array(reduction_blocks, dtype=np.float64), block_rho_min=cuda.device_array(reduction_blocks, dtype=np.float64), block_rho_max=cuda.device_array(reduction_blocks, dtype=np.float64), source_budget=cuda.device_array(4, dtype=np.float64), segregation_budget=cuda.device_array(1, dtype=np.float64))

def _backend_run_ritter_cuda(nx: int, order: int, *, limiter: str='superbee', xmin: float=-500.0, xmax: float=500.0, dam_x: float=0.0, h0: float=10.0, t_end: float=15.0, cfl: float=0.25, fixed_dt: Optional[float]=None, h_dry: float=1e-06, allow_cudasim: bool=False) -> _backend_RitterResult:
    _backend__cuda_device_label(allow_cudasim)
    if nx < 16:
        raise ValueError('nx must be at least 16')
    if order not in (1, 2):
        raise ValueError('order must be 1 or 2')
    if xmax <= xmin:
        raise ValueError('xmax must be greater than xmin')
    if h0 <= 0.0 or t_end <= 0.0:
        raise ValueError('h0 and t_end must be positive')
    if fixed_dt is not None and fixed_dt <= 0.0:
        raise ValueError('fixed_dt must be positive')
    cfg = _backend__ritter_config(order, limiter, t_end, cfl, h_dry, 'cuda')
    ng = 2
    ny_core = 1
    dx = (xmax - xmin) / float(nx)
    dy = 1000000.0
    x = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    shape = (ny_core + 2 * ng, nx + 2 * ng)
    U = np.zeros((NV, *shape), dtype=np.float64)
    zb = np.zeros(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.uint8)
    h_init = np.where(x < dam_x, h0, 0.0)
    U[HF, ng, ng:ng + nx] = h_init
    d_U = cuda.to_device(U)
    d_U1 = cuda.device_array_like(d_U)
    d_U2 = cuda.device_array_like(d_U)
    d_zb = cuda.to_device(zb)
    d_active = cuda.to_device(active)
    threads_1d = int(cfg.compute.cuda_threads_1d)
    block = (int(cfg.compute.cuda_block_x), int(cfg.compute.cuda_block_y))
    grid = _grid2(shape, block)
    ws = _backend__allocate_workspace(shape, nx, ny_core, threads_1d)
    if fixed_dt is not None:
        max_characteristic = 2.0 * math.sqrt(cfg.numerics.g * h0)
        estimated_cfl = fixed_dt * max_characteristic / dx
        if estimated_cfl > 0.9:
            raise ValueError(f'fixed_dt={fixed_dt:g} s is too large for nx={nx}; estimated CFL={estimated_cfl:.3f} exceeds 0.9')
    t = 0.0
    steps = 0
    fallback_faces = 0
    global_fallback_steps = 0
    retries = 0
    used_dts: List[float] = []
    while t < t_end - 1e-13:
        if fixed_dt is None:
            dt = _compute_dt_device(d_U, ws, cfg, dx, dy, ng, nx, ny_core, threads_1d)
        else:
            dt = float(fixed_dt)
        dt = min(dt, t_end - t)
        accepted = False
        for _ in range(cfg.numerics.max_step_retries + 1):
            final_state, fb, global_fb, _ = _transport_step_device(d_U, d_U1, d_U2, dt, d_zb, d_active, cfg, ws, dx, dy, grid, block, threads_1d, ng, nx, ny_core)
            if final_state is None:
                if fixed_dt is not None:
                    raise RuntimeError(f'Fixed time step {fixed_dt:g} s produced a non-admissible state at t={t:g} s')
                dt *= 0.5
                retries += 1
                continue
            old_U = d_U
            d_U = final_state
            if final_state is d_U1:
                d_U1 = old_U
            else:
                d_U2 = old_U
            fallback_faces += int(fb)
            global_fallback_steps += int(global_fb)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f'Ritter CUDA step failed at t={t:g} s')
        t += dt
        steps += 1
        used_dts.append(dt)
    cuda.synchronize()
    U_final = d_U.copy_to_host()[:, ng:ng + ny_core, ng:ng + nx]
    h = np.mean(np.maximum(U_final[HF] + U_final[HSU] + U_final[HSL], 0.0), axis=0)
    mass = cfg.material.rho_fluid * np.maximum(U_final[HF], 0.0)
    mass += cfg.material.rho_solid * (np.maximum(U_final[HSU], 0.0) + np.maximum(U_final[HSL], 0.0))
    momentum = np.mean(U_final[MX], axis=0)
    mass_line = np.mean(mass, axis=0)
    u = np.zeros_like(h)
    wet = h > h_dry
    u[wet] = momentum[wet] / np.maximum(mass_line[wet], 1e-30)
    qx = h * u
    h_exact, u_exact, qx_exact = _backend_ritter_exact(x, t_end, h0, cfg.numerics.g, dam_x)
    dts = np.asarray(used_dts, dtype=np.float64)
    return _backend_RitterResult(nx=nx, order=order, limiter=limiter, x=x, h=h, u=u, qx=qx, h_exact=h_exact, u_exact=u_exact, qx_exact=qx_exact, dt_mode='fixed' if fixed_dt is not None else 'CFL', requested_dt=fixed_dt, min_dt=float(np.min(dts)), max_dt=float(np.max(dts)), mean_dt=float(np.mean(dts)), steps=steps, fallback_faces=fallback_faces, first_order_fallback_steps=global_fallback_steps, retries=retries)

def _backend_run_ritter_cpu(nx: int, order: int, *, limiter: str='superbee', xmin: float=-500.0, xmax: float=500.0, dam_x: float=0.0, h0: float=10.0, t_end: float=15.0, cfl: float=0.25, fixed_dt: Optional[float]=None, h_dry: float=1e-06) -> _backend_RitterResult:
    if nx < 16:
        raise ValueError('nx must be at least 16')
    if order not in (1, 2):
        raise ValueError('order must be 1 or 2')
    if xmax <= xmin:
        raise ValueError('xmax must be greater than xmin')
    if h0 <= 0.0 or t_end <= 0.0:
        raise ValueError('h0 and t_end must be positive')
    if fixed_dt is not None and fixed_dt <= 0.0:
        raise ValueError('fixed_dt must be positive')
    cfg = _backend__ritter_config(order, limiter, t_end, cfl, h_dry, 'cpu')
    ng = 2
    ny_core = 1
    dx = (xmax - xmin) / float(nx)
    dy = 1000000.0
    x = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    shape = (ny_core + 2 * ng, nx + 2 * ng)
    U = np.zeros((NV, *shape), dtype=np.float64)
    zb = np.zeros(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.uint8)
    U[HF, ng, ng:ng + nx] = np.where(x < dam_x, h0, 0.0)
    core = (slice(ng, ng + ny_core), slice(ng, ng + nx))
    workspace = FluxWorkspace(shape)
    if fixed_dt is not None:
        max_characteristic = 2.0 * math.sqrt(cfg.numerics.g * h0)
        estimated_cfl = fixed_dt * max_characteristic / dx
        if estimated_cfl > 0.9:
            raise ValueError(f'fixed_dt={fixed_dt:g} s is too large for nx={nx}; estimated CFL={estimated_cfl:.3f} exceeds 0.9')
    t = 0.0
    steps = 0
    fallback_faces = 0
    global_fallback_steps = 0
    retries = 0
    used_dts: List[float] = []
    while t < t_end - 1e-13:
        dt = compute_dt(U, core, dx, dy, cfg) if fixed_dt is None else float(fixed_dt)
        dt = min(dt, t_end - t)
        accepted = False
        for _ in range(cfg.numerics.max_step_retries + 1):
            final_state, fb, global_fb, _ = transport_step_ssprk2(U, dt, zb, active, cfg, workspace, dx, dy, core)
            if final_state is None:
                if fixed_dt is not None:
                    raise RuntimeError(f'Fixed time step {fixed_dt:g} s produced a non-admissible state at t={t:g} s')
                dt *= 0.5
                retries += 1
                continue
            U = final_state
            fallback_faces += int(fb)
            global_fallback_steps += int(global_fb)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f'Ritter CPU step failed at t={t:g} s')
        t += dt
        steps += 1
        used_dts.append(dt)
    U_final = U[:, core[0], core[1]]
    h = np.mean(np.maximum(U_final[HF] + U_final[HSU] + U_final[HSL], 0.0), axis=0)
    mass = cfg.material.rho_fluid * np.maximum(U_final[HF], 0.0)
    mass += cfg.material.rho_solid * (np.maximum(U_final[HSU], 0.0) + np.maximum(U_final[HSL], 0.0))
    momentum = np.mean(U_final[MX], axis=0)
    mass_line = np.mean(mass, axis=0)
    u = np.zeros_like(h)
    wet = h > h_dry
    u[wet] = momentum[wet] / np.maximum(mass_line[wet], 1e-30)
    qx = h * u
    h_exact, u_exact, qx_exact = _backend_ritter_exact(x, t_end, h0, cfg.numerics.g, dam_x)
    dts = np.asarray(used_dts, dtype=np.float64)
    return _backend_RitterResult(nx=nx, order=order, limiter=limiter, x=x, h=h, u=u, qx=qx, h_exact=h_exact, u_exact=u_exact, qx_exact=qx_exact, dt_mode='fixed' if fixed_dt is not None else 'CFL', requested_dt=fixed_dt, min_dt=float(np.min(dts)), max_dt=float(np.max(dts)), mean_dt=float(np.mean(dts)), steps=steps, fallback_faces=fallback_faces, first_order_fallback_steps=global_fallback_steps, retries=retries)

def _backend_run_ritter_case(nx: int, order: int, *, backend: str, allow_cudasim: bool=False, **kwargs: object) -> _backend_RitterResult:
    selected = str(backend).strip().lower()
    if selected == 'cuda':
        return _backend_run_ritter_cuda(nx, order, allow_cudasim=allow_cudasim, **kwargs)
    if selected == 'cpu':
        return _backend_run_ritter_cpu(nx, order, **kwargs)
    raise ValueError("backend must be either 'cuda' or 'cpu'")

def _backend__relative_errors(numerical: np.ndarray, exact: np.ndarray, dx: float) -> Dict[str, float]:
    diff = np.asarray(numerical) - np.asarray(exact)
    l1_den = max(float(np.sum(np.abs(exact)) * dx), 1e-30)
    l2_den = max(float(math.sqrt(np.sum(exact * exact) * dx)), 1e-30)
    return {'rel_l1': float(np.sum(np.abs(diff)) * dx / l1_den), 'rel_l2': float(math.sqrt(np.sum(diff * diff) * dx) / l2_den), 'linf': float(np.max(np.abs(diff)))}

def _backend__observed_orders(rows: List[Dict[str, float]], error_key: str) -> None:
    rows.sort(key=lambda row: row['dx'], reverse=True)
    previous = None
    for row in rows:
        row[f'order_{error_key}'] = float('nan')
        if previous is not None and row[error_key] > 0.0 and (previous[error_key] > 0.0):
            row[f'order_{error_key}'] = math.log(previous[error_key] / row[error_key]) / math.log(previous['dx'] / row['dx'])
        previous = row

def _backend__monitor_depth(result: _backend_RitterResult, monitor_x: float) -> float:
    return float(np.interp(monitor_x, result.x, result.h))

def _backend__save_profile_csv(path: Path, result: _backend_RitterResult) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_m', 'h_numerical_m', 'h_exact_m', 'u_numerical_ms', 'u_exact_ms', 'qx_numerical_m2s', 'qx_exact_m2s'])
        for values in zip(result.x, result.h, result.h_exact, result.u, result.u_exact, result.qx, result.qx_exact):
            writer.writerow([f'{float(v):.16g}' for v in values])

def _backend__save_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _backend__artist_points_in_axes(source_ax, reference_ax, max_points_per_artist: int=800) -> np.ndarray:
    chunks: List[np.ndarray] = []
    to_axes = reference_ax.transAxes.inverted()
    for line in source_ax.lines:
        try:
            x = np.ma.asarray(line.get_xdata(orig=False)).filled(np.nan).ravel()
            y = np.ma.asarray(line.get_ydata(orig=False)).filled(np.nan).ravel()
            n = min(x.size, y.size)
            if n == 0:
                continue
            x, y = (x[:n], y[:n])
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = (x[finite], y[finite])
            if x.size == 0:
                continue
            if x.size > max_points_per_artist:
                take = np.linspace(0, x.size - 1, max_points_per_artist, dtype=int)
                x, y = (x[take], y[take])
            display = line.get_transform().transform(np.column_stack((x, y)))
            chunks.append(to_axes.transform(display))
        except Exception:
            continue
    for collection in source_ax.collections:
        try:
            offsets = np.ma.asarray(collection.get_offsets()).filled(np.nan)
            if offsets.ndim != 2 or offsets.shape[1] < 2 or offsets.size == 0:
                continue
            offsets = offsets[:, :2]
            finite = np.all(np.isfinite(offsets), axis=1)
            offsets = offsets[finite]
            if offsets.shape[0] > max_points_per_artist:
                take = np.linspace(0, offsets.shape[0] - 1, max_points_per_artist, dtype=int)
                offsets = offsets[take]
            display = collection.get_offset_transform().transform(offsets)
            chunks.append(to_axes.transform(display))
        except Exception:
            continue
    if not chunks:
        return np.empty((0, 2), dtype=float)
    points = np.vstack(chunks)
    return points[np.all(np.isfinite(points), axis=1)]

def _backend__smart_legend(ax, *, ncol: int=1, frameon: bool=False, fontsize=None, allow_outside: bool=False, loc=None, bbox_to_anchor=None):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if loc is not None:
        kwargs = {"borderaxespad": 0.0}
        if bbox_to_anchor is not None:
            kwargs["bbox_to_anchor"] = bbox_to_anchor
            kwargs["bbox_transform"] = ax.transAxes
        return ax.legend(handles, labels, loc=loc, ncol=ncol, frameon=frameon, fontsize=fontsize, **kwargs)
    fig = ax.figure
    fig.canvas.draw()
    points = _backend__artist_points_in_axes(ax, ax)
    candidates = ('upper right', 'lower right', 'lower center')
    penalties = {'upper right': 0.0, 'lower right': 0.004, 'lower center': 0.008}
    best_loc, best_score = (candidates[0], float('inf'))
    for loc in candidates:
        legend = ax.legend(handles, labels, loc=loc, ncol=ncol, frameon=frameon, fontsize=fontsize)
        fig.canvas.draw()
        bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        bbox = bbox.expanded(1.05, 1.1)
        if points.size:
            inside = (points[:, 0] >= bbox.x0) & (points[:, 0] <= bbox.x1) & (points[:, 1] >= bbox.y0) & (points[:, 1] <= bbox.y1)
            score = float(np.count_nonzero(inside)) / max(1.0, math.sqrt(float(points.shape[0]))) + penalties[loc]
        else:
            score = penalties[loc]
        legend.remove()
        if score < best_score:
            best_loc, best_score = (loc, score)
    if allow_outside and best_score > 0.025:
        return ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0, ncol=ncol, frameon=frameon, fontsize=fontsize)
    return ax.legend(handles, labels, loc=best_loc, ncol=ncol, frameon=frameon, fontsize=fontsize)

def _backend__panel_label(ax, label: str) -> None:
    ax.annotate(label, xy=(-0.015, 1.02), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points', ha='right', va='bottom', fontsize=11, fontweight='bold', annotation_clip=False, clip_on=False)

def _backend__save_figure(fig, out_dir: Path, filename: str) -> List[str]:
    png = out_dir / f'{filename}.png'
    pdf = out_dir / f'{filename}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    return [png.name, pdf.name]

def _backend__plot_profiles(out_dir: Path, results: Dict[Tuple[int, int], _backend_RitterResult], nx_values: Sequence[int], field: str, exact_field: str, ylabel: str, title_name: str, filename: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True, sharey=True)
    for panel_index, (ax, order) in enumerate(zip(axes, (1, 2))):
        for nx in nx_values:
            r = results[order, nx]
            ax.plot(r.x, getattr(r, field), linestyle='--', linewidth=0.9, label=f'Nx={nx}')
        finest = results[order, nx_values[-1]]
        ax.plot(finest.x, getattr(finest, exact_field), linewidth=1.8, label='Exact (Ritter)')
        ordinal = '1st-order' if order == 1 else '2nd-order'
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        _backend__panel_label(ax, f"({chr(ord('a') + panel_index)})")
    handles, labels = axes[1].get_legend_handles_labels()
    if 'velocity' in filename.lower():
        for ax in axes:
            ax.set_ylim(-5.0, 21.0)
        axes[1].legend(
            handles, labels, loc='upper left', bbox_to_anchor=(0.02, 0.96),
            bbox_transform=axes[1].transAxes, borderaxespad=0.0,
            ncol=2, fontsize=8, frameon=False,
        )
    elif 'discharge' in filename.lower():
        axes[1].legend(
            handles, labels, loc='upper left', bbox_to_anchor=(0.02, 0.96),
            bbox_transform=axes[1].transAxes, borderaxespad=0.0,
            ncol=2, fontsize=8, frameon=False,
        )
    else:
        axes[1].legend(
            handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.96),
            bbox_transform=axes[1].transAxes, borderaxespad=0.0,
            ncol=2, fontsize=8, frameon=False,
        )
    fig.tight_layout()
    _backend__save_figure(fig, out_dir, filename)
    plt.close(fig)

def _backend__plot_hstar(out_dir: Path, spatial_rows: Sequence[Dict[str, object]], temporal_rows: Sequence[Dict[str, object]], monitor_x: float, t_end: float) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for order in (1, 2):
        s = sorted((r for r in spatial_rows if int(r['order']) == order), key=lambda r: float(r['dx_m']))
        axes[0].plot([float(r['dx_m']) for r in s], [float(r['h_star_m']) for r in s], marker='o', label=f"{order}{('st' if order == 1 else 'nd')}-order")
        q = sorted((r for r in temporal_rows if int(r['order']) == order), key=lambda r: float(r['dt_s']))
        axes[1].plot([float(r['dt_s']) for r in q], [float(r['h_star_m']) for r in q], marker='o', label=f"{order}{('st' if order == 1 else 'nd')}-order")
    exact = float(spatial_rows[0]['h_star_exact_m'])
    axes[0].axhline(exact, linestyle=':', linewidth=1.2, label='Exact (Ritter)')
    axes[1].axhline(exact, linestyle=':', linewidth=1.2, label='Exact (Ritter)')
    subtitle = f'at $x^*={monitor_x:g}$ m and $t^*={t_end:g}$ s'
    axes[0].set_xlabel('$\\Delta x$ [m]')
    axes[1].set_xlabel('$\\Delta t$ [s]')
    all_hstar = [float(r['h_star_m']) for r in spatial_rows] + [float(r['h_star_m']) for r in temporal_rows] + [exact]
    hmin, hmax = min(all_hstar), max(all_hstar)
    hspan = max(hmax - hmin, 1.0e-6)
    for index, ax in enumerate(axes):
        ax.set_ylabel('$h^*$ [m]')
        ax.set_ylim(hmin - 0.08 * hspan, hmax + 0.62 * hspan)
        ax.grid(True, alpha=0.25)
        _backend__smart_legend(
            ax, frameon=False, allow_outside=False, loc='upper right',
            bbox_to_anchor=(0.98, 0.96),
        )
        _backend__panel_label(ax, f"({chr(ord('a') + index)})")
    fig.tight_layout()
    _backend__save_figure(fig, out_dir, 'figure_4_hstar_discretization')
    plt.close(fig)

def _backend__normalise_limiters(limiters: Sequence[str]) -> Tuple[str, ...]:
    aliases = {'van_leer': 'vanleer'}
    allowed = {'minmod', 'mc', 'vanleer', 'superbee'}
    output: List[str] = []
    for raw in limiters:
        name = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if name not in allowed:
            raise ValueError(f'Unknown limiter {raw!r}; expected one of minmod, mc, vanleer, or superbee')
        if name not in output:
            output.append(name)
    if not output:
        raise ValueError('At least one limiter must be selected')
    return tuple(output)

def _backend__plot_limiter_profiles(out_dir: Path, results: Dict[Tuple[str, int], _backend_RitterResult], limiters: Sequence[str], profile_nx: int, field: str, exact_field: str, ylabel: str, title_name: str, filename: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    for limiter in limiters:
        result = results[limiter, profile_nx]
        ax.plot(result.x, getattr(result, field), linewidth=1.25, label=limiter.upper() if limiter == 'mc' else limiter.capitalize())
    reference = results[limiters[0], profile_nx]
    ax.plot(reference.x, getattr(reference, exact_field), color='black', linestyle='--', linewidth=1.8, label='Exact (Ritter)')
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if 'velocity' in filename.lower():
        ax.set_ylim(-5.0, 21.0)
        _backend__smart_legend(
            ax, frameon=False, ncol=2, allow_outside=False,
            loc='upper left', bbox_to_anchor=(0.02, 0.96),
        )
    else:
        _backend__smart_legend(
            ax, frameon=False, ncol=2, allow_outside=False,
            loc='upper right', bbox_to_anchor=(0.98, 0.96),
        )
    fig.tight_layout()
    fig.savefig(out_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

def _backend__plot_limiter_hstar(out_dir: Path, spatial_rows: Sequence[Dict[str, object]], temporal_rows: Sequence[Dict[str, object]], limiters: Sequence[str], monitor_x: float, t_end: float) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for limiter in limiters:
        label = limiter.upper() if limiter == 'mc' else limiter.capitalize()
        spatial = sorted((row for row in spatial_rows if str(row['limiter']) == limiter), key=lambda row: float(row['dx_m']))
        temporal = sorted((row for row in temporal_rows if str(row['limiter']) == limiter), key=lambda row: float(row['dt_s']))
        axes[0].plot([float(row['dx_m']) for row in spatial], [float(row['h_star_m']) for row in spatial], marker='o', linewidth=1.2, label=label)
        axes[1].plot([float(row['dt_s']) for row in temporal], [float(row['h_star_m']) for row in temporal], marker='o', linewidth=1.2, label=label)
    exact = float(spatial_rows[0]['h_star_exact_m'])
    subtitle = f'at $x^*={monitor_x:g}$ m and $t^*={t_end:g}$ s'
    for index, ax in enumerate(axes):
        ax.axhline(exact, color='black', linestyle='--', linewidth=1.2, label='Exact (Ritter)')
        line_values = []
        for line in ax.lines:
            ydata = np.asarray(line.get_ydata(), dtype=float)
            line_values.extend(ydata[np.isfinite(ydata)].tolist())
        ymin, ymax = min(line_values), max(line_values)
        yspan = max(ymax - ymin, 1.0e-6)
        ax.set_ylabel('$h^*$ [m]')
        ax.set_ylim(ymin - 0.08 * yspan, ymax + 0.95 * yspan)
        ax.grid(True, alpha=0.25)
        _backend__smart_legend(
            ax, frameon=False, ncol=2, allow_outside=False,
            loc='upper right', bbox_to_anchor=(0.98, 0.96),
        )
        _backend__panel_label(ax, f"({chr(ord('a') + index)})")
    axes[0].set_xlabel('$\\Delta x$ [m]')
    axes[1].set_xlabel('$\\Delta t$ [s]')
    fig.tight_layout()
    _backend__save_figure(fig, out_dir, 'figure_8_limiter_hstar_discretization')
    plt.close(fig)

def _backend_run_backend_verification_suite(out_dir: str, *, limiter: str='superbee', limiter_values: Sequence[str]=('minmod', 'mc', 'vanleer', 'superbee'), limiter_profile_nx: Optional[int]=None, nx_values: Sequence[int]=(100, 200, 400, 800, 1200, 2400), dt_values: Sequence[float]=(0.1, 0.05, 0.025, 0.0125, 0.00625), temporal_nx: int=100, xmin: float=-500.0, xmax: float=500.0, dam_x: float=0.0, h0: float=10.0, t_end: float=15.0, monitor_x: float=0.0, cfl: float=0.25, allow_cudasim: bool=False, backend: str='cuda', device_label: Optional[str]=None) -> Dict[str, object]:
    backend = str(backend).strip().lower()
    if backend not in {'cuda', 'cpu'}:
        raise ValueError("backend must be either 'cuda' or 'cpu'")
    if backend == 'cuda':
        device = device_label or _backend__cuda_device_label(allow_cudasim)
    else:
        device = device_label or 'NumPy/Numba CPU backend'
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    nx_values = tuple(sorted({int(value) for value in nx_values}))
    dt_values = tuple(sorted({float(value) for value in dt_values}, reverse=True))
    limiters = _backend__normalise_limiters(limiter_values)
    limiter = _backend__normalise_limiters((limiter,))[0]
    if not nx_values or not dt_values:
        raise ValueError('nx_values and dt_values cannot be empty')
    if temporal_nx < 16:
        raise ValueError('temporal_nx must be at least 16')
    if limiter_profile_nx is None:
        limiter_profile_nx = max(nx_values)
    limiter_profile_nx = int(limiter_profile_nx)
    if limiter_profile_nx not in nx_values:
        nx_values = tuple(sorted(set(nx_values + (limiter_profile_nx,))))
    h_exact_monitor, _, _ = _backend_ritter_exact(np.array([monitor_x]), t_end, h0, 9.81, dam_x)
    h_star_exact = float(h_exact_monitor[0])
    order_results: Dict[Tuple[int, int], _backend_RitterResult] = {}
    order_temporal_results: Dict[Tuple[int, float], _backend_RitterResult] = {}
    error_rows: List[Dict[str, object]] = []
    spatial_rows: List[Dict[str, object]] = []
    for order in (1, 2):
        order_errors: Dict[str, List[Dict[str, float]]] = {'h': [], 'qx': [], 'u': []}
        for nx in nx_values:
            print(f'[VERIFY] Order comparison: order={order}, Nx={nx}, limiter={limiter}')
            result = _backend_run_ritter_case(nx, order, limiter=limiter, xmin=xmin, xmax=xmax, dam_x=dam_x, h0=h0, t_end=t_end, cfl=cfl, allow_cudasim=allow_cudasim, backend=backend)
            order_results[order, nx] = result
            dx = (xmax - xmin) / nx
            for field, exact_field in (('h', 'h_exact'), ('qx', 'qx_exact'), ('u', 'u_exact')):
                errors = _backend__relative_errors(getattr(result, field), getattr(result, exact_field), dx)
                row = {'dx': dx, **errors}
                order_errors[field].append(row)
                error_rows.append({'order': order, 'limiter': limiter if order == 2 else 'not_used', 'nx': nx, 'dx_m': dx, 'field': field, **errors, 'steps': result.steps, 'dt_min_s': result.min_dt, 'dt_max_s': result.max_dt, 'dt_mean_s': result.mean_dt, 'hll_hllc_fallback_faces': result.fallback_faces, 'global_first_order_fallback_steps': result.first_order_fallback_steps, 'retries': result.retries})
            spatial_rows.append({'order': order, 'limiter': limiter if order == 2 else 'not_used', 'nx': nx, 'dx_m': dx, 'h_star_m': _backend__monitor_depth(result, monitor_x), 'h_star_exact_m': h_star_exact, 'monitor_x_m': monitor_x, 'time_s': t_end})
            _backend__save_profile_csv(output / f'ritter_profile_order{order}_nx{nx}.csv', result)
        for field in ('h', 'qx', 'u'):
            _backend__observed_orders(order_errors[field], 'rel_l1')
            _backend__observed_orders(order_errors[field], 'rel_l2')
            lookup = {round(float(row['dx']), 14): row for row in order_errors[field]}
            for target in error_rows:
                if int(target['order']) == order and target['field'] == field:
                    source = lookup[round(float(target['dx_m']), 14)]
                    target['observed_order_l1'] = source['order_rel_l1']
                    target['observed_order_l2'] = source['order_rel_l2']
    temporal_rows: List[Dict[str, object]] = []
    for order in (1, 2):
        for dt in dt_values:
            print(f'[VERIFY] Order h* sensitivity: order={order}, dt={dt:g} s, Nx={temporal_nx}')
            result = _backend_run_ritter_case(temporal_nx, order, limiter=limiter, xmin=xmin, xmax=xmax, dam_x=dam_x, h0=h0, t_end=t_end, cfl=cfl, fixed_dt=dt, allow_cudasim=allow_cudasim, backend=backend)
            order_temporal_results[order, dt] = result
            temporal_rows.append({'order': order, 'limiter': limiter if order == 2 else 'not_used', 'nx': temporal_nx, 'dx_m': (xmax - xmin) / temporal_nx, 'dt_s': dt, 'h_star_m': _backend__monitor_depth(result, monitor_x), 'h_star_exact_m': h_star_exact, 'monitor_x_m': monitor_x, 'time_s': t_end, 'steps': result.steps, 'hll_hllc_fallback_faces': result.fallback_faces, 'global_first_order_fallback_steps': result.first_order_fallback_steps})
    _backend__plot_profiles(output, order_results, nx_values, 'h', 'h_exact', '$h$ [m]', f'Ritter flow-depth profile at $t={t_end:g}$ s', 'figure_1_ritter_depth')
    _backend__plot_profiles(output, order_results, nx_values, 'qx', 'qx_exact', '$q_x$ [$\\mathrm{m^2\\,s^{-1}}$]', f'Ritter unit discharge at $t={t_end:g}$ s', 'figure_2_ritter_discharge')
    _backend__plot_profiles(output, order_results, nx_values, 'u', 'u_exact', '$u$ [$\\mathrm{m\\,s^{-1}}$]', f'Ritter velocity profile at $t={t_end:g}$ s', 'figure_3_ritter_velocity')
    _backend__plot_hstar(output, spatial_rows, temporal_rows, monitor_x, t_end)
    _backend__save_rows_csv(output / 'ritter_error_convergence.csv', error_rows)
    _backend__save_rows_csv(output / 'hstar_spatial_resolution.csv', spatial_rows)
    _backend__save_rows_csv(output / 'hstar_time_step.csv', temporal_rows)
    limiter_results: Dict[Tuple[str, int], _backend_RitterResult] = {}
    limiter_temporal_results: Dict[Tuple[str, float], _backend_RitterResult] = {}
    limiter_error_rows: List[Dict[str, object]] = []
    limiter_spatial_rows: List[Dict[str, object]] = []
    limiter_temporal_rows: List[Dict[str, object]] = []
    for current_limiter in limiters:
        field_errors: Dict[str, List[Dict[str, float]]] = {'h': [], 'qx': [], 'u': []}
        for nx in nx_values:
            if current_limiter == limiter:
                result = order_results[2, nx]
            else:
                print(f'[VERIFY] Limiter comparison: limiter={current_limiter}, order=2, Nx={nx}')
                result = _backend_run_ritter_case(nx, 2, limiter=current_limiter, xmin=xmin, xmax=xmax, dam_x=dam_x, h0=h0, t_end=t_end, cfl=cfl, allow_cudasim=allow_cudasim, backend=backend)
            limiter_results[current_limiter, nx] = result
            dx = (xmax - xmin) / nx
            for field, exact_field in (('h', 'h_exact'), ('qx', 'qx_exact'), ('u', 'u_exact')):
                errors = _backend__relative_errors(getattr(result, field), getattr(result, exact_field), dx)
                local_row = {'dx': dx, **errors}
                field_errors[field].append(local_row)
                limiter_error_rows.append({'order': 2, 'limiter': current_limiter, 'nx': nx, 'dx_m': dx, 'field': field, **errors, 'steps': result.steps, 'dt_min_s': result.min_dt, 'dt_max_s': result.max_dt, 'dt_mean_s': result.mean_dt, 'hll_hllc_fallback_faces': result.fallback_faces, 'global_first_order_fallback_steps': result.first_order_fallback_steps, 'retries': result.retries})
            limiter_spatial_rows.append({'order': 2, 'limiter': current_limiter, 'nx': nx, 'dx_m': dx, 'h_star_m': _backend__monitor_depth(result, monitor_x), 'h_star_exact_m': h_star_exact, 'monitor_x_m': monitor_x, 'time_s': t_end})
            _backend__save_profile_csv(output / f'ritter_limiter_{current_limiter}_nx{nx}.csv', result)
        for field in ('h', 'qx', 'u'):
            _backend__observed_orders(field_errors[field], 'rel_l1')
            _backend__observed_orders(field_errors[field], 'rel_l2')
            lookup = {round(float(row['dx']), 14): row for row in field_errors[field]}
            for target in limiter_error_rows:
                if target['limiter'] == current_limiter and target['field'] == field:
                    source = lookup[round(float(target['dx_m']), 14)]
                    target['observed_order_l1'] = source['order_rel_l1']
                    target['observed_order_l2'] = source['order_rel_l2']
        for dt in dt_values:
            if current_limiter == limiter:
                result = order_temporal_results[2, dt]
            else:
                print(f'[VERIFY] Limiter h* sensitivity: limiter={current_limiter}, order=2, dt={dt:g} s, Nx={temporal_nx}')
                result = _backend_run_ritter_case(temporal_nx, 2, limiter=current_limiter, xmin=xmin, xmax=xmax, dam_x=dam_x, h0=h0, t_end=t_end, cfl=cfl, fixed_dt=dt, allow_cudasim=allow_cudasim, backend=backend)
            limiter_temporal_results[current_limiter, dt] = result
            limiter_temporal_rows.append({'order': 2, 'limiter': current_limiter, 'nx': temporal_nx, 'dx_m': (xmax - xmin) / temporal_nx, 'dt_s': dt, 'h_star_m': _backend__monitor_depth(result, monitor_x), 'h_star_exact_m': h_star_exact, 'monitor_x_m': monitor_x, 'time_s': t_end, 'steps': result.steps, 'hll_hllc_fallback_faces': result.fallback_faces, 'global_first_order_fallback_steps': result.first_order_fallback_steps})
    _backend__plot_limiter_profiles(output, limiter_results, limiters, limiter_profile_nx, 'h', 'h_exact', '$h$ [m]', f'Second-order limiter comparison for $h(x)$ at $t={t_end:g}$ s, $N_x={limiter_profile_nx}$', 'figure_5_limiter_depth')
    _backend__plot_limiter_profiles(output, limiter_results, limiters, limiter_profile_nx, 'qx', 'qx_exact', '$q_x$ [$\\mathrm{m^2\\,s^{-1}}$]', f'Second-order limiter comparison for $q_x(x)$ at $t={t_end:g}$ s, $N_x={limiter_profile_nx}$', 'figure_6_limiter_discharge')
    _backend__plot_limiter_profiles(output, limiter_results, limiters, limiter_profile_nx, 'u', 'u_exact', '$u$ [$\\mathrm{m\\,s^{-1}}$]', f'Second-order limiter comparison for $u(x)$ at $t={t_end:g}$ s, $N_x={limiter_profile_nx}$', 'figure_7_limiter_velocity')
    _backend__plot_limiter_hstar(output, limiter_spatial_rows, limiter_temporal_rows, limiters, monitor_x, t_end)
    _backend__save_rows_csv(output / 'ritter_limiter_error_convergence.csv', limiter_error_rows)
    _backend__save_rows_csv(output / 'hstar_limiter_spatial_resolution.csv', limiter_spatial_rows)
    _backend__save_rows_csv(output / 'hstar_limiter_time_step.csv', limiter_temporal_rows)
    figures = ['figure_1_ritter_depth.png', 'figure_2_ritter_discharge.png', 'figure_3_ritter_velocity.png', 'figure_4_hstar_discretization.png', 'figure_5_limiter_depth.png', 'figure_6_limiter_discharge.png', 'figure_7_limiter_velocity.png', 'figure_8_limiter_hstar_discretization.png']
    summary: Dict[str, object] = {'backend': backend, 'device': device, 'order_comparison_second_order_limiter': limiter, 'limiter_comparison_order': 2, 'limiters_compared': list(limiters), 'limiter_profile_nx': limiter_profile_nx, 'orders': [1, 2], 'nx_values': list(nx_values), 'dt_values_s': list(dt_values), 'temporal_nx': temporal_nx, 'domain_m': [xmin, xmax], 'dam_x_m': dam_x, 'reservoir_depth_m': h0, 'final_time_s': t_end, 'monitor_x_m': monitor_x, 'h_star_definition': 'linearly interpolated numerical flow depth at monitor_x and final_time', 'h_star_exact_m': h_star_exact, 'figures': figures}
    with (output / 'backend_verification_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    return summary


def run_backend_suite(
    settings: Dict[str, object],
    test_config: Path,
    *,
    backend: str,
    device_label: str,
) -> Dict[str, object]:
    out_dir = _resolve_from_config(
        test_config,
        str(settings.get("figure_output_dir", "../outputs/figures")),
    )
    limiter = str(settings.get("limiter", "superbee")).lower()
    limiters = tuple(str(value).lower() for value in settings.get("limiters", LIMITER_CHOICES))
    invalid = [value for value in (limiter, *limiters) if value not in LIMITER_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported limiter values in config_tests.yaml: {invalid}")

    nx_values = [int(value) for value in settings.get("nx", [100, 200, 400, 800, 1200, 2400])]
    dt_values = [float(value) for value in settings.get("dt", [0.1, 0.05, 0.025, 0.0125, 0.00625])]
    temporal_nx = int(settings.get("temporal_nx", 100))
    limiter_profile_nx = int(settings.get("limiter_profile_nx", max(nx_values)))

    print(f"[BACKEND] Running the complete suite on {backend.upper()}: {device_label}")
    return _backend_run_backend_verification_suite(
        str(out_dir),
        limiter=limiter,
        limiter_values=limiters,
        limiter_profile_nx=limiter_profile_nx,
        nx_values=nx_values,
        dt_values=dt_values,
        temporal_nx=temporal_nx,
        xmin=float(settings.get("xmin", -500.0)),
        xmax=float(settings.get("xmax", 500.0)),
        dam_x=float(settings.get("dam_x", 0.0)),
        h0=float(settings.get("h0", 10.0)),
        t_end=float(settings.get("t_end", 15.0)),
        monitor_x=float(settings.get("monitor_x", 0.0)),
        cfl=float(settings.get("cfl", 0.25)),
        backend=backend,
        device_label=device_label,
    )


def main(argv=None) -> int:
    global OUT_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Run all debris-flow tests. CUDA is attempted first and the complete "
            "suite falls back automatically to CPU when CUDA is unavailable."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TEST_CONFIG),
        help="General test YAML configuration.",
    )
    parser.add_argument(
        "--marsicano-config",
        default=str(DEFAULT_MARSICANO_CONFIG),
        help="Marsicano YAML configuration validated by the suite.",
    )
    args = parser.parse_args(argv)

    test_config = Path(args.config).resolve()
    marsicano_config = Path(args.marsicano_config).resolve()
    settings = _read_test_settings(test_config)

    print("[CONFIG] Validating repository configuration files...")
    config_results = validate_configuration_files(test_config, marsicano_config)
    print(json.dumps(config_results, indent=2))

    test_cfg = load_config(str(test_config))
    backend_info = resolve_backend(test_cfg)
    if backend_info.selected == "cpu":
        active_threads = configure_cpu_threads(test_cfg)
        device_label = f"CPU backend with {active_threads} Numba threads"
        print(f"[BACKEND] CUDA unavailable: {backend_info.reason}")
        print(f"[BACKEND] Automatic CPU fallback enabled ({active_threads} threads).")
    else:
        device_label = backend_info.cuda_device_name or "CUDA device"
        print(f"[BACKEND] CUDA selected: {device_label}")

    print("[TESTS] Running the general numerical verification suite...")
    numerical_results = self_test(verbose=True)

    OUT_DIR = _resolve_from_config(
        test_config,
        str(settings.get("figure_output_dir", "../outputs/figures")),
    )
    print(f"[FIGURES] Writing general verification figures to {OUT_DIR}")
    figure_results = generate_all(verbose=True)

    backend_results = run_backend_suite(
        settings,
        test_config,
        backend=backend_info.selected,
        device_label=device_label,
    )
    print(json.dumps(backend_results, indent=2))

    summary = {
        "configurations": config_results,
        "requested_backend": backend_info.requested,
        "selected_backend": backend_info.selected,
        "backend_reason": backend_info.reason,
        "numerical_test_count": len(numerical_results),
        "figures_generated": 2 * len(figure_results) + 1,
        "complete_backend_suite_executed": True,
    }
    print("[SUMMARY]")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
