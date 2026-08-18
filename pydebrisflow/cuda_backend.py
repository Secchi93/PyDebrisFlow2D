from __future__ import annotations

"""CUDA-resident finite-volume backend implemented with Numba CUDA."""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from numba import cuda, float64, int32

from .config import SolverConfig, _bc_id, _limiter_id
from .constants import (
    BC_PERIODIC,
    BC_REFLECTIVE,
    FLUX_HLL,
    FLUX_HLLC,
    HCL,
    HCU,
    HF,
    HSL,
    HSU,
    LIMITER_MINMOD,
    LIMITER_VANLEER,
    LIMITER_SUPERBEE,
    MX,
    MY,
    NV,
)
from .dem import load_or_build_dem
from .geometry import prepare_interactive_geometry, save_setup_preview
from .outputs import make_depth_gif, save_maps, save_snapshot
from .physics import (
    build_release,
    composition_fields,
    ferguson_church_settling_velocity,
    material_budget,
    terrain_geometry,
    terrain_velocity_components,
)


# -----------------------------------------------------------------------------
# Scalar device utilities.
# -----------------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@cuda.jit(device=True, inline=True)
def _minmod2(a: float, b: float) -> float:
    if a * b <= 0.0:
        return 0.0
    return math.copysign(min(abs(a), abs(b)), a)


@cuda.jit(device=True, inline=True)
def _limited_slope(dl: float, dr: float, limiter: int) -> float:
    if limiter == LIMITER_MINMOD:
        return _minmod2(dl, dr)
    if limiter == LIMITER_VANLEER:
        if dl * dr <= 0.0:
            return 0.0
        return 2.0 * dl * dr / (dl + dr + 1.0e-30)
    if limiter == LIMITER_SUPERBEE:
        if dl * dr <= 0.0:
            return 0.0
        magnitude = max(min(2.0 * abs(dl), abs(dr)), min(abs(dl), 2.0 * abs(dr)))
        return math.copysign(magnitude, dl)
    # Monotonized-central (MC) limiter.
    return _minmod2(0.5 * (dl + dr), _minmod2(2.0 * dl, 2.0 * dr))


@cuda.jit(device=True, inline=True)
def _derived(U, rho_f: float, rho_s: float, h_dry: float):
    hf = max(U[HF], 0.0)
    hsu = max(U[HSU], 0.0)
    hcu = _clamp(U[HCU], 0.0, hsu)
    hsl = max(U[HSL], 0.0)
    hcl = _clamp(U[HCL], 0.0, hsl)
    hs = hsu + hsl
    h = hf + hs
    mass = rho_f * hf + rho_s * hs
    if h <= h_dry or mass <= 0.0:
        return h, mass, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, rho_f
    u = U[MX] / mass
    v = U[MY] / mass
    cs = _clamp(hs / h, 0.0, 1.0)
    lam = _clamp(hsu / (hs + 1.0e-30), 0.0, 1.0) if hs > 0.0 else 0.5
    fcu = _clamp(hcu / (hsu + 1.0e-30), 0.0, 1.0) if hsu > 0.0 else 0.0
    fcl = _clamp(hcl / (hsl + 1.0e-30), 0.0, 1.0) if hsl > 0.0 else 0.0
    rho = mass / h
    return h, mass, u, v, cs, lam, fcu, fcl, rho


@cuda.jit(device=True, inline=True)
def _primitive_state(
    h: float,
    u: float,
    v: float,
    cs: float,
    lam: float,
    fcu: float,
    fcl: float,
    rho_f: float,
    rho_s: float,
    max_cs: float,
    out,
) -> None:
    h = max(h, 0.0)
    cs = _clamp(cs, 0.0, max_cs)
    lam = _clamp(lam, 0.0, 1.0)
    fcu = _clamp(fcu, 0.0, 1.0)
    fcl = _clamp(fcl, 0.0, 1.0)
    hs = h * cs
    hf = h - hs
    hsu = hs * lam
    hsl = hs - hsu
    hcu = hsu * fcu
    hcl = hsl * fcl
    mass = rho_f * hf + rho_s * hs
    out[HF] = hf
    out[HSU] = hsu
    out[HCU] = hcu
    out[HSL] = hsl
    out[HCL] = hcl
    out[MX] = mass * u
    out[MY] = mass * v


@cuda.jit(device=True, inline=True)
def _physical_flux(U, nxn: float, nyn: float, g: float, rho_f: float, rho_s: float, h_dry: float, F) -> None:
    for k in range(NV):
        F[k] = 0.0
    h, mass, u, v, _, _, _, _, rho = _derived(U, rho_f, rho_s, h_dry)
    if h <= h_dry or mass <= 0.0:
        return
    un = u * nxn + v * nyn
    for k in range(5):
        F[k] = U[k] * un
    p = 0.5 * rho * g * h * h
    F[MX] = U[MX] * un + p * nxn
    F[MY] = U[MY] * un + p * nyn


@cuda.jit(device=True, inline=True)
def _hll_flux(UL, UR, nxn: float, nyn: float, g: float, rho_f: float, rho_s: float, h_dry: float, F) -> None:
    hL, _, uL, vL, _, _, _, _, _ = _derived(UL, rho_f, rho_s, h_dry)
    hR, _, uR, vR, _, _, _, _, _ = _derived(UR, rho_f, rho_s, h_dry)
    if hL <= h_dry and hR <= h_dry:
        for k in range(NV):
            F[k] = 0.0
        return
    unL = uL * nxn + vL * nyn
    unR = uR * nxn + vR * nyn
    cL = math.sqrt(g * max(hL, 0.0))
    cR = math.sqrt(g * max(hR, 0.0))
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)
    FL = cuda.local.array(NV, dtype=float64)
    FR = cuda.local.array(NV, dtype=float64)
    _physical_flux(UL, nxn, nyn, g, rho_f, rho_s, h_dry, FL)
    _physical_flux(UR, nxn, nyn, g, rho_f, rho_s, h_dry, FR)
    if SL >= 0.0:
        for k in range(NV):
            F[k] = FL[k]
    elif SR <= 0.0:
        for k in range(NV):
            F[k] = FR[k]
    else:
        den = SR - SL + 1.0e-30
        for k in range(NV):
            F[k] = (SR * FL[k] - SL * FR[k] + SL * SR * (UR[k] - UL[k])) / den


@cuda.jit(device=True, inline=True)
def _hllc_flux(
    UL,
    UR,
    nxn: float,
    nyn: float,
    g: float,
    rho_f: float,
    rho_s: float,
    h_dry: float,
    dry_factor: float,
    max_froude: float,
    F,
) -> int:
    hL, mL, uL, vL, _, _, _, _, rhoL = _derived(UL, rho_f, rho_s, h_dry)
    hR, mR, uR, vR, _, _, _, _, rhoR = _derived(UR, rho_f, rho_s, h_dry)
    if hL <= h_dry and hR <= h_dry:
        for k in range(NV):
            F[k] = 0.0
        return 0
    if hL <= dry_factor * h_dry or hR <= dry_factor * h_dry:
        _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
        return 1
    unL = uL * nxn + vL * nyn
    unR = uR * nxn + vR * nyn
    utL = -uL * nyn + vL * nxn
    utR = -uR * nyn + vR * nxn
    cL = math.sqrt(g * hL)
    cR = math.sqrt(g * hR)
    if abs(unL) / (cL + 1.0e-30) > max_froude or abs(unR) / (cR + 1.0e-30) > max_froude:
        _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
        return 1
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)
    FL = cuda.local.array(NV, dtype=float64)
    FR = cuda.local.array(NV, dtype=float64)
    _physical_flux(UL, nxn, nyn, g, rho_f, rho_s, h_dry, FL)
    _physical_flux(UR, nxn, nyn, g, rho_f, rho_s, h_dry, FR)
    if SL >= 0.0:
        for k in range(NV):
            F[k] = FL[k]
        return 0
    if SR <= 0.0:
        for k in range(NV):
            F[k] = FR[k]
        return 0
    pL = 0.5 * rhoL * g * hL * hL
    pR = 0.5 * rhoR * g * hR * hR
    den = mL * (SL - unL) - mR * (SR - unR)
    if abs(den) < 1.0e-14:
        _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
        return 1
    Sstar = (pR - pL + mL * unL * (SL - unL) - mR * unR * (SR - unR)) / den
    if not math.isfinite(Sstar) or Sstar <= SL or Sstar >= SR:
        _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
        return 1
    Ustar = cuda.local.array(NV, dtype=float64)
    if Sstar >= 0.0:
        ratio = (SL - unL) / (SL - Sstar + 1.0e-30)
        if ratio <= 0.0:
            _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
            return 1
        for k in range(5):
            Ustar[k] = UL[k] * ratio
        mstar = mL * ratio
        mn = mstar * Sstar
        mt = mstar * utL
        Ustar[MX] = mn * nxn - mt * nyn
        Ustar[MY] = mn * nyn + mt * nxn
        for k in range(NV):
            if not math.isfinite(Ustar[k]):
                _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
                return 1
            F[k] = FL[k] + SL * (Ustar[k] - UL[k])
        return 0
    ratio = (SR - unR) / (SR - Sstar + 1.0e-30)
    if ratio <= 0.0:
        _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
        return 1
    for k in range(5):
        Ustar[k] = UR[k] * ratio
    mstar = mR * ratio
    mn = mstar * Sstar
    mt = mstar * utR
    Ustar[MX] = mn * nxn - mt * nyn
    Ustar[MY] = mn * nyn + mt * nxn
    for k in range(NV):
        if not math.isfinite(Ustar[k]):
            _hll_flux(UL, UR, nxn, nyn, g, rho_f, rho_s, h_dry, F)
            return 1
        F[k] = FR[k] + SR * (Ustar[k] - UR[k])
    return 0


@cuda.jit(device=True, inline=True)
def _build_face_state(
    eta: float,
    u: float,
    v: float,
    cs: float,
    lam: float,
    fcu: float,
    fcl: float,
    z_local: float,
    z_star: float,
    rho_f: float,
    rho_s: float,
    max_cs: float,
    g: float,
    Ustar,
) -> float:
    h_orig = max(eta - z_local, 0.0)
    h_star = max(eta - z_star, 0.0)
    _primitive_state(h_star, u, v, cs, lam, fcu, fcl, rho_f, rho_s, max_cs, Ustar)
    c = _clamp(cs, 0.0, max_cs)
    rho = rho_f * (1.0 - c) + rho_s * c
    return 0.5 * rho * g * (h_orig * h_orig - h_star * h_star)


# -----------------------------------------------------------------------------
# Transport kernels.
# -----------------------------------------------------------------------------
@cuda.jit
def _boundary_x(U, zb, active, ng: int, left_bc: int, right_bc: int) -> None:
    j = cuda.grid(1)
    ny = U.shape[1]
    nx = U.shape[2]
    if j >= ny:
        return
    for q in range(ng):
        if left_bc == BC_PERIODIC:
            src = nx - 2 * ng + q
        else:
            src = ng
        for k in range(NV):
            value = U[k, j, src]
            if left_bc == BC_REFLECTIVE and k == MX:
                value = -value
            U[k, j, q] = value
        zb[j, q] = zb[j, src]
        active[j, q] = active[j, src]

        dst = nx - ng + q
        if right_bc == BC_PERIODIC:
            src = ng + q
        else:
            src = nx - ng - 1
        for k in range(NV):
            value = U[k, j, src]
            if right_bc == BC_REFLECTIVE and k == MX:
                value = -value
            U[k, j, dst] = value
        zb[j, dst] = zb[j, src]
        active[j, dst] = active[j, src]


@cuda.jit
def _boundary_y(U, zb, active, ng: int, bottom_bc: int, top_bc: int) -> None:
    i = cuda.grid(1)
    ny = U.shape[1]
    nx = U.shape[2]
    if i >= nx:
        return
    for q in range(ng):
        if bottom_bc == BC_PERIODIC:
            src = ny - 2 * ng + q
        else:
            src = ng
        for k in range(NV):
            value = U[k, src, i]
            if bottom_bc == BC_REFLECTIVE and k == MY:
                value = -value
            U[k, q, i] = value
        zb[q, i] = zb[src, i]
        active[q, i] = active[src, i]

        dst = ny - ng + q
        if top_bc == BC_PERIODIC:
            src = ng + q
        else:
            src = ny - ng - 1
        for k in range(NV):
            value = U[k, src, i]
            if top_bc == BC_REFLECTIVE and k == MY:
                value = -value
            U[k, dst, i] = value
        zb[dst, i] = zb[src, i]
        active[dst, i] = active[src, i]


@cuda.jit
def _compute_primitives(U, zb, active, P, rho_f: float, rho_s: float, h_dry: float) -> None:
    i, j = cuda.grid(2)
    ny, nx = zb.shape
    if i >= nx or j >= ny:
        return
    if active[j, i] == 0:
        for k in range(7):
            P[k, j, i] = 0.0
        P[0, j, i] = zb[j, i]
        P[5, j, i] = 0.5
        return
    local_u = cuda.local.array(NV, dtype=float64)
    for k in range(NV):
        local_u[k] = U[k, j, i]
    h, _, u, v, cs, lam, fcu, fcl, _ = _derived(local_u, rho_f, rho_s, h_dry)
    P[0, j, i] = h + zb[j, i]
    P[1, j, i] = u
    P[2, j, i] = v
    P[3, j, i] = cs
    P[4, j, i] = lam
    P[5, j, i] = fcu
    P[6, j, i] = fcl


@cuda.jit
def _compute_slopes(P, active, sx, sy, limiter: int) -> None:
    i, j = cuda.grid(2)
    _, ny, nx = P.shape
    if i >= nx or j >= ny:
        return
    for k in range(7):
        sx[k, j, i] = 0.0
        sy[k, j, i] = 0.0
    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1:
        return
    if active[j, i] == 0 or active[j, i - 1] == 0 or active[j, i + 1] == 0 or active[j - 1, i] == 0 or active[j + 1, i] == 0:
        return
    for k in range(7):
        sx[k, j, i] = _limited_slope(P[k, j, i] - P[k, j, i - 1], P[k, j, i + 1] - P[k, j, i], limiter)
        sy[k, j, i] = _limited_slope(P[k, j, i] - P[k, j - 1, i], P[k, j + 1, i] - P[k, j, i], limiter)


@cuda.jit
def _flux_x(
    zb,
    active,
    P,
    sx,
    fx,
    cxl,
    cxr,
    order: int,
    flux_id: int,
    g: float,
    rho_f: float,
    rho_s: float,
    max_cs: float,
    h_dry: float,
    dry_factor: float,
    max_froude: float,
    hybrid_jump: float,
    fallback_count,
) -> None:
    i, j = cuda.grid(2)
    ny, nx = zb.shape
    if i <= 0 or i >= nx or j <= 0 or j >= ny - 1:
        return
    aL = active[j, i - 1]
    aR = active[j, i]
    if aL == 0 and aR == 0:
        for k in range(NV):
            fx[k, j, i] = 0.0
        cxl[j, i] = 0.0
        cxr[j, i] = 0.0
        return
    PL = cuda.local.array(7, dtype=float64)
    PR = cuda.local.array(7, dtype=float64)
    if aL == 1:
        for k in range(7):
            PL[k] = P[k, j, i - 1] + (0.5 * sx[k, j, i - 1] if order == 2 else 0.0)
    if aR == 1:
        for k in range(7):
            PR[k] = P[k, j, i] - (0.5 * sx[k, j, i] if order == 2 else 0.0)
    zL = zb[j, i - 1]
    zR = zb[j, i]
    if aL == 0:
        for k in range(7):
            PL[k] = PR[k]
        PL[1] = -PR[1]
        zL = zR
    elif aR == 0:
        for k in range(7):
            PR[k] = PL[k]
        PR[1] = -PL[1]
        zR = zL
    zstar = max(zL, zR)
    UL = cuda.local.array(NV, dtype=float64)
    UR = cuda.local.array(NV, dtype=float64)
    dpL = _build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g, UL)
    dpR = _build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g, UR)
    hL, _, _, _, _, _, _, _, _ = _derived(UL, rho_f, rho_s, h_dry)
    hR, _, _, _, _, _, _, _, _ = _derived(UR, rho_f, rho_s, h_dry)
    rel_jump = abs(hR - hL) / max(max(hR, hL), h_dry)
    F = cuda.local.array(NV, dtype=float64)
    fb = 0
    if flux_id == FLUX_HLL:
        _hll_flux(UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry, F)
    elif rel_jump > hybrid_jump:
        fb = 1
    else:
        fb = _hllc_flux(UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry, dry_factor, max_froude, F)

    if fb:
        # Complete local fallback: discard MUSCL face values, reconstruct both
        # adjacent cells as piecewise constants, rebuild hydrostatic states,
        # and only then evaluate HLL.
        if aL == 1:
            for k in range(7):
                PL[k] = P[k, j, i - 1]
        if aR == 1:
            for k in range(7):
                PR[k] = P[k, j, i]
        zL = zb[j, i - 1]
        zR = zb[j, i]
        if aL == 0:
            for k in range(7):
                PL[k] = PR[k]
            PL[1] = -PR[1]
            zL = zR
        elif aR == 0:
            for k in range(7):
                PR[k] = PL[k]
            PR[1] = -PL[1]
            zR = zL
        zstar = max(zL, zR)
        dpL = _build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g, UL)
        dpR = _build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g, UR)
        _hll_flux(UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry, F)
        cuda.atomic.add(fallback_count, 0, 1)
    for k in range(NV):
        fx[k, j, i] = F[k]
    cxl[j, i] = dpL
    cxr[j, i] = dpR


@cuda.jit
def _flux_y(
    zb,
    active,
    P,
    sy,
    fy,
    cyl,
    cyr,
    order: int,
    flux_id: int,
    g: float,
    rho_f: float,
    rho_s: float,
    max_cs: float,
    h_dry: float,
    dry_factor: float,
    max_froude: float,
    hybrid_jump: float,
    fallback_count,
) -> None:
    i, j = cuda.grid(2)
    ny, nx = zb.shape
    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny:
        return
    aL = active[j - 1, i]
    aR = active[j, i]
    if aL == 0 and aR == 0:
        for k in range(NV):
            fy[k, j, i] = 0.0
        cyl[j, i] = 0.0
        cyr[j, i] = 0.0
        return
    PL = cuda.local.array(7, dtype=float64)
    PR = cuda.local.array(7, dtype=float64)
    if aL == 1:
        for k in range(7):
            PL[k] = P[k, j - 1, i] + (0.5 * sy[k, j - 1, i] if order == 2 else 0.0)
    if aR == 1:
        for k in range(7):
            PR[k] = P[k, j, i] - (0.5 * sy[k, j, i] if order == 2 else 0.0)
    zL = zb[j - 1, i]
    zR = zb[j, i]
    if aL == 0:
        for k in range(7):
            PL[k] = PR[k]
        PL[2] = -PR[2]
        zL = zR
    elif aR == 0:
        for k in range(7):
            PR[k] = PL[k]
        PR[2] = -PL[2]
        zR = zL
    zstar = max(zL, zR)
    UL = cuda.local.array(NV, dtype=float64)
    UR = cuda.local.array(NV, dtype=float64)
    dpL = _build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g, UL)
    dpR = _build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g, UR)
    hL, _, _, _, _, _, _, _, _ = _derived(UL, rho_f, rho_s, h_dry)
    hR, _, _, _, _, _, _, _, _ = _derived(UR, rho_f, rho_s, h_dry)
    rel_jump = abs(hR - hL) / max(max(hR, hL), h_dry)
    F = cuda.local.array(NV, dtype=float64)
    fb = 0
    if flux_id == FLUX_HLL:
        _hll_flux(UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry, F)
    elif rel_jump > hybrid_jump:
        fb = 1
    else:
        fb = _hllc_flux(UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry, dry_factor, max_froude, F)

    if fb:
        if aL == 1:
            for k in range(7):
                PL[k] = P[k, j - 1, i]
        if aR == 1:
            for k in range(7):
                PR[k] = P[k, j, i]
        zL = zb[j - 1, i]
        zR = zb[j, i]
        if aL == 0:
            for k in range(7):
                PL[k] = PR[k]
            PL[2] = -PR[2]
            zL = zR
        elif aR == 0:
            for k in range(7):
                PR[k] = PL[k]
            PR[2] = -PL[2]
            zR = zL
        zstar = max(zL, zR)
        dpL = _build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g, UL)
        dpR = _build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g, UR)
        _hll_flux(UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry, F)
        cuda.atomic.add(fallback_count, 0, 1)
    for k in range(NV):
        fy[k, j, i] = F[k]
    cyl[j, i] = dpL
    cyr[j, i] = dpR


@cuda.jit
def _divergence(fx, fy, cxl, cxr, cyl, cyr, active, rhs, dx: float, dy: float, ng: int) -> None:
    i, j = cuda.grid(2)
    ny, nx = active.shape
    if i < ng or i >= nx - ng or j < ng or j >= ny - ng:
        return
    if active[j, i] == 0:
        for k in range(NV):
            rhs[k, j, i] = 0.0
        return
    for k in range(NV):
        rhs[k, j, i] = -(fx[k, j, i + 1] - fx[k, j, i]) / dx - (fy[k, j + 1, i] - fy[k, j, i]) / dy
    rhs[MX, j, i] += -(cxl[j, i + 1] - cxr[j, i]) / dx
    rhs[MY, j, i] += -(cyl[j + 1, i] - cyr[j, i]) / dy


@cuda.jit
def _boundary_budget(fx, fy, out, dx: float, dy: float, ng: int, ny_core: int, nx_core: int) -> None:
    q = cuda.grid(1)
    if q < ny_core:
        j = ng + q
        for k in range(5):
            cuda.atomic.add(out, k, dy * (fx[k, j, ng] - fx[k, j, ng + nx_core]))
    if q < nx_core:
        i = ng + q
        for k in range(5):
            cuda.atomic.add(out, k, dx * (fy[k, ng, i] - fy[k, ng + ny_core, i]))


@cuda.jit
def _euler_update(U, rhs, out, dt: float) -> None:
    i, j = cuda.grid(2)
    ny, nx = U.shape[1], U.shape[2]
    if i >= nx or j >= ny:
        return
    for k in range(NV):
        out[k, j, i] = U[k, j, i] + dt * rhs[k, j, i]


@cuda.jit
def _ssprk2_combine(U0, U1, rhs, out, dt: float) -> None:
    i, j = cuda.grid(2)
    ny, nx = U0.shape[1], U0.shape[2]
    if i >= nx or j >= ny:
        return
    for k in range(NV):
        out[k, j, i] = 0.5 * (U0[k, j, i] + U1[k, j, i] + dt * rhs[k, j, i])


@cuda.jit
def _admissibility(U, flag, ng: int, tol: float) -> None:
    i, j = cuda.grid(2)
    ny, nx = U.shape[1], U.shape[2]
    if i < ng or i >= nx - ng or j < ng or j >= ny - ng:
        return
    for k in range(NV):
        if not math.isfinite(U[k, j, i]):
            cuda.atomic.max(flag, 0, 1)
            return
    if U[HF, j, i] < -tol or U[HSU, j, i] < -tol or U[HSL, j, i] < -tol:
        cuda.atomic.max(flag, 0, 1)
    elif U[HCU, j, i] < -tol or U[HCL, j, i] < -tol:
        cuda.atomic.max(flag, 0, 1)
    elif U[HCU, j, i] - U[HSU, j, i] > tol or U[HCL, j, i] - U[HSL, j, i] > tol:
        cuda.atomic.max(flag, 0, 1)


@cuda.jit
def _roundoff_repair(U, ng: int, h_dry: float, tol: float) -> None:
    i, j = cuda.grid(2)
    ny, nx = U.shape[1], U.shape[2]
    if i < ng or i >= nx - ng or j < ng or j >= ny - ng:
        return
    indices = cuda.local.array(5, dtype=int32)
    indices[0] = HF
    indices[1] = HSU
    indices[2] = HSL
    indices[3] = HCU
    indices[4] = HCL
    for q in range(5):
        k = indices[q]
        value = U[k, j, i]
        if value < 0.0 and value >= -tol:
            U[k, j, i] = 0.0
    U[HCU, j, i] = min(max(U[HCU, j, i], 0.0), U[HSU, j, i])
    U[HCL, j, i] = min(max(U[HCL, j, i], 0.0), U[HSL, j, i])
    h = U[HF, j, i] + U[HSU, j, i] + U[HSL, j, i]
    if h <= h_dry:
        U[MX, j, i] = 0.0
        U[MY, j, i] = 0.0


@cuda.jit
def _wave_speed_blocks(U, out_x, out_y, ng: int, nx_core: int, ny_core: int, rho_f: float, rho_s: float, h_dry: float, g: float) -> None:
    shared_x = cuda.shared.array(1024, dtype=float64)
    shared_y = cuda.shared.array(1024, dtype=float64)
    tid = cuda.threadIdx.x
    q = cuda.grid(1)
    n = nx_core * ny_core
    sx = 0.0
    sy = 0.0
    if q < n:
        jc = q // nx_core
        ic = q - jc * nx_core
        j = jc + ng
        i = ic + ng
        hf = max(U[HF, j, i], 0.0)
        hs = max(U[HSU, j, i], 0.0) + max(U[HSL, j, i], 0.0)
        h = hf + hs
        mass = rho_f * hf + rho_s * hs
        if h > h_dry and mass > 0.0:
            u = U[MX, j, i] / mass
            v = U[MY, j, i] / mass
            c = math.sqrt(g * h)
            sx = abs(u) + c
            sy = abs(v) + c
    shared_x[tid] = sx
    shared_y[tid] = sy
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared_x[tid] = max(shared_x[tid], shared_x[tid + stride])
            shared_y[tid] = max(shared_y[tid], shared_y[tid + stride])
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        out_x[cuda.blockIdx.x] = shared_x[0]
        out_y[cuda.blockIdx.x] = shared_y[0]


@cuda.jit
def _diagnostic_blocks(U, out_h, out_speed, out_rho_min, out_rho_max, ng: int, nx_core: int, ny_core: int, rho_f: float, rho_s: float, h_dry: float) -> None:
    sh_h = cuda.shared.array(1024, dtype=float64)
    sh_s = cuda.shared.array(1024, dtype=float64)
    sh_rmin = cuda.shared.array(1024, dtype=float64)
    sh_rmax = cuda.shared.array(1024, dtype=float64)
    tid = cuda.threadIdx.x
    q = cuda.grid(1)
    n = nx_core * ny_core
    h = 0.0
    speed = 0.0
    rmin = 1.0e300
    rmax = 0.0
    if q < n:
        jc = q // nx_core
        ic = q - jc * nx_core
        j = jc + ng
        i = ic + ng
        hf = max(U[HF, j, i], 0.0)
        hs = max(U[HSU, j, i], 0.0) + max(U[HSL, j, i], 0.0)
        h = hf + hs
        mass = rho_f * hf + rho_s * hs
        if h > h_dry and mass > 0.0:
            u = U[MX, j, i] / mass
            v = U[MY, j, i] / mass
            speed = math.sqrt(u * u + v * v)
            rho = mass / h
            rmin = rho
            rmax = rho
    sh_h[tid] = h
    sh_s[tid] = speed
    sh_rmin[tid] = rmin
    sh_rmax[tid] = rmax
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sh_h[tid] = max(sh_h[tid], sh_h[tid + stride])
            sh_s[tid] = max(sh_s[tid], sh_s[tid + stride])
            sh_rmin[tid] = min(sh_rmin[tid], sh_rmin[tid + stride])
            sh_rmax[tid] = max(sh_rmax[tid], sh_rmax[tid + stride])
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        b = cuda.blockIdx.x
        out_h[b] = sh_h[0]
        out_speed[b] = sh_s[0]
        out_rho_min[b] = sh_rmin[0]
        out_rho_max[b] = sh_rmax[0]


# -----------------------------------------------------------------------------
# Source-term kernels.
# -----------------------------------------------------------------------------
@cuda.jit
def _terrain_cosbeta(zb, cosbeta, ng: int, nx_core: int, ny_core: int, dx: float, dy: float) -> None:
    ic, jc = cuda.grid(2)
    if ic >= nx_core or jc >= ny_core:
        return
    i = ic + ng
    j = jc + ng
    if ic == 0:
        dzdx = (zb[j, i + 1] - zb[j, i]) / dx
    elif ic == nx_core - 1:
        dzdx = (zb[j, i] - zb[j, i - 1]) / dx
    else:
        dzdx = (zb[j, i + 1] - zb[j, i - 1]) / (2.0 * dx)
    if jc == 0:
        dzdy = (zb[j + 1, i] - zb[j, i]) / dy
    elif jc == ny_core - 1:
        dzdy = (zb[j, i] - zb[j - 1, i]) / dy
    else:
        dzdy = (zb[j + 1, i] - zb[j - 1, i]) / (2.0 * dy)
    cosbeta[jc, ic] = 1.0 / math.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)


@cuda.jit(device=True, inline=True)
def _cell_fields(U, j: int, i: int, rho_f: float, rho_s: float, h_dry: float, mu_fine: float, mu_coarse: float, xi_fine: float, xi_coarse: float, mu_override, xi_override, jc: int, ic: int):
    hf = max(U[HF, j, i], 0.0)
    hsu = max(U[HSU, j, i], 0.0)
    hsl = max(U[HSL, j, i], 0.0)
    hcu = _clamp(U[HCU, j, i], 0.0, hsu)
    hcl = _clamp(U[HCL, j, i], 0.0, hsl)
    hs = hsu + hsl
    hc = hcu + hcl
    h = hf + hs
    mass = rho_f * hf + rho_s * hs
    u = 0.0
    v = 0.0
    if h > h_dry and mass > 0.0:
        u = U[MX, j, i] / mass
        v = U[MY, j, i] / mass
    speed = math.sqrt(u * u + v * v)
    cs = hs / h if h > h_dry else 0.0
    fc = hc / hs if hs > 0.0 else 0.0
    rho = mass / h if h > h_dry else rho_f
    mu = mu_fine * (1.0 - fc) + mu_coarse * fc
    xi = 1.0 / ((1.0 - fc) / max(xi_fine, 1.0e-12) + fc / max(xi_coarse, 1.0e-12))
    mo = mu_override[jc, ic]
    xo = xi_override[jc, ic]
    if math.isfinite(mo):
        mu = mo
    if math.isfinite(xo):
        xi = xo
    return hf, hsu, hsl, hcu, hcl, hs, h, mass, u, v, speed, cs, fc, rho, mu, xi


@cuda.jit(device=True, inline=True)
def _basal_tau(h: float, mass: float, speed: float, rho: float, mu: float, xi: float, cosbeta: float, g: float, h_dry: float, yield_n0: float) -> float:
    if h <= h_dry:
        return 0.0
    normal = mass * g * cosbeta
    tau = mu * normal + rho * g * speed * speed / max(xi, 1.0e-12)
    if yield_n0 > 0.0:
        tau += max(0.0, 1.0 - mu) * yield_n0 * (1.0 - math.exp(-normal / yield_n0))
    return tau


@cuda.jit
def _voellmy_friction(
    U,
    cosbeta,
    mu_override,
    xi_override,
    ng: int,
    nx_core: int,
    ny_core: int,
    dt: float,
    g: float,
    h_dry: float,
    rho_f: float,
    rho_s: float,
    mu_fine: float,
    mu_coarse: float,
    xi_fine: float,
    xi_coarse: float,
    yield_n0: float,
) -> None:
    ic, jc = cuda.grid(2)
    if ic >= nx_core or jc >= ny_core:
        return
    i, j = ic + ng, jc + ng
    _, _, _, _, _, _, h, mass, _, _, speed, _, _, _, mu, xi = _cell_fields(
        U, j, i, rho_f, rho_s, h_dry, mu_fine, mu_coarse, xi_fine, xi_coarse, mu_override, xi_override, jc, ic
    )
    if h <= h_dry or mass <= 0.0 or speed <= 0.0:
        return
    normal = mass * g * cosbeta[jc, ic]
    yield_acc = 0.0
    if yield_n0 > 0.0:
        yield_stress = max(0.0, 1.0 - mu) * yield_n0 * (1.0 - math.exp(-normal / yield_n0))
        yield_acc = yield_stress / mass
    a = mu * g * cosbeta[jc, ic] + yield_acc
    b = g / (max(xi, 1.0e-12) * max(h, h_dry))
    rem = speed - dt * a
    s1 = 0.0
    if rem > 0.0:
        if b * dt < 1.0e-12:
            s1 = rem
        else:
            s1 = (math.sqrt(1.0 + 4.0 * b * dt * rem) - 1.0) / (2.0 * b * dt)
    fac = s1 / max(speed, 1.0e-30)
    U[MX, j, i] *= fac
    U[MY, j, i] *= fac


@cuda.jit
def _speed_cap(U, ng: int, nx_core: int, ny_core: int, cap: float, rho_f: float, rho_s: float, h_dry: float) -> None:
    ic, jc = cuda.grid(2)
    if ic >= nx_core or jc >= ny_core or cap <= 0.0:
        return
    i, j = ic + ng, jc + ng
    hf = max(U[HF, j, i], 0.0)
    hs = max(U[HSU, j, i], 0.0) + max(U[HSL, j, i], 0.0)
    h = hf + hs
    mass = rho_f * hf + rho_s * hs
    if h <= h_dry or mass <= 0.0:
        return
    u = U[MX, j, i] / mass
    v = U[MY, j, i] / mass
    speed = math.sqrt(u * u + v * v)
    if speed > cap:
        fac = cap / speed
        U[MX, j, i] *= fac
        U[MY, j, i] *= fac


@cuda.jit
def _segregation(
    U,
    budget,
    ng: int,
    nx_core: int,
    ny_core: int,
    dt: float,
    enabled: int,
    h_dry: float,
    rho_f: float,
    rho_s: float,
    segregation_length: float,
    max_speed: float,
    remix_diffusivity: float,
    min_layer: float,
) -> None:
    ic, jc = cuda.grid(2)
    if ic >= nx_core or jc >= ny_core or enabled == 0 or dt <= 0.0:
        return
    i, j = ic + ng, jc + ng
    hsu = max(U[HSU, j, i], 0.0)
    hsl = max(U[HSL, j, i], 0.0)
    hcu = _clamp(U[HCU, j, i], 0.0, hsu)
    hcl = _clamp(U[HCL, j, i], 0.0, hsl)
    fine_u = max(hsu - hcu, 0.0)
    fine_l = max(hsl - hcl, 0.0)
    phi_u = hcu / max(hsu, 1.0e-30) if hsu > 0.0 else 0.0
    phi_l = hcl / max(hsl, 1.0e-30) if hsl > 0.0 else 0.0
    hf = max(U[HF, j, i], 0.0)
    h = hf + hsu + hsl
    mass = rho_f * hf + rho_s * (hsu + hsl)
    speed = 0.0
    if h > h_dry and mass > 0.0:
        u = U[MX, j, i] / mass
        v = U[MY, j, i] / mass
        speed = math.sqrt(u * u + v * v)
    shear_rate = speed / max(h, min_layer)
    wseg = min(segregation_length * shear_rate, max_speed)
    j_kin = wseg * phi_l * (1.0 - phi_u)
    dz = max(0.5 * h, min_layer)
    j_mix = remix_diffusivity * (phi_u - phi_l) / dz
    delta = (j_kin - j_mix) * dt
    pos_cap = min(hcl, fine_u)
    neg_cap = min(hcu, fine_l)
    delta = min(max(delta, -neg_cap), pos_cap)
    U[HCU, j, i] += delta
    U[HCL, j, i] -= delta
    cuda.atomic.add(budget, 0, delta)


@cuda.jit
def _erosion_deposition(
    U,
    zb,
    bed_fine,
    bed_coarse,
    peak_tau,
    cosbeta,
    mu_override,
    xi_override,
    budget,
    ng: int,
    nx_core: int,
    ny_core: int,
    dt: float,
    g: float,
    h_dry: float,
    rho_f: float,
    rho_s: float,
    max_solid_fraction: float,
    mu_fine: float,
    mu_coarse: float,
    xi_fine: float,
    xi_coarse: float,
    yield_n0: float,
    erosion_enabled: int,
    erosion_model: int,
    bed_porosity: float,
    bed_coarse_default: float,
    critical_shear: float,
    excess_rate: float,
    excess_exponent: float,
    max_erosion_rate: float,
    velocity_coeff: float,
    depth_per_kpa: float,
    velocity: float,
    entrainment_beta: float,
    update_bed: int,
    deposition_enabled: int,
    deposition_critical_shear: float,
    hindered_exponent: float,
    deposition_max_rate: float,
    minimum_mobile_depth: float,
    settling_fine: float,
    settling_coarse: float,
) -> None:
    ic, jc = cuda.grid(2)
    if ic >= nx_core or jc >= ny_core:
        return
    i, j = ic + ng, jc + ng
    fields = _cell_fields(U, j, i, rho_f, rho_s, h_dry, mu_fine, mu_coarse, xi_fine, xi_coarse, mu_override, xi_override, jc, ic)
    hf, _, _, _, _, hs, h, mass, u, v, speed, cs, _, rho, mu, xi = fields
    tau = _basal_tau(h, mass, speed, rho, mu, xi, cosbeta[jc, ic], g, h_dry, yield_n0)
    if tau > peak_tau[jc, ic]:
        peak_tau[jc, ic] = tau
    one_minus_por = 1.0 - bed_porosity

    if erosion_enabled:
        available_solid = bed_fine[jc, ic] + bed_coarse[jc, ic]
        available_bulk = available_solid / one_minus_por
        ebulk = 0.0
        if erosion_model == 1:
            ebulk = velocity_coeff * speed
        elif erosion_model == 2:
            potential = depth_per_kpa * max(peak_tau[jc, ic] - critical_shear, 0.0) / 1000.0
            if potential > 0.0:
                ebulk = min(velocity, potential / max(dt, 1.0e-30))
        else:
            excess = max(tau / max(critical_shear, 1.0e-12) - 1.0, 0.0)
            ebulk = excess_rate * math.pow(excess, excess_exponent)
        ebulk = min(ebulk, max_erosion_rate)
        ebulk = min(ebulk, available_bulk / max(dt, 1.0e-30))
        if h <= h_dry:
            ebulk = 0.0
        dB = max(ebulk * dt, 0.0)
        solid_add = one_minus_por * dB
        bed_fc = bed_coarse[jc, ic] / max(available_solid, 1.0e-30) if available_solid > 0.0 else bed_coarse_default
        coarse_add = min(solid_add * bed_fc, bed_coarse[jc, ic])
        fine_add = min(solid_add - solid_add * bed_fc, bed_fine[jc, ic])
        solid_add = fine_add + coarse_add
        dB = solid_add / one_minus_por
        fluid_add = bed_porosity * dB
        bed_fine[jc, ic] -= fine_add
        bed_coarse[jc, ic] -= coarse_add
        U[HF, j, i] += fluid_add
        U[HSL, j, i] += solid_add
        U[HCL, j, i] += coarse_add
        beta = _clamp(entrainment_beta, 0.0, 1.0)
        if beta > 0.0:
            mass_add = rho_f * fluid_add + rho_s * solid_add
            U[MX, j, i] += beta * mass_add * u
            U[MY, j, i] += beta * mass_add * v
        if update_bed:
            zb[j, i] -= dB
        cuda.atomic.add(budget, 0, dB)
        cuda.atomic.add(budget, 2, solid_add)

    if deposition_enabled:
        fields = _cell_fields(U, j, i, rho_f, rho_s, h_dry, mu_fine, mu_coarse, xi_fine, xi_coarse, mu_override, xi_override, jc, ic)
        hf, _, hsl, _, hcl, _, h, mass, _, _, speed, cs, _, rho, mu, xi = fields
        tau = _basal_tau(h, mass, speed, rho, mu, xi, cosbeta[jc, ic], g, h_dry, yield_n0)
        shear_factor = max(1.0 - tau / max(deposition_critical_shear, 1.0e-12), 0.0)
        hinder = math.pow(max(1.0 - cs / max(max_solid_fraction, 1.0e-12), 0.0), hindered_exponent)
        fine_lower = max(hsl - hcl, 0.0)
        coarse_lower = max(hcl, 0.0)
        cf_bed = min(2.0 * fine_lower / max(h, h_dry), max_solid_fraction)
        cc_bed = min(2.0 * coarse_lower / max(h, h_dry), max_solid_fraction)
        Df = min(settling_fine * cf_bed * hinder * shear_factor, deposition_max_rate)
        Dc = min(settling_coarse * cc_bed * hinder * shear_factor, deposition_max_rate)
        if h <= minimum_mobile_depth:
            Df = 0.0
            Dc = 0.0
        df = min(Df * dt, fine_lower)
        dc = min(Dc * dt, coarse_lower)
        solid_dep = df + dc
        bulk_dep = solid_dep / one_minus_por
        fluid_need = bed_porosity * bulk_dep
        if fluid_need > hf:
            scale = hf / max(fluid_need, 1.0e-30)
            df *= scale
            dc *= scale
            solid_dep = df + dc
            bulk_dep = solid_dep / one_minus_por
            fluid_need = bed_porosity * bulk_dep
        U[HSL, j, i] -= solid_dep
        U[HCL, j, i] -= dc
        U[HF, j, i] -= fluid_need
        bed_fine[jc, ic] += df
        bed_coarse[jc, ic] += dc
        if update_bed:
            zb[j, i] += bulk_dep
        cuda.atomic.add(budget, 1, bulk_dep)
        cuda.atomic.add(budget, 3, solid_dep)


# -----------------------------------------------------------------------------
# CUDA workspace and launch helpers.
# -----------------------------------------------------------------------------
@dataclass
class CudaWorkspace:
    P: Any
    sx: Any
    sy: Any
    fx: Any
    fy: Any
    cxl: Any
    cxr: Any
    cyl: Any
    cyr: Any
    rhs: Any
    fallback: Any
    boundary_rate: Any
    invalid: Any
    block_sx: Any
    block_sy: Any
    block_h: Any
    block_speed: Any
    block_rho_min: Any
    block_rho_max: Any
    source_budget: Any
    segregation_budget: Any


def _grid2(shape: Tuple[int, int], block: Tuple[int, int]) -> Tuple[int, int]:
    ny, nx = shape
    return ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])


def _apply_boundary_device(U, zb, active, cfg: SolverConfig, threads_1d: int, ng: int = 2) -> None:
    n = cfg.numerics
    ny, nx = U.shape[1], U.shape[2]
    _boundary_x[(ny + threads_1d - 1) // threads_1d, threads_1d](
        U, zb, active, ng, _bc_id(n.bc_left), _bc_id(n.bc_right)
    )
    _boundary_y[(nx + threads_1d - 1) // threads_1d, threads_1d](
        U, zb, active, ng, _bc_id(n.bc_bottom), _bc_id(n.bc_top)
    )


def _is_admissible_device(U, ws: CudaWorkspace, grid, block, ng: int, tol: float) -> bool:
    ws.invalid.copy_to_device(np.zeros(1, dtype=np.int32))
    _admissibility[grid, block](U, ws.invalid, ng, tol)
    return int(ws.invalid.copy_to_host()[0]) == 0


def _transport_rhs_device(
    U,
    zb,
    active,
    cfg: SolverConfig,
    ws: CudaWorkspace,
    dx: float,
    dy: float,
    order: int,
    flux_name: str,
    grid,
    block,
    threads_1d: int,
    ng: int,
    nx_core: int,
    ny_core: int,
) -> Tuple[int, np.ndarray]:
    n, m = cfg.numerics, cfg.material
    _compute_primitives[grid, block](U, zb, active, ws.P, m.rho_fluid, m.rho_solid, n.h_dry)
    _compute_slopes[grid, block](ws.P, active, ws.sx, ws.sy, _limiter_id(n.limiter))
    ws.fallback.copy_to_device(np.zeros(1, dtype=np.int64))
    flux_id = FLUX_HLLC if flux_name.lower() == "hllc" else FLUX_HLL
    _flux_x[grid, block](
        zb, active, ws.P, ws.sx, ws.fx, ws.cxl, ws.cxr, order, flux_id,
        n.g, m.rho_fluid, m.rho_solid, m.max_solid_fraction, n.h_dry,
        n.hllc_dry_factor, n.hllc_max_froude, n.hybrid_depth_rel_jump, ws.fallback,
    )
    _flux_y[grid, block](
        zb, active, ws.P, ws.sy, ws.fy, ws.cyl, ws.cyr, order, flux_id,
        n.g, m.rho_fluid, m.rho_solid, m.max_solid_fraction, n.h_dry,
        n.hllc_dry_factor, n.hllc_max_froude, n.hybrid_depth_rel_jump, ws.fallback,
    )
    _divergence[grid, block](ws.fx, ws.fy, ws.cxl, ws.cxr, ws.cyl, ws.cyr, active, ws.rhs, dx, dy, ng)
    ws.boundary_rate.copy_to_device(np.zeros(5, dtype=np.float64))
    count = max(nx_core, ny_core)
    _boundary_budget[(count + threads_1d - 1) // threads_1d, threads_1d](
        ws.fx, ws.fy, ws.boundary_rate, dx, dy, ng, ny_core, nx_core
    )
    fallback = int(ws.fallback.copy_to_host()[0])
    boundary_rate = ws.boundary_rate.copy_to_host()
    return fallback, boundary_rate


def _transport_step_device(
    U0,
    U1,
    U2,
    dt: float,
    zb,
    active,
    cfg: SolverConfig,
    ws: CudaWorkspace,
    dx: float,
    dy: float,
    grid,
    block,
    threads_1d: int,
    ng: int,
    nx_core: int,
    ny_core: int,
):
    n = cfg.numerics
    fallback_used = False
    _apply_boundary_device(U0, zb, active, cfg, threads_1d, ng)
    fb1, br1 = _transport_rhs_device(U0, zb, active, cfg, ws, dx, dy, n.space_order, n.flux, grid, block, threads_1d, ng, nx_core, ny_core)
    _euler_update[grid, block](U0, ws.rhs, U1, dt)
    if not _is_admissible_device(U1, ws, grid, block, ng, n.positivity_tolerance):
        fb1, br1 = _transport_rhs_device(U0, zb, active, cfg, ws, dx, dy, 1, "hll", grid, block, threads_1d, ng, nx_core, ny_core)
        _euler_update[grid, block](U0, ws.rhs, U1, dt)
        fallback_used = True
    if not _is_admissible_device(U1, ws, grid, block, ng, n.positivity_tolerance):
        return None, fb1, True, np.zeros(5, dtype=np.float64)
    _roundoff_repair[grid, block](U1, ng, n.h_dry, n.positivity_tolerance)
    if n.time_order == 1:
        return U1, fb1, fallback_used, br1

    _apply_boundary_device(U1, zb, active, cfg, threads_1d, ng)
    fb2, br2 = _transport_rhs_device(U1, zb, active, cfg, ws, dx, dy, n.space_order, n.flux, grid, block, threads_1d, ng, nx_core, ny_core)
    _ssprk2_combine[grid, block](U0, U1, ws.rhs, U2, dt)
    if not _is_admissible_device(U2, ws, grid, block, ng, n.positivity_tolerance):
        fb2, br2 = _transport_rhs_device(U1, zb, active, cfg, ws, dx, dy, 1, "hll", grid, block, threads_1d, ng, nx_core, ny_core)
        _ssprk2_combine[grid, block](U0, U1, ws.rhs, U2, dt)
        fallback_used = True
    if not _is_admissible_device(U2, ws, grid, block, ng, n.positivity_tolerance):
        return None, fb1 + fb2, True, np.zeros(5, dtype=np.float64)
    _roundoff_repair[grid, block](U2, ng, n.h_dry, n.positivity_tolerance)
    return U2, fb1 + fb2, fallback_used, 0.5 * (br1 + br2)


def _compute_dt_device(U, ws: CudaWorkspace, cfg: SolverConfig, dx: float, dy: float, ng: int, nx_core: int, ny_core: int, threads_1d: int) -> float:
    n, m = cfg.numerics, cfg.material
    cells = nx_core * ny_core
    blocks = (cells + threads_1d - 1) // threads_1d
    _wave_speed_blocks[blocks, threads_1d](
        U, ws.block_sx, ws.block_sy, ng, nx_core, ny_core,
        m.rho_fluid, m.rho_solid, n.h_dry, n.g,
    )
    sx = float(np.max(ws.block_sx.copy_to_host()))
    sy = float(np.max(ws.block_sy.copy_to_host()))
    if sx <= 0.0 and sy <= 0.0:
        return n.dt_max
    dt = n.cfl * min(dx / max(sx, 1.0e-12), dy / max(sy, 1.0e-12))
    return min(float(dt), n.dt_max)


def _diagnostics_device(U, ws: CudaWorkspace, cfg: SolverConfig, ng: int, nx_core: int, ny_core: int, threads_1d: int) -> Dict[str, float]:
    n, m = cfg.numerics, cfg.material
    cells = nx_core * ny_core
    blocks = (cells + threads_1d - 1) // threads_1d
    _diagnostic_blocks[blocks, threads_1d](
        U, ws.block_h, ws.block_speed, ws.block_rho_min, ws.block_rho_max,
        ng, nx_core, ny_core, m.rho_fluid, m.rho_solid, n.h_dry,
    )
    hmax = float(np.max(ws.block_h.copy_to_host()))
    vmax = float(np.max(ws.block_speed.copy_to_host()))
    rmin_values = ws.block_rho_min.copy_to_host()
    rmax_values = ws.block_rho_max.copy_to_host()
    finite = rmin_values < 1.0e299
    rmin = float(np.min(rmin_values[finite])) if np.any(finite) else m.rho_fluid
    rmax = float(np.max(rmax_values)) if np.max(rmax_values) > 0.0 else m.rho_fluid
    return {"hmax": hmax, "vmax": vmax, "rho_min": rmin, "rho_max": rmax}


def _source_kernel_args(cfg: SolverConfig):
    m, n, e, d = cfg.material, cfg.numerics, cfg.erosion, cfg.deposition
    model = str(e.model).strip().lower()
    model_id = 1 if model == "velocity" else 2 if model == "peak" else 0
    wf = ferguson_church_settling_velocity(m.grain_diameter_fine_m, m.rho_solid, m.rho_fluid, m.kinematic_viscosity_m2s, n.g)
    wc = ferguson_church_settling_velocity(m.grain_diameter_coarse_m, m.rho_solid, m.rho_fluid, m.kinematic_viscosity_m2s, n.g)
    return model_id, wf, wc


def _apply_sources_first_half(U, zb, bed_fine, bed_coarse, peak_tau, cosbeta, mu_override, xi_override, ws, cfg, dt, grid_core, block, ng, nx_core, ny_core):
    n, m, e, d, s = cfg.numerics, cfg.material, cfg.erosion, cfg.deposition, cfg.segregation
    model_id, wf, wc = _source_kernel_args(cfg)
    _voellmy_friction[grid_core, block](
        U, cosbeta, mu_override, xi_override, ng, nx_core, ny_core, dt,
        n.g, n.h_dry, m.rho_fluid, m.rho_solid, m.mu_fine, m.mu_coarse,
        m.xi_fine, m.xi_coarse, m.yield_N0_pa,
    )
    ws.source_budget.copy_to_device(np.zeros(4, dtype=np.float64))
    _erosion_deposition[grid_core, block](
        U, zb, bed_fine, bed_coarse, peak_tau, cosbeta, mu_override, xi_override, ws.source_budget,
        ng, nx_core, ny_core, dt, n.g, n.h_dry, m.rho_fluid, m.rho_solid,
        m.max_solid_fraction, m.mu_fine, m.mu_coarse, m.xi_fine, m.xi_coarse, m.yield_N0_pa,
        int(e.enabled), model_id, e.bed_porosity, e.bed_coarse_fraction, e.critical_shear_pa,
        e.excess_shear_rate_ms, e.excess_shear_exponent, e.max_erosion_rate_ms,
        e.velocity_erosion_coefficient, e.potential_depth_per_kpa, e.erosion_velocity_ms,
        e.entrainment_velocity_fraction, int(e.update_bed), int(d.enabled), d.critical_shear_pa,
        d.hindered_settling_exponent, d.max_rate_ms, d.minimum_mobile_depth_m, wf, wc,
    )
    source = ws.source_budget.copy_to_host()
    ws.segregation_budget.copy_to_device(np.zeros(1, dtype=np.float64))
    _segregation[grid_core, block](
        U, ws.segregation_budget, ng, nx_core, ny_core, dt, int(s.enabled), n.h_dry,
        m.rho_fluid, m.rho_solid, s.segregation_length_m, s.max_segregation_speed_ms,
        s.remix_diffusivity_m2s, s.min_layer_thickness_m,
    )
    upward = float(ws.segregation_budget.copy_to_host()[0])
    _speed_cap[grid_core, block](U, ng, nx_core, ny_core, n.speed_cap_ms, m.rho_fluid, m.rho_solid, n.h_dry)
    return source, upward


def _apply_sources_second_half(U, zb, bed_fine, bed_coarse, peak_tau, cosbeta, mu_override, xi_override, ws, cfg, dt, grid_core, block, ng, nx_core, ny_core):
    n, m, e, d, s = cfg.numerics, cfg.material, cfg.erosion, cfg.deposition, cfg.segregation
    model_id, wf, wc = _source_kernel_args(cfg)
    ws.source_budget.copy_to_device(np.zeros(4, dtype=np.float64))
    _erosion_deposition[grid_core, block](
        U, zb, bed_fine, bed_coarse, peak_tau, cosbeta, mu_override, xi_override, ws.source_budget,
        ng, nx_core, ny_core, dt, n.g, n.h_dry, m.rho_fluid, m.rho_solid,
        m.max_solid_fraction, m.mu_fine, m.mu_coarse, m.xi_fine, m.xi_coarse, m.yield_N0_pa,
        int(e.enabled), model_id, e.bed_porosity, e.bed_coarse_fraction, e.critical_shear_pa,
        e.excess_shear_rate_ms, e.excess_shear_exponent, e.max_erosion_rate_ms,
        e.velocity_erosion_coefficient, e.potential_depth_per_kpa, e.erosion_velocity_ms,
        e.entrainment_velocity_fraction, int(e.update_bed), int(d.enabled), d.critical_shear_pa,
        d.hindered_settling_exponent, d.max_rate_ms, d.minimum_mobile_depth_m, wf, wc,
    )
    source = ws.source_budget.copy_to_host()
    ws.segregation_budget.copy_to_device(np.zeros(1, dtype=np.float64))
    _segregation[grid_core, block](
        U, ws.segregation_budget, ng, nx_core, ny_core, dt, int(s.enabled), n.h_dry,
        m.rho_fluid, m.rho_solid, s.segregation_length_m, s.max_segregation_speed_ms,
        s.remix_diffusivity_m2s, s.min_layer_thickness_m,
    )
    upward = float(ws.segregation_budget.copy_to_host()[0])
    return source, upward


def _host_core(U_device, zb_device, ng: int, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    U_full = U_device.copy_to_host()
    zb_full = zb_device.copy_to_host()
    return U_full[:, ng:ng + ny, ng:ng + nx].copy(), zb_full[ng:ng + ny, ng:ng + nx].copy()


def _device_memory_mib(arrays: List[Any]) -> float:
    return sum(int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize for a in arrays) / (1024.0 ** 2)


# -----------------------------------------------------------------------------
# Public CUDA solver.
# -----------------------------------------------------------------------------
def run_solver_cuda(cfg: SolverConfig, device_name: Optional[str] = None) -> Dict[str, Any]:
    if not cuda.is_available() and not bool(getattr(cuda.config, "ENABLE_CUDASIM", False)):
        raise RuntimeError("CUDA backend requested but no CUDA runtime is available")
    if not bool(getattr(cuda.config, "ENABLE_CUDASIM", False)):
        cuda.select_device(int(cfg.compute.cuda_device))
    device_label = device_name or ("CUDA simulator" if bool(getattr(cuda.config, "ENABLE_CUDASIM", False)) else "CUDA device")
    print(f"[BACKEND] CUDA / Numba, device={device_label}")

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
    z_initial = zb[core].copy()

    with open(os.path.join(cfg.output.out_dir, "config_resolved.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=lambda o: o.__dict__)), stream, sort_keys=False)

    # Host boundary initialization is performed once; subsequent updates remain on device.
    from .numerics import apply_boundary
    apply_boundary(U, zb, active, cfg.numerics, ng)

    d_U = cuda.to_device(U)
    d_Utrial = cuda.device_array_like(d_U)
    d_U1 = cuda.device_array_like(d_U)
    d_U2 = cuda.device_array_like(d_U)
    d_zb = cuda.to_device(zb)
    d_zbtrial = cuda.device_array_like(d_zb)
    d_active = cuda.to_device(active)
    d_bed_fine = cuda.to_device(bed_fine)
    d_bed_fine_trial = cuda.device_array_like(d_bed_fine)
    d_bed_coarse = cuda.to_device(bed_coarse)
    d_bed_coarse_trial = cuda.device_array_like(d_bed_coarse)
    d_peak = cuda.to_device(peak_tau)
    d_peak_trial = cuda.device_array_like(d_peak)
    d_mu = cuda.to_device(mu_override.astype(np.float64))
    d_xi = cuda.to_device(xi_override.astype(np.float64))
    d_cosbeta = cuda.device_array((ny, nx), dtype=np.float64)

    P = cuda.device_array((7, *shape), dtype=np.float64)
    sx = cuda.device_array_like(P)
    sy = cuda.device_array_like(P)
    fx = cuda.device_array((NV, shape[0], shape[1] + 1), dtype=np.float64)
    fy = cuda.device_array((NV, shape[0] + 1, shape[1]), dtype=np.float64)
    cxl = cuda.device_array((shape[0], shape[1] + 1), dtype=np.float64)
    cxr = cuda.device_array_like(cxl)
    cyl = cuda.device_array((shape[0] + 1, shape[1]), dtype=np.float64)
    cyr = cuda.device_array_like(cyl)
    rhs = cuda.device_array((NV, *shape), dtype=np.float64)

    threads_1d = int(cfg.compute.cuda_threads_1d)
    cells = nx * ny
    reduction_blocks = (cells + threads_1d - 1) // threads_1d
    ws = CudaWorkspace(
        P=P, sx=sx, sy=sy, fx=fx, fy=fy, cxl=cxl, cxr=cxr, cyl=cyl, cyr=cyr, rhs=rhs,
        fallback=cuda.device_array(1, dtype=np.int64),
        boundary_rate=cuda.device_array(5, dtype=np.float64),
        invalid=cuda.device_array(1, dtype=np.int32),
        block_sx=cuda.device_array(reduction_blocks, dtype=np.float64),
        block_sy=cuda.device_array(reduction_blocks, dtype=np.float64),
        block_h=cuda.device_array(reduction_blocks, dtype=np.float64),
        block_speed=cuda.device_array(reduction_blocks, dtype=np.float64),
        block_rho_min=cuda.device_array(reduction_blocks, dtype=np.float64),
        block_rho_max=cuda.device_array(reduction_blocks, dtype=np.float64),
        source_budget=cuda.device_array(4, dtype=np.float64),
        segregation_budget=cuda.device_array(1, dtype=np.float64),
    )

    block = (int(cfg.compute.cuda_block_x), int(cfg.compute.cuda_block_y))
    grid = _grid2(shape, block)
    grid_core = _grid2((ny, nx), block)
    arrays = [d_U, d_Utrial, d_U1, d_U2, d_zb, d_zbtrial, d_active, d_bed_fine, d_bed_fine_trial,
              d_bed_coarse, d_bed_coarse_trial, d_peak, d_peak_trial, d_mu, d_xi, d_cosbeta,
              P, sx, sy, fx, fy, cxl, cxr, cyl, cyr, rhs]
    cuda_allocated_mib_estimate = _device_memory_mib(arrays)
    print(f"[CUDA] grid={nx}x{ny}, block={block[0]}x{block[1]}, allocated≈{cuda_allocated_mib_estimate:.1f} MiB")

    # Timed interval starts after allocation/setup. CUDA synchronization at the end
    # ensures queued device work is included in the reported wall time.
    wall_start = time.perf_counter()
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

    while t < cfg.numerics.t_end - 1.0e-12:
        if t + 1.0e-12 >= next_output:
            if cfg.output.save_snapshots:
                U_host, zb_host = _host_core(d_U, d_zb, ng, ny, nx)
                snapshots.append(save_snapshot(cfg.output.out_dir, output_index, t, U_host, zb_host, x, y, cfg))
            output_index += 1
            next_output += cfg.numerics.output_dt

        dt = min(_compute_dt_device(d_U, ws, cfg, dx, dy, ng, nx, ny, threads_1d), cfg.numerics.t_end - t)
        accepted = False
        for _retry in range(cfg.numerics.max_step_retries + 1):
            d_Utrial.copy_to_device(d_U)
            d_zbtrial.copy_to_device(d_zb)
            d_bed_fine_trial.copy_to_device(d_bed_fine)
            d_bed_coarse_trial.copy_to_device(d_bed_coarse)
            d_peak_trial.copy_to_device(d_peak)

            _terrain_cosbeta[grid_core, block](d_zbtrial, d_cosbeta, ng, nx, ny, dx, dy)
            source1, seg1 = _apply_sources_first_half(
                d_Utrial, d_zbtrial, d_bed_fine_trial, d_bed_coarse_trial, d_peak_trial,
                d_cosbeta, d_mu, d_xi, ws, cfg, 0.5 * dt, grid_core, block, ng, nx, ny,
            )
            _apply_boundary_device(d_Utrial, d_zbtrial, d_active, cfg, threads_1d, ng)
            final_state, fb, global_fb, boundary_rate = _transport_step_device(
                d_Utrial, d_U1, d_U2, dt, d_zbtrial, d_active, cfg, ws, dx, dy,
                grid, block, threads_1d, ng, nx, ny,
            )
            if final_state is None:
                dt *= 0.5
                retries_total += 1
                continue

            _terrain_cosbeta[grid_core, block](d_zbtrial, d_cosbeta, ng, nx, ny, dx, dy)
            source2, seg2 = _apply_sources_second_half(
                final_state, d_zbtrial, d_bed_fine_trial, d_bed_coarse_trial, d_peak_trial,
                d_cosbeta, d_mu, d_xi, ws, cfg, 0.5 * dt, grid_core, block, ng, nx, ny,
            )
            # Erosion/deposition can modify z_b. Refresh the local slope before the
            # terminal Voellmy half-step so CPU and CUDA use the current bed geometry.
            _terrain_cosbeta[grid_core, block](d_zbtrial, d_cosbeta, ng, nx, ny, dx, dy)
            _voellmy_friction[grid_core, block](
                final_state, d_cosbeta, d_mu, d_xi, ng, nx, ny, 0.5 * dt,
                cfg.numerics.g, cfg.numerics.h_dry, cfg.material.rho_fluid, cfg.material.rho_solid,
                cfg.material.mu_fine, cfg.material.mu_coarse, cfg.material.xi_fine, cfg.material.xi_coarse,
                cfg.material.yield_N0_pa,
            )
            _speed_cap[grid_core, block](
                final_state, ng, nx, ny, cfg.numerics.speed_cap_ms,
                cfg.material.rho_fluid, cfg.material.rho_solid, cfg.numerics.h_dry,
            )
            if not _is_admissible_device(final_state, ws, grid, block, ng, cfg.numerics.positivity_tolerance):
                dt *= 0.5
                retries_total += 1
                continue
            _roundoff_repair[grid, block](final_state, ng, cfg.numerics.h_dry, cfg.numerics.positivity_tolerance)

            old_U = d_U
            d_U = final_state
            if final_state is d_U1:
                d_U1 = old_U
            else:
                d_U2 = old_U
            d_zb, d_zbtrial = d_zbtrial, d_zb
            d_bed_fine, d_bed_fine_trial = d_bed_fine_trial, d_bed_fine
            d_bed_coarse, d_bed_coarse_trial = d_bed_coarse_trial, d_bed_coarse
            d_peak, d_peak_trial = d_peak_trial, d_peak

            fallback_faces += fb
            boundary_component_change += dt * boundary_rate
            global_fallback_steps += int(global_fb)
            source_totals["eroded_bulk_m3"] += (source1[0] + source2[0]) * cell_area
            source_totals["deposited_bulk_m3"] += (source1[1] + source2[1]) * cell_area
            source_totals["upward_coarse_m3"] += (seg1 + seg2) * cell_area
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Step failed after {cfg.numerics.max_step_retries} retries at t={t:.6g}")

        t += dt
        step += 1
        if cfg.compute.cuda_sync_each_step:
            cuda.synchronize()
        if cfg.output.progress_every_steps > 0 and step % cfg.output.progress_every_steps == 0:
            stats = _diagnostics_device(d_U, ws, cfg, ng, nx, ny, threads_1d)
            print(
                f"[RUN][CUDA] step={step} t={t:.3f}/{cfg.numerics.t_end:.3f} dt={dt:.3g} "
                f"hmax={stats['hmax']:.3g} vmax={stats['vmax']:.3g} "
                f"rho=[{stats['rho_min']:.1f},{stats['rho_max']:.1f}] retries={retries_total}"
            )

    cuda.synchronize()
    elapsed_wall_s = time.perf_counter() - wall_start
    Uc, zb_core = _host_core(d_U, d_zb, ng, ny, nx)
    bed_fine = d_bed_fine.copy_to_host()
    bed_coarse = d_bed_coarse.copy_to_host()
    peak_tau = d_peak.copy_to_host()
    if cfg.output.save_snapshots:
        snapshots.append(save_snapshot(cfg.output.out_dir, output_index, t, Uc, zb_core, x, y, cfg))
    if cfg.output.make_png:
        save_maps(cfg.output.out_dir, t, Uc, zb_core, x, y, cfg, "final")
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
    fields = composition_fields(Uc, cfg)
    wet = fields["wet"]
    dzdx_final, dzdy_final, _ = terrain_geometry(zb_core, dx, dy)
    v_down, v_cross = terrain_velocity_components(fields["u"], fields["v"], dzdx_final, dzdy_final)
    metrics = {
        "backend": "cuda",
        "cuda_device": device_label,
        "cuda_block": [block[0], block[1]],
        "cuda_threads_1d": threads_1d,
        "elapsed_wall_s": float(elapsed_wall_s),
        "cuda_allocated_mib_estimate": float(cuda_allocated_mib_estimate),
        "t_final_s": t,
        "steps": step,
        "dx_m": dx,
        "dy_m": dy,
        "shape": [ny, nx],
        "release_cells": int(np.sum(release_mask)),
        "friction_override_cells": int(np.sum(np.isfinite(mu_override))),
        "friction_regions": cfg.friction.regions,
        "release_volume_m3": cfg.release.volume_m3,
        "h_max_m": float(np.max(fields["h"])),
        "speed_max_ms": float(np.max(fields["speed"])),
        "velocity_x_abs_max_ms": float(np.max(np.abs(fields["u"]))),
        "velocity_y_abs_max_ms": float(np.max(np.abs(fields["v"]))),
        "velocity_downslope_abs_max_ms": float(np.max(np.abs(v_down))),
        "velocity_crossslope_abs_max_ms": float(np.max(np.abs(v_cross))),
        "rho_min_wet_kgm3": float(np.min(fields["rho"][wet])) if np.any(wet) else cfg.material.rho_fluid,
        "rho_max_wet_kgm3": float(np.max(fields["rho"][wet])) if np.any(wet) else cfg.material.rho_fluid,
        "bed_elevation_change_min_m": float(np.min(zb_core - z_initial)),
        "bed_elevation_change_max_m": float(np.max(zb_core - z_initial)),
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
    with open(os.path.join(cfg.output.out_dir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    np.savez_compressed(
        os.path.join(cfg.output.out_dir, "final_state.npz"), x=x, y=y, zb=zb_core, U=Uc,
        bed_fine_solid=bed_fine, bed_coarse_solid=bed_coarse, peak_tau=peak_tau,
        mu_override=mu_override, xi_override=xi_override, release_mask=release_mask.astype(np.uint8),
    )
    print(json.dumps(metrics, indent=2))
    return metrics