from __future__ import annotations

"""Finite-volume kernels, HLL/HLLC fluxes, boundaries, and time stepping."""

import math
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    prange = range

from .config import NumericsConfig, SolverConfig, _bc_id, _limiter_id
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

# -----------------------------------------------------------------------------
# Primitive/conservative conversions and scalar utilities.
# -----------------------------------------------------------------------------
@njit(cache=True)
def minmod2(a: float, b: float) -> float:
    if a * b <= 0.0:
        return 0.0
    return math.copysign(min(abs(a), abs(b)), a)


@njit(cache=True)
def limited_slope(dl: float, dr: float, limiter: int) -> float:
    if limiter == LIMITER_MINMOD:
        return minmod2(dl, dr)
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
    return minmod2(0.5 * (dl + dr), minmod2(2.0 * dl, 2.0 * dr))


@njit(cache=True)
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@njit(cache=True)
def state_derived(
    U: np.ndarray,
    rho_f: float,
    rho_s: float,
    h_dry: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    hf = max(U[HF], 0.0)
    hsu = max(U[HSU], 0.0)
    hcu = clamp(U[HCU], 0.0, hsu)
    hsl = max(U[HSL], 0.0)
    hcl = clamp(U[HCL], 0.0, hsl)
    hs = hsu + hsl
    h = hf + hs
    mass = rho_f * hf + rho_s * hs
    if h <= h_dry or mass <= 0.0:
        return h, mass, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, rho_f
    u = U[MX] / mass
    v = U[MY] / mass
    cs = clamp(hs / h, 0.0, 1.0)
    lam = clamp(hsu / (hs + 1.0e-30), 0.0, 1.0) if hs > 0.0 else 0.5
    fcu = clamp(hcu / (hsu + 1.0e-30), 0.0, 1.0) if hsu > 0.0 else 0.0
    fcl = clamp(hcl / (hsl + 1.0e-30), 0.0, 1.0) if hsl > 0.0 else 0.0
    rho = mass / h
    return h, mass, u, v, cs, lam, fcu, fcl, rho


@njit(cache=True)
def primitive_to_state(
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
) -> np.ndarray:
    out = np.zeros(NV, dtype=np.float64)
    h = max(h, 0.0)
    cs = clamp(cs, 0.0, max_cs)
    lam = clamp(lam, 0.0, 1.0)
    fcu = clamp(fcu, 0.0, 1.0)
    fcl = clamp(fcl, 0.0, 1.0)
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
    return out


@njit(cache=True)
def physical_flux(U: np.ndarray, nx: float, ny: float, g: float, rho_f: float, rho_s: float, h_dry: float) -> np.ndarray:
    F = np.zeros(NV, dtype=np.float64)
    h, mass, u, v, _, _, _, _, rho = state_derived(U, rho_f, rho_s, h_dry)
    if h <= h_dry or mass <= 0.0:
        return F
    un = u * nx + v * ny
    for k in range(5):
        F[k] = U[k] * un
    p = 0.5 * rho * g * h * h
    F[MX] = U[MX] * un + p * nx
    F[MY] = U[MY] * un + p * ny
    return F


@njit(cache=True)
def hll_flux_state(
    UL: np.ndarray,
    UR: np.ndarray,
    nx: float,
    ny: float,
    g: float,
    rho_f: float,
    rho_s: float,
    h_dry: float,
) -> np.ndarray:
    hL, mL, uL, vL, _, _, _, _, _ = state_derived(UL, rho_f, rho_s, h_dry)
    hR, mR, uR, vR, _, _, _, _, _ = state_derived(UR, rho_f, rho_s, h_dry)
    if hL <= h_dry and hR <= h_dry:
        return np.zeros(NV, dtype=np.float64)
    unL = uL * nx + vL * ny
    unR = uR * nx + vR * ny
    cL = math.sqrt(g * max(hL, 0.0))
    cR = math.sqrt(g * max(hR, 0.0))
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)
    FL = physical_flux(UL, nx, ny, g, rho_f, rho_s, h_dry)
    FR = physical_flux(UR, nx, ny, g, rho_f, rho_s, h_dry)
    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR
    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1.0e-30)


@njit(cache=True)
def hllc_flux_state(
    UL: np.ndarray,
    UR: np.ndarray,
    nx: float,
    ny: float,
    g: float,
    rho_f: float,
    rho_s: float,
    h_dry: float,
    dry_factor: float,
    max_froude: float,
) -> Tuple[np.ndarray, int]:
    """Barotropic-mixture HLLC. Returns (flux, used_hll_fallback)."""
    hL, mL, uL, vL, _, _, _, _, rhoL = state_derived(UL, rho_f, rho_s, h_dry)
    hR, mR, uR, vR, _, _, _, _, rhoR = state_derived(UR, rho_f, rho_s, h_dry)
    if hL <= h_dry and hR <= h_dry:
        return np.zeros(NV, dtype=np.float64), 0
    if hL <= dry_factor * h_dry or hR <= dry_factor * h_dry:
        return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1
    unL = uL * nx + vL * ny
    unR = uR * nx + vR * ny
    utL = -uL * ny + vL * nx
    utR = -uR * ny + vR * nx
    cL = math.sqrt(g * hL)
    cR = math.sqrt(g * hR)
    FrL = abs(unL) / (cL + 1.0e-30)
    FrR = abs(unR) / (cR + 1.0e-30)
    if FrL > max_froude or FrR > max_froude:
        return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)
    FL = physical_flux(UL, nx, ny, g, rho_f, rho_s, h_dry)
    FR = physical_flux(UR, nx, ny, g, rho_f, rho_s, h_dry)
    if SL >= 0.0:
        return FL, 0
    if SR <= 0.0:
        return FR, 0
    pL = 0.5 * rhoL * g * hL * hL
    pR = 0.5 * rhoR * g * hR * hR
    den = mL * (SL - unL) - mR * (SR - unR)
    if abs(den) < 1.0e-14:
        return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1
    Sstar = (pR - pL + mL * unL * (SL - unL) - mR * unR * (SR - unR)) / den
    if not math.isfinite(Sstar) or Sstar <= SL or Sstar >= SR:
        return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1

    if Sstar >= 0.0:
        ratio = (SL - unL) / (SL - Sstar + 1.0e-30)
        Ustar = np.empty(NV, dtype=np.float64)
        for k in range(5):
            Ustar[k] = UL[k] * ratio
        mstar = mL * ratio
        mn = mstar * Sstar
        mt = mstar * utL
        Ustar[MX] = mn * nx - mt * ny
        Ustar[MY] = mn * ny + mt * nx
        if ratio <= 0.0 or not np.all(np.isfinite(Ustar)):
            return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1
        return FL + SL * (Ustar - UL), 0
    ratio = (SR - unR) / (SR - Sstar + 1.0e-30)
    Ustar = np.empty(NV, dtype=np.float64)
    for k in range(5):
        Ustar[k] = UR[k] * ratio
    mstar = mR * ratio
    mn = mstar * Sstar
    mt = mstar * utR
    Ustar[MX] = mn * nx - mt * ny
    Ustar[MY] = mn * ny + mt * nx
    if ratio <= 0.0 or not np.all(np.isfinite(Ustar)):
        return hll_flux_state(UL, UR, nx, ny, g, rho_f, rho_s, h_dry), 1
    return FR + SR * (Ustar - UR), 0


@njit(parallel=True, cache=True)
def compute_primitives(
    U: np.ndarray,
    zb: np.ndarray,
    active: np.ndarray,
    P: np.ndarray,
    rho_f: float,
    rho_s: float,
    h_dry: float,
) -> None:
    ny, nx = zb.shape
    for j in prange(ny):
        for i in range(nx):
            if active[j, i] == 0:
                for k in range(7):
                    P[k, j, i] = 0.0
                P[0, j, i] = zb[j, i]
                P[5, j, i] = 0.5
                continue
            h, _, u, v, cs, lam, fcu, fcl, _ = state_derived(U[:, j, i], rho_f, rho_s, h_dry)
            P[0, j, i] = h + zb[j, i]
            P[1, j, i] = u
            P[2, j, i] = v
            P[3, j, i] = cs
            P[4, j, i] = lam
            P[5, j, i] = fcu
            P[6, j, i] = fcl


@njit(parallel=True, cache=True)
def compute_slopes(P: np.ndarray, active: np.ndarray, sx: np.ndarray, sy: np.ndarray, limiter: int) -> None:
    nv, ny, nx = P.shape
    # Parallelize over grid rows rather than the seven state channels.
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            usable = not (
                active[j, i] == 0
                or active[j, i - 1] == 0
                or active[j, i + 1] == 0
                or active[j - 1, i] == 0
                or active[j + 1, i] == 0
            )
            for k in range(nv):
                if not usable:
                    sx[k, j, i] = 0.0
                    sy[k, j, i] = 0.0
                else:
                    sx[k, j, i] = limited_slope(
                        P[k, j, i] - P[k, j, i - 1],
                        P[k, j, i + 1] - P[k, j, i],
                        limiter,
                    )
                    sy[k, j, i] = limited_slope(
                        P[k, j, i] - P[k, j - 1, i],
                        P[k, j + 1, i] - P[k, j, i],
                        limiter,
                    )


@njit(cache=True)
def build_face_state(
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
) -> Tuple[np.ndarray, float]:
    h_orig = max(eta - z_local, 0.0)
    h_star = max(eta - z_star, 0.0)
    Ustar = primitive_to_state(h_star, u, v, cs, lam, fcu, fcl, rho_f, rho_s, max_cs)
    rho = rho_f * (1.0 - clamp(cs, 0.0, max_cs)) + rho_s * clamp(cs, 0.0, max_cs)
    dp = 0.5 * rho * g * (h_orig * h_orig - h_star * h_star)
    return Ustar, dp


@njit(parallel=True, cache=True)
def compute_face_fluxes(
    U: np.ndarray,
    zb: np.ndarray,
    active: np.ndarray,
    P: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    corr_x_L: np.ndarray,
    corr_x_R: np.ndarray,
    corr_y_L: np.ndarray,
    corr_y_R: np.ndarray,
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
    fallback_count: np.ndarray,
) -> None:
    ny, nx = zb.shape
    # x-faces: face i lies between cells i-1 and i.
    for j in prange(1, ny - 1):
        local_fallback = 0
        for i in range(1, nx):
            aL = active[j, i - 1]
            aR = active[j, i]
            if aL == 0 and aR == 0:
                for k in range(NV):
                    fx[k, j, i] = 0.0
                corr_x_L[j, i] = 0.0
                corr_x_R[j, i] = 0.0
                continue
            PL = np.empty(7, dtype=np.float64)
            PR = np.empty(7, dtype=np.float64)
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
            UL, dpL = build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g)
            UR, dpR = build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g)
            hL, _, _, _, _, _, _, _, _ = state_derived(UL, rho_f, rho_s, h_dry)
            hR, _, _, _, _, _, _, _, _ = state_derived(UR, rho_f, rho_s, h_dry)
            rel_jump = abs(hR - hL) / max(hR, hL, h_dry)

            # A configured HLL flux may still use MUSCL reconstruction.  By
            # contrast, an HLLC safety fallback is a complete local fallback:
            # reconstruct the two adjacent cells as piecewise constants and
            # recompute both the hydrostatic corrections and the HLL flux.
            fallback_face = False
            if flux_id == FLUX_HLL:
                F = hll_flux_state(UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry)
            elif rel_jump > hybrid_jump:
                fallback_face = True
                F = np.zeros(NV, dtype=np.float64)
            else:
                F, fb = hllc_flux_state(
                    UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry,
                    dry_factor, max_froude,
                )
                fallback_face = fb == 1

            if fallback_face:
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
                UL, dpL = build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g)
                UR, dpR = build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g)
                F = hll_flux_state(UL, UR, 1.0, 0.0, g, rho_f, rho_s, h_dry)
                local_fallback += 1
            for k in range(NV):
                fx[k, j, i] = F[k]
            corr_x_L[j, i] = dpL
            corr_x_R[j, i] = dpR
        fallback_count[j] = local_fallback

    # y-faces: face j lies between cells j-1 and j.
    offset = ny
    for i in prange(1, nx - 1):
        local_fallback = 0
        for j in range(1, ny):
            aL = active[j - 1, i]
            aR = active[j, i]
            if aL == 0 and aR == 0:
                for k in range(NV):
                    fy[k, j, i] = 0.0
                corr_y_L[j, i] = 0.0
                corr_y_R[j, i] = 0.0
                continue
            PL = np.empty(7, dtype=np.float64)
            PR = np.empty(7, dtype=np.float64)
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
            UL, dpL = build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g)
            UR, dpR = build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g)
            hL, _, _, _, _, _, _, _, _ = state_derived(UL, rho_f, rho_s, h_dry)
            hR, _, _, _, _, _, _, _, _ = state_derived(UR, rho_f, rho_s, h_dry)
            rel_jump = abs(hR - hL) / max(hR, hL, h_dry)

            fallback_face = False
            if flux_id == FLUX_HLL:
                F = hll_flux_state(UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry)
            elif rel_jump > hybrid_jump:
                fallback_face = True
                F = np.zeros(NV, dtype=np.float64)
            else:
                F, fb = hllc_flux_state(
                    UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry,
                    dry_factor, max_froude,
                )
                fallback_face = fb == 1

            if fallback_face:
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
                UL, dpL = build_face_state(PL[0], PL[1], PL[2], PL[3], PL[4], PL[5], PL[6], zL, zstar, rho_f, rho_s, max_cs, g)
                UR, dpR = build_face_state(PR[0], PR[1], PR[2], PR[3], PR[4], PR[5], PR[6], zR, zstar, rho_f, rho_s, max_cs, g)
                F = hll_flux_state(UL, UR, 0.0, 1.0, g, rho_f, rho_s, h_dry)
                local_fallback += 1
            for k in range(NV):
                fy[k, j, i] = F[k]
            corr_y_L[j, i] = dpL
            corr_y_R[j, i] = dpR
        fallback_count[offset + i] = local_fallback


@njit(parallel=True, cache=True)
def divergence_rhs(
    fx: np.ndarray,
    fy: np.ndarray,
    corr_x_L: np.ndarray,
    corr_x_R: np.ndarray,
    corr_y_L: np.ndarray,
    corr_y_R: np.ndarray,
    active: np.ndarray,
    rhs: np.ndarray,
    dx: float,
    dy: float,
    ng: int,
) -> None:
    ny, nx = active.shape
    for j in prange(ng, ny - ng):
        for i in range(ng, nx - ng):
            if active[j, i] == 0:
                for k in range(NV):
                    rhs[k, j, i] = 0.0
                continue
            for k in range(NV):
                rhs[k, j, i] = -(fx[k, j, i + 1] - fx[k, j, i]) / dx - (fy[k, j + 1, i] - fy[k, j, i]) / dy
            # Hydrostatic pressure corrections are side-specific.
            rhs[MX, j, i] += -(corr_x_L[j, i + 1] - corr_x_R[j, i]) / dx
            rhs[MY, j, i] += -(corr_y_L[j + 1, i] - corr_y_R[j, i]) / dy


# -----------------------------------------------------------------------------
# Boundary conditions, admissibility, and time stepping.
# -----------------------------------------------------------------------------
def apply_boundary(U: np.ndarray, zb: np.ndarray, active: np.ndarray, cfg: NumericsConfig, ng: int = 2) -> None:
    left, right = _bc_id(cfg.bc_left), _bc_id(cfg.bc_right)
    bottom, top = _bc_id(cfg.bc_bottom), _bc_id(cfg.bc_top)

    def fill_x(side: str, bc: int) -> None:
        if side == "left":
            dst = slice(0, ng)
            src0 = ng
            if bc == BC_PERIODIC:
                U[:, :, dst] = U[:, :, -2 * ng:-ng]
                zb[:, dst] = zb[:, -2 * ng:-ng]
                active[:, dst] = active[:, -2 * ng:-ng]
            else:
                for k in range(ng):
                    U[:, :, k] = U[:, :, src0]
                    zb[:, k] = zb[:, src0]
                    active[:, k] = active[:, src0]
                    if bc == BC_REFLECTIVE:
                        U[MX, :, k] *= -1.0
        else:
            if bc == BC_PERIODIC:
                U[:, :, -ng:] = U[:, :, ng:2 * ng]
                zb[:, -ng:] = zb[:, ng:2 * ng]
                active[:, -ng:] = active[:, ng:2 * ng]
            else:
                src0 = -ng - 1
                for k in range(ng):
                    idx = -ng + k
                    U[:, :, idx] = U[:, :, src0]
                    zb[:, idx] = zb[:, src0]
                    active[:, idx] = active[:, src0]
                    if bc == BC_REFLECTIVE:
                        U[MX, :, idx] *= -1.0

    def fill_y(side: str, bc: int) -> None:
        if side == "bottom":
            if bc == BC_PERIODIC:
                U[:, :ng, :] = U[:, -2 * ng:-ng, :]
                zb[:ng, :] = zb[-2 * ng:-ng, :]
                active[:ng, :] = active[-2 * ng:-ng, :]
            else:
                src0 = ng
                for k in range(ng):
                    U[:, k, :] = U[:, src0, :]
                    zb[k, :] = zb[src0, :]
                    active[k, :] = active[src0, :]
                    if bc == BC_REFLECTIVE:
                        U[MY, k, :] *= -1.0
        else:
            if bc == BC_PERIODIC:
                U[:, -ng:, :] = U[:, ng:2 * ng, :]
                zb[-ng:, :] = zb[ng:2 * ng, :]
                active[-ng:, :] = active[ng:2 * ng, :]
            else:
                src0 = -ng - 1
                for k in range(ng):
                    idx = -ng + k
                    U[:, idx, :] = U[:, src0, :]
                    zb[idx, :] = zb[src0, :]
                    active[idx, :] = active[src0, :]
                    if bc == BC_REFLECTIVE:
                        U[MY, idx, :] *= -1.0

    fill_x("left", left)
    fill_x("right", right)
    fill_y("bottom", bottom)
    fill_y("top", top)


def check_admissible(U: np.ndarray, core: Tuple[slice, slice], tol: float) -> bool:
    C = U[:, core[0], core[1]]
    if not np.all(np.isfinite(C)):
        return False
    if np.min(C[[HF, HSU, HSL]]) < -tol:
        return False
    if np.min(C[HCU]) < -tol or np.min(C[HCL]) < -tol:
        return False
    if np.max(C[HCU] - C[HSU]) > tol or np.max(C[HCL] - C[HSL]) > tol:
        return False
    return True


def conservative_roundoff_repair(U: np.ndarray, core: Tuple[slice, slice], h_dry: float, tol: float) -> Dict[str, float]:
    C = U[:, core[0], core[1]]
    correction = {"fluid": 0.0, "solid": 0.0, "coarse": 0.0}
    for idx, key in ((HF, "fluid"), (HSU, "solid"), (HSL, "solid"), (HCU, "coarse"), (HCL, "coarse")):
        neg = (C[idx] < 0.0) & (C[idx] >= -tol)
        if np.any(neg):
            correction[key] += float(-np.sum(C[idx][neg]))
            C[idx][neg] = 0.0
    C[HCU] = np.minimum(np.maximum(C[HCU], 0.0), C[HSU])
    C[HCL] = np.minimum(np.maximum(C[HCL], 0.0), C[HSL])
    h = C[HF] + C[HSU] + C[HSL]
    dry = h <= h_dry
    C[MX][dry] = 0.0
    C[MY][dry] = 0.0
    return correction


class FluxWorkspace:
    def __init__(self, shape: Tuple[int, int]):
        ny, nx = shape
        self.P = np.zeros((7, ny, nx), dtype=np.float64)
        self.sx = np.zeros_like(self.P)
        self.sy = np.zeros_like(self.P)
        self.fx = np.zeros((NV, ny, nx + 1), dtype=np.float64)
        self.fy = np.zeros((NV, ny + 1, nx), dtype=np.float64)
        self.cxl = np.zeros((ny, nx + 1), dtype=np.float64)
        self.cxr = np.zeros((ny, nx + 1), dtype=np.float64)
        self.cyl = np.zeros((ny + 1, nx), dtype=np.float64)
        self.cyr = np.zeros((ny + 1, nx), dtype=np.float64)
        self.rhs = np.zeros((NV, ny, nx), dtype=np.float64)
        self.fallback_count = np.zeros(ny + nx + 4, dtype=np.int64)


def transport_rhs(
    U: np.ndarray,
    zb: np.ndarray,
    active: np.ndarray,
    cfg: SolverConfig,
    ws: FluxWorkspace,
    dx: float,
    dy: float,
    order: Optional[int] = None,
    flux_name: Optional[str] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    n = cfg.numerics
    m = cfg.material
    order = n.space_order if order is None else order
    flux_name = n.flux if flux_name is None else flux_name
    compute_primitives(U, zb, active, ws.P, m.rho_fluid, m.rho_solid, n.h_dry)
    ws.sx.fill(0.0)
    ws.sy.fill(0.0)
    if order == 2:
        compute_slopes(ws.P, active, ws.sx, ws.sy, _limiter_id(n.limiter))
    ws.fx.fill(0.0)
    ws.fy.fill(0.0)
    ws.cxl.fill(0.0)
    ws.cxr.fill(0.0)
    ws.cyl.fill(0.0)
    ws.cyr.fill(0.0)
    ws.fallback_count.fill(0)
    compute_face_fluxes(
        U, zb, active, ws.P, ws.sx, ws.sy, ws.fx, ws.fy,
        ws.cxl, ws.cxr, ws.cyl, ws.cyr,
        order, FLUX_HLLC if flux_name.lower() == "hllc" else FLUX_HLL,
        n.g, m.rho_fluid, m.rho_solid, m.max_solid_fraction, n.h_dry,
        n.hllc_dry_factor, n.hllc_max_froude, n.hybrid_depth_rel_jump,
        ws.fallback_count,
    )
    divergence_rhs(ws.fx, ws.fy, ws.cxl, ws.cxr, ws.cyl, ws.cyr, active, ws.rhs, dx, dy, 2)
    ng = 2
    ny_core = active.shape[0] - 2 * ng
    nx_core = active.shape[1] - 2 * ng
    boundary_rate = np.zeros(5, dtype=np.float64)
    for k in range(5):
        boundary_rate[k] = (
            dy * (np.sum(ws.fx[k, ng:ng + ny_core, ng]) - np.sum(ws.fx[k, ng:ng + ny_core, ng + nx_core]))
            + dx * (np.sum(ws.fy[k, ng, ng:ng + nx_core]) - np.sum(ws.fy[k, ng + ny_core, ng:ng + nx_core]))
        )
    return ws.rhs, int(np.sum(ws.fallback_count)), boundary_rate


def euler_transport_candidate(
    U: np.ndarray,
    dt: float,
    zb: np.ndarray,
    active: np.ndarray,
    cfg: SolverConfig,
    ws: FluxWorkspace,
    dx: float,
    dy: float,
    order: int,
    flux_name: str,
) -> Tuple[np.ndarray, int, np.ndarray]:
    apply_boundary(U, zb, active, cfg.numerics)
    rhs, fb, boundary_rate = transport_rhs(U, zb, active, cfg, ws, dx, dy, order, flux_name)
    out = U + dt * rhs
    return out, fb, boundary_rate


def transport_step_ssprk2(
    U: np.ndarray,
    dt: float,
    zb: np.ndarray,
    active: np.ndarray,
    cfg: SolverConfig,
    ws: FluxWorkspace,
    dx: float,
    dy: float,
    core: Tuple[slice, slice],
) -> Tuple[Optional[np.ndarray], int, bool, np.ndarray]:
    n = cfg.numerics
    fallback_used = False
    U0 = U.copy()
    U1, fb1, br1 = euler_transport_candidate(U0.copy(), dt, zb, active, cfg, ws, dx, dy, n.space_order, n.flux)
    if not check_admissible(U1, core, n.positivity_tolerance):
        U1, fb1, br1 = euler_transport_candidate(U0.copy(), dt, zb, active, cfg, ws, dx, dy, 1, "hll")
        fallback_used = True
    if not check_admissible(U1, core, n.positivity_tolerance):
        return None, fb1, True, np.zeros(5, dtype=np.float64)
    conservative_roundoff_repair(U1, core, n.h_dry, n.positivity_tolerance)
    if n.time_order == 1:
        return U1, fb1, fallback_used, br1
    apply_boundary(U1, zb, active, n)
    U2e, fb2, br2 = euler_transport_candidate(U1.copy(), dt, zb, active, cfg, ws, dx, dy, n.space_order, n.flux)
    U2 = 0.5 * (U0 + U2e)
    if not check_admissible(U2, core, n.positivity_tolerance):
        U2e, fb2, br2 = euler_transport_candidate(U1.copy(), dt, zb, active, cfg, ws, dx, dy, 1, "hll")
        U2 = 0.5 * (U0 + U2e)
        fallback_used = True
    if not check_admissible(U2, core, n.positivity_tolerance):
        return None, fb1 + fb2, True, np.zeros(5, dtype=np.float64)
    conservative_roundoff_repair(U2, core, n.h_dry, n.positivity_tolerance)
    return U2, fb1 + fb2, fallback_used, 0.5 * (br1 + br2)


def compute_dt(U: np.ndarray, core: Tuple[slice, slice], dx: float, dy: float, cfg: SolverConfig) -> float:
    n, m = cfg.numerics, cfg.material
    C = U[:, core[0], core[1]]
    hf = np.maximum(C[HF], 0.0)
    hs = np.maximum(C[HSU], 0.0) + np.maximum(C[HSL], 0.0)
    h = hf + hs
    mass = m.rho_fluid * hf + m.rho_solid * hs
    wet = h > n.h_dry
    if not np.any(wet):
        return n.dt_max
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    u[wet] = C[MX][wet] / mass[wet]
    v[wet] = C[MY][wet] / mass[wet]
    c = np.sqrt(n.g * h[wet])
    sx = np.max(np.abs(u[wet]) + c)
    sy = np.max(np.abs(v[wet]) + c)
    dt = n.cfl * min(dx / max(sx, 1.0e-12), dy / max(sy, 1.0e-12))
    return min(float(dt), n.dt_max)
