from __future__ import annotations

"""Mixture composition, Voellmy friction, erosion/deposition, and segregation."""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .config import SolverConfig
from .constants import HCL, HCU, HF, HSL, HSU, MX, MY, NV
from .numerics import clamp, primitive_to_state

# -----------------------------------------------------------------------------
def terrain_geometry(zb_core: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dzdy, dzdx = np.gradient(zb_core.astype(np.float64), dy, dx)
    cosbeta = 1.0 / np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
    return dzdx, dzdy, cosbeta


def terrain_velocity_components(
    u: np.ndarray,
    v: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    gradient_epsilon: float = 1.0e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Cartesian velocity onto local downslope/cross-slope directions.

    This is a diagnostic transform only: the governing equations continue to
    evolve the Cartesian momentum components Mx and My.  On locally flat cells
    the topographic frame is undefined, so both diagnostic components are set
    to zero.
    """
    grad = np.hypot(dzdx, dzdy)
    valid = grad > gradient_epsilon
    ex = np.zeros_like(grad, dtype=np.float64)
    ey = np.zeros_like(grad, dtype=np.float64)
    # Unit vector of steepest descent in the horizontal plane.
    ex[valid] = -dzdx[valid] / grad[valid]
    ey[valid] = -dzdy[valid] / grad[valid]
    v_down = u * ex + v * ey
    v_cross = -u * ey + v * ex
    v_down[~valid] = 0.0
    v_cross[~valid] = 0.0
    return v_down, v_cross


def composition_fields(
    Uc: np.ndarray,
    cfg: SolverConfig,
    mu_override: Optional[np.ndarray] = None,
    xi_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    m = cfg.material
    hf = np.maximum(Uc[HF], 0.0)
    hsu = np.maximum(Uc[HSU], 0.0)
    hsl = np.maximum(Uc[HSL], 0.0)
    hcu = np.clip(Uc[HCU], 0.0, hsu)
    hcl = np.clip(Uc[HCL], 0.0, hsl)
    hs = hsu + hsl
    hc = hcu + hcl
    h = hf + hs
    mass = m.rho_fluid * hf + m.rho_solid * hs
    wet = h > cfg.numerics.h_dry
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    u[wet] = Uc[MX][wet] / mass[wet]
    v[wet] = Uc[MY][wet] / mass[wet]
    speed = np.hypot(u, v)
    cs = np.zeros_like(h)
    fc = np.zeros_like(h)
    cs[wet] = hs[wet] / h[wet]
    solid = hs > 0.0
    fc[solid] = hc[solid] / hs[solid]
    rho = np.full_like(h, m.rho_fluid)
    rho[wet] = mass[wet] / h[wet]
    mu = m.mu_fine * (1.0 - fc) + m.mu_coarse * fc
    xi = 1.0 / ((1.0 - fc) / max(m.xi_fine, 1.0e-12) + fc / max(m.xi_coarse, 1.0e-12))
    if mu_override is not None:
        use = np.isfinite(mu_override)
        mu = np.where(use, mu_override, mu)
    if xi_override is not None:
        use = np.isfinite(xi_override)
        xi = np.where(use, xi_override, xi)
    return dict(hf=hf, hsu=hsu, hsl=hsl, hcu=hcu, hcl=hcl, hs=hs, hc=hc, h=h, mass=mass,
                wet=wet, u=u, v=v, speed=speed, cs=cs, fc=fc, rho=rho, mu=mu, xi=xi)



def _rheology_model(cfg: SolverConfig) -> str:
    model = str(cfg.rheology.model).strip().lower().replace("-", "_")
    return "obrien_julien" if model in ("obrien_julien", "oj") else "voellmy"


def obrien_julien_properties(fields: Dict[str, np.ndarray], cfg: SolverConfig) -> Dict[str, np.ndarray]:
    """Return concentration-dependent O'Brien-Julien rheological properties.

    Literature basis
    ----------------
    O'Brien & Julien (1988) measured exponential concentration dependence for
    mud-matrix yield stress and dynamic viscosity. O'Brien, Julien & Fullerton
    (1993) embedded that rheology in a depth-averaged quadratic-resistance
    routing model. The FLO-2D formulation writes

        tau_y = alpha_y exp(beta_y C_v)
        eta   = alpha_eta exp(beta_eta C_v)

    with a depth-integrated yield + viscous + turbulent/dispersive friction
    slope. The empirical alpha/beta coefficients are material-specific.

    PyDebrisFlow2D provides two interpretations of the empirical laws. The
    legacy ``literature_piecewise`` option reproduces a clear-water/debris-flow
    regime switch used in EDDA-style applications. The recommended
    ``continuous_asymptotic`` option instead preserves the O'Brien-Julien
    exponential concentration sensitivity as an *excess* above the carrier-
    fluid limit:

        eta(C)   = eta_w + alpha_eta [exp(beta_eta C) - 1]
        tau_y(C) =         alpha_y   [exp(beta_y C)   - 1]

    and anchors the turbulent/dispersive roughness continuously to n_w at
    zero solids. This anchoring is a PyDebrisFlow2D regularization, not a
    verbatim equation from O'Brien & Julien. Its asymptotes are constrained by
    suspension-rheology literature: Boyer, Guazzelli & Pouliquen (2011) recover
    the carrier-fluid/Einstein limit at low particle volume fraction and a
    continuously increasing frictional/hydrodynamic resistance toward dense
    suspension; Pellegrino & Schippa (2018) likewise describe debris and
    hyperconcentrated mixtures with concentration as a continuous state
    variable. The construction removes the artificial resistance jump produced
    by a hard C_v threshold while retaining the calibrated O'Brien-Julien slopes.
    """
    r = cfg.rheology
    h = fields["h"]
    hs = fields["hs"]
    hc = fields["hc"]
    cs = np.clip(fields["cs"], 0.0, cfg.material.max_solid_fraction)
    fine = np.maximum(hs - hc, 0.0)
    cv_fine = np.zeros_like(h)
    wet = h > cfg.numerics.h_dry
    cv_fine[wet] = fine[wet] / h[wet]
    basis = str(r.oj_concentration_basis).strip().lower()
    cv_rheo = cs if basis == "total" else cv_fine

    # Empirical O'Brien-Julien coefficients are commonly tabulated in poise
    # and dyn/cm^2. Convert to SI before applying the depth-integrated source.
    alpha_eta_si = 0.1 * float(r.oj_alpha_viscosity_poise)  # Pa s
    alpha_tau_si = 0.1 * float(r.oj_alpha_yield_dyn_cm2)    # Pa
    eta_raw = alpha_eta_si * np.exp(float(r.oj_beta_viscosity) * cv_rheo)
    tau_raw = alpha_tau_si * np.exp(float(r.oj_beta_yield) * cv_rheo)

    transition = str(r.oj_transition_mode).strip().lower()
    if transition in ("literature_piecewise", "piecewise"):
        # Literature-backed clear-water / hyperconcentrated switch used in
        # EDDA-style implementations: Manning below Cv=0.20, O'Brien-Julien
        # quadratic rheology above the threshold. The threshold is configurable
        # because the empirical classification is not universal.
        clear = cs < float(r.oj_clear_water_cv_threshold)
        eta = np.where(clear, float(r.oj_water_dynamic_viscosity_pas), eta_raw)
        tau_y = np.where(clear, 0.0, tau_raw)
        n_debris = float(r.oj_water_manning_n) * float(r.oj_turbulent_b) * np.exp(float(r.oj_turbulent_m) * cs)
        n_td = np.where(clear, float(r.oj_water_manning_n), n_debris)
    elif transition in ("continuous_asymptotic", "regularized"):
        # Literature-constrained asymptotic anchoring.  O'Brien-Julien supplies
        # the exponential concentration sensitivity; the zero-solids limits are
        # imposed from suspension rheology (eta -> carrier-fluid viscosity,
        # tau_y -> 0) instead of introducing a discontinuous regime threshold.
        eta = float(r.oj_water_dynamic_viscosity_pas) + alpha_eta_si * np.expm1(float(r.oj_beta_viscosity) * cv_rheo)
        tau_y = alpha_tau_si * np.expm1(float(r.oj_beta_yield) * cv_rheo)
        # Treat the published exponential roughness factor as a concentration-
        # dependent *excess* over clear-water roughness. This is continuous and
        # differentiable at cs=0 and preserves the empirical exponential slope.
        n_td = float(r.oj_water_manning_n) * (
            1.0 + float(r.oj_turbulent_b) * np.expm1(float(r.oj_turbulent_m) * cs)
        )
    else:
        eta = eta_raw
        tau_y = tau_raw
        n_td = float(r.oj_water_manning_n) * float(r.oj_turbulent_b) * np.exp(float(r.oj_turbulent_m) * cs)

    tau_y = np.where(wet, tau_y, 0.0)
    eta = np.where(wet, eta, float(r.oj_water_dynamic_viscosity_pas))
    n_td = np.where(wet, n_td, float(r.oj_water_manning_n))
    return {
        "cv_rheology": cv_rheo,
        "cv_total": cs,
        "yield_stress_pa": tau_y,
        "dynamic_viscosity_pas": eta,
        "n_td": n_td,
    }

def basal_shear(fields: Dict[str, np.ndarray], cosbeta: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Basal shear stress for the selected resistance closure."""
    n, m = cfg.numerics, cfg.material
    h = fields["h"]
    mass = fields["mass"]
    rho = fields["rho"]
    speed = fields["speed"]

    if _rheology_model(cfg) == "obrien_julien":
        oj = obrien_julien_properties(fields, cfg)
        hh = np.maximum(h, n.h_dry)
        # O'Brien-Julien/FLO-2D depth-integrated quadratic resistance:
        # tau_b = tau_y + K eta V/(8h) + rho g n_td^2 V^2 / h^(1/3).
        tau = (
            oj["yield_stress_pa"]
            + float(cfg.rheology.oj_laminar_K) * oj["dynamic_viscosity_pas"] * speed / (8.0 * hh)
            + rho * n.g * oj["n_td"] ** 2 * speed * speed / np.power(hh, 1.0 / 3.0)
        )
        tau[h <= n.h_dry] = 0.0
        return tau

    mu = fields["mu"]
    xi = fields["xi"]
    normal = mass * n.g * cosbeta
    tau = mu * normal + rho * n.g * speed * speed / np.maximum(xi, 1.0e-12)
    if m.yield_N0_pa > 0.0:
        tau += np.maximum(0.0, 1.0 - mu) * m.yield_N0_pa * (1.0 - np.exp(-normal / m.yield_N0_pa))
    tau[h <= n.h_dry] = 0.0
    return tau


def apply_voellmy_friction(
    Uc: np.ndarray,
    cosbeta: np.ndarray,
    dt: float,
    cfg: SolverConfig,
    mu_override: Optional[np.ndarray] = None,
    xi_override: Optional[np.ndarray] = None,
) -> None:
    """Apply the selected basal-resistance source step.

    The function name is retained for API compatibility. With
    ``rheology.model=obrien_julien`` it applies the concentration-aware
    O'Brien-Julien quadratic closure instead of Voellmy resistance.
    """
    n, m = cfg.numerics, cfg.material
    f = composition_fields(Uc, cfg, mu_override, xi_override)
    h, mass, speed = f["h"], f["mass"], f["speed"]
    wet = f["wet"] & (speed > 0.0)
    if not np.any(wet):
        return

    if _rheology_model(cfg) == "obrien_julien":
        oj = obrien_julien_properties(f, cfg)
        hh = np.maximum(h, n.h_dry)
        rho = np.maximum(f["rho"], 1.0e-30)
        # ds/dt = -(a + c*s + b*s^2), with yield treated as a stopping
        # threshold and the linear/quadratic terms treated by backward Euler.
        a = oj["yield_stress_pa"] / np.maximum(mass, 1.0e-30)
        c = float(cfg.rheology.oj_laminar_K) * oj["dynamic_viscosity_pas"] / (8.0 * rho * hh * hh)
        b = n.g * oj["n_td"] ** 2 / np.power(hh, 4.0 / 3.0)
        rem = speed - dt * a
        s1 = np.zeros_like(speed)
        moving = wet & (rem > 0.0)
        if np.any(moving):
            A = dt * b[moving]
            B = 1.0 + dt * c[moving]
            R = rem[moving]
            disc = np.sqrt(B * B + 4.0 * A * R)
            # Cancellation-safe positive quadratic root; also valid as A->0.
            vals = np.where(A > 1.0e-14, 2.0 * R / (B + disc), R / B)
            s1[moving] = vals
        fac = np.zeros_like(speed)
        fac[wet] = s1[wet] / np.maximum(speed[wet], 1.0e-30)
        Uc[MX] *= fac
        Uc[MY] *= fac
        return

    mu, xi = f["mu"], f["xi"]
    normal = mass * n.g * cosbeta
    yield_acc = np.zeros_like(h)
    if m.yield_N0_pa > 0.0:
        ystress = np.maximum(0.0, 1.0 - mu) * m.yield_N0_pa * (1.0 - np.exp(-normal / m.yield_N0_pa))
        yield_acc[wet] = ystress[wet] / mass[wet]
    a = mu * n.g * cosbeta + yield_acc
    b = n.g / (np.maximum(xi, 1.0e-12) * np.maximum(h, n.h_dry))
    s0 = speed
    rem = s0 - dt * a
    s1 = np.zeros_like(s0)
    moving = wet & (rem > 0.0)
    if np.any(moving):
        bm = b[moving]
        rm = rem[moving]
        small = bm * dt < 1.0e-12
        vals = np.empty_like(rm)
        vals[small] = rm[small]
        vals[~small] = (np.sqrt(1.0 + 4.0 * bm[~small] * dt * rm[~small]) - 1.0) / (2.0 * bm[~small] * dt)
        s1[moving] = vals
    fac = np.zeros_like(s0)
    fac[wet] = s1[wet] / np.maximum(s0[wet], 1.0e-30)
    Uc[MX] *= fac
    Uc[MY] *= fac


def apply_basal_resistance(
    Uc: np.ndarray,
    cosbeta: np.ndarray,
    dt: float,
    cfg: SolverConfig,
    mu_override: Optional[np.ndarray] = None,
    xi_override: Optional[np.ndarray] = None,
) -> None:
    """Preferred closure-neutral alias for :func:`apply_voellmy_friction`."""
    apply_voellmy_friction(Uc, cosbeta, dt, cfg, mu_override, xi_override)


def ferguson_church_settling_velocity(d: float, rho_s: float, rho_f: float, nu: float, g: float = 9.81, C1: float = 18.0, C2: float = 1.0) -> float:
    R = max(rho_s / rho_f - 1.0, 0.0)
    if d <= 0.0 or R <= 0.0:
        return 0.0
    return R * g * d * d / (C1 * nu + math.sqrt(0.75 * C2 * R * g * d * d * d))


def apply_erosion_deposition(
    Uc: np.ndarray,
    zb_core: np.ndarray,
    bed_fine_solid: np.ndarray,
    bed_coarse_solid: np.ndarray,
    peak_tau: np.ndarray,
    cosbeta: np.ndarray,
    dt: float,
    cfg: SolverConfig,
    mu_override: Optional[np.ndarray] = None,
    xi_override: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    e = cfg.erosion
    dcfg = cfg.deposition
    m = cfg.material
    n = cfg.numerics
    f = composition_fields(Uc, cfg, mu_override, xi_override)
    tau = basal_shear(f, cosbeta, cfg)
    peak_tau[:] = np.maximum(peak_tau, tau)
    por = e.bed_porosity
    one_minus_por = 1.0 - por
    budget = {"eroded_bulk": 0.0, "deposited_bulk": 0.0, "eroded_solid": 0.0, "deposited_solid": 0.0}

    if e.enabled:
        available_solid = bed_fine_solid + bed_coarse_solid
        available_bulk = available_solid / one_minus_por
        if e.model.lower() == "velocity":
            Ebulk = e.velocity_erosion_coefficient * f["speed"]
        elif e.model.lower() == "peak":
            potential = e.potential_depth_per_kpa * np.maximum(peak_tau - e.critical_shear_pa, 0.0) / 1000.0
            Ebulk = np.where(potential > 0.0, e.erosion_velocity_ms, 0.0)
            Ebulk = np.minimum(Ebulk, np.maximum(potential, 0.0) / max(dt, 1.0e-30))
        else:
            excess = np.maximum(tau / max(e.critical_shear_pa, 1.0e-12) - 1.0, 0.0)
            Ebulk = e.excess_shear_rate_ms * np.power(excess, e.excess_shear_exponent)
        Ebulk = np.minimum(Ebulk, e.max_erosion_rate_ms)
        Ebulk = np.minimum(Ebulk, available_bulk / max(dt, 1.0e-30))
        Ebulk[f["h"] <= n.h_dry] = 0.0
        dB = np.maximum(Ebulk * dt, 0.0)
        solid_add = one_minus_por * dB
        bed_total = available_solid
        bed_fc = np.where(bed_total > 0.0, bed_coarse_solid / np.maximum(bed_total, 1.0e-30), e.bed_coarse_fraction)
        coarse_add = solid_add * bed_fc
        fine_add = solid_add - coarse_add
        # Remove from bed inventories exactly.
        fine_add = np.minimum(fine_add, bed_fine_solid)
        coarse_add = np.minimum(coarse_add, bed_coarse_solid)
        solid_add = fine_add + coarse_add
        dB = solid_add / one_minus_por
        fluid_add = por * dB
        bed_fine_solid -= fine_add
        bed_coarse_solid -= coarse_add
        Uc[HF] += fluid_add
        Uc[HSL] += solid_add
        Uc[HCL] += coarse_add
        # Entrainment momentum. beta=0 means static bed material and momentum dilution.
        beta = clamp(e.entrainment_velocity_fraction, 0.0, 1.0)
        if beta > 0.0:
            mass_add = m.rho_fluid * fluid_add + m.rho_solid * solid_add
            Uc[MX] += beta * mass_add * f["u"]
            Uc[MY] += beta * mass_add * f["v"]
        if e.update_bed:
            zb_core -= dB
        budget["eroded_bulk"] = float(np.sum(dB))
        budget["eroded_solid"] = float(np.sum(solid_add))

    # Recompute after erosion because concentration and velocity changed.
    f = composition_fields(Uc, cfg, mu_override, xi_override)
    tau = basal_shear(f, cosbeta, cfg)
    if dcfg.enabled:
        h = f["h"]
        cs = f["cs"]
        shear_factor = np.maximum(1.0 - tau / max(dcfg.critical_shear_pa, 1.0e-12), 0.0)
        hinder = np.power(np.maximum(1.0 - cs / max(m.max_solid_fraction, 1.0e-12), 0.0), dcfg.hindered_settling_exponent)
        wf = ferguson_church_settling_velocity(m.grain_diameter_fine_m, m.rho_solid, m.rho_fluid, m.kinematic_viscosity_m2s, n.g)
        wc = ferguson_church_settling_velocity(m.grain_diameter_coarse_m, m.rho_solid, m.rho_fluid, m.kinematic_viscosity_m2s, n.g)
        fine_lower = np.maximum(Uc[HSL] - Uc[HCL], 0.0)
        coarse_lower = np.maximum(Uc[HCL], 0.0)
        # Approximate near-bed concentration from the lower solid layer.
        cf_bed = np.minimum(2.0 * fine_lower / np.maximum(h, n.h_dry), m.max_solid_fraction)
        cc_bed = np.minimum(2.0 * coarse_lower / np.maximum(h, n.h_dry), m.max_solid_fraction)
        Df = np.minimum(wf * cf_bed * hinder * shear_factor, dcfg.max_rate_ms)
        Dc = np.minimum(wc * cc_bed * hinder * shear_factor, dcfg.max_rate_ms)
        Df[h <= dcfg.minimum_mobile_depth_m] = 0.0
        Dc[h <= dcfg.minimum_mobile_depth_m] = 0.0
        df = np.minimum(Df * dt, fine_lower)
        dc = np.minimum(Dc * dt, coarse_lower)
        solid_dep = df + dc
        bulk_dep = solid_dep / one_minus_por
        fluid_need = por * bulk_dep
        # Saturated deposition cannot consume more mobile fluid than available.
        scale = np.ones_like(solid_dep)
        mask = fluid_need > Uc[HF]
        scale[mask] = Uc[HF][mask] / np.maximum(fluid_need[mask], 1.0e-30)
        df *= scale
        dc *= scale
        solid_dep = df + dc
        bulk_dep = solid_dep / one_minus_por
        fluid_need = por * bulk_dep
        Uc[HSL] -= solid_dep
        Uc[HCL] -= dc
        Uc[HF] -= fluid_need
        bed_fine_solid += df
        bed_coarse_solid += dc
        if e.update_bed:
            zb_core += bulk_dep
        budget["deposited_bulk"] = float(np.sum(bulk_dep))
        budget["deposited_solid"] = float(np.sum(solid_dep))
    return budget


def apply_segregation(Uc: np.ndarray, dt: float, cfg: SolverConfig) -> Dict[str, float]:
    s = cfg.segregation
    if not s.enabled or dt <= 0.0:
        return {"upward_coarse": 0.0}
    f = composition_fields(Uc, cfg)
    hsu = f["hsu"]
    hsl = f["hsl"]
    hcu = f["hcu"]
    hcl = f["hcl"]
    fine_u = np.maximum(hsu - hcu, 0.0)
    fine_l = np.maximum(hsl - hcl, 0.0)
    phi_u = np.where(hsu > 0.0, hcu / np.maximum(hsu, 1.0e-30), 0.0)
    phi_l = np.where(hsl > 0.0, hcl / np.maximum(hsl, 1.0e-30), 0.0)
    shear_rate = f["speed"] / np.maximum(f["h"], s.min_layer_thickness_m)
    wseg = np.minimum(s.segregation_length_m * shear_rate, s.max_segregation_speed_ms)
    j_kin = wseg * phi_l * (1.0 - phi_u)
    dz = np.maximum(0.5 * f["h"], s.min_layer_thickness_m)
    j_mix = s.remix_diffusivity_m2s * (phi_u - phi_l) / dz
    J = j_kin - j_mix  # positive: coarse rises and an equal fine volume descends
    delta = J * dt
    pos_cap = np.minimum(hcl, fine_u)
    neg_cap = np.minimum(hcu, fine_l)
    delta = np.minimum(np.maximum(delta, -neg_cap), pos_cap)
    Uc[HCU] += delta
    Uc[HCL] -= delta
    return {"upward_coarse": float(np.sum(delta))}


def apply_speed_cap(Uc: np.ndarray, cfg: SolverConfig) -> None:
    cap = cfg.numerics.speed_cap_ms
    if cap <= 0.0:
        return
    f = composition_fields(Uc, cfg)
    mask = f["speed"] > cap
    fac = np.ones_like(f["speed"])
    fac[mask] = cap / f["speed"][mask]
    Uc[MX] *= fac
    Uc[MY] *= fac


# -----------------------------------------------------------------------------
# Initialization, budgets, output.
# -----------------------------------------------------------------------------
def build_release(Uc: np.ndarray, x: np.ndarray, y: np.ndarray, valid: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    r = cfg.release
    m = cfg.material
    X, Y = np.meshgrid(x, y)
    if not r.enabled:
        return np.zeros_like(valid, dtype=bool)
    if str(r.mode).strip().lower() == "polygon":
        if len(r.polygon) < 3:
            raise RuntimeError("release.mode=polygon requires at least three polygon vertices")
        from matplotlib.path import Path as MplPath
        pts = np.column_stack((X.ravel(), Y.ravel()))
        mask = MplPath(np.asarray(r.polygon, dtype=float)).contains_points(pts).reshape(X.shape) & valid
        geometry_name = "polygon"
    else:
        mask = (((X - r.center_x) / r.radius_x) ** 2 + ((Y - r.center_y) / r.radius_y) ** 2 <= 1.0) & valid
        geometry_name = "ellipse"
    if not np.any(mask):
        raise RuntimeError(f"Release {geometry_name} contains no valid DEM cells")
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    h0 = r.volume_m3 / (float(np.sum(mask)) * dx * dy)
    cs = m.initial_solid_fraction if r.solid_fraction is None else float(r.solid_fraction)
    fc = m.initial_coarse_fraction if r.coarse_fraction is None else float(r.coarse_fraction)
    lam = m.initial_upper_solid_fraction
    angle = math.radians(r.direction_deg)
    state = primitive_to_state(h0, r.initial_speed_ms * math.cos(angle), r.initial_speed_ms * math.sin(angle),
                               cs, lam, fc, fc, m.rho_fluid, m.rho_solid, m.max_solid_fraction)
    for k in range(NV):
        Uc[k][mask] = state[k]
    return mask


def material_budget(
    Uc: np.ndarray,
    bed_fine: np.ndarray,
    bed_coarse: np.ndarray,
    cell_area: float,
    cfg: SolverConfig,
) -> Dict[str, float]:
    m = cfg.material
    mobile_fluid = float(np.sum(Uc[HF], dtype=np.float64) * cell_area)
    mobile_fine = float(np.sum((Uc[HSU] - Uc[HCU]) + (Uc[HSL] - Uc[HCL]), dtype=np.float64) * cell_area)
    mobile_coarse = float(np.sum(Uc[HCU] + Uc[HCL], dtype=np.float64) * cell_area)
    bed_fine_v = float(np.sum(bed_fine, dtype=np.float64) * cell_area)
    bed_coarse_v = float(np.sum(bed_coarse, dtype=np.float64) * cell_area)
    por = cfg.erosion.bed_porosity
    bed_pore_fluid = por / max(1.0 - por, 1.0e-30) * (bed_fine_v + bed_coarse_v)
    mobile_mass = m.rho_fluid * mobile_fluid + m.rho_solid * (mobile_fine + mobile_coarse)
    return {
        "mobile_fluid_m3": mobile_fluid,
        "bed_pore_fluid_m3": bed_pore_fluid,
        "total_fluid_m3": mobile_fluid + bed_pore_fluid,
        "mobile_fine_m3": mobile_fine,
        "mobile_coarse_m3": mobile_coarse,
        "bed_fine_solid_m3": bed_fine_v,
        "bed_coarse_solid_m3": bed_coarse_v,
        "total_fine_m3": mobile_fine + bed_fine_v,
        "total_coarse_m3": mobile_coarse + bed_coarse_v,
        "mobile_mass_kg": mobile_mass,
    }
