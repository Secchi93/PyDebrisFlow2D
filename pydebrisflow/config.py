from __future__ import annotations

"""Typed solver configuration, YAML loading, and validation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from .constants import (
    BC_OUTFLOW,
    BC_PERIODIC,
    BC_REFLECTIVE,
    LIMITER_MC,
    LIMITER_MINMOD,
    LIMITER_VANLEER,
    LIMITER_SUPERBEE,
)



@dataclass
class ComputeConfig:
    """Execution backend and parallel-computing controls."""

    backend: str = "auto"  # auto | cpu | cuda
    cpu_threads: int = 0  # 0 = Numba default
    cuda_block_x: int = 16
    cuda_block_y: int = 16
    cuda_threads_1d: int = 256
    cuda_device: int = 0
    cuda_allow_fallback: bool = True
    cuda_sync_each_step: bool = False

@dataclass
class GridConfig:
    dem_path: str = "../DEM.zip"
    zip_member: str = "DEM/DemMosaic_WGS84_33N.asc"
    cache_path: str = "cache/morino_dx5.npz"
    target_dx: float = 5.0
    clip_enabled: bool = True
    xmin: float = 373350.0
    xmax: float = 374650.0
    ymin: float = 4633230.0
    ymax: float = 4634420.0
    min_valid_fraction: float = 0.5
    nodata_barrier_m: float = 100.0


@dataclass
class NumericsConfig:
    g: float = 9.81
    cfl: float = 0.20
    dt_max: float = 0.50
    t_end: float = 1000.0
    output_dt: float = 10.0
    h_dry: float = 1.0e-4
    space_order: int = 2
    time_order: int = 2
    limiter: str = "mc"
    flux: str = "hllc"
    hllc_fallback_hll: bool = True
    hllc_dry_factor: float = 5.0
    hllc_max_froude: float = 8.0
    hybrid_depth_rel_jump: float = 0.75
    max_step_retries: int = 10
    positivity_tolerance: float = 1.0e-10
    speed_cap_ms: float = 0.0
    bc_left: str = "outflow"
    bc_right: str = "outflow"
    bc_bottom: str = "outflow"
    bc_top: str = "outflow"


@dataclass
class MaterialConfig:
    rho_fluid: float = 1000.0
    rho_solid: float = 2650.0
    initial_solid_fraction: float = 0.55
    initial_coarse_fraction: float = 0.40
    initial_upper_solid_fraction: float = 0.50
    max_solid_fraction: float = 0.68
    mu_fine: float = 0.010
    mu_coarse: float = 0.100
    xi_fine: float = 200.0
    xi_coarse: float = 200.0
    yield_N0_pa: float = 1000.0
    grain_diameter_fine_m: float = 0.002
    grain_diameter_coarse_m: float = 0.050
    kinematic_viscosity_m2s: float = 1.0e-6


@dataclass
class FrictionConfig:
    """Spatial overrides for the composition-dependent Voellmy coefficients.

    Cells outside the configured polygons continue to use the effective mu and xi
    computed from the local fine/coarse composition.  Inside a polygon, the
    region values below override those effective coefficients.  Later regions
    overwrite earlier ones where polygons overlap.
    """
    interactive: bool = False
    region_1_mu: float = 0.010
    region_1_xi: float = 200.0
    region_2_mu: float = 0.100
    region_2_xi: float = 200.0
    regions: list = field(default_factory=list)


@dataclass
class ReleaseConfig:
    enabled: bool = True
    mode: str = "ellipse"  # ellipse | polygon
    interactive_polygon: bool = False
    polygon: list = field(default_factory=list)
    volume_m3: float = 200.0
    center_x: float = 373805.33
    center_y: float = 4631362.39
    radius_x: float = 20.0
    radius_y: float = 20.0
    solid_fraction: Optional[float] = None
    coarse_fraction: Optional[float] = None
    initial_speed_ms: float = 0.0
    direction_deg: float = 0.0


@dataclass
class ErosionConfig:
    enabled: bool = True
    model: str = "excess_shear"  # excess_shear | velocity | peak
    erodible_depth_m: float = 1.0
    bed_porosity: float = 0.40
    bed_coarse_fraction: float = 0.45
    critical_shear_pa: float = 1000.0
    excess_shear_rate_ms: float = 0.010
    excess_shear_exponent: float = 1.0
    max_erosion_rate_ms: float = 0.05
    velocity_erosion_coefficient: float = 1.0e-3
    potential_depth_per_kpa: float = 0.10
    erosion_velocity_ms: float = 0.025
    entrainment_velocity_fraction: float = 0.0
    update_bed: bool = True


@dataclass
class DepositionConfig:
    enabled: bool = True
    critical_shear_pa: float = 500.0
    hindered_settling_exponent: float = 4.65
    max_rate_ms: float = 0.10
    minimum_mobile_depth_m: float = 0.005


@dataclass
class SegregationConfig:
    enabled: bool = True
    segregation_length_m: float = 0.010
    max_segregation_speed_ms: float = 0.20
    remix_diffusivity_m2s: float = 1.0e-4
    min_layer_thickness_m: float = 0.01


@dataclass
class OutputConfig:
    out_dir: str = "output_multiphysics"
    save_snapshots: bool = True
    make_png: bool = True
    make_gif: bool = False
    gif_fps: int = 10
    progress_every_steps: int = 50
    use_hillshade: bool = True
    hillshade_alpha: float = 0.28
    save_setup_preview: bool = True


@dataclass
class SolverConfig:
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    deposition: DepositionConfig = field(default_factory=DepositionConfig)
    segregation: SegregationConfig = field(default_factory=SegregationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _update_dataclass(obj: Any, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown configuration key {obj.__class__.__name__}.{key}")
        setattr(obj, key, value)


def load_config(path: str) -> SolverConfig:
    cfg = SolverConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for section in ("compute", "grid", "numerics", "material", "friction", "release", "erosion", "deposition", "segregation", "output"):
        if section in data:
            _update_dataclass(getattr(cfg, section), data[section] or {})
    base = os.path.dirname(os.path.abspath(path))
    for attr in ("dem_path", "cache_path"):
        value = getattr(cfg.grid, attr)
        if not os.path.isabs(value):
            setattr(cfg.grid, attr, os.path.abspath(os.path.join(base, value)))
    if not os.path.isabs(cfg.output.out_dir):
        cfg.output.out_dir = os.path.abspath(os.path.join(base, cfg.output.out_dir))
    validate_config(cfg)
    return cfg


def validate_config(cfg: SolverConfig) -> None:
    c = cfg.compute
    n = cfg.numerics
    backend = str(c.backend).strip().lower()
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("compute.backend must be auto, cpu, or cuda")
    if int(c.cpu_threads) < 0:
        raise ValueError("compute.cpu_threads must be >= 0")
    if int(c.cuda_block_x) <= 0 or int(c.cuda_block_y) <= 0:
        raise ValueError("CUDA block dimensions must be positive")
    if int(c.cuda_block_x) * int(c.cuda_block_y) > 1024:
        raise ValueError("CUDA block dimensions exceed 1024 threads")
    if int(c.cuda_threads_1d) <= 0 or int(c.cuda_threads_1d) > 1024:
        raise ValueError("compute.cuda_threads_1d must be in [1, 1024]")
    if int(c.cuda_threads_1d) & (int(c.cuda_threads_1d) - 1):
        raise ValueError("compute.cuda_threads_1d must be a power of two")
    if int(c.cuda_device) < 0:
        raise ValueError("compute.cuda_device must be >= 0")
    m = cfg.material
    e = cfg.erosion
    if n.space_order not in (1, 2):
        raise ValueError("space_order must be 1 or 2")
    if n.time_order not in (1, 2):
        raise ValueError("time_order must be 1 or 2")
    if not (0.0 < n.cfl <= 0.5):
        raise ValueError("cfl must be in (0, 0.5]")
    if n.flux.lower() not in ("hll", "hllc"):
        raise ValueError("flux must be hll or hllc")
    if n.limiter.lower() not in ("minmod", "mc", "vanleer", "van_leer", "superbee"):
        raise ValueError("limiter must be minmod, mc, vanleer, or superbee")
    for name in ("bc_left", "bc_right", "bc_bottom", "bc_top"):
        if getattr(n, name).lower() not in ("outflow", "reflective", "wall", "periodic"):
            raise ValueError(f"invalid {name}")
    if not (0.0 <= m.initial_solid_fraction <= m.max_solid_fraction < 1.0):
        raise ValueError("solid fractions are inconsistent")
    if not (0.0 <= m.initial_coarse_fraction <= 1.0):
        raise ValueError("initial_coarse_fraction must be in [0,1]")
    if not (0.0 < e.bed_porosity < 1.0):
        raise ValueError("bed_porosity must be in (0,1)")
    if cfg.release.volume_m3 <= 0.0:
        raise ValueError("release.volume_m3 must be positive")
    if cfg.release.mode.strip().lower() not in ("ellipse", "polygon"):
        raise ValueError("release.mode must be ellipse or polygon")
    fr = cfg.friction
    for name, value in (("region_1_mu", fr.region_1_mu), ("region_2_mu", fr.region_2_mu)):
        if not (0.0 <= float(value) <= 2.0):
            raise ValueError(f"friction.{name} must be in [0, 2]")
    for name, value in (("region_1_xi", fr.region_1_xi), ("region_2_xi", fr.region_2_xi)):
        if float(value) <= 0.0:
            raise ValueError(f"friction.{name} must be positive")
    if not (0.0 <= cfg.output.hillshade_alpha <= 1.0):
        raise ValueError("output.hillshade_alpha must be in [0, 1]")


def _bc_id(name: str) -> int:
    v = name.strip().lower()
    if v in ("reflective", "wall"):
        return BC_REFLECTIVE
    if v == "periodic":
        return BC_PERIODIC
    return BC_OUTFLOW


def _limiter_id(name: str) -> int:
    v = name.strip().lower()
    if v == "minmod":
        return LIMITER_MINMOD
    if v in ("vanleer", "van_leer"):
        return LIMITER_VANLEER
    if v == "superbee":
        return LIMITER_SUPERBEE
    return LIMITER_MC