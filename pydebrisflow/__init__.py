from __future__ import annotations

"""Public scientific API for PyDebrisFlow2D."""

from .config import (
    ComputeConfig,
    DepositionConfig,
    ErosionConfig,
    FrictionConfig,
    GridConfig,
    MaterialConfig,
    NumericsConfig,
    OutputConfig,
    ReleaseConfig,
    SegregationConfig,
    SolverConfig,
    load_config,
    validate_config,
)
from .constants import *
from .dem import load_esri_ascii_cropped, load_or_build_dem, resample_dem
from .geometry import (
    click_polygon,
    polygon_mask,
    prepare_interactive_geometry,
    save_setup_preview,
)
from .numerics import (
    FluxWorkspace,
    apply_boundary,
    build_face_state,
    compute_face_fluxes,
    compute_primitives,
    compute_slopes,
    check_admissible,
    compute_dt,
    divergence_rhs,
    euler_transport_candidate,
    conservative_roundoff_repair,
    hll_flux_state,
    limited_slope,
    minmod2,
    hllc_flux_state,
    physical_flux,
    clamp,
    primitive_to_state,
    state_derived,
    transport_rhs,
    transport_step_ssprk2,
)
from .outputs import make_depth_gif, save_maps, save_snapshot
from .physics import (
    apply_erosion_deposition,
    apply_segregation,
    apply_speed_cap,
    apply_voellmy_friction,
    basal_shear,
    build_release,
    composition_fields,
    ferguson_church_settling_velocity,
    material_budget,
    terrain_geometry,
)
from .simulation import run_solver

__version__ = "1.1.1"
