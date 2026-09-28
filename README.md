<div align="center">

<img src="docs/images/PyDebrisFlow2D_banner.png" width="100%" alt="PyDebrisFlow2D banner">

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)
![CPU](https://img.shields.io/badge/Backend-CPU-555555)
![CUDA](https://img.shields.io/badge/Backend-CUDA-76B900?logo=nvidia&logoColor=white)
![Research software](https://img.shields.io/badge/Status-Research%20Software-7B2CBF)

</div>

---

**PyDebrisFlow2D** is an open-source Python finite-volume research solver for two-dimensional, depth-averaged debris-flow propagation on raster digital elevation models (DEMs).

The model combines hydrostatic reconstruction, HLLC transport with local HLL fallback, optional MUSCL reconstruction, first- or second-order time integration, variable-density constituent bookkeeping, conservative bed exchange, grain-size segregation, and CPU/CUDA execution.

The repository includes the solver, verification tests, Marsicano real-topography experiments, reviewer-oriented diagnostics, grid- and time-step-sensitivity workflows, rheological sensitivity tools, and CPU/CUDA performance benchmarks.

> **Research software.** PyDebrisFlow2D is intended for scientific research, numerical experimentation, teaching, and method development. It is not a certified operational forecasting, engineering, hazard-warning, or early-warning system.

## Main capabilities

- Cartesian finite-volume solution on real or synthetic DEMs.
- Depth-averaged two-dimensional flow with horizontal mixture velocity components `(u, v)`.
- Variable-density constituent bookkeeping.
- Hydrostatic reconstruction and topographic source balancing.
- HLLC approximate Riemann solver with local HLL fallback.
- First-order or MUSCL spatial reconstruction.
- First-order Euler or SSPRK2 time integration.
- Minmod, MC, van Leer, and superbee slope limiters.
- Positivity safeguards, conservative repair, full-stage fallback, and adaptive step retry.
- Voellmy-type basal resistance.
- O'Brien–Julien-style rheological closure.
- Conservative erosion and entrainment.
- Class-resolved deposition.
- Reduced two-layer grain-size segregation.
- Diffusive remixing.
- CPU execution through Numba.
- CUDA execution on compatible NVIDIA GPUs.
- Solver metrics, material budgets, final-state files, maps, setup previews, and optional GIFs.
- Automated verification and publication-oriented diagnostic workflows.

## Governing-equation form

PyDebrisFlow2D solves a conservative depth-averaged balance-law system of the generic form

\[
\frac{\partial \mathbf{U}}{\partial t}
+
\nabla \cdot \mathbf{F}(\mathbf{U})
=
\mathbf{S}(\mathbf{U}),
\]

or, explicitly in two horizontal dimensions,

\[
\frac{\partial \mathbf{U}}{\partial t}
+
\frac{\partial \mathbf{F}(\mathbf{U})}{\partial x}
+
\frac{\partial \mathbf{G}(\mathbf{U})}{\partial y}
=
\mathbf{S}(\mathbf{U}).
\]

The state vector contains the conservative mobile-mixture quantities required by the active physical configuration, while the source terms represent topographic forcing, basal resistance, bed exchange, deposition, segregation, and related processes.

## Important physical limitations

PyDebrisFlow2D uses one common depth-averaged mixture-velocity vector for all transported constituents. The two Cartesian velocity components describe map-plane motion, not fluid-solid phase slip.

The current formulation does not explicitly resolve:

- phase-specific momentum;
- fluid-solid slip velocity;
- dynamic pore pressure;
- non-hydrostatic vertical acceleration;
- continuous vertical concentration profiles;
- individual boulder impacts;
- fully three-dimensional free-fall dynamics.

The fixed Cartesian grid also limits the representation of very narrow channels and sharp evolving-bed features.

Grid refinement cannot recover topographic information that is absent from the source DEM. For the distributed Marsicano case, the native DEM resolution is 5 m. Real-topography grid-sensitivity experiments therefore use terrain-informed resolutions down to 5 m and do not treat sub-5 m interpolation of the same DEM as additional geomorphic information.

## Repository structure

```text
PyDebrisFlow2D/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── run_simulation.py
├── run_full_suite.py
├── run_marsicano_studies.py
├── run_reviewer_diagnostics.py
├── configs/
├── data/
├── docs/
│   └── images/
│       └── PyDebrisFlow2D_banner.png
├── pydebrisflow/
├── tests/
└── utils/
```

Generated `cache/` and `outputs/` directories are runtime products and should not be committed to the repository.

## Installation

Python 3.10 or later is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/Secchi93/PyDebrisFlow2D.git
cd PyDebrisFlow2D
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For editable development installation:

```bash
python -m pip install -e .
```

## Quick start

Run a configured simulation:

```bash
python run_simulation.py
```

Run the full verification/publication workflow:

```bash
python run_full_suite.py
```

Inspect the complete suite without running all phases:

```bash
python run_full_suite.py --dry-run
```

## Main workflows

### `run_simulation.py`

Runs a configured PyDebrisFlow2D simulation from YAML input.

Typical usage:

```bash
python run_simulation.py
python run_simulation.py --config configs/config_marsicano.yaml
python run_simulation.py --backend cpu
python run_simulation.py --backend cuda
```

### `run_full_suite.py`

Runs the project-wide verification and publication workflow.

The suite is organized into multiple automated phases covering core numerical verification, multiphysics checks, reviewer-oriented tests, Marsicano experiments, sensitivity analyses, and implementation-performance diagnostics.

Use:

```bash
python run_full_suite.py
```

or inspect the execution plan with:

```bash
python run_full_suite.py --dry-run
```

### `run_marsicano_studies.py`

Runs the real-topography Marsicano numerical experiments used for sensitivity and model-behavior analysis.

Depending on the selected configuration, the workflow can include:

- first- vs second-order comparisons;
- multiphysics ablation;
- real-topography grid sensitivity;
- rheological sensitivity;
- parameter studies;
- publication-oriented maps and summary tables.

### `run_reviewer_diagnostics.py`

Runs focused diagnostic tests designed to document numerical robustness, assumptions, convergence behavior, and reviewer-requested sensitivity analyses.

## Verification

The project includes verification and regression tests covering the major numerical and physical components.

Representative checks include:

- HLLC flux consistency;
- lake-at-rest balance;
- wet/dry robustness;
- positivity preservation;
- segregation conservation;
- erosion/deposition budgets;
- density closure;
- Ritter dam-break comparison;
- composition advection;
- open-boundary budgets;
- grid convergence;
- limiter sensitivity;
- time-step sensitivity;
- rheological-response checks;
- CPU/CUDA consistency where applicable.

Run the project tests with:

```bash
python -m pytest
```

Specific tests can also be executed directly from the `tests/` directory.

## Marsicano real-topography experiments

The Marsicano case is distributed with a 5 m DEM and is used as the main real-topography research example.

A typical full-duration simulation uses:

```text
t_end = 275 s
```

The associated workflows can report:

- maximum final depth;
- maximum speed;
- wet area;
- mobile volume;
- runout-related diagnostics;
- solver wall time;
- step count;
- depth differences across grid resolutions;
- speed differences across grid resolutions;
- wet-support Intersection over Union;
- material-budget diagnostics.

### Real-topography grid sensitivity

The native terrain resolution is 5 m.

The controlled grid-sensitivity workflow uses resolutions such as:

```text
20 m
15 m
10 m
8 m
6.5 m
5 m
```

The 5 m run is the finest terrain-informed realization when the source DEM has 5 m spacing.

Interpolation between grids is used only for cross-grid diagnostics and does not create additional terrain information.

## Rheology

PyDebrisFlow2D supports alternative basal-resistance formulations for research comparison.

### Voellmy-type resistance

The Voellmy formulation combines frictional and velocity-dependent resistance and supports composition-dependent end-member parameters.

### O'Brien–Julien-style closure

An alternative debris-flow rheological formulation is available for sensitivity studies and model comparison.

The exact rheological configuration used in a scientific experiment should always be archived together with the corresponding YAML file and software version.

## CPU and CUDA

PyDebrisFlow2D supports:

- CPU execution through Numba;
- CUDA execution on compatible NVIDIA GPUs.

The selected backend should be documented for reproducibility.

Performance studies should distinguish physical/model sensitivity from hardware-performance benchmarking.

Where CPU/CUDA comparisons are performed, report at least:

- wall-clock time;
- accepted step count;
- time per accepted step;
- speedup;
- selected resolution;
- numerical order;
- hardware information;
- software/dependency versions.

## Configuration

Simulation configuration is controlled through YAML files.

Principal configuration blocks include:

- `compute`
- `grid`
- `numerics`
- `material`
- `friction`
- `release`
- `erosion`
- `deposition`
- `segregation`
- `output`

Depending on the workflow, additional blocks may control ablation, grid sensitivity, reviewer diagnostics, rheological sensitivity, or benchmark settings.

## Typical outputs

A simulation may create:

```text
config_resolved.yaml
runtime_geometry.yaml
interactive_setup.png
state_*.npz
metrics.json
final_state.npz
depth_evolution.gif
```

as well as maps of quantities such as:

- flow depth;
- speed;
- Cartesian velocity components;
- terrain-aligned velocities;
- density;
- solid fraction;
- coarse fraction;
- segregation-related fields.

Outputs depend on the selected configuration.

Do not interpret maps alone. Scientific assessment should also include conservation residuals, material budgets, fallback counts, retries, wet/dry behavior, grid sensitivity, time-step sensitivity, parameter sensitivity, and the physical assumptions relevant to the application.

## Reproducible scientific use

For every published or shared simulation, retain:

- the exact PyDebrisFlow2D release or commit;
- the complete input YAML configuration;
- the resolved configuration;
- original and prepared DEMs;
- DEM coordinate reference system;
- terrain-processing history;
- release geometry;
- friction geometry;
- initial conditions;
- boundary conditions;
- rheological parameters;
- erosion/deposition parameters;
- segregation parameters;
- selected CPU/CUDA backend;
- hardware information;
- Python version;
- dependency versions;
- relevant random seeds;
- conservation diagnostics;
- mesh-sensitivity results;
- time-step-sensitivity results;
- verification results;
- any local code modifications.

## Scientific interpretation

Successful numerical execution does not imply physical validity.

Results depend on:

- model assumptions;
- DEM quality;
- release geometry;
- initial conditions;
- boundary conditions;
- material parameters;
- rheological parameters;
- erosion/deposition parameterization;
- segregation settings;
- numerical resolution;
- time-step controls;
- calibration;
- validation data;
- software version;
- hardware and backend.

Independent validation is required before using model results in applied hazard or engineering contexts.

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

When using PyDebrisFlow2D in scientific work:

1. cite the exact software release or commit used;
2. archive the associated configuration and input data when possible;
3. cite the related peer-reviewed scientific article when available.

Keep the version and release date in `CITATION.cff` synchronized with the version exposed by the Python package before publishing a new release.

## License

PyDebrisFlow2D is distributed under the **Apache License 2.0**.

See the `LICENSE` file for the complete terms.

The license permits use, modification, and redistribution, including commercial use, subject to its conditions.

## Research software and limitation-of-liability disclaimer

PyDebrisFlow2D is research software intended for scientific research, numerical experimentation, education, and method development.

It is **not** a certified engineering tool, operational forecasting system, emergency-management platform, hazard-warning system, early-warning system, or other safety-critical system.

Simulation outputs depend on model assumptions, input data, DEM quality, release geometry, initial and boundary conditions, material and rheological parameters, source-term parameterizations, calibration, numerical resolution, time-step selection, software dependencies, hardware, and user-defined configurations.

Numerical stability, successful execution, or satisfaction of the included verification tests does not establish that a simulated scenario is physically correct or suitable for a particular real-world application.

Results must be independently reviewed and validated by appropriately qualified professionals before being used in engineering design, hazard assessment, land-use planning, emergency management, regulatory procedures, or decisions affecting people, property, infrastructure, or the environment.

The software and its outputs must not be used as the sole basis for safety-critical or operational decisions.

PyDebrisFlow2D is provided on an **“AS IS”** and **“AS AVAILABLE”** basis, without warranties or conditions of any kind, whether express, implied, statutory, or otherwise, to the maximum extent permitted by applicable law.

Users are responsible for determining whether the software is suitable for their intended purpose, selecting and verifying input data and parameters, reviewing and validating all results, and complying with applicable laws, professional standards, institutional procedures, and safety requirements.

This disclaimer supplements the project documentation but does not replace, amend, or override the terms of the Apache License 2.0.

---

<div align="center">

**PyDebrisFlow2D**

Research software for conservative two-dimensional debris-flow simulation.

</div>
