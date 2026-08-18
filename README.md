# PyDebrisFlow2D

**PyDebrisFlow2D** is an open-source Python finite-volume research solver for two-dimensional, depth-averaged debris-flow propagation on raster digital elevation models (DEMs). The implementation combines hydrostatic reconstruction, HLLC transport with local HLL fallback, optional MUSCL reconstruction, first- or second-order time integration, Voellmy basal resistance, variable-density constituent bookkeeping, conservative bed exchange, and a reduced two-layer grain-size segregation model.

The repository includes the solver, a unified verification suite, the Marsicano real-topography experiments used in the accompanying manuscript, and a standalone CPU/CUDA scaling benchmark.

> **Research software.** PyDebrisFlow2D is intended for scientific research, numerical experimentation, teaching, and method development. It is not a certified operational forecasting, engineering, or early-warning system. Numerical stability and successful verification tests do not establish field predictive validity.

## Main capabilities

- Cartesian finite-volume solution on real or synthetic DEMs.
- Seven conservative state variables for mobile fluid, layered fine/coarse solids, and two horizontal mixture-momentum components.
- Fully two-dimensional bulk velocity `(u, v)` plus terrain-aligned downslope and cross-slope diagnostics.
- Hydrostatic reconstruction and topographic source balancing.
- HLLC approximate Riemann solver with local HLL fallback near difficult wet/dry or inadmissible reconstructed states.
- First-order or MUSCL spatial reconstruction with first-order Euler or SSPRK2 time integration.
- Minmod, MC, van Leer, and superbee limiters.
- Positivity safeguards, conservative repair, complete-stage fallback, and adaptive step retry.
- Voellmy resistance with fine/coarse solid end-member parameters.
- Conservative erosion/entrainment, class-resolved deposition, two-layer segregation, and diffusive remixing.
- CPU execution through Numba and CUDA execution on compatible NVIDIA GPUs.
- Solver metrics, material budgets, final-state NPZ files, maps, setup previews, and optional GIFs.
- Publication-scale plotting typography shared by verification, Marsicano, CPU/CUDA, setup-preview, and solver-map outputs.

## Important physical limitations

PyDebrisFlow2D uses one common depth-averaged mixture-velocity vector for all transported constituents. The two Cartesian velocity components describe map-plane motion, not fluid-solid phase slip. The current formulation does not resolve phase-specific momentum, dynamic pore pressure, non-hydrostatic vertical acceleration, continuous vertical concentration profiles, individual boulder impacts, or three-dimensional free-fall dynamics.

The fixed Cartesian grid also limits the representation of narrow channels and sharp evolving-bed features. Grid refinement cannot recover topographic information that is absent from the source DEM. For the distributed Marsicano case, the DEM is natively 5 m; therefore the reviewer-oriented real-topography sensitivity experiment uses six controlled resolutions, **20, 15, 10, 8, 6.5, and 5 m**, with 5 m as the finest terrain-informed calculation. No sub-5 m pseudo-refinement is used, because interpolation of the same 5 m DEM would add cells without adding independently observed geomorphic information.

## Repository structure

```text
PyDebrisFlow2D/
├── main.py
├── marsicano_ablation.py
├── test.py
├── tests/
│   └── test_cpu_cuda.py
├── configs/
│   ├── config_marsicano.yaml
│   ├── config_marsicano_ablation.yaml
│   └── config_tests.yaml
├── data/
│   ├── w46090_s10_Marsicano_UTM33_5m.asc
│   ├── w46090_s10_Marsicano_UTM33_5m.prj
│   └── ...
├── pydebrisflow/
│   ├── __init__.py
│   ├── compute.py
│   ├── config.py
│   ├── constants.py
│   ├── cuda_backend.py
│   ├── dem.py
│   ├── geometry.py
│   ├── numerics.py
│   ├── outputs.py
│   ├── physics.py
│   └── simulation.py
├── utils/
│   ├── __init__.py
│   └── prepare_dem.py
├── requirements.txt
└── LICENSE
```

Generated `cache/` and `outputs/` directories are runtime products and can be removed when a clean rerun is required.

## Installation

Python 3.10 or later is recommended.

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

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The distributed configurations request CUDA where appropriate but permit CPU fallback unless a benchmark explicitly disables it.

## Primary entry points

```bash
python main.py
python test.py
python marsicano_ablation.py
python tests/test_cpu_cuda.py
```

### `main.py`
Runs a configured simulation. By default it loads `configs/config_marsicano.yaml`.

```bash
python main.py
python main.py --config configs/config_marsicano.yaml
python main.py --backend cpu
python main.py --backend cuda
python main.py --cuda-info
```

### `test.py`
Runs the complete verification workflow and the matched Ritter first-/second-order accuracy-versus-cost analysis.

```bash
python test.py
```

Verification products are written to:

```text
outputs/verification_figures/
```

The workflow includes HLLC consistency, lake-at-rest balance, segregation conservation, erosion/deposition budgets, density closure, wet/dry fallback, Ritter dam-break profiles, composition advection, open-boundary budgets, grid convergence, limiter sensitivity, time-step sensitivity, and Ritter O1/O2 timing. Figures use enlarged publication-oriented typography.

### `marsicano_ablation.py`
This existing Marsicano test script contains two complementary workflows; no separate grid-test source file is required.

#### A. 16-run multiphysics ablation

The standard command executes the full `2^3` erosion/deposition/segregation matrix for both numerical orders:

```bash
python marsicano_ablation.py
```

Physical cases:

| ID | Erosion | Deposition | Segregation |
|---|---:|---:|---:|
| `P000` | off | off | off |
| `E100` | on | off | off |
| `D010` | off | on | off |
| `S001` | off | off | on |
| `ED110` | on | on | off |
| `ES101` | on | off | on |
| `DS011` | off | on | on |
| `EDS111` | on | on | on |

Each case is run as O1 and O2. The continuously updated table is:

```text
outputs/Marsicano_Ablation/marsicano_ablation_results.csv
```

Useful options:

```bash
python marsicano_ablation.py --dry-run
python marsicano_ablation.py --orders O2
python marsicano_ablation.py --cases P000 EDS111
python marsicano_ablation.py --backend cpu
python marsicano_ablation.py --backend cuda
python marsicano_ablation.py --restart
python marsicano_ablation.py --no-resume
```

#### B. Marsicano 20/15/10/8/6.5/5 m real-topography grid-sensitivity experiment

To answer the real-topography grid-resolution reviewer comment without creating a new test source file, run:

```bash
python marsicano_ablation.py --grid-sensitivity-only
```

The experiment loads the propagation-only `configs/config_marsicano.yaml`, uses nominal second-order numerics by default, and runs the same Marsicano scenario at:

```text
20 m -> 15 m -> 10 m -> 8 m -> 6.5 m -> 5 m
```

The 5 m run is the finest available terrain-informed solution because the distributed DEM itself has 5 m spacing. Coarser DEMs are generated from the native DEM through the repository's existing DEM-resampling path. Sub-5 m pseudo-refinement is intentionally rejected.

The workflow reports for each spacing:

- maximum final depth and speed;
- wet area;
- mobile volume;
- maximum distance from the release centroid;
- solver wall time and step count;
- depth RMSE relative to the 5 m solution after diagnostic interpolation to the 5 m cell centres;
- speed RMSE relative to the 5 m solution;
- wet-support IoU relative to the 5 m solution.

The interpolation is used **only for cross-grid diagnostics**. It does not create a finer DEM and is never used by the solver.

Outputs are written under:

```text
outputs/Marsicano_Ablation/grid_sensitivity/
```

Key products:

```text
marsicano_grid_sensitivity_results.csv
marsicano_grid_sensitivity_results.json
marsicano_grid_depth_comparison.png
marsicano_grid_depth_comparison.pdf
marsicano_grid_sensitivity_metrics.png
marsicano_grid_sensitivity_metrics.pdf
```

Each spacing also receives the normal final solver maps using manuscript-scale typography.

For a quick smoke test only:

```bash
python marsicano_ablation.py --grid-sensitivity-only --t-end 2
```

For the manuscript/reviewer result, use the full default final time of 275 s.

To execute the 16-run ablation and then the grid-sensitivity experiment in one command:

```bash
python marsicano_ablation.py --grid-sensitivity
```

## CPU/CUDA benchmark

The hardware benchmark remains separate from physical/grid sensitivity because it answers a different question: implementation performance.

```bash
python tests/test_cpu_cuda.py --dx 20 10 5 --t-end 20 --order 2
```

Products are written to:

```text
outputs/CPU_CUDA_Benchmark/
```

The benchmark warms each backend, disables CUDA-to-CPU fallback during GPU timing, and reports wall time, time per accepted step, speedup, and implementation-level working-array memory estimates.

## Plotting and figure typography

Reviewer-oriented plotting has been standardized across the project. The following outputs now use enlarged manuscript-scale labels, tick labels, legends, titles, and markers:

- all figures generated by `test.py`;
- Ritter accuracy-versus-cost figures;
- all solver final maps and GIF frames from `pydebrisflow/outputs.py`;
- setup and interactive geometry previews from `pydebrisflow/geometry.py`;
- Marsicano order-cost and grid-sensitivity figures from `marsicano_ablation.py`;
- CPU/CUDA performance figures from `tests/test_cpu_cuda.py`.

This keeps figure symbols legible after insertion at journal column/page width and directly addresses the reviewer request concerning Figures 3, 4, and related plots.

## Configuration notes

YAML configuration is divided into these principal blocks:

- `compute`: backend, CPU threads, CUDA device/block settings, fallback policy;
- `grid`: DEM, cache, target spacing, clipping, valid-cell threshold;
- `numerics`: gravity, CFL, maximum time step, final time, order, limiter, flux, boundaries, positivity and fallback controls;
- `material`: intrinsic densities, initial composition, Voellmy end-member parameters, grain properties;
- `friction`: uniform or region-specific resistance;
- `release`: geometry, volume, composition, initial velocity;
- `erosion`, `deposition`, `segregation`: source-term parameters;
- `output`: snapshots, PNG/GIF generation, progress, hillshade, setup preview.

`configs/config_marsicano_ablation.yaml` additionally contains top-level `ablation` and `grid_sensitivity` sections consumed by `marsicano_ablation.py`.

## Main simulation outputs

Depending on the configuration, a run may create:

- `config_resolved.yaml`;
- `runtime_geometry.yaml` and `interactive_setup.png`;
- `state_*.npz` snapshots;
- final maps for depth, speed, Cartesian velocities, terrain-aligned velocities, density, solid fraction, coarse fraction, and segregation;
- `depth_evolution.gif`;
- `metrics.json`;
- `final_state.npz`.

Do not interpret maps alone. Review conservation residuals, material budgets, fallback counts, retries, wet/dry behavior, mesh/time-step sensitivity, and the physical assumptions relevant to the application.

## Scientific interpretation of the 20/15/10/8/6.5/5 m study

The grid-sensitivity experiment should be described as a **real-topography resolution-sensitivity test**, not as proof of asymptotic convergence to an exact field solution. The recommended sequence is **20, 15, 10, 8, 6.5, and 5 m**, all generated from the same native terrain dataset and run with the same propagation-only physical configuration.

1. The 20, 15, 10, 8, and 6.5 m cases are controlled coarsenings of the native 5 m DEM and intentionally discard progressively less terrain detail.
2. The 5 m calculation uses the finest independent terrain information available in the distributed dataset and is therefore the reference for the cross-grid field metrics.
3. Refinement should be assessed with several diagnostics together, including depth and speed RMSE over the union of wet supports, wet-support IoU, runout-distance discrepancy, wet area, volume, and extrema. Non-monotonic behavior in local maxima or shallow margins must be reported rather than hidden.
4. A progressive reduction of field RMSE and runout discrepancy toward the 5 m calculation supports increasing solution consistency over the terrain-informed resolution range, but it must not be described as proof that the 5 m solution is grid independent.
5. A genuine physical-resolution study below 5 m requires independently finer topography and/or an adaptive or unstructured representation supported by finer terrain data. Interpolating the existing 5 m DEM alone would create additional numerical cells without adding new geomorphic information.

## Reproducibility

Precomputed per-run JSON/NPZ result files are intentionally not required as part of the distributed repository. The published experiments are intended to be reproduced by executing the supplied code with the documented configurations and input DEM. Generated metrics and state files are created locally during those runs.

For every reproduced or newly published simulation, retain:

- the exact code version/commit;
- input and resolved YAML files;
- original/prepared DEM and CRS information;
- release and friction geometry;
- rheological and source-term parameters;
- selected backend and hardware information;
- Python/dependency versions;
- conservation and fallback diagnostics;
- grid/time-step/order sensitivity outputs.

## License

PyDebrisFlow2D is distributed under the Apache License 2.0. See `LICENSE` for the complete terms.

## Disclaimer

PyDebrisFlow2D is provided as research software without a guarantee that a particular simulation is physically correct, complete, field validated, or suitable for engineering or operational decision-making. Users are responsible for calibration, uncertainty assessment, independent validation, professional review, and compliance with applicable standards and regulations.
