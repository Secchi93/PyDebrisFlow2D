# PyDebrisFlow2D

<div align="center">

**A conservative Python solver for variable-density debris flows, bed exchange, and grain-size segregation**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)
![CPU](https://img.shields.io/badge/Backend-CPU-555555)
![CUDA](https://img.shields.io/badge/Backend-CUDA-76B900?logo=nvidia&logoColor=white)
![Version](https://img.shields.io/badge/version-1.4.0-blue)
![Research software](https://img.shields.io/badge/Status-Research%20Software-7B2CBF)

</div>

**PyDebrisFlow2D** is an open-source finite-volume research solver for two-dimensional, depth-averaged debris-flow propagation over raster digital elevation models (DEMs). It combines hydrostatic reconstruction, HLL/HLLC transport, optional MUSCL reconstruction, first- or second-order time integration, variable-density constituent bookkeeping, conservative bed exchange, grain-size segregation, and CPU/CUDA execution.

The repository contains the solver, numerical verification tests, the Marsicano real-topography experiments used for manuscript analyses, reviewer-oriented diagnostics, temporal and grid-sensitivity studies, and a matched CPU/CUDA benchmark.

> **Research software.** PyDebrisFlow2D is intended for scientific research, numerical experimentation, teaching, and method development. It is not a certified operational forecasting, engineering-design, emergency-management, or early-warning system.

## Highlights

- Cartesian finite-volume solver on synthetic or real DEMs.
- Seven conservative state variables for mobile fluid, layered fine/coarse solids, and two horizontal mixture-momentum components.
- Hydrostatic reconstruction and topographic source balancing.
- HLLC approximate Riemann solver with local HLL fallback for difficult wet/dry or inadmissible reconstructed states.
- First-order or MUSCL reconstruction with Euler or SSPRK2 time integration.
- Minmod, MC, van Leer, and superbee slope limiters.
- Positivity safeguards, conservative repair, stage fallback, and adaptive step retry.
- Variable-density mixture bookkeeping.
- Conservative erosion/entrainment and class-resolved deposition.
- Reduced two-layer grain-size segregation with diffusive remixing.
- CPU execution through Numba and CUDA acceleration on compatible NVIDIA GPUs.
- Publication-oriented maps, figures, GIFs, metrics, budgets, and final-state files.
- Automated verification, grid/CFL studies, reviewer diagnostics, and performance benchmarking.

## Basal-resistance closures

PyDebrisFlow2D exposes two **alternative** basal-resistance closures. They are selected per simulation and are not summed together.

### Voellmy

Voellmy resistance is the reference mobility closure for the distributed Marsicano benchmark, multiphysics ablation matrix, grid study, CFL study, and CPU/CUDA benchmark.

### Concentration-aware O'Brien–Julien

A concentration-dependent O'Brien–Julien-type quadratic resistance is also implemented. The recommended `continuous_asymptotic` transition provides a continuous carrier-fluid limit while preserving the selected concentration trends.

The distributed short O'Brien–Julien Marsicano run is an **integration test**, not a field calibration or independent observational validation.

## Physical scope and limitations

PyDebrisFlow2D uses one common depth-averaged mixture-velocity vector for all transported constituents. The two Cartesian velocity components describe map-plane motion and do not represent fluid-solid phase slip.

The current formulation does not resolve phase-specific momentum, dynamic pore pressure, non-hydrostatic vertical acceleration, continuous vertical concentration profiles, individual boulder impacts, or fully three-dimensional free-fall dynamics. Results should therefore be interpreted within the assumptions of a depth-averaged shallow-flow model.

The fixed Cartesian grid also limits representation of narrow channels and sharp evolving-bed features. For the distributed Marsicano case, the terrain data have a native spacing of 5 m. Grid-sensitivity tests therefore use **20, 15, 10, 8, 6.5, and 5 m**, with 5 m as the finest terrain-informed calculation. Sub-5 m pseudo-refinement is intentionally avoided because it would interpolate the same source DEM without adding independent terrain information.

## Repository layout

```text
PyDebrisFlow2D/
├── pydebrisflow/                  # solver package
│   ├── compute.py
│   ├── config.py
│   ├── constants.py
│   ├── cuda_backend.py
│   ├── dem.py
│   ├── geometry.py
│   ├── numerics.py
│   ├── outputs.py
│   ├── physics.py
│   ├── simulation.py
│   └── __init__.py
├── configs/
│   ├── marsicano_voellmy.yaml
│   ├── marsicano_multiphysics.yaml
│   ├── obrien_julien_quick.yaml
│   └── verification.yaml
├── data/
│   └── w46090_s10_Marsicano_UTM33_5m.tif
├── tests/
│   ├── test_cfl_regression.py
│   ├── test_concentration_rheology.py
│   ├── test_concentration_sweep.py
│   ├── test_cpu_cuda_benchmark.py
│   ├── test_oj_end_to_end.py
│   ├── test_reviewer_regression.py
│   └── test_windows_console_compat.py
├── run_full_suite.py              # complete publication/reproducibility workflow
├── run_simulation.py              # run one configured simulation
├── run_marsicano_studies.py       # ablation + real-topography grid studies
├── run_reviewer_diagnostics.py    # reviewer-requested diagnostics
├── run_cfl_study.py               # temporal CFL robustness study
├── verify_core.py                 # controlled numerical verification
├── prepare_dem.py                 # DEM preparation helper
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── .gitignore
└── LICENSE
```

`cache/`, `outputs/`, Python bytecode, and local virtual environments are generated locally and should not be committed.

## Installation

Python **3.10 or later** is recommended.

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

For editable development installation:

```bash
python -m pip install -e .
```

## Quick start

Run the default Marsicano Voellmy configuration:

```bash
python run_simulation.py --config configs/marsicano_voellmy.yaml --backend auto
```

Explicit CPU or CUDA execution:

```bash
python run_simulation.py --config configs/marsicano_voellmy.yaml --backend cpu
python run_simulation.py --config configs/marsicano_voellmy.yaml --backend cuda
```

Inspect the CUDA runtime/device configuration:

```bash
python run_simulation.py --cuda-info
```

Run the short concentration-aware O'Brien–Julien integration case:

```bash
python run_simulation.py --config configs/obrien_julien_quick.yaml --backend auto
```

## Full publication suite

The complete reproducibility workflow is driven by `run_full_suite.py`.

```bash
python run_full_suite.py --backend cuda --fresh
```

`--fresh` removes previous generated outputs and DEM caches before starting. The suite uses a fresh temporary working directory for expensive intermediate simulations and copies publication-facing products back into `outputs/`.

The full workflow contains 12 stages:

1. reviewer/GCI regression checks;
2. CFL-driver regression checks;
3. concentration-aware O'Brien–Julien regression checks;
4. Windows redirected-console compatibility;
5. dense concentration sweep;
6. controlled numerical verification and Ritter accuracy/cost products;
7. 16-case Marsicano Voellmy multiphysics ablation matrix;
8. Marsicano real-topography grid/GCI/bottleneck study;
9. reviewer-requested physical and parameter diagnostics;
10. native-5 m CFL robustness study;
11. short O'Brien–Julien real-DEM end-to-end integration check;
12. matched CPU/CUDA performance benchmark.

Useful alternatives:

```bash
# Print all commands without running simulations
python run_full_suite.py --backend cuda --fresh --dry-run

# Short software-wiring check; not publication evidence
python run_full_suite.py --backend cuda --fresh --smoke --skip-performance

# Full scientific suite without hardware benchmarking
python run_full_suite.py --backend cuda --fresh --skip-performance
```

At completion, inspect the products written under `outputs/`, including the suite summary, artifact index, logs, tables, and publication figures.

## Numerical verification

Run the controlled verification suite with:

```bash
python verify_core.py --backend auto
```

The suite covers the principal numerical building blocks and repository configurations, including HLL/HLLC behavior, balance and wet/dry handling, transport/reconstruction behavior, Ritter dam-break accuracy/cost diagnostics, and relevant conservation checks.

## Marsicano studies

### Multiphysics ablation

Run the full first-/second-order multiphysics matrix:

```bash
python run_marsicano_studies.py --backend cuda
```

Selected examples:

```bash
python run_marsicano_studies.py --dry-run
python run_marsicano_studies.py --orders O2
python run_marsicano_studies.py --cases P000 EDS111
python run_marsicano_studies.py --restart
python run_marsicano_studies.py --no-resume
```

The eight physical combinations are:

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

Each physical case can be evaluated with first- and second-order numerics.

### Real-topography grid sensitivity

Run only the Marsicano grid-sensitivity experiment:

```bash
python run_marsicano_studies.py --grid-sensitivity-only --backend cuda
```

The default terrain-informed sequence is:

```text
20 m -> 15 m -> 10 m -> 8 m -> 6.5 m -> 5 m
```

A short smoke test can be executed with:

```bash
python run_marsicano_studies.py --grid-sensitivity-only --t-end 2 --backend cuda
```

For publication/reviewer results, use the full configured final time rather than the smoke-test setting.

## Reviewer-oriented diagnostics

Synthetic and parameter-sensitivity diagnostics can be run with:

```bash
python run_reviewer_diagnostics.py --all --backend cuda
```

The hydrostatic/curvature diagnostic additionally requires a solver `final_state.npz` supplied through `--hydrostatic-state`.

## CFL robustness

Run the native-5 m temporal CFL study with:

```bash
python run_cfl_study.py --backend cuda
```

The default tested CFL values are `0.25`, `0.15`, `0.10`, and `0.075` at a 5 m grid spacing and 275 s final time.

## CPU/CUDA benchmark

Run the standalone matched hardware benchmark with:

```bash
python tests/test_cpu_cuda_benchmark.py --dx 20 10 5 --t-end 20 --order 2
```

The benchmark compares matched solver configurations and reports wall time, timing per accepted step, speedup, and implementation-level memory estimates. CUDA-to-CPU fallback is disabled during GPU timing so that the reported GPU measurements correspond to actual CUDA execution.

## Fast regression checks

The following checks do not rerun the complete publication suite:

```bash
python tests/test_reviewer_regression.py
python tests/test_concentration_rheology.py
python tests/test_concentration_sweep.py
python tests/test_cfl_regression.py
python tests/test_windows_console_compat.py
```

The O'Brien–Julien end-to-end test executes the real solver over the Marsicano DEM and is therefore heavier:

```bash
python tests/test_oj_end_to_end.py --backend cuda --output-root outputs/oj_check
```

## Configuration overview

The YAML files expose the main model and numerical controls, including:

- `compute`: backend, CPU threading, CUDA device/block settings, fallback policy;
- `grid`: DEM source, cache, target spacing, clipping, and valid-cell threshold;
- `numerics`: gravity, CFL, maximum time step, final time, numerical order, limiter, flux, boundaries, positivity, and fallback controls;
- `material`: intrinsic densities, initial composition, resistance parameters, and grain properties;
- resistance/rheology selection;
- `release`: release geometry, volume, composition, and initial velocity;
- `erosion`, `deposition`, `segregation`: multiphysics source-term parameters;
- `output`: snapshots, PNG/GIF generation, progress logging, hillshade, and setup previews.

## Reproducible scientific use

For every published or shared simulation, retain at least:

- the exact PyDebrisFlow2D release or commit;
- the complete input and resolved YAML configurations;
- the original/prepared DEM and its coordinate reference system;
- release and resistance geometries;
- initial and boundary conditions;
- rheological and multiphysics parameters;
- CPU/CUDA backend and device information;
- Python and dependency versions;
- relevant random seeds;
- conservation and material-budget diagnostics;
- mesh and time-step sensitivity results;
- numerical verification results;
- any local source-code modifications.

Numerical stability or successful completion of the verification suite does not by itself establish field predictive validity. Marsicano products distributed or reproduced by this repository should be described as numerical benchmark/back-analysis products unless independent field observations are used for validation.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

When using PyDebrisFlow2D in scientific work:

1. cite the exact software release or commit used in the analysis;
2. archive the associated configuration and input data when possible;
3. cite the associated peer-reviewed scientific article when available.

The release metadata in `CITATION.cff` should remain synchronized with `pydebrisflow.__version__`.

## License

PyDebrisFlow2D is distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for the complete terms.

## Disclaimer

PyDebrisFlow2D is provided for scientific research and education. Simulation results depend on model assumptions, input data, DEM quality, release conditions, rheological and source-term parameters, numerical resolution, software dependencies, and hardware. Outputs must be independently reviewed and validated before use in engineering design, hazard assessment, planning, emergency management, regulatory procedures, or any decision affecting people, property, infrastructure, or the environment.
