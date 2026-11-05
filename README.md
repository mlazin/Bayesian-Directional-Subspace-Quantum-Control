# Bayesian Directional Subspace Optimization for Quantum Control

Code and data supporting the manuscript:

**“Bayesian Optimization for Constructing Quantum Control Fields in Optically-Excited Heterostructure Nanomaterials”**

This repository contains the quantum-control simulation code, optimization methods, stored convergence data, and plotting scripts used for the results reported in the manuscript.

The optimization problem is formulated in terms of a two-component time-dependent electric field,

$$
\mathbf{E}(t) = \left(E_x(t), E_y(t)\right),
$$

with the objective of maximizing the state-transition probability

$$
P = \left|\langle \psi_f \mid \psi(T) \rangle\right|^2.
$$

The optimization routines minimize $f=-P$.

The three optimization approaches considered in the manuscript are:

- **Line Search**
- **BO-DSO**: Bayesian Optimization with Directional Subspace Optimization
- **SAASBO-DSO**: SAAS Bayesian Optimization with Directional Subspace Optimization

## Repository structure

```text
Bayesian-Directional-Subspace-Quantum-Control/
├── config/
│   ├── __init__.py
│   └── paths.py
│
├── data/
│   ├── convergence_logs/
│   └── system_files/
│
├── figure_scripts/
│   └── generate_convergence_plots.py
│
├── figures/
│   └── convergence/
│
├── optimization/
│   ├── __init__.py
│   ├── run_experiment.py
│   ├── BOopt.py
│   ├── EIacquisition.py
│   ├── GPyGP.py
│   ├── opts.py
│   ├── saasopt.py
│   ├── saasgp.py
│   └── saasboecq.py
│
├── simulation/
│   ├── __init__.py
│   ├── harness.py
│   ├── qfun.py
│   └── qsimnp.py
│
└── README.md
```

### `simulation/`

Contains the quantum-control simulator and the objective-function interface.

- `qsimnp.py` — quantum dynamics and transition-probability calculations.
- `qfun.py` — objective-function wrapper used by the optimizers.
- `harness.py` — records evaluated fields, objective values, gradients, and timing information.

### `optimization/`

Contains the optimization methods used in the manuscript.

- `run_experiment.py` — unified command-line interface for running the three optimization methods.
- `opts.py` — directional-subspace wrappers and line-search implementation.
- `BOopt.py` — Gaussian-process Bayesian optimization loop.
- `GPyGP.py` — gradient-enhanced and standard Gaussian-process models.
- `EIacquisition.py` — expected-improvement acquisition optimization.
- `saasopt.py` — SAASBO optimization interface.
- `saasgp.py` — SAAS Gaussian-process model.
- `saasboecq.py` — expected-improvement optimization used by SAASBO.

### `data/system_files/`

Contains the system-specific input files used by the quantum-control simulator.

The mapping between geometry names and system files is defined in:

```text
config/paths.py
```

Alternative system-file locations can also be specified through the environment-variable overrides defined in that file.

### `data/convergence_logs/`

Contains the stored optimization logs used to construct the convergence plots reported in the manuscript.

### `figure_scripts/`

Contains scripts that process the stored optimization logs and reproduce manuscript figures.

`generate_convergence_plots.py` reproduces the convergence plots shown in **Figures 3, 6, and 9** of the manuscript.

## Software requirements

The experiments reported in the manuscript were performed using **Python 3.11.11**.

The principal software packages used were:

```text id="rxjrto"
numpy          1.26.4
scipy          1.13.1
matplotlib     3.10.0
torch          2.2.0+cu121
gpytorch       1.13
jax            0.4.35
jaxlib         0.4.35
numpyro        0.13.2
scikit-sparse  0.4.14
SuiteSparse    5.10.1
```

The quantum-control simulator uses the CHOLMOD interface provided by `scikit-sparse`, which requires SuiteSparse/CHOLMOD to be available on the system.

The PyTorch installation used for the reported experiments was built with CUDA 12.1 support (`torch 2.2.0+cu121`); however, GPU acceleration is not required for the provided optimization workflow, which can be executed on CPU.

These version numbers document the software environment used for the reported calculations. Other compatible package versions may also work, but the versions above correspond to the environment used for the manuscript results.

## Running an optimization experiment

Run experiments from the **repository root** using:

```bash
python -m optimization.run_experiment \
    --method METHOD \
    --seed-index SEED_INDEX \
    --problem PROBLEM \
    --psi-init INITIAL_STATE \
    --psi-target TARGET_STATE
```

Available methods are:

```text
line-search
bo-dso
saasbo-dso
```

Available geometry identifiers are:

```text
hex
triG
triN
```

For example, to run BO-DSO for the $1 \rightarrow 2$ transition in the hexagonal geometry using the first random seed:

```bash
python -m optimization.run_experiment \
    --method bo-dso \
    --seed-index 0 \
    --problem hex \
    --psi-init 1 \
    --psi-target 2
```

The equivalent SAASBO-DSO experiment is:

```bash
python -m optimization.run_experiment \
    --method saasbo-dso \
    --seed-index 0 \
    --problem hex \
    --psi-init 1 \
    --psi-target 2
```

The line-search baseline can be run using:

```bash
python -m optimization.run_experiment \
    --method line-search \
    --seed-index 0 \
    --problem hex \
    --psi-init 1 \
    --psi-target 2
```

Generated experiment outputs are written to the output directory configured by the repository unless an alternative directory is supplied to `run_experiment.py`.

## Random seeds

The ten random seeds used for the reported experiments are:

```text
5304
4160
2598
7027
6471
8653
4776
1135
3525
9814
```

The command-line argument `--seed-index` selects an entry from this list, with indices ranging from `0` to `9`.

## Optimization settings

The main settings used for the reported experiments are encoded directly in `optimization/run_experiment.py`.

### Line Search

- Evaluation budget: 200
- Initial field: zero field
- Search direction: normalized negative gradient
- SciPy line search with Wolfe conditions
- Fallback step size: $10^{-2}$

### BO-DSO

- Evaluation budget: 100
- Inner optimization length: 10
- Gradient-enhanced Gaussian process
- Matérn-5/2 covariance model
- Expected improvement acquisition function
- GP model-training parameter: `mloop = 1000`

### SAASBO-DSO

- Evaluation budget: 100
- Inner optimization length: 10
- Initial evaluations: 4
- HMC warmup samples: 512
- Posterior samples: 256
- Gradient-enhanced SAAS Gaussian process
- Expected improvement acquisition function

## Directional-subspace bounds

The directional-coordinate bounds used in the experiments are geometry dependent:

```text
hex   : [-1e-3, 1e-3]
triG  : [-5e-2, 5e-2]
triN  : [-5e-2, 5e-2]
```

These are directional-subspace coordinate bounds and are distinct from the physical electric-field bounds used by the simulator.

## Reproducing the convergence figures

The stored convergence logs are provided in:

```text
data/convergence_logs/
```

From the repository root, run:

```bash
python figure_scripts/generate_convergence_plots.py
```

The script processes the stored results for the ten random seeds and generates the convergence plots corresponding to:

- **Figure 3** — Hexagonal geometry
- **Figure 6** — Triangular Ga-face geometry
- **Figure 9** — Triangular N-face geometry

Generated figures are written to:

```text
figures/convergence/
```

The plotted curves show the mean best-so-far transition probability across the ten runs. Shaded regions correspond to the sample standard deviation across runs.

## Reproducibility note

The optimization implementations in this repository retain the computational behavior used to generate the reported results. 

The stored convergence logs can be used to reproduce the manuscript convergence figures without rerunning the computationally expensive optimization experiments.

## Data availability

The data and code supporting the findings of this study are provided in this repository. The repository includes the quantum-control simulation code, optimization implementations, system input files, stored optimization results used for the convergence analysis, and scripts for processing the data and reproducing the corresponding figures.
