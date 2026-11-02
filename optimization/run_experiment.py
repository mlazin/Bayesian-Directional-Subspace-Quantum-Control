#!/usr/bin/env python3
"""
Experiment runner for the quantum-control optimization benchmarks.

Methods
-------
- line-search : normalized-gradient SciPy line-search baseline
- bo-dso      : Bayesian optimization with directional subspace optimization
- saasbo-dso  : SAAS Bayesian optimization with directional subspace optimization

Examples
--------
Run Hexagonal 1 -> 2, seed index 0:

    python optimization/run_experiment.py \
        --method line-search --seed-index 0 --problem hex --psi-init 1 --psi-target 2

    python optimization/run_experiment.py \
        --method bo-dso --seed-index 0 --problem hex --psi-init 1 --psi-target 2

    python optimization/run_experiment.py \
        --method saasbo-dso --seed-index 0 --problem hex --psi-init 1 --psi-target 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Respect the CPU allocation when running under SLURM.  Outside SLURM, leave
# the user's existing thread settings unchanged.
_slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if _slurm_cpus:
    os.environ["OMP_NUM_THREADS"] = _slurm_cpus
    os.environ["MKL_NUM_THREADS"] = _slurm_cpus
    os.environ["OPENBLAS_NUM_THREADS"] = _slurm_cpus
    os.environ["NUMEXPR_NUM_THREADS"] = _slurm_cpus

import numpy as np
from scipy.io import savemat

from config.paths import OUTPUT_DIR, SFILES

from simulation.harness import harness
from simulation.qfun import Sqfun
from simulation.qsimnp import SQsim

from .opts import (
    DirectionalAlphaWrapper,
    DirectionalAlphaWrapper1D,
    scipyopt_linesearch,
)
from .BOopt import BayesOpt_wrapper
from .saasopt import saasbomodigDir


# ---------------------------------------------------------------------------
# Manuscript experiment configuration
# ---------------------------------------------------------------------------

SEEDS = [5304, 4160, 2598, 7027, 6471, 8653, 4776, 1135, 3525, 9814]

# DSO coordinate ranges used for the geometries.
ALPHA_BOUNDS = {
    "hex": (-0.001, 0.001),
    "triG": (-0.05, 0.05),
    "triN": (-0.05, 0.05),
}

# Bounds on the physical time-domain control field passed to Sqfun.
# These are distinct from the DSO alpha-coordinate bounds above.
FIELD_BOUNDS = np.array([0.001, 0.001])

LINE_SEARCH_BUDGET = 200

BO_DSO_BUDGET = 100
BO_DSO_INNER_ITERS = 10
BO_DSO_MODEL_STEPS = 1000

SAASBO_DSO_BUDGET = 100
SAASBO_DSO_INNER_ITERS = 10
SAASBO_NUM_INIT_EVALS = 4
SAASBO_NUM_SAMPLES = 256

# Historical values retained because the original BO/SAASBO scripts generated
# these random arrays before optimization.  Although the arrays were not used
# as the zero-field anchor, the draws advance NumPy's RNG state and therefore
# may affect later randomized optimizer behavior.
HISTORICAL_NPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one quantum-control optimization experiment."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("line-search", "bo-dso", "saasbo-dso"),
        help="Optimization method to run.",
    )
    parser.add_argument(
        "--seed-index",
        required=True,
        type=int,
        choices=range(10),
        metavar="0-9",
        help="Index into the 10 manuscript random seeds.",
    )
    parser.add_argument(
        "--problem",
        required=True,
        choices=("hex", "triG", "triN"),
        help="Nanowire geometry.",
    )
    parser.add_argument("--psi-init", required=True, type=int, help="Initial state index.")
    parser.add_argument("--psi-target", required=True, type=int, help="Target state index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for experiment outputs (default: paths.OUTPUT_DIR).",
    )
    return parser.parse_args()


def make_problem(problem: str, psi_init: int, psi_target: int):
    """Construct the simulator objective and its logging harness."""
    if problem not in SFILES:
        raise ValueError(
            f"Unknown problem {problem!r}. Available entries in SFILES: {list(SFILES)}"
        )

    qfunconfig = {
        "sfile": str(SFILES[problem]),
        "psi_init": psi_init,
        "psi_target": psi_target,
    }

    # Build once to obtain the experiment-specific time grid/dimension.
    q = SQsim(**qfunconfig)
    _, fld = q.startup()

    rawfn = Sqfun(
        **qfunconfig,
        bound=FIELD_BOUNDS,
        device="cpu",
        flatten_grad=True,
    )
    wrapped = harness(rawfn)
    bounds = wrapped.bounds()

    return qfunconfig, fld, wrapped, bounds


def legacy_rng_advance(seed_index: int, bounds: np.ndarray, nsteps: int) -> None:
    """
    Reproduce the NumPy RNG draws made by the original BO/SAASBO scripts.

    The generated arrays were not used as the final zero-field anchor, but
    retaining these draws helps preserve the subsequent random-number state.
    """
    np.random.seed(SEEDS[seed_index])

    ndim = bounds[0].shape[-1]
    xinit = np.random.randn(HISTORICAL_NPTS, ndim)
    xinit /= np.linalg.norm(xinit, axis=1, keepdims=True)
    xinit *= np.random.rand(HISTORICAL_NPTS, 1) ** (1 / ndim) * bounds[1]

    # Historical alpha_init draw from the original scripts.
    _alpha_init = np.random.rand(2 * nsteps)

    # Keep references alive until the function exits and make intent explicit.
    _ = xinit, _alpha_init


def best_so_far_probabilities(fs) -> list[float]:
    """Convert objective values f=-P into a best-so-far probability trace."""
    ys = -np.asarray(fs, dtype=float)
    best = 0.0
    trace = []
    for y in ys:
        if y > best:
            best = float(y)
        trace.append(best)
    return trace


def save_directional_history(
    savefileplot: Path,
    opt,
    f: harness,
    *,
    include_bestx_key: bool = False,
) -> None:
    """Save DSO history in a consistent format."""
    X_opt, Y_opt, G_opt, D_hist_list, expand_iters = opt.get_history()

    if len(D_hist_list) > 0:
        D_hist_all = np.asarray(D_hist_list[-1], float)
    else:
        width = X_opt.shape[1] if getattr(X_opt, "size", 0) else 0
        D_hist_all = np.empty((0, width))

    D_all = np.asarray(opt.D, float)

    alpha_eval = getattr(opt, "alpha_eval_log", [])
    alpha_eval_vals = getattr(opt, "alpha_eval_vals", [])

    X_f = np.asarray(f.xs, float)
    P_f = -np.asarray(f.fs, float).ravel()

    best_idx_f = int(np.argmin(f.fs))
    best_x_f = np.asarray(f.xs[best_idx_f], float)

    mat_data = {
        "bestX_f": best_x_f,
        "X_all_f": X_f,
        "Y_vals_f": P_f,
        "X_all_opt": np.asarray(X_opt, float),
        "Y_vals_opt": -np.asarray(Y_opt, float),
        "D_all": D_all,
        "expand_iters": np.asarray(expand_iters, int),
        "G_all": np.asarray(G_opt, dtype=object),
    }
    if include_bestx_key:
        # Preserve the key used by the original SAASBO experiment output.
        mat_data["bestX"] = best_x_f

    savemat(savefileplot, mat_data, do_compression=False)

    stem = savefileplot.stem
    np.save(savefileplot.with_name(stem + "_D_hist_all.npy"), D_hist_all)
    np.save(savefileplot.with_name(stem + "_D_all.npy"), D_all)

    alpha_eval_obj = np.array(
        [np.asarray(a, float) for a in alpha_eval],
        dtype=object,
    )
    np.save(
        savefileplot.with_name(stem + "_alpha_eval.npy"),
        alpha_eval_obj,
        allow_pickle=True,
    )
    np.save(
        savefileplot.with_name(stem + "_alpha_eval_vals.npy"),
        np.asarray(alpha_eval_vals, float),
        allow_pickle=True,
    )

    bundle = {
        "D_hist_all": D_hist_all,
        "D_all": D_all,
        "alpha_eval": np.array(alpha_eval, dtype=object),
        "alpha_eval_vals": np.asarray(alpha_eval_vals, float),
        "X_all_f": X_f,
        "Y_vals_f": P_f,
        "X_all_opt": np.asarray(X_opt, float),
        "Y_vals_opt": -np.asarray(Y_opt, float),
        "G_all": np.asarray(G_opt, dtype=object),
        "expand_iters": np.asarray(expand_iters, int),
    }
    np.save(
        savefileplot.with_name(stem + "_bundle.npy"),
        bundle,
        allow_pickle=True,
    )


def run_line_search(
    *,
    seed_index: int,
    problem: str,
    psi_init: int,
    psi_target: int,
    output_dir: Path,
) -> None:
    """Run the manuscript line-search baseline."""
    label = f"{problem}_{psi_init}to{psi_target}"
    savefile = output_dir / (
        f"{label}_linesearchtimedomain_{LINE_SEARCH_BUDGET}iter_{seed_index + 1}.txt"
    )
    savefileplot = output_dir / (
        f"{label}_linesearchtimedomainplot{LINE_SEARCH_BUDGET}iter_{seed_index + 1}.mat"
    )

    _, _, f, bounds = make_problem(problem, psi_init, psi_target)

    # The original line-search experiment set the manuscript seed even though
    # the actual starting field was deterministic (zero).
    np.random.seed(SEEDS[seed_index])
    ndim = bounds[0].shape[-1]
    xinit = np.zeros((HISTORICAL_NPTS, ndim))

    optimizer = scipyopt_linesearch(
        bounds=bounds,
        options={
            "maxfun": LINE_SEARCH_BUDGET,
            "normalize_dirs": True,
        },
    )
    x_final, f_final = optimizer.run(
        f,
        xinit,
        maxitt=LINE_SEARCH_BUDGET,
    )

    fvals_ls = [entry["f"] for entry in optimizer.gds]
    grads_ls = [entry["grad_norm"] for entry in optimizer.gds]
    dirs_ls = np.asarray(optimizer.dirs, float)
    dirs_ls_unit = dirs_ls / np.linalg.norm(dirs_ls, axis=1, keepdims=True)

    best_idx = int(np.argmin(f.fs))
    # Original line-search output saved the 2 x Nt field rather than flattened x.
    best_field = np.asarray(f.Es[best_idx])

    savemat(
        savefileplot,
        {
            "bestX": best_field,
            "fvals_ls": fvals_ls,
            "grads_ls": grads_ls,
            "dirs_ls": dirs_ls,
            "dirs_ls_unit": dirs_ls_unit,
            "x_ls_final": x_final,
            "f_ls_final": f_final,
        },
    )

    probabilities = (-np.asarray(f.fs, float)).tolist()

    # Keep the exact labels consumed by generate_convergence_plots.py.
    with open(savefile, "w", encoding="utf-8") as output_file:
        output_file.write("\nBest P:" + str(f.bestPs))
        output_file.write("\nY:" + str(probabilities))
        output_file.write("\nEval Time:" + str(f.evalt))
        output_file.write("\nTotal Time:" + str(f.t))

    print(f"Best P: {-min(f.fs):.12g}")
    print(f"Saved: {savefile}")
    print(f"Saved: {savefileplot}")


def run_bo_dso(
    *,
    seed_index: int,
    problem: str,
    psi_init: int,
    psi_target: int,
    output_dir: Path,
) -> None:
    """Run the BO-DSO experiment using the original BO-DSO wrapper path."""
    label = f"{problem}_{psi_init}to{psi_target}"
    usegrad = True

    savefile = output_dir / (
        f"{label}_bodaw_{BO_DSO_BUDGET}iter_"
        f"ittexp{BO_DSO_INNER_ITERS}_mloop{BO_DSO_MODEL_STEPS}_"
        f"grad{usegrad}_{seed_index + 1}.txt"
    )
    savefileplot = output_dir / (
        f"{label}_bodawplot{BO_DSO_BUDGET}iter_"
        f"ittexp{BO_DSO_INNER_ITERS}_mloop{BO_DSO_MODEL_STEPS}_"
        f"grad{usegrad}_{seed_index + 1}.mat"
    )

    _, fld, f, bounds = make_problem(problem, psi_init, psi_target)

    nsteps = int(len(fld["tgrid"]))
    legacy_rng_advance(seed_index, bounds, nsteps)

    baseopt = BayesOpt_wrapper(
        bounds=bounds,
        mloop=BO_DSO_MODEL_STEPS,
        grad=usegrad,
        maxiter=BO_DSO_INNER_ITERS,
    )
    optimizer = DirectionalAlphaWrapper1D(
        baseopt=baseopt,
        sim=f,
        alpha_bounds=ALPHA_BOUNDS[problem],
        max_outer_iters=BO_DSO_BUDGET,
    )

    ndim = bounds[0].shape[-1]
    optimizer.set_anchor(np.zeros(ndim))
    optimizer.run1d(f=f, xinit=None, iter=BO_DSO_BUDGET)

    save_directional_history(savefileplot, optimizer, f)

    best_ps = best_so_far_probabilities(f.fs)
    probabilities = (-np.asarray(f.fs, float)).tolist()

    with open(savefile, "w", encoding="utf-8") as output_file:
        output_file.write("\nbest found:" + str(-min(f.fs)))
        output_file.write("\nExpanditercount:" + str(optimizer.expanditercount))
        output_file.write("\nbest Ps:" + str(best_ps))
        output_file.write("\nY-vals f:" + str(probabilities))
        output_file.write("\neval time:" + str(f.evalt))
        output_file.write("\ntotal time:" + str(f.t))
        output_file.write(
            "\nalpha_eval:"
            + str([a.tolist() for a in getattr(optimizer, "alpha_eval_log", [])])
        )
        output_file.write(
            "\nalpha_eval_vals:"
            + str(getattr(optimizer, "alpha_eval_vals", []))
        )
        output_file.write(
            "\nused grad? " + str([g is not None for g in f.gs])
        )

    print(f"Best P: {-min(f.fs):.12g}")
    print(f"Final DSO dimension: {optimizer.D.shape[1]}")
    print(f"Alpha bounds: {ALPHA_BOUNDS[problem]}")
    print(f"Saved: {savefile}")
    print(f"Saved: {savefileplot}")


def run_saasbo_dso(
    *,
    seed_index: int,
    problem: str,
    psi_init: int,
    psi_target: int,
    output_dir: Path,
) -> None:
    """Run the SAASBO-DSO experiment using the original SAASBO wrapper path."""
    label = f"{problem}_{psi_init}to{psi_target}"
    usegrad = True

    savefile = output_dir / (
        f"{label}_saasbononaxisaligned_{SAASBO_DSO_BUDGET}iter_"
        f"ittexp{SAASBO_DSO_INNER_ITERS}_{seed_index + 1}.txt"
    )
    savefileplot = output_dir / (
        f"{label}_saasbononaxisalignedplot{SAASBO_DSO_BUDGET}iter_"
        f"ittexp{SAASBO_DSO_INNER_ITERS}_{seed_index + 1}.mat"
    )

    _, fld, f, bounds = make_problem(problem, psi_init, psi_target)

    nsteps = int(len(fld["tgrid"]))
    legacy_rng_advance(seed_index, bounds, nsteps)

    baseopt = saasbomodigDir(
        bounds=bounds,
        num_init_evals=SAASBO_NUM_INIT_EVALS,
        seed=None,
        maxiter=SAASBO_DSO_INNER_ITERS,
        grad=usegrad,
        gradcomb=usegrad,
        num_samples=SAASBO_NUM_SAMPLES,
    )
    optimizer = DirectionalAlphaWrapper(
        baseopt=baseopt,
        sim=f,
        alpha_bounds=ALPHA_BOUNDS[problem],
        max_outer_iters=SAASBO_DSO_BUDGET,
    )

    ndim = bounds[0].shape[-1]
    optimizer.set_anchor(np.zeros(ndim))
    optimizer.run(f=f, xinit=None, iter=SAASBO_DSO_BUDGET)

    save_directional_history(
        savefileplot,
        optimizer,
        f,
        include_bestx_key=True,
    )

    best_ps = best_so_far_probabilities(f.fs)
    probabilities = (-np.asarray(f.fs, float)).tolist()

    with open(savefile, "w", encoding="utf-8") as output_file:
        output_file.write("\nbest found:" + str(-min(f.fs)))
        output_file.write("\nnum evals:" + str(len(f.xs)))
        output_file.write("\nExpanditercount:" + str(optimizer.expanditercount))
        output_file.write("\nbest Ps:" + str(best_ps))
        output_file.write("\nY-vals:" + str(probabilities))
        output_file.write("\neval time:" + str(f.evalt))
        output_file.write("\ntotal time:" + str(f.t))
        output_file.write(
            "\nalpha_eval:"
            + str([a.tolist() for a in getattr(optimizer, "alpha_eval_log", [])])
        )
        output_file.write(
            "\nalpha_eval_vals:"
            + str(getattr(optimizer, "alpha_eval_vals", []))
        )
        output_file.write(
            "\nused grad?" + str([g is not None for g in f.gs])
        )

    print(f"Best P: {-min(f.fs):.12g}")
    print(f"Final DSO dimension: {optimizer.D.shape[1]}")
    print(f"Alpha bounds: {ALPHA_BOUNDS[problem]}")
    print(f"Saved: {savefile}")
    print(f"Saved: {savefileplot}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Quantum-control experiment")
    print("=" * 72)
    print(f"method      : {args.method}")
    print(f"problem     : {args.problem}")
    print(f"transition  : {args.psi_init} -> {args.psi_target}")
    print(f"seed index  : {args.seed_index}")
    print(f"seed        : {SEEDS[args.seed_index]}")
    if args.method != "line-search":
        print(f"alpha bounds: {ALPHA_BOUNDS[args.problem]}")
    print(f"output dir  : {args.output_dir}")
    print("=" * 72)

    common = dict(
        seed_index=args.seed_index,
        problem=args.problem,
        psi_init=args.psi_init,
        psi_target=args.psi_target,
        output_dir=args.output_dir,
    )

    if args.method == "line-search":
        run_line_search(**common)
    elif args.method == "bo-dso":
        run_bo_dso(**common)
    elif args.method == "saasbo-dso":
        run_saasbo_dso(**common)
    else:  # argparse choices should make this unreachable.
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
