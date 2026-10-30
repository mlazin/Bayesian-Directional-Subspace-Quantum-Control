#!/usr/bin/env python3
"""
Reproduce the manuscript convergence plots.

This script reads the stored optimization logs for 10 independent runs of
Line Search, BO-DSO, and SAASBO-DSO and generates the grouped convergence
figures used in the manuscript.

Statistics shown:
    - solid curve: mean best-so-far transition probability across runs
    - shaded band: ±1 sample standard deviation across runs
    - x-axis: cumulative simulation time

Expected repository layout:
    data/
        convergence_logs/
            <log files>
    figure_scripts/
        generate_convergence_plots.py
    figures/
        convergence/

The script can be run from the repository root with:

    python figure_scripts/generate_convergence_plots.py

Optional custom paths:
    python figure_scripts/generate_convergence_plots.py \
        --data-dir path/to/logs \
        --output-dir path/to/output

Update this header with the final manuscript figure numbers before release.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np

# PGF/LaTeX configuration used for manuscript-ready figures.
mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "savefig.bbox": "tight",
    }
)
plt.ioff()


N_RUNS = 10
MAX_POINTS = {
    "Line Search": 100,
    "BO-DSO": 100,
    "SAASBO-DSO": 100,
}

COLORS = {
    "Line Search": "#D55E00",
    "BO-DSO": "#0072B2",
    "SAASBO-DSO": "#009E73",
}

MARKERS = {
    "Line Search": "^",
    "BO-DSO": "o",
    "SAASBO-DSO": "s",
}

LINESTYLES = {
    "Line Search": "-",
    "BO-DSO": "-",
    "SAASBO-DSO": "-",
}

# Prefixes correspond to the stored log-file names.
GROUPS = {
    "Hex": [
        ("hex_1to2", "1", "2"),
        ("hex_4to6", "4", "6"),
    ],
    "TriG": [
        ("triG_1to2", "1", "2"),
        ("triG_2to3", "2", "3"),
        ("triG_3to4", "3", "4"),
    ],
    "TriN": [
        ("triN_1to2", "1", "2"),
        ("triN_1to3", "1", "3"),
        ("triN_2to4", "2", "4"),
        ("triN_4to6", "4", "6"),
        ("triN_9to10", "9", "10"),
    ],
}

METHOD_FILES = {
    "Line Search": {
        "filename": "{prefix}_linesearchtimedomain_200iter_{run}.txt",
        "value_marker": "Best P:",
        "time_marker": "Eval Time:",
    },
    "BO-DSO": {
        "filename": "{prefix}_bodaw_100iter_ittexp10_mloop1000_gradTrue_{run}.txt",
        "value_marker": "best Ps:",
        "time_marker": "eval time:",
    },
    "SAASBO-DSO": {
        "filename": "{prefix}_saasbononaxisaligned_100iter_ittexp10_{run}.txt",
        "value_marker": "best Ps:",
        "time_marker": "eval time:",
    },
}

LOG_LABELS = (
    "best found:",
    "best Ps:",
    "y-vals:",
    "eval time:",
    "total time:",
    "Best P:",
    "Eval Time:",
)


def repository_root() -> Path:
    """Infer the repository root from this script's location."""
    return Path(__file__).resolve().parents[1]


def read_array_from_log(path: Path, marker: str) -> np.ndarray:
    """
    Return the array from the last line in *path* containing *marker*.

    This preserves the behavior of the original notebook while adding explicit
    errors when a required marker or file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required log file not found: {path}")

    matched_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                matched_line = line.rstrip()

    if matched_line is None:
        raise ValueError(f"Marker {marker!r} not found in {path}")

    cleaned = matched_line
    for label in LOG_LABELS:
        cleaned = cleaned.replace(label, " ")
    cleaned = " ".join(cleaned.split())

    try:
        values = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse array from {path}: {cleaned}") from exc

    return np.asarray(values)


def mean_and_sd(value_runs: list[np.ndarray], time_runs: list[np.ndarray]):
    """
    Truncate runs to their common evaluation length and compute:
      - mean best-so-far transition probability,
      - mean cumulative simulation time,
      - sample standard deviation across runs.
    """
    min_len = min(
        min(len(values) for values in value_runs),
        min(len(times) for times in time_runs),
    )

    values = np.asarray([run[:min_len] for run in value_runs])
    times = np.asarray([run[:min_len] for run in time_runs])

    mean_values = np.mean(values, axis=0)
    mean_times = np.mean(times, axis=0)
    sd_values = np.std(values, axis=0, ddof=1)

    return mean_times, mean_values, sd_values


def load_transition(data_dir: Path, prefix: str):
    """Load and summarize all three optimization methods for one transition."""
    summary = {}

    for method, spec in METHOD_FILES.items():
        value_runs = []
        time_runs = []

        for run in range(1, N_RUNS + 1):
            path = data_dir / spec["filename"].format(prefix=prefix, run=run)
            value_runs.append(read_array_from_log(path, spec["value_marker"]))
            time_runs.append(read_array_from_log(path, spec["time_marker"]))

        summary[method] = mean_and_sd(value_runs, time_runs)

    return summary


def truncate_to_time(
    times: np.ndarray,
    values: np.ndarray,
    sd: np.ndarray,
    end_time: float,
):
    """Keep only points at or before the common simulation-time horizon."""
    mask = np.asarray(times) <= end_time
    return np.asarray(times)[mask], np.asarray(values)[mask], np.asarray(sd)[mask]


def plot_transition(ax, summary, initial_state: str, final_state: str):
    """
    Plot one transition using a common cumulative simulation-time horizon
    across Line Search, BO-DSO, and SAASBO-DSO.
    """
    prepared = {}

    # Preserve the original notebook's 100-point cap for each method.
    for method in METHOD_FILES:
        times, values, sd = summary[method]
        n = min(
            MAX_POINTS[method],
            len(times),
            len(values),
            len(sd),
        )
        prepared[method] = (
            np.asarray(times[:n]),
            np.asarray(values[:n]),
            np.asarray(sd[:n]),
        )

    end_time = min(prepared[method][0][-1] for method in METHOD_FILES)

    for method in METHOD_FILES:
        times, values, sd = truncate_to_time(*prepared[method], end_time)
        prepared[method] = (times, values, sd)

        ax.fill_between(
            times,
            np.clip(values - sd, 0.0, 1.0),
            np.clip(values + sd, 0.0, 1.0),
            color=COLORS[method],
            alpha=0.30,
            linewidth=0,
        )

        ax.plot(
            times,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            label=method,
            markevery=max(1, len(times) // 12),
            zorder=3,
        )

    ax.set_xlim(0.0, float(end_time))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(rf"{initial_state} $\rightarrow$ {final_state} transition")
    ax.set_xlabel("Simulation Time (sec)")
    ax.set_ylabel(r"$P^{\ast}$")
    ax.grid(True, alpha=0.3)


def save_legend(output_dir: Path):
    """Save one reusable legend for all convergence-plot groups."""
    fig = plt.figure(figsize=(6.8, 0.45))
    ax = fig.add_subplot(111)
    ax.axis("off")

    handles = []
    labels = list(METHOD_FILES)
    for label in labels:
        handle, = ax.plot(
            [],
            [],
            color=COLORS[label],
            marker=MARKERS[label],
            linestyle=LINESTYLES[label],
            label=label,
        )
        handles.append(handle)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=3,
        frameon=True,
        handlelength=2.0,
        columnspacing=1.5,
    )

    fig.savefig(output_dir / "method_legend.pgf", bbox_inches="tight")
    fig.savefig(output_dir / "method_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_group(group_name: str, transitions, data_dir: Path, output_dir: Path):
    """Generate one grouped manuscript figure."""
    cols = 3
    rows = (len(transitions) + cols - 1) // cols

    fig_width = 6.8
    fig_height_per_row = 2.1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, rows * fig_height_per_row),
    )
    axes = np.asarray(axes).reshape(-1)

    for index, (prefix, initial_state, final_state) in enumerate(transitions):
        summary = load_transition(data_dir, prefix)
        plot_transition(
            axes[index],
            summary,
            initial_state=initial_state,
            final_state=final_state,
        )

    for index in range(len(transitions), len(axes)):
        fig.delaxes(axes[index])

    plt.tight_layout()

    output_base = output_dir / f"{group_name.lower()}_group"
    fig.savefig(str(output_base) + ".pgf", bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def validate_inputs(data_dir: Path):
    """Check that all 300 log files required by the manuscript plots exist."""
    missing = []

    for transitions in GROUPS.values():
        for prefix, _, _ in transitions:
            for method, spec in METHOD_FILES.items():
                for run in range(1, N_RUNS + 1):
                    path = data_dir / spec["filename"].format(prefix=prefix, run=run)
                    if not path.exists():
                        missing.append(path)

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        extra = ""
        if len(missing) > 20:
            extra = f"\n  ... and {len(missing) - 20} more."
        raise FileNotFoundError(
            f"{len(missing)} required log files are missing:\n{preview}{extra}"
        )

    print("Input validation passed: all 300 required log files are present.")


def parse_args():
    root = repository_root()

    parser = argparse.ArgumentParser(
        description="Generate convergence plots for the quantum-control manuscript."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "convergence_logs",
        help="Directory containing the stored optimization text logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "figures" / "convergence",
        help="Directory in which generated PDF/PGF figures are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validate_inputs(args.data_dir)

    for group_name, transitions in GROUPS.items():
        print(f"Generating {group_name} convergence figure...")
        plot_group(group_name, transitions, args.data_dir, args.output_dir)

    save_legend(args.output_dir)
    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
