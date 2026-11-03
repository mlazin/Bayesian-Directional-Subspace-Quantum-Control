from pathlib import Path
import os


# Repository root:
# Bayesian-Directional-Subspace-Quantum-Control/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Default locations inside the repository
DATA_DIR = REPO_ROOT / "data"
SYSTEM_DIR = DATA_DIR / "system_files"
OUTPUT_DIR = DATA_DIR / "experiment_outputs"

# Create the output directory automatically if it does not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Quantum-system input files.
#
# Environment variables can override these locations when running on HPCC
# or another machine without modifying the source code.
SFILES = {
    "hex": Path(
        os.environ.get(
            "QCONTROL_HEX_SFILE",
            SYSTEM_DIR / "geom_hex.mat",
        )
    ),
    "triG": Path(
        os.environ.get(
            "QCONTROL_TRIG_SFILE",
            SYSTEM_DIR / "geom_triG.mat",
        )
    ),
    "triN": Path(
        os.environ.get(
            "QCONTROL_TRIN_SFILE",
            SYSTEM_DIR / "geom_triN.mat",
        )
    ),
}
