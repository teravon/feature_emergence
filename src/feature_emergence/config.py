"""Central project path configuration.

All data, model and output artifacts live outside the package in clearly
named directories at the repository root:
    data/    -> raw/preprocessed trace datasets (*.h5)
    models/  -> trained model checkpoints
    outputs/ -> generated figures, tables and analysis artifacts
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ORIGINAL_NOTEBOOKS_DIR = NOTEBOOKS_DIR / "original"


def ensure_dirs() -> None:
    """Create artifact directories if they do not exist yet."""
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)