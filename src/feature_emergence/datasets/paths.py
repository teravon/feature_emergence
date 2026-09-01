"""Resolve dataset file paths.

The optional ``dataset_root_folder`` argument is kept for backward
compatibility with the original notebooks. When omitted, files are looked up
under the project's ``data/`` directory (see :mod:`feature_emergence.config`).
"""

from pathlib import Path

from .config import DATA_DIR


def get_dataset_filepath(dataset_root_folder, dataset_name, npoi, leakage_model):

    root = Path(dataset_root_folder) if dataset_root_folder is not None else DATA_DIR

    dataset_dict = {
        "ascad-variable": {
            1400: f"{root}/ascad-variable.h5",
            2000: f"{root}/ASCADr/ascad-variable_70k-90k_20.h5",
            250000: f"{root}/ASCADr/atmega8515-raw-traces.h5",
        },
        "eshard": {
            1400: f"{root}/eshard.h5",
        },
        "ches_ctf": {
            15000: f"{root}/ches_ctf/ches_ctf_nopoi_window_20.h5"
        },
    }

    return dataset_dict[dataset_name][npoi]