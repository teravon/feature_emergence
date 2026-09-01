"""Smoke tests: the package must import and resolve its config paths.

These tests only exercise imports and path resolution — running them does not
require tensorflow or the datasets.
"""

from pathlib import Path

from feature_emergence import config


def test_config_paths_point_inside_repo():
    assert config.REPO_ROOT.name == "feature_emergence"
    assert config.DATA_DIR == config.REPO_ROOT / "data"
    assert config.MODELS_DIR == config.REPO_ROOT / "models"
    assert config.OUTPUTS_DIR == config.REPO_ROOT / "outputs"


def test_directories_are_created():
    config.ensure_dirs()
    assert config.DATA_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()
    assert config.OUTPUTS_DIR.is_dir()


def test_package_version():
    import feature_emergence

    assert hasattr(feature_emergence, "__version__")