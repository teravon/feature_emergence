#!/bin/bash

# Script to set up the environment for the "You Have To be Realistic" paper experiments.

set -euo pipefail

# --- Configuration ---
# URLs for the Zenodo record containing datasets and model checkpoints
DATASETS_URL="https://zenodo.org/records/15410792/files/extracted_datasets.tar?download=1"
CHECKPOINTS_URL="https://zenodo.org/records/15410792/files/model_checkpoints.tar?download=1"

# Local names for the downloaded archives
DATASETS_ARCHIVE_NAME="extracted_datasets.tar"
CHECKPOINTS_ARCHIVE_NAME="model_checkpoints.tar"

# Target directories (standard project layout)
DATA_DIR="data"
MODELS_DIR="models"

echo "Creating '$DATA_DIR' and '$MODELS_DIR' directories..."
mkdir -p "$DATA_DIR" "$MODELS_DIR"

echo "Downloading datasets archive from '$DATASETS_URL'..."
wget -O "$DATASETS_ARCHIVE_NAME" "$DATASETS_URL"

echo "Downloading model checkpoints archive from '$CHECKPOINTS_URL'..."
wget -O "$CHECKPOINTS_ARCHIVE_NAME" "$CHECKPOINTS_URL"

echo "Extracting '$DATASETS_ARCHIVE_NAME'..."
tar -xf "$DATASETS_ARCHIVE_NAME" -C "$DATA_DIR/"

echo "Extracting '$CHECKPOINTS_ARCHIVE_NAME'..."
tar -xf "$CHECKPOINTS_ARCHIVE_NAME" -C "$MODELS_DIR/"

echo "Cleaning up downloaded archives..."
rm "$DATASETS_ARCHIVE_NAME" "$CHECKPOINTS_ARCHIVE_NAME"

echo "Setup complete."
echo "Activate the virtual environment and install the package with:"
echo "  pip install -e ."
echo "You can now run the Jupyter notebooks in notebooks/."