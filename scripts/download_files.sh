#!/bin/bash

# Downloads the trace datasets and model checkpoints into data/ and models/.
# Run from the repository root.

set -euo pipefail

# --- Configuration ---
# Zenodo record containing the preprocessed (trimmed) trace datasets and the
# trained model checkpoints.
DATASETS_URL="https://zenodo.org/records/15410792/files/extracted_datasets.tar?download=1"
CHECKPOINTS_URL="https://zenodo.org/records/15410792/files/model_checkpoints.tar?download=1"

DATASETS_ARCHIVE_NAME="extracted_datasets.tar"
CHECKPOINTS_ARCHIVE_NAME="model_checkpoints.tar"

# Target directories (standard project layout)
DATA_DIR="data"
MODELS_DIR="models"

echo "Creating '$DATA_DIR' and '$MODELS_DIR' directories..."
mkdir -p "$DATA_DIR" "$MODELS_DIR"

fetch() {
    local url="$1"
    local out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
    else
        curl -L -o "$out" "$url"
    fi
}

echo "Downloading datasets archive from '$DATASETS_URL'..."
fetch "$DATASETS_URL" "$DATASETS_ARCHIVE_NAME"

echo "Extracting '$DATASETS_ARCHIVE_NAME' into '$DATA_DIR/'..."
tar -xf "$DATASETS_ARCHIVE_NAME" -C "$DATA_DIR/"

echo "Downloading model checkpoints archive from '$CHECKPOINTS_URL'..."
fetch "$CHECKPOINTS_URL" "$CHECKPOINTS_ARCHIVE_NAME"

echo "Extracting '$CHECKPOINTS_ARCHIVE_NAME' into '$MODELS_DIR/'..."
# The archive wraps everything in a top-level model_checkpoints/ folder; strip
# it so checkpoints land flat in models/, where the code looks for them.
tar --strip-components=1 -xf "$CHECKPOINTS_ARCHIVE_NAME" -C "$MODELS_DIR/"

echo "Cleaning up downloaded archives..."
rm "$DATASETS_ARCHIVE_NAME" "$CHECKPOINTS_ARCHIVE_NAME"

echo "Setup complete."
echo "Trace datasets are in '$DATA_DIR/', model checkpoints in '$MODELS_DIR/'."