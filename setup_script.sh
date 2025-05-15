#!/bin/bash

# Script to set up the environment for the "You Have To be Realistic" paper experiments.

# --- Configuration ---
# URLs for the Zenodo record containing datasets and model checkpoints
# Based on the provided image, there are two separate files.
DATASETS_URL="https://zenodo.org/records/15410792/files/extracted_datasets.tar?download=1"
CHECKPOINTS_URL="https://zenodo.org/records/15410792/files/model_checkpoints.tar?download=1"

# Local names for the downloaded archives
DATASETS_ARCHIVE_NAME="extracted_datasets.tar"
CHECKPOINTS_ARCHIVE_NAME="model_checkpoints.tar"

VENV_NAME=".venv"                    # Name for the virtual environment
REQUIREMENTS_FILE="requirements.txt" # Name of the requirements file

# --- Setup ---

echo "Starting setup process..."

# 1. Create and activate a virtual environment
echo "Creating virtual environment '$VENV_NAME'..."
python3 -m venv $VENV_NAME

echo "Activating virtual environment..."
# Source the activate script. Note: This activation is only effective within this script's execution.
# You will need to activate it manually in your terminal session later: source .venv/bin/activate
source $VENV_NAME/bin/activate

# 2. Install dependencies from requirements.txt
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing dependencies from '$REQUIREMENTS_FILE'..."
  pip install --upgrade pip
  pip install -r $REQUIREMENTS_FILE
  if [ $? -ne 0 ]; then
    echo "Error installing dependencies. Please check '$REQUIREMENTS_FILE' and your environment."
    exit 1
  fi
else
  echo "Error: '$REQUIREMENTS_FILE' not found. Please make sure it exists in the current directory."
  exit 1
fi

# 3. Create necessary directories
echo "Creating 'model_checkpoints' and 'datasets' directories..."
mkdir -p model_checkpoints
mkdir -p datasets

# 4. Download the archives
echo "Downloading datasets archive from '$DATASETS_URL'..."
wget -O "$DATASETS_ARCHIVE_NAME" "$DATASETS_URL"
if [ $? -ne 0 ]; then
  echo "Error downloading the datasets archive. Please check the URL and your internet connection."
  # Exit might be too aggressive, maybe continue and report later? For now, exit.
  exit 1
fi

echo "Downloading model checkpoints archive from '$CHECKPOINTS_URL'..."
wget -O "$CHECKPOINTS_ARCHIVE_NAME" "$CHECKPOINTS_URL"
if [ $? -ne 0 ]; then
  echo "Error downloading the model checkpoints archive. Please check the URL and your internet connection."
  exit 1
fi

# 5. Extract the archives
echo "Extracting '$DATASETS_ARCHIVE_NAME'..."
# Assuming the tar contains the contents that should go into the 'datasets' folder
tar -xf "$DATASETS_ARCHIVE_NAME" -C datasets/ # Extract into the datasets directory

if [ $? -ne 0 ]; then
  echo "Error extracting the datasets archive. Please check the archive format and the extraction command."
  exit 1
fi

echo "Extracting '$CHECKPOINTS_ARCHIVE_NAME'..."
# Assuming the tar contains the contents that should go into the 'model_checkpoints' folder
tar -xf "$CHECKPOINTS_ARCHIVE_NAME" -C . # Extract into the model_checkpoints directory

if [ $? -ne 0 ]; then
  echo "Error extracting the model checkpoints archive. Please check the archive format and the extraction command."
  exit 1
fi

# Clean up the downloaded archives
echo "Cleaning up downloaded archives..."
rm "$DATASETS_ARCHIVE_NAME" "$CHECKPOINTS_ARCHIVE_NAME"

echo "Setup complete."
echo "Please activate the virtual environment manually in your terminal: source $VENV_NAME/bin/activate"
echo "You can now run the Jupyter notebooks."

# Note: If the paths in the notebooks or src/datasets/paths.py do not match after extraction,
# you may need to manually adjust src/datasets/paths.py or symlink the directories.
