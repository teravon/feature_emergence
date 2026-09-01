#!/bin/bash

# Name of your conda environment
ENV_NAME="you-have-to-be-realistic"

# Create the conda environment with Python 3.9
conda create -y -n $ENV_NAME python=3.9

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install packages with conda where possible
conda install -y \
  h5py \
  matplotlib \
  numpy \
  scikit-learn \
  scipy \
  jupyter \
  tqdm=4.63.0

# Install pip-only packages (tensorflow, experiments, etc.)
pip install tensorflow==2.15.0.post1

# Install the project package in editable mode (src/ layout)
pip install -e .

# Register the kernel for Jupyter
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo "✅ Conda environment '$ENV_NAME' created and ready with all packages."