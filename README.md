# Code for "I know what your layers did"

In this repository scripts to generate results for the paper "I Know What Your Layers Did: Layer-wise Explainability of Deep Learning Side-channel Analysis" are provided.

## Setup
We advise creating a virtualenvironment and installing the dependencies in the requirements.txt file. 

Our python version is 3.9.12, requirements file was generated using pipreqs

Note that the python/package versions in the requirements file are probably not extremely tight, but these are the ones we used.

Some functions in the repo assume the existence of several folders: figures, model_checkpoints, data

Besides this, the paths in src/datasets/paths.py will need updating to accurately point to datasets

# Download Model Checkpoints:
Model checkpoints are available at anonymized link at https://zenodo.org/records/15395878 These should be extracted to a folder called model_checkpoints in the root of this directory.

# Create Datasets:
Note that generating the datasets from raw traces to work with will take some time and storage space (ASCADr raw trace file is ~80GB, CHES_CTF is 24 GB in total). The raw traces can be deleted after running the scripts to generate h5 files. ESHARD is < 1GB.
## ASCADr
After downloading the raw traces file as descirbed in https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key run the generate_new_ascadr.py file on this (again update paths in script)

## Creating eshard.h5
Download eshard non-shuffled  ets file from https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled/-/blob/main/Nucleo_AES_masked_non_shuffled.ets?ref_type=heads and use generate_eshard.py with updated locations to point to appropriate file locations.

## Create CHES_CTF.h5
Follow instructions in generate_ches_ctf.py (this file is adapted from https://github.com/AISyLab/feature_selection_dlsca/blob/master/experiments/CHESCTF/generate_dataset.py)


