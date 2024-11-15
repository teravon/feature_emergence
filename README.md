# Code for "I know what your layers did"

In this repository scripts to generate results for the paper "I Know What Your Layers Did: Layer-wise Explainability of Deep Learning Side-channel Analysis" are provided.

## Setup
We advise creating a virtualenvironment and installing the dependencies in the requirements.txt file. 

Our python version is 3.9.12, requirements file was generated using pipreqs

Note that the python/package versions in the requirements file are probably not extremely tight, but these are the ones we used.

Some functions in the repo assume the existence of several folders: figures, model_checkpoints, data

Besides this, the paths in src/datasets/paths.py might need updating to accurately point to datasets

## Creating 2000 sample ASCADr
After downloading the raw traces file as descirbed in https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key run the generate_new_ascadr.py file on this (again update paths in script)


## Training Models to Analyze
In "train_model.py" a script to train a model to analyze is provided. The models used in the paper are defined in the "profiling_and_attack.py" script, and checkpoints are saved in the model_checkpoints folder

## Generating Ex-DLSCA probing results
The file pi_plots_eshard_mlp.py contains a script to generate npz files containing PI results. These files get stored in /data folder by default

## Patching
In activation_patching_eshard_mlp.py there is an annotated file for patching. The other patching results are in corresponding files