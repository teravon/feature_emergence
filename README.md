# Code for "You have to be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis"

In this noteboooks for the experiments and results for the paper "You Have To be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis" are provided.

## Setup
We advise creating a virtualenvironment and installing the dependencies in the requirements.txt file. 

Our python version is 3.9.12, requirements file was generated using pipreqs

Note that the python/package versions in the requirements file are probably not tight, but these are the ones we used.

Some functions in the repo assume the existence of several folders: figures, model_checkpoints, datasets


# Download Model Checkpoints:
Model checkpoints are available at anonymized link at https://zenodo.org/records/15410792 . These should be extracted to a folder called model_checkpoints in the root of this directory. Model retraining is also possible, but can impact the resulting structures in PCs and patching experiments will likely need manual adaption (e.g., different rotation)

# Create Datasets:
Note that generating the datasets from raw traces to work with will take some time and storage space (ASCADr raw trace file is ~80GB, CHES_CTF is 24 GB in total). The final used datasets are significantly smaller (ches 1.1 GB ASCADr 500MB, ESHARD 550MB) We also provide pre-extracted datasets at https://zenodo.org/records/15410792. Using these is recommended if resources are an issue (note also that trsfile and estraces can be removed from required packages as these are only used for raw dataset extraction). If paths do not match please check the src/datasets/paths.py file and update accordingly. 

For each of the datasets a shell script is provided that makes downloads raw traces and extracts new trace sets to appropriate locations. 

Otherwise raw traces can be downloaded and put in datasets folder manually from the included urls.


