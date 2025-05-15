# Code for "You have to be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis"

In this jupyter notebooks for the experiments and results for the paper "You Have To be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis" are provided. 

After setting up and downloading datasets to the datasets/ folder and checkpoints to model_checkpoints/ folder the *.ipynb should run and reproduce results for the models in the paper. We advise downloading the extracted datasets directly from https://zenodo.org/records/15410792 as the direct extraction requires significantly more disk space (70GB vs. 500MB for ASCADr).

It is also possible to retrain models, the code for this is also in the notebooks. In this case we advise changing the model name to not overwrite the original checkpoints. 

Results for each of the considered models are produced in the appropriate notebooks. ASCADr results from the main paper are in ascadr_clean_mlp.ipynb. The CNN results from the appendix are in ascadr_clean_cnn.ipynb. ESHARD and CHES_CTF results are in eshard_clean.ipynb and ches_ctf_clean.ipynb respectively.ts.


# Setup
We advise creating a virtualenvironment and installing the dependencies in the requirements.txt file. 

Our python version is 3.9.12, requirements file was generated using pipreqs

Note that the python/package versions in the requirements file are not tight. A preexisting tensorflow installation (with missing packages installed) **should** work

Some functions in the repo assume the existence of several folders: model_checkpoints, datasets which can be downloaded from https://zenodo.org/records/15410792 as described below.


## Download Model Checkpoints:
Model checkpoints are available at anonymized link at https://zenodo.org/records/15410792 . These should be extracted to a folder called model_checkpoints in the root of this directory. Model retraining is also possible, but can impact the resulting structures in PCs and patching experiments will likely need manual adaption (e.g., different rotation)

## Creating/Downloading Extracted Datasets:
Note that generating the datasets from raw traces to work with will take some time and storage space (ASCADr raw trace file is ~80GB, CHES_CTF is 24 GB in total). The final used datasets are significantly smaller (ches 1.1 GB ASCADr 500MB, ESHARD 550MB). We also provide pre-extracted datasets at https://zenodo.org/records/15410792. Using these is recommended if resources are an issue (note also that trsfile and estraces can be removed from required packages as these are only used for raw dataset extraction). If paths do not match please check the src/datasets/paths.py file and update accordingly. 

For each of the datasets a shell script is provided that makes downloads raw traces and extracts new trace sets to appropriate locations. 

Otherwise raw traces can be downloaded and put in datasets folder manually from the included urls.


