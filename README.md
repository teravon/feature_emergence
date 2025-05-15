# Code for "You have to be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis"

In this jupyter notebooks for the experiments and results for the paper "You Have To be Realistic: On Investigating Feature Emergence in Deep Learning-based Side-channel Analysis" are provided. 

After setting up and downloading datasets using below code you can run the jupyter notebook command to run the jupyter server (or use your preffered editor for notebooks). The install script should register the kernel under ``you-have-to-be-realistic''. please make sure to select the right kernel.

It is also possible to retrain models, the code for this is also in the notebooks. In this case we advise changing the model name to not overwrite the original checkpoints. 

Results for each of the considered models are produced in the appropriate notebooks. ASCADr results from the main paper are in ascadr_clean_mlp.ipynb. The CNN results from the appendix are in ascadr_clean_cnn.ipynb. ESHARD and CHES_CTF results are in eshard_clean.ipynb and ches_ctf_clean.ipynb respectively.ts.


# Setup

## Download Model Checkpoints and Extracted datasets:
Model checkpoints are available at anonymized link at https://zenodo.org/records/15410792 . The script in download_files.sh (ps1) will download and extract files to the appropriate locations. These should be extracted to a folder called model_checkpoints in the root of this directory. Model retraining is also possible, but can impact the resulting structures in PCs and patching experiments will likely need manual adaption (e.g., different rotation)


## Docker
The most error proof method is to use the provided docker file given you have a docker installation. First download the model checkpoints and datasets using the download_files.sh/ps1 scripts for bash/powershell. Then build and run the docker container using:: 
```
docker build -t realistic-env . 
docker run -it --rm -p 8888:8888 realistic-env
```
Then going to localhost:8888 in your chosen browser where the notebooks will be available.

## Conda 
If you dont want to use docker we advise creating a virtualenvironment using conda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) with the appropriate python version (3.9). The install_environments.sh script contains an installation script. You can also manually install the dependencies in the requirements.txt file. We also provide powershell script for windows users although these are untested and we recommend using WSL for these.

## Manually Creating/Downloading Extracted Datasets:
Note that generating the datasets from raw traces to work with will take some time and storage space (ASCADr raw trace file is ~80GB, CHES_CTF is 24 GB in total). The final used datasets are significantly smaller (ches 1.1 GB ASCADr 500MB, ESHARD 550MB). 

For each of the datasets a shell script is provided that makes downloads raw traces and extracts new trace sets to appropriate locations. 

Otherwise raw traces can be downloaded and put in datasets folder manually from the included urls.


