
# Setup
We advise creating a virtualenvironment and installing the dependencies in the requirements.txt file. 

Our python version is 3.9.12, requirements file was generated using pipreqs

Note that the python/package versions in the requirements file are probably not extremely tight, but these are the ones we used.

Some functions in the repo assume the existence of several folders: figures, model_checkpoints, data

Besides this, the paths in src/datasets/paths.py might need updating to accurately point to dataset locations

# Creating 2000 sample ASCADr
After downloading the raw traces file as descirbed in https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key run the generate_new_ascadr.py file on this (again update paths in script)


