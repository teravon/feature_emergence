# PowerShell script to set up the conda environment for the "You Have To be Realistic" project

# --- Configuration ---
$ENV_NAME = "you-have-to-be-realistic"

# 1. Create the conda environment with Python 3.9
Write-Host "Creating conda environment '$ENV_NAME' with Python 3.9..."
conda create -y -n $ENV_NAME python=3.9

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create the conda environment."
    exit 1
}

# 2. Activate the environment
# Note: PowerShell does not use 'source'; use 'conda activate' directly if Conda shell integration is set up.
Write-Host "Activating the conda environment..."
conda activate $ENV_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate the conda environment. Make sure conda is initialized for PowerShell."
    exit 1
}

# 3. Install packages with conda
Write-Host "Installing core packages with conda..."
conda install -y `
    h5py `
    matplotlib `
    numpy `
    scikit-learn `
    scipy `
    jupyter `
    tqdm=4.63.0

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install some conda packages."
    exit 1
}

# 4. Install pip-only packages
Write-Host "Installing pip-only packages..."
pip install tensorflow==2.15.0.post1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install pip packages."
    exit 1
}

# 5. Register the environment kernel for Jupyter
Write-Host "Registering the Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to register the Jupyter kernel."
    exit 1
}

Write-Host "`n✅ Conda environment '$ENV_NAME' created and ready with all packages."