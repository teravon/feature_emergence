# PowerShell script to set up the environment for the "You Have To be Realistic" paper experiments

# --- Configuration ---
# URLs for datasets and model checkpoints
$DATASETS_URL = "https://zenodo.org/records/15410792/files/extracted_datasets.tar?download=1"
$CHECKPOINTS_URL = "https://zenodo.org/records/15410792/files/model_checkpoints.tar?download=1"

# Local filenames
$DATASETS_ARCHIVE_NAME = "extracted_datasets.tar"
$CHECKPOINTS_ARCHIVE_NAME = "model_checkpoints.tar"

# 1. Create necessary directories
$DATA_DIR = "data"
$MODELS_DIR = "models"
Write-Host "Creating '$MODELS_DIR' and '$DATA_DIR' directories..."
New-Item -ItemType Directory -Force -Path $MODELS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null

# 2. Download the archives
Write-Host "Downloading datasets archive from '$DATASETS_URL'..."
Invoke-WebRequest -Uri $DATASETS_URL -OutFile $DATASETS_ARCHIVE_NAME
if (-not (Test-Path $DATASETS_ARCHIVE_NAME)) {
    Write-Error "Failed to download datasets archive."
    exit 1
}

Write-Host "Downloading model checkpoints archive from '$CHECKPOINTS_URL'..."
Invoke-WebRequest -Uri $CHECKPOINTS_URL -OutFile $CHECKPOINTS_ARCHIVE_NAME
if (-not (Test-Path $CHECKPOINTS_ARCHIVE_NAME)) {
    Write-Error "Failed to download model checkpoints archive."
    exit 1
}

# 3. Extract the archives (requires tar available on system — included in Windows 10+)
Write-Host "Extracting '$DATASETS_ARCHIVE_NAME' to '$DATA_DIR/'..."
tar -xf $DATASETS_ARCHIVE_NAME -C $DATA_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error extracting datasets archive."
    exit 1
}

Write-Host "Extracting '$CHECKPOINTS_ARCHIVE_NAME' to '$MODELS_DIR/'..."
tar -xf $CHECKPOINTS_ARCHIVE_NAME -C $MODELS_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error extracting model checkpoints archive."
    exit 1
}

# 4. Clean up downloaded archives
Write-Host "Cleaning up downloaded archives..."
Remove-Item $DATASETS_ARCHIVE_NAME, $CHECKPOINTS_ARCHIVE_NAME

# 5. Final message
Write-Host "`nSetup complete."
Write-Host "Please activate your Python virtual environment manually (e.g., `& .\venv\Scripts\Activate.ps1`)."
Write-Host "You can now run the Jupyter notebooks."
