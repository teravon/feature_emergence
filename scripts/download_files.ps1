# PowerShell script to download the trace datasets and model checkpoints into
# data/ and models/. Run from the repository root.

# --- Configuration ---
# Zenodo record containing the preprocessed (trimmed) trace datasets and the
# trained model checkpoints.
$DATASETS_URL = "https://zenodo.org/records/15410792/files/extracted_datasets.tar?download=1"
$CHECKPOINTS_URL = "https://zenodo.org/records/15410792/files/model_checkpoints.tar?download=1"

$DATASETS_ARCHIVE_NAME = "extracted_datasets.tar"
$CHECKPOINTS_ARCHIVE_NAME = "model_checkpoints.tar"

# Target directories (standard project layout)
$DATA_DIR = "data"
$MODELS_DIR = "models"

# 1. Create the target directories
Write-Host "Creating '$DATA_DIR' and '$MODELS_DIR' directories..."
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $MODELS_DIR | Out-Null

# 2. Download the datasets archive
Write-Host "Downloading datasets archive from '$DATASETS_URL'..."
Invoke-WebRequest -Uri $DATASETS_URL -OutFile $DATASETS_ARCHIVE_NAME
if (-not (Test-Path $DATASETS_ARCHIVE_NAME)) {
    Write-Error "Failed to download datasets archive."
    exit 1
}

Write-Host "Extracting '$DATASETS_ARCHIVE_NAME' to '$DATA_DIR/'..."
tar -xf $DATASETS_ARCHIVE_NAME -C $DATA_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error extracting datasets archive."
    exit 1
}

# 3. Download the checkpoints archive
Write-Host "Downloading model checkpoints archive from '$CHECKPOINTS_URL'..."
Invoke-WebRequest -Uri $CHECKPOINTS_URL -OutFile $CHECKPOINTS_ARCHIVE_NAME
if (-not (Test-Path $CHECKPOINTS_ARCHIVE_NAME)) {
    Write-Error "Failed to download model checkpoints archive."
    exit 1
}

# 4. Extract checkpoints, stripping the top-level model_checkpoints/ folder so
#    files land flat in models/, where the code looks for them.
Write-Host "Extracting '$CHECKPOINTS_ARCHIVE_NAME' to '$MODELS_DIR/'..."
tar --strip-components=1 -xf $CHECKPOINTS_ARCHIVE_NAME -C $MODELS_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error extracting model checkpoints archive."
    exit 1
}

# 5. Clean up downloaded archives
Write-Host "Cleaning up downloaded archives..."
Remove-Item $DATASETS_ARCHIVE_NAME, $CHECKPOINTS_ARCHIVE_NAME

# 6. Final message
Write-Host "`nSetup complete."
Write-Host "Trace datasets are in '$DATA_DIR/', model checkpoints in '$MODELS_DIR/'."