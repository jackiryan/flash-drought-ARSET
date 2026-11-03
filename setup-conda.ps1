# Create (or update) the flash-drought conda environment and open the first notebook.
Set-Location $PSScriptRoot

$envName = "flash-drought"

function Fail($message) {
    Write-Host "`nERROR: $message" -ForegroundColor Red
    Write-Host "`nPress any key to exit..." -ForegroundColor Gray
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch {}
    exit 1
}

# Locate conda. In a regular PowerShell window conda is usually not on the PATH, because the
# Anaconda installer does not add it by default, so also check the usual install locations.
$conda = $env:CONDA_EXE
if (-not $conda) {
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { $conda = $cmd.Source }
}
if (-not $conda) {
    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\anaconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe"
    )
    $conda = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $conda) {
    Fail ("Could not find conda. Install Anaconda from https://www.anaconda.com/download, " +
          "then run this script from the 'Anaconda PowerShell Prompt' in the Start menu.")
}
Write-Host "Using conda at $conda" -ForegroundColor Cyan

# Check for the environment first, rather than letting "conda env create" fail, so that
# real errors (e.g. a failed package download) are reported clearly.
$envExists = (& $conda env list) | Where-Object { ($_ -split '\s+')[0] -eq $envName }
if ($envExists) {
    Write-Host "Updating the existing $envName environment..." -ForegroundColor Green
    & $conda env update --prune -n $envName -f environment.yml
} else {
    & $conda env create -f environment.yml
}
if ($LASTEXITCODE -ne 0) { Fail "Creating the $envName environment failed (see the messages above)." }

# Load conda's PowerShell support so that "conda activate" works in this script
(& $conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $envName
if ($env:CONDA_DEFAULT_ENV -ne $envName) { Fail "Could not activate the $envName environment." }

python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
if ($LASTEXITCODE -ne 0) { Fail "Installing the Jupyter kernel failed." }

jupyter lab notebooks\1_drought_indicators.ipynb
