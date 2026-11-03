# Create the flash-drought Python environment with pip and open the first notebook.
# Running this script again reuses the environment, so it is also how you reopen the notebook.
Set-Location $PSScriptRoot

$venv = "flash-drought"

function Fail($message) {
    Write-Host "`nERROR: $message" -ForegroundColor Red
    Write-Host "`nPress any key to exit..." -ForegroundColor Gray
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch {}
    exit 1
}

# Succeeds if the given command runs Python 3.12 or newer.
# On Windows, "python" is often missing from the PATH (the python.org installer leaves that
# box unchecked by default) or is the Microsoft Store placeholder, which does not run Python.
function Test-Python($exe) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $false }
    try {
        $null = & $exe -c "import sys; sys.exit(sys.version_info < (3, 12))" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$venvPython = [IO.Path]::Combine($PSScriptRoot, $venv, "Scripts", "python.exe")
if (Test-Path $venvPython) {
    Write-Host "Using the existing $venv environment" -ForegroundColor Green
} else {
    # Try the "py" launcher first, since python.org installs it even when "python" is not on the PATH
    $pyExe = @("py", "python") | Where-Object { Test-Python $_ } | Select-Object -First 1
    if (-not $pyExe) {
        Fail ("This course needs Python 3.12 or newer, which was not found. Install Python from " +
              "https://www.python.org/downloads/ and run this script again in a new PowerShell " +
              "window. Or install Anaconda and run .\setup-conda.ps1.")
    }
    Write-Host "Creating the $venv environment with $(& $pyExe --version)" -ForegroundColor Green
    & $pyExe -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "Creating the $venv environment failed." }
}

# Use the environment's own python.exe for every step instead of activating the environment,
# so packages can never be installed into another Python by mistake.
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail "Upgrading pip failed (see the messages above)." }

# --prefer-binary picks an older pre-built release over compiling a newer one from source,
# which would fail because it needs libraries (such as GDAL) that are not installed
& $venvPython -m pip install --prefer-binary -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installing the packages in requirements.txt failed (see the messages above)." }

& $venvPython -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
if ($LASTEXITCODE -ne 0) { Fail "Installing the Jupyter kernel failed." }

& $venvPython -m jupyter lab notebooks\1_drought_indicators.ipynb
