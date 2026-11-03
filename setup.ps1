# Create the flash-drought Python environment with pip and open the first notebook.
# Running this script again reuses the environment, so it is also how you reopen the notebook.
Set-Location $PSScriptRoot

$venv = "flash-drought"
# The pinned packages in requirements.txt only have pre-built versions for these Python
# releases. Newer releases (3.14+) would need fastparquet to be compiled, which usually fails.
$supported = @("3.13", "3.12")

function Fail($message) {
    Write-Host "`nERROR: $message" -ForegroundColor Red
    Write-Host "`nPress any key to exit..." -ForegroundColor Gray
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch {}
    exit 1
}

# Returns the "major.minor" version of a Python command, or $null if it does not run.
# On Windows, "python" is often missing from the PATH (the python.org installer leaves that
# box unchecked by default) or is the Microsoft Store placeholder, which does not run Python.
function Get-PythonVersion($exe, $exeArgs) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $null }
    try {
        $version = & $exe @exeArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version) { return "$version".Trim() }
    } catch {}
    return $null
}

$venvPython = [IO.Path]::Combine($PSScriptRoot, $venv, "Scripts", "python.exe")
if ((Test-Path $venvPython) -and ($supported -contains (Get-PythonVersion $venvPython @()))) {
    Write-Host "Using the existing $venv environment" -ForegroundColor Green
} else {
    # Try the "py" launcher first, since it can pick a specific installed version
    $candidates = @(
        @("py", @("-3.13")),
        @("py", @("-3.12")),
        @("python", @()),
        @("python3", @())
    )
    $pyExe = $null
    foreach ($candidate in $candidates) {
        if ($supported -contains (Get-PythonVersion $candidate[0] $candidate[1])) {
            $pyExe = $candidate[0]
            $pyArgs = $candidate[1]
            break
        }
    }
    if (-not $pyExe) {
        Fail ("This course needs Python 3.13 or 3.12, which was not found. Install Python 3.13 " +
              "from https://www.python.org/downloads/ (not 3.14 or newer) and run this script " +
              "again in a new PowerShell window. Or install Anaconda and run .\setup-conda.ps1.")
    }
    Write-Host "Creating the $venv environment with Python $(Get-PythonVersion $pyExe $pyArgs)" -ForegroundColor Green
    # --clear replaces an environment left over from an unsupported Python version
    & $pyExe @pyArgs -m venv --clear $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "Creating the $venv environment failed." }
}

# Use the environment's own python.exe for every step instead of activating the environment,
# so packages can never be installed into another Python by mistake.
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail "Upgrading pip failed (see the messages above)." }

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installing the packages in requirements.txt failed (see the messages above)." }

& $venvPython -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
if ($LASTEXITCODE -ne 0) { Fail "Installing the Jupyter kernel failed." }

& $venvPython -m jupyter lab notebooks\1_drought_indicators.ipynb
