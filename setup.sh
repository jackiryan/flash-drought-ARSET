#!/bin/bash
# Create the flash-drought Python environment with pip and open the first notebook.
# Running this script again reuses the environment, so it is also how you reopen the notebook.
set -e
cd "$(dirname "$0")"

VENV=flash-drought

# rasterio and pyproj only provide pre-built macOS packages for macOS 14+ on Apple Silicon
# and macOS 15+ on Intel Macs. Anaconda provides them for older versions of macOS.
if [ "$(uname)" = "Darwin" ]; then
    macos_major=$(sw_vers -productVersion | cut -d. -f1)
    if [ "$(uname -m)" = "arm64" ]; then min_macos=14; else min_macos=15; fi
    if [ "$macos_major" -lt "$min_macos" ]; then
        echo "Error: setup.sh needs macOS $min_macos or newer on this Mac (found macOS $(sw_vers -productVersion))." >&2
        echo "Please install Anaconda from https://www.anaconda.com/download and run setup-conda.sh instead." >&2
        exit 1
    fi
fi

if [ -x "$VENV/bin/python" ]; then
    echo "Using the existing $VENV environment"
else
    # The python3 that comes with macOS is too old (3.9), so check the version first
    if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
        echo "Error: this course needs Python 3.12 or newer, which was not found." >&2
        if command -v python3 >/dev/null 2>&1; then
            echo "(python3 is $(python3 --version 2>&1), which will not work.)" >&2
        fi
        echo "Install Python from https://www.python.org/downloads/, open a new Terminal window," >&2
        echo "and run this script again. Or install Anaconda and run setup-conda.sh." >&2
        exit 1
    fi
    echo "Creating the $VENV environment with $(python3 --version)"
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks/1_drought_indicators.ipynb
