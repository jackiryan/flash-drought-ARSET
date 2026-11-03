#!/bin/bash
# Create the flash-drought Python environment with pip and open the first notebook.
# Running this script again reuses the environment, so it is also how you reopen the notebook.
set -e
cd "$(dirname "$0")"

VENV=flash-drought
# The pinned packages in requirements.txt only have pre-built versions for these Python
# releases. Newer releases (3.14+) would need fastparquet to be compiled, which usually fails.
SUPPORTED="3.13 3.12"

# Succeeds if the given Python command runs and is one of the SUPPORTED versions
is_supported() {
    local version
    version=$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || return 1
    case " $SUPPORTED " in
        *" $version "*) return 0 ;;
        *) return 1 ;;
    esac
}

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

if [ -x "$VENV/bin/python" ] && is_supported "$VENV/bin/python"; then
    echo "Using the existing $VENV environment"
else
    PYTHON=""
    for cmd in python3.13 python3.12 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1 && is_supported "$cmd"; then
            PYTHON="$cmd"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: this course needs Python 3.13 or 3.12, which was not found." >&2
        if command -v python3 >/dev/null 2>&1; then
            echo "(python3 is $(python3 --version 2>&1), which will not work.)" >&2
        fi
        echo "Install Python 3.13 from https://www.python.org/downloads/ (not 3.14 or newer), open a" >&2
        echo "new Terminal window, and run this script again. Or install Anaconda and run setup-conda.sh." >&2
        exit 1
    fi
    echo "Creating the $VENV environment with $("$PYTHON" --version)"
    # --clear replaces an environment left over from an unsupported Python version
    "$PYTHON" -m venv --clear "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks/1_drought_indicators.ipynb
