#!/bin/bash
# Create (or update) the flash-drought conda environment and open the first notebook.
set -e
cd "$(dirname "$0")"

ENV_NAME=flash-drought

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda was not found." >&2
    echo "Install Anaconda from https://www.anaconda.com/download, open a new Terminal window," >&2
    echo "and run this script again." >&2
    exit 1
fi

# Check for the environment first, rather than letting "conda env create" fail, so that
# real errors (e.g. a failed package download) are shown instead of being hidden.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Updating the existing $ENV_NAME environment..."
    conda env update --prune -n "$ENV_NAME" -f environment.yml
else
    conda env create -f environment.yml
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks/1_drought_indicators.ipynb
