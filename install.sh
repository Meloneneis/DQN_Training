#!/bin/bash
set -e  # Exit immediately if any command fails

echo "--- Starting Installation ---"

# 1. Initialize Conda Environment 🐍
# This boilerplate ensures 'conda activate' works correctly inside the script.
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Error: Conda is not found. Please ensure it is installed and in your PATH." >&2
    exit 1
fi

# 2. Create and Activate Conda environment 'sdc' with python=3.8.10
ENV_NAME="sdc"
PYTHON_VERSION="3.8.10"

if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Creating Conda environment '${ENV_NAME}' with python=${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo "Activating Conda environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# 3. Install uv inside the Conda environment
# We use standard pip, which is now the pip inside the 'sdc' environment.
echo "Installing uv..."
pip install uv

# --- The rest of the uv commands now execute within the 'sdc' environment ---

# 4. Install swig
echo "Installing swig..."
uv pip install swig

# 5. Upgrade typing_extensions
echo "Upgrading typing_extensions..."
uv pip install typing_extensions --upgrade

# 6. Install Torch/Vision with CUDA 12.1 Index
# Note: Ensure these versions are compatible with cu121.
echo "Installing Torch/Torchvision from CUDA 12.1 index..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 7. Install remaining requirements (Safe Mode)
# This filters out torch/torchvision to prevent the CUDA version from being overwritten by CPU versions from PyPI.
echo "Installing remaining requirements..."
grep -vE "^torch==|^torchvision==" requirements.txt | uv pip install -r /dev/stdin

echo "--- Installation Complete ---"