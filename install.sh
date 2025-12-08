#!/bin/bash
set -e  # Exit immediately if any command fails

echo "--- Starting Installation ---"

# 1. Install uv (using standard pip)
pip install uv

# 2. Create and activate a virtual environment (Required for uv)
# If a venv folder doesn't exist, create it.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi
# Activate the environment
source .venv/bin/activate

# 3. Install swig
echo "Installing swig..."
uv pip install swig

# 4. Upgrade typing_extensions (Corrected from 'extension')
echo "Upgrading typing_extensions..."
uv pip install typing_extensions --upgrade

# 5. Install Torch/Vision with CUDA 12.1 Index
# Note: As discussed, Torch 2.0.1 + cu121 is a rare combination.
# This command attempts to find it; if it fails, consider changing versions.
echo "Installing Torch/Torchvision from CUDA 12.1 index..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Install requirements.txt (Safe Mode)
# We use grep to filter out torch/torchvision lines from the file
# and pipe the result to uv. This prevents uv from reinstalling the
# Wrong (CPU) versions of torch over the CUDA ones you just installed.
echo "Installing remaining requirements..."
grep -vE "^torch==|^torchvision==" requirements.txt | uv pip install -r /dev/stdin

echo "--- Installation Complete ---"