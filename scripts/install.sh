#!/bin/bash

# Activate virtual environment if needed
# source your_virtual_env/bin/activate

# Install CPU-based dependencies from requirements.txt
echo "Installing CPU-based packages..."
pip install -r requirements.txt

# Check if a GPU is available
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected. Installing GPU-based packages..."

    # Example: Replace CPU-based packages with GPU versions
    pip uninstall -y torch  # Uninstall CPU versions if installed

    # Install GPU versions
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected. Keeping CPU-based packages."
fi
echo "Installation complete."