#!/bin/bash

# Activate virtual environment if needed
# source your_virtual_env/bin/activate

# Install CPU-based dependencies from requirements.txt
echo "Installing CPU-based packages..."

# Check if a GPU is available
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected. Installing GPU-based packages..."
    # Install GPU versions
    pip install torch torchvision torchaudio
    pip install transformers==4.48.3 streamlit==1.43.2 nltk==3.9.1

else
    echo "No GPU detected. Keeping CPU-based packages."
    pip install -r requirements.txt
fi
echo "Installation complete."