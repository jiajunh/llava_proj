#!/bin/bash

# Activate virtual environment if needed
# source your_virtual_env/bin/activate

# Install CPU-based dependencies from requirements.txt

# Check if a GPU is available
if command -v nvidia-smi &> /dev/null
then
    echo "GPU detected. Installing GPU-based packages..."
    # Install GPU versions
    pip install torch torchvision torchaudio
    pip install transformers==4.48.3 streamlit==1.43.2 nltk==3.2.4
    pip install -U bitsandbytes

else
    echo "No GPU detected. Installing CPU-based packages..."
    pip install -r requirements.txt
fi
echo "Installation complete."