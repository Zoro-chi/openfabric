#!/bin/bash

# Create models directory in the project
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

# Default model
DEFAULT_MODEL="meta-llama/Llama-3.2-3B-Instruct"

echo "Setting up Hugging Face model in project directory: $MODELS_DIR"
echo "Note: This script will download the model files directly to your project."

# Check if Hugging Face CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    pip install -q huggingface_hub
fi

# Login to Hugging Face (if needed)
echo "Checking Hugging Face login status..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "You need to login to Hugging Face to download the model."
    echo "Please enter your Hugging Face token when prompted:"
    huggingface-cli login
fi

# Download the model to the project directory
echo "Downloading model $DEFAULT_MODEL to $MODELS_DIR..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$DEFAULT_MODEL', local_dir='$MODELS_DIR/$DEFAULT_MODEL', local_dir_use_symlinks=False)"

# Update environment file to point to the local model
echo "MODEL_ID=$MODELS_DIR/$DEFAULT_MODEL" >> .env

echo "Model setup complete!"
echo "The model has been downloaded to: $MODELS_DIR/$DEFAULT_MODEL"
echo "This location has been added to your .env file"
echo "When running the Docker container, this directory will be mounted and available to the application."
