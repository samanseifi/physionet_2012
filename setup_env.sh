#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create the environment from YAML file
echo "Creating conda environment..."
conda env create -f environment.yml

# Extract environment name from the YAML file
ENV_NAME=$(head -n 1 environment.yml | cut -d ' ' -f 2)

# Activate environment
echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

# Install any additional pip packages (if a requirements file exists)
if [ -f "requirements.txt" ]; then
    echo "Installing pip packages..."
    pip install -r requirements.txt
fi

echo "Environment setup complete!"
