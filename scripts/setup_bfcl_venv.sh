#!/bin/bash

# BFCL Virtual Environment Setup Script
set -e

echo "Setting up BFCL virtual environment..."

# Create virtual environment
echo "Creating virtual environment at .venvbfcl..."
python -m venv .venvbfcl

# Activate virtual environment
echo "Activating virtual environment..."
source ./.venvbfcl/bin/activate

# Navigate to BFCL directory
echo "Navigating to BFCL directory..."
cd third_party/gorilla_56d7a7c/berkeley-function-call-leaderboard/

# Install BFCL in editable mode
echo "Installing BFCL in editable mode..."
pip install -e .

# Install vllm
echo "Installing vllm..."
pip install vllm

# Install optional evaluation dependencies
echo "Installing OSS evaluation dependencies..."
pip install -e .[oss_eval_vllm]

echo "BFCL virtual environment setup complete!"
echo "To activate: source ./.venvbfcl/bin/activate"
