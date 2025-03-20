#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Create directories if they don't exist
mkdir -p results

# Run the simulation
python3 /workspace/EZ-diffusion-model/src/simulation_main.py

# Acknowledging referance to and help from some website tools for fixing codes/ errors
