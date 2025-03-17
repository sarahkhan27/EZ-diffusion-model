# Acknowledging reference to and help from ChatGPT

#!/bin/bash

# Test if the main simulation script runs without errors
echo "Testing simulation script..."
bash src/main.sh

# Test if the result files are generated
if [ -f "results/simulation_results_N10.txt" ] && [ -f "results/simulation_results_N40.txt" ] && [ -f "results/simulation_results_N4000.txt" ]; then
    echo "Simulation results files found!"
else
    echo "Error: Simulation result files not found."
    exit 1
fi