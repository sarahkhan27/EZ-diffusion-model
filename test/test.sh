#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Run the tests
python3 -m unittest discover -s test

python3 -m unittest discover -s test -p "test_simulation.py"

# Acknowledging reference to and help from ChatGPT

