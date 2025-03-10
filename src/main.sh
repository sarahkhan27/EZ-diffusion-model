# Acknowledging reference to and help from ChatGPT

#!/bin/bash

# Function to simulate data and recover parameters
simulate_and_recover() {
    N=$1  # Number of trials
    iterations=1000  # Total number of iterations for each N

    # Run 1000 iterations for the current value of N
    for i in $(seq 1 $iterations); do
        # Set the parameters to the specified values
        a=1.5  # Boundary separation (fixed value - choose between 0.5 and 2)
        v=1.3  # Drift rate (fixed value - choose between 0.5 and 2)
        t=0.3  # Nondecision time (fixed value - choose between 0.1 and 0.5)

        # Call the Python script to simulate and recover parameters
        python3 /workspace/EZ-diffusion-model/src/simulate_and_recover.py $a $v $t $N
    done
}

# Run simulations for different values of N (10, 40, and 4000)
simulate_and_recover (10)
simulate_and_recover (40)
simulate_and_recover (4000)
