#!/bin/bash

# Create directories to store results
mkdir -p results

# Set the number of iterations and the sample sizes
iterations=1000
sample_sizes=(10 40 4000)

# Loop through different sample sizes
for N in "${sample_sizes[@]}"; do
    echo "Running simulations for sample size N=$N..."

    # Run simulations for each iteration
    for i in $(seq 1 $iterations); do
        # Randomly select true parameters
        alpha=$(awk -v min=0.5 -v max=2 'BEGIN{srand(); print min+rand()*(max-min)}')
        nu=$(awk -v min=0.5 -v max=2 'BEGIN{srand(); print min+rand()*(max-min)}')
        tau=$(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')

        # Simulate predicted parameters
        read R_pred M_pred V_pred <<< $(python3 -c "
import numpy as np
from src.simulate_and_recover import simulate_predicted_parameters
alpha = $alpha
nu = $nu
tau = $tau
R_pred, M_pred, V_pred = simulate_predicted_parameters(alpha, nu, tau)
print(R_pred, M_pred, V_pred)
")

        # Simulate observed parameters
        read R_obs M_obs V_obs <<< $(python3 -c "
import numpy as np
from src.simulate_and_recover import simulate_observed_statistics
R_pred = $R_pred
M_pred = $M_pred
V_pred = $V_pred
N = $N
R_obs, M_obs, V_obs = simulate_observed_statistics(R_pred, M_pred, V_pred, N)
print(R_obs, M_obs, V_obs)
")

        # Estimate parameters from observed data
        read nu_est alpha_est tau_est <<< $(python3 -c "
import numpy as np
from src.simulate_and_recover import estimate_parameters
R_obs = $R_obs
M_obs = $M_obs
V_obs = $V_obs
nu_est, alpha_est, tau_est = estimate_parameters(R_obs, M_obs, V_obs)
print(nu_est, alpha_est, tau_est)
")

        # Compute bias and squared error
        bias=$(python3 -c "
true_params = [$nu, $alpha, $tau]
estimated_params = [$nu_est, $alpha_est, $tau_est]
bias = np.array(true_params) - np.array(estimated_params)
print(' '.join(map(str, bias)))
")

        squared_error=$(python3 -c "
true_params = [$nu, $alpha, $tau]
estimated_params = [$nu_est, $alpha_est, $tau_est]
squared_error = np.sum((np.array(true_params) - np.array(estimated_params)) ** 2)
print(squared_error)
")

        # Save the results
        echo "$alpha $nu $tau $nu_est $alpha_est $tau_est $bias $squared_error" >> results/simulation_results_N${N}.txt
    done
done