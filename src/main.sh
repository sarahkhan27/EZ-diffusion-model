#!/bin/bash

# This is the main script to run the complete simulate-and-recover exercise for 3000 iterations

# Define the number of iterations and sample sizes
iterations=1000
sample_sizes=(10 40 4000)

# Run the simulation and recovery for each sample size
for N in "${sample_sizes[@]}"
do
  echo "Running simulation for N=$N"

  # Generate true parameters for the simulation (randomly selected within specified ranges)
  for i in $(seq 1 $iterations)
  do
    a=$(awk -v min=0.5 -v max=2 'BEGIN{srand(); print min+((max-min)*rand())}')
    v=$(awk -v min=0.5 -v max=2 'BEGIN{srand(); print min+((max-min)*rand())}')
    t=$(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+((max-min)*rand())}')

    # Simulate predicted parameters
    predicted_params=$(python3 src/simulate_and_recover.py simulate_predicted_parameters --a $a --v $v --t $t)
    R_pred=$(echo $predicted_params | cut -d' ' -f1)
    M_pred=$(echo $predicted_params | cut -d' ' -f2)
    V_pred=$(echo $predicted_params | cut -d' ' -f3)

    # Simulate observed statistics
    observed_stats=$(python3 src/simulate_and_recover.py simulate_observed_statistics --R_pred $R_pred --M_pred $M_pred --V_pred $V_pred --N $N)
    R_obs=$(echo $observed_stats | cut -d' ' -f1)
    M_obs=$(echo $observed_stats | cut -d' ' -f2)
    V_obs=$(echo $observed_stats | cut -d' ' -f3)

    # Estimate parameters based on observed stats
    estimated_params=$(python3 src/simulate_and_recover.py estimate_parameters --R_obs $R_obs --M_obs $M_obs --V_obs $V_obs)

    # Calculate bias and squared error
    bias=$(python3 src/simulate_and_recover.py compute_bias --true_params $a,$v,$t --estimated_params $estimated_params)
    squared_error=$(python3 src/simulate_and_recover.py compute_squared_error --true_params $a,$v,$t --estimated_params $estimated_params)

    # Store results in results directory
    echo "$a,$v,$t,$R_pred,$M_pred,$V_pred,$R_obs,$M_obs,$V_obs,$estimated_params,$bias,$squared_error" >> results/results_N_$N.csv
  done
done

# Acknowledging reference to and help from ChatGPT
