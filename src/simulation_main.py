# Acknowledging reference to and help from ChatGPT

import sys
import os
import numpy as np
import random
import scipy.stats as stats

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the correct module
from src.simulate_and_recover import forward_ez, inverse_ez, simulate_observed_stats, simulate_and_recover

def run_simulation(iterations=1000, N_values=[10, 40, 4000], seed=None):
    """
    Run the simulation for different sample sizes.
    
    Parameters:
    iterations (int): Number of simulation iterations for each N
    N_values (list): List of sample sizes to test
    seed (int): Random seed for reproducibility
    
    Returns:
    dict: Results for each N value
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    results = {}
    
    for N in N_values:
        print(f"Running simulation for N = {N}")
        
        v_true = []
        alpha_true = []
        tau_true = []
        
        v_est = []
        alpha_est = []
        tau_est = []
        
        biases = []
        squared_errors = []
        
        for i in range(iterations):
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{iterations}")
            
            # Generate random parameters within the specified ranges
            v = random.uniform(0.5, 2.0)
            alpha = random.uniform(0.5, 2.0) 
            tau = random.uniform(0.1, 0.5)
            
            # Simulate and recover
            try:
                v_e, alpha_e, tau_e, bias, squared_error = simulate_and_recover(v, alpha, tau, N)
                
                # Store results
                v_true.append(v)
                alpha_true.append(alpha)
                tau_true.append(tau)
                
                v_est.append(v_e)
                alpha_est.append(alpha_e)
                tau_est.append(tau_e)
                
                biases.append(bias)
                squared_errors.append(squared_error)
            except Exception as e:
                print(f"  Error in iteration {i+1}: {e}")
                # Continue with next iteration
                continue
        
        # Calculate average bias and squared error
        avg_bias = np.mean(biases, axis=0)
        avg_squared_error = np.mean(squared_errors)
        
        # Store results for this N
        results[N] = {
            'v_true': v_true,
            'alpha_true': alpha_true,
            'tau_true': tau_true,
            'v_est': v_est,
            'alpha_est': alpha_est,
            'tau_est': tau_est,
            'biases': biases,
            'squared_errors': squared_errors,
            'avg_bias_v': avg_bias[0],
            'avg_bias_alpha': avg_bias[1],
            'avg_bias_tau': avg_bias[2],
            'avg_squared_error': avg_squared_error
        }
        
        print(f"  Average bias for N = {N}: v = {avg_bias[0]:.4f}, alpha = {avg_bias[1]:.4f}, tau = {avg_bias[2]:.4f}")
        print(f"  Average squared error for N = {N}: {avg_squared_error:.4f}")
    
    return results

def save_results(results, output_dir):
    """
    Save simulation results to files.
    
    Parameters:
    results (dict): Simulation results
    output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("EZ Diffusion Model Simulation Results\n")
        f.write("===================================\n\n")
        
        for N, data in results.items():
            f.write(f"Sample size N = {N}\n")
            f.write(f"Average bias for v: {data['avg_bias_v']:.6f}\n")
            f.write(f"Average bias for alpha: {data['avg_bias_alpha']:.6f}\n")
            f.write(f"Average bias for tau: {data['avg_bias_tau']:.6f}\n")
            f.write(f"Average squared error: {data['avg_squared_error']:.6f}\n\n")
    
    # Save detailed results for each N
    for N, data in results.items():
        output_file = os.path.join(output_dir, f'results_N{N}.txt')
        with open(output_file, 'w') as f:
            f.write(f"Results for N = {N}\n")
            f.write("Iteration,v_true,alpha_true,tau_true,v_est,alpha_est,tau_est,bias_v,bias_alpha,bias_tau,squared_error\n")
            
            for i in range(len(data['v_true'])):
                f.write(f"{i+1},{data['v_true'][i]:.6f},{data['alpha_true'][i]:.6f},{data['tau_true'][i]:.6f},")
                f.write(f"{data['v_est'][i]:.6f},{data['alpha_est'][i]:.6f},{data['tau_est'][i]:.6f},")
                f.write(f"{data['biases'][i][0]:.6f},{data['biases'][i][1]:.6f},{data['biases'][i][2]:.6f},")
                f.write(f"{data['squared_errors'][i]:.6f}\n")

def main():
    """
    Main function to run the simulation and save results.
    """
    # Set random seed for reproducibility
    seed = 42
    
    # Set output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    
    # Run simulation
    results = run_simulation(iterations=1000, N_values=[10, 40, 4000], seed=seed)
    
    # Save results
    save_results(results, output_dir)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()