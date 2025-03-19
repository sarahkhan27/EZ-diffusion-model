import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from simulate_and_recover import generate_predicted_statistics, generate_observed_statistics, recover_parameters, calculate_bias_and_squared_error

class TestSimulation(unittest.TestCase):
    
    def test_generate_predicted_statistics(self):
        # Set the true parameters (alpha, v, tau)
        true_params = (1.0, 1.0, 0.2)
        alpha, v, tau = true_params
        
        # Generate predicted statistics
        R_pred, M_pred, V_pred = generate_predicted_statistics(alpha, v, tau)

        # Debugging: Print the results of the predicted statistics
        print(f"Test: Generating Predicted Statistics")
        print(f"alpha: {alpha}, v: {v}, tau: {tau}")
        print(f"R_pred: {R_pred}, M_pred: {M_pred}, V_pred: {V_pred}")

        # Assert that R_pred is close to 0.5
        self.assertAlmostEqual(R_pred, 0.5, delta=0.1)
    
    def test_generate_observed_statistics(self):
        # Set the true parameters (alpha, v, tau)
        true_params = (1.0, 1.0, 0.2)
        alpha, v, tau = true_params
        
        # Generate predicted statistics
        R_pred, M_pred, V_pred = generate_predicted_statistics(alpha, v, tau)
        
        # Set sample size
        N = 10
        
        # Generate observed statistics
        R_obs, M_obs, V_obs = generate_observed_statistics(R_pred, M_pred, V_pred, N)

        # Debugging: Print the results of the observed statistics
        print(f"Test: Generating Observed Statistics")
        print(f"R_pred: {R_pred}, M_pred: {M_pred}, V_pred: {V_pred}, N: {N}")
        print(f"R_obs: {R_obs}, M_obs: {M_obs}, V_obs: {V_obs}")

        # Assert that R_obs, M_obs, and V_obs are within reasonable bounds
        self.assertTrue(0 <= R_obs <= 1)
        self.assertTrue(M_obs >= 0)
        self.assertTrue(V_obs >= 0)

    def test_recover_parameters(self):
        # Set the true parameters (alpha, v, tau)
        true_params = (1.0, 1.0, 0.2)
        alpha, v, tau = true_params
        
        # Generate predicted statistics
        R_pred, M_pred, V_pred = generate_predicted_statistics(alpha, v, tau)
        
        # Set sample size
        N = 10
        
        # Generate observed statistics
        R_obs, M_obs, V_obs = generate_observed_statistics(R_pred, M_pred, V_pred, N)
        
        # Recover parameters from observed data
        v_est, alpha_est, tau_est = recover_parameters(R_obs, M_obs, V_obs)
        
        # Debugging: Print the recovered parameters and compare to true values
        print(f"Test: Recovering Parameters")
        print(f"True parameters: (alpha: {alpha}, v: {v}, tau: {tau})")
        print(f"Recovered parameters: (v_est: {v_est}, alpha_est: {alpha_est}, tau_est: {tau_est})")

        # Assert that recovered v_est is close to true v
        self.assertAlmostEqual(v_est, v, delta=0.1)
    
    def test_calculate_bias_and_squared_error(self):
        # Set the true parameters (alpha, v, tau)
        true_params = (1.0, 1.0, 0.2)
        alpha, v, tau = true_params
        
        # Generate predicted statistics
        R_pred, M_pred, V_pred = generate_predicted_statistics(alpha, v, tau)
        
        # Set sample size
        N = 10
        
        # Generate observed statistics
        R_obs, M_obs, V_obs = generate_observed_statistics(R_pred, M_pred, V_pred, N)
        
        # Recover parameters from observed data
        v_est, alpha_est, tau_est = recover_parameters(R_obs, M_obs, V_obs)
        
        # Calculate bias and squared error
        bias, squared_error = calculate_bias_and_squared_error(true_params, (v_est, alpha_est, tau_est))

        # Debugging: Print the bias and squared error
        print(f"Test: Calculating Bias and Squared Error")
        print(f"True parameters: {true_params}")
        print(f"Estimated parameters: {(v_est, alpha_est, tau_est)}")
        print(f"Bias: {bias}, Squared Error: {squared_error}")

        # Assert that the bias for each parameter is less than 0.1
        self.assertLess(abs(bias[0]), 0.1)  # Check bias for v
        self.assertLess(abs(bias[1]), 0.1)  # Check bias for alpha
        self.assertLess(abs(bias[2]), 0.1)  # Check bias for tau

        # Assert that squared error is less than a threshold
        self.assertLess(squared_error, 1.0)

    def test_large_sample_size(self):
        # Set the true parameters (alpha, v, tau)
        true_params = (1.0, 1.0, 0.2)
        alpha, v, tau = true_params
        
        # Set large sample size
        N = 4000
        
        # Simulate and recover parameters for large sample size
        bias, squared_error = simulate_and_recover(true_params, N)
        
        # Debugging: Print bias and squared error for large sample size
        print(f"Test: Large Sample Size (N={N})")
        print(f"Bias (first few): {bias[:5]}")  # Print first 5 for example
        print(f"Squared Error (first few): {squared_error[:5]}")

        # Assert that v_est is close to true v
        self.assertAlmostEqual(bias[0], 0, delta=0.1)  # Check bias for v
        self.assertAlmostEqual(bias[1], 0, delta=0.1)  # Check bias for alpha
        self.assertAlmostEqual(bias[2], 0, delta=0.1)  # Check bias for tau


if __name__ == "__main__":
    unittest.main()