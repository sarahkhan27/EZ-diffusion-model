# Acknowledging reference to and help from ChatGPT

import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from simulate_and_recover import simulate_predicted_parameters, simulate_observed_statistics, estimate_parameters, compute_bias, compute_squared_error

class TestEZDiffusionModel(unittest.TestCase):

    def test_simulate_predicted_parameters(self):
        alpha, nu, tau = 1.0, 1.0, 0.3
        R_pred, M_pred, V_pred = simulate_predicted_parameters(alpha, nu, tau)
        
        # Check if the values returned are floats (you can also add a more specific check here)
        self.assertIsInstance(R_pred, float)
        self.assertIsInstance(M_pred, float)
        self.assertIsInstance(V_pred, float)

    def test_simulate_observed_statistics(self):
        R_pred, M_pred, V_pred = 0.7, 0.4, 0.5
        N = 40
        R_obs, M_obs, V_obs = simulate_observed_statistics(R_pred, M_pred, V_pred, N)
        
        # Check if the observed statistics are floats
        self.assertIsInstance(R_obs, float)
        self.assertIsInstance(M_obs, float)
        self.assertIsInstance(V_obs, float)

    def test_estimate_parameters(self):
        R_obs, M_obs, V_obs = 0.7, 0.4, 0.5
        nu_est, alpha_est, tau_est = estimate_parameters(R_obs, M_obs, V_obs)
        
        # Check if the estimated parameters are floats
        self.assertIsInstance(nu_est, float)
        self.assertIsInstance(alpha_est, float)
        self.assertIsInstance(tau_est, float)

    def test_compute_bias(self):
        true_params = [1.0, 1.0, 0.3]
        estimated_params = [1.1, 1.1, 0.3]
        bias = compute_bias(true_params, estimated_params)
        
        # Check if bias is a numpy array and has 3 elements
        self.assertIsInstance(bias, np.ndarray)
        self.assertEqual(len(bias), 3)

    def test_compute_squared_error(self):
        true_params = [1.0, 1.0, 0.3]
        estimated_params = [1.1, 1.1, 0.3]
        squared_error = compute_squared_error(true_params, estimated_params)
        
        # Check if squared_error is a float
        self.assertIsInstance(squared_error, float)

if __name__ == '__main__':
    unittest.main()