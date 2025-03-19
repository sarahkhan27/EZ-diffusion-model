# Acknowledging reference to and help from ChatGPT

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from simulate_and_recover import simulate_predicted_parameters, simulate_observed_statistics, estimate_parameters, compute_bias, compute_squared_error

class TestEZDiffusionModel(unittest.TestCase):

    def test_simulate_predicted_parameters(self):
        # Example test case
        params = {'a': 1.5, 'v': 1.0, 't': 0.3}
        nu = 1.0  # Drift rate
        tau = 0.3  # Non-decision time
        predicted = simulate_predicted_parameters(params, nu, tau)
        self.assertEqual(len(predicted), 3)  # Should return 3 predicted values

    def test_simulate_observed_statistics(self):
        # Example test case
        params = {'a': 1.5, 'v': 1.0, 't': 0.3}
        V_pred = 0.4  # Predicted variance
        N = 100  # Number of trials
        observed = simulate_observed_statistics(params, V_pred, N)
        self.assertEqual(len(observed), 3)  # Should return 3 observed values

    def test_estimate_parameters(self):
        # Example test case
        observed_stats = {'R': 0.75, 'M': 0.5, 'V': 0.4}
        M_obs = 0.5  # Example observed mean RT
        V_obs = 0.4  # Example observed variance
        estimated = estimate_parameters(observed_stats, M_obs, V_obs)
        self.assertTrue('v' in estimated)  # Should return estimated drift rate 'v'
        self.assertTrue('a' in estimated)  # Should return estimated boundary separation 'a'
        self.assertTrue('t' in estimated)  # Should return estimated non-decision time 't'

    def test_compute_bias(self):
        # Example test case
        true_params = {'a': 1.5, 'v': 1.0, 't': 0.3}
        estimated_params = {'a': 1.4, 'v': 1.1, 't': 0.32}
        bias = compute_bias(true_params, estimated_params)
        self.assertEqual(len(bias), 3)  # Should return 3 bias values
        self.assertAlmostEqual(bias['a'], 0.1, places=5)  # Example bias check for 'a'
        self.assertAlmostEqual(bias['v'], -0.1, places=5)  # Example bias check for 'v'
        self.assertAlmostEqual(bias['t'], -0.02, places=5)  # Example bias check for 't'

    def test_compute_squared_error(self):
        # Example test case
        true_params = {'a': 1.5, 'v': 1.0, 't': 0.3}
        estimated_params = {'a': 1.4, 'v': 1.1, 't': 0.32}
        squared_error = compute_squared_error(true_params, estimated_params)
        self.assertEqual(len(squared_error), 3)  # Should return 3 squared error values
        self.assertAlmostEqual(squared_error['a'], 0.01, places=5)  # Example squared error check for 'a'
        self.assertAlmostEqual(squared_error['v'], 0.01, places=5)  # Example squared error check for 'v'
        self.assertAlmostEqual(squared_error['t'], 0.0004, places=5)  # Example squared error check for 't'

if __name__ == '__main__':
    unittest.main()