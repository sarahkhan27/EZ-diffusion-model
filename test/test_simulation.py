# Acknowledging reference to and help from ChatGPT/AI tools

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from simulate_and_recover import forward_ez, inverse_ez, simulate_observed_stats, simulate_and_recover

class TestEZDiffusion(unittest.TestCase):
    def test_forward_ez(self):
        """Test forward EZ equations with some known values"""
        v, alpha, tau = 1.0, 1.0, 0.3
        R_pred, M_pred, V_pred = forward_ez(v, alpha, tau)
        
        # Check if the output is within reasonable bounds
        self.assertTrue(0 < R_pred < 1)
        self.assertTrue(tau < M_pred)
        self.assertTrue(V_pred > 0)
        
        # Check if accuracy is 0.5 when drift rate is 0
        R_pred_zero, _, _ = forward_ez(0.001, 1.0, 0.3)  # Using very small v instead of 0
        self.assertAlmostEqual(R_pred_zero, 0.5, places=2)
    
    def test_inverse_ez(self):
        """Test inverse EZ equations by checking if we can recover known parameters"""
        # Start with some true parameters
        v_true, alpha_true, tau_true = 1.0, 1.0, 0.3
        
        # Get predicted summary statistics
        R_pred, M_pred, V_pred = forward_ez(v_true, alpha_true, tau_true)
        
        # Recover parameters
        v_est, alpha_est, tau_est = inverse_ez(R_pred, M_pred, V_pred)
        
        # Check if recovered parameters are close to true parameters
        self.assertAlmostEqual(v_est, v_true, places=4)
        self.assertAlmostEqual(alpha_est, alpha_true, places=4)
        self.assertAlmostEqual(tau_est, tau_true, places=4)
    
    def test_parameter_recovery(self):
        """Test parameter recovery with different parameter values"""
        test_params = [
            (0.5, 0.5, 0.1),
            (1.0, 1.0, 0.2),
            (1.5, 1.5, 0.3),
            (2.0, 2.0, 0.4)
        ]
        
        for v_true, alpha_true, tau_true in test_params:
            # Get predicted summary statistics
            R_pred, M_pred, V_pred = forward_ez(v_true, alpha_true, tau_true)
            
            # Recover parameters
            v_est, alpha_est, tau_est = inverse_ez(R_pred, M_pred, V_pred)
            
            # Check if recovered parameters are close to true parameters
            self.assertAlmostEqual(v_est, v_true, places=4)
            self.assertAlmostEqual(alpha_est, alpha_true, places=4)
            self.assertAlmostEqual(tau_est, tau_true, places=4)
            
    def test_edge_cases(self):
        """Test edge cases for EZ diffusion model"""
        # Test with very high drift rate (should result in near-perfect accuracy)
        v_high, alpha, tau = 10.0, 1.0, 0.3
        R_pred, M_pred, V_pred = forward_ez(v_high, alpha, tau)
        self.assertGreater(R_pred, 0.99)  # Accuracy should be very high

        # Test with drift rate near zero (should result in accuracy near 0.5)
        v_zero, alpha, tau = 0.001, 1.0, 0.3
        R_pred, M_pred, V_pred = forward_ez(v_zero, alpha, tau)
        self.assertAlmostEqual(R_pred, 0.5, places=2)  # Accuracy should be close to 0.5

        # Test with very large boundary separation (should result in longer RT)
        v, alpha_high, tau = 1.0, 5.0, 0.3
        _, M_pred_high, _ = forward_ez(v, alpha_high, tau)
        _, M_pred_normal, _ = forward_ez(v, 1.0, tau)
        self.assertGreater(M_pred_high, M_pred_normal)  # RT should be longer with higher boundary

        # Test with very small boundary separation (should result in shorter RT)
        v, alpha_low, tau = 1.0, 0.1, 0.3
        _, M_pred_low, _ = forward_ez(v, alpha_low, tau)
        self.assertLess(M_pred_low, M_pred_normal)  # RT should be shorter with lower boundary

        # Test preservation of non-decision time
        v, alpha, tau_high = 1.0, 1.0, 0.5
        _, M_pred_high_tau, _ = forward_ez(v, alpha, tau_high)
        _, M_pred_normal_tau, _ = forward_ez(v, alpha, 0.3)
        self.assertAlmostEqual(M_pred_high_tau - M_pred_normal_tau, 0.2, places=4)  # Difference in RT should equal difference in tau
    
    def test_complete_simulation_workflow(self):
        """Test the complete simulate-and-recover process with various sample sizes"""
        # Test parameters
        v, alpha, tau = 1.0, 1.0, 0.3
        
        # Test different sample sizes
        for N in [100, 1000]:
            v_est, alpha_est, tau_est, bias, squared_error = simulate_and_recover(v, alpha, tau, N)
            
            # Check if parameters are valid (not NaN or inf)
            self.assertFalse(np.isnan(v_est) or np.isinf(v_est))
            self.assertFalse(np.isnan(alpha_est) or np.isinf(alpha_est))
            self.assertFalse(np.isnan(tau_est) or np.isinf(tau_est))
            
            # Check if bias is calculated correctly
            expected_bias = np.array([v, alpha, tau]) - np.array([v_est, alpha_est, tau_est])
            np.testing.assert_array_almost_equal(bias, expected_bias)
            
            # Check if squared error is calculated correctly
            expected_squared_error = np.sum((np.array([v, alpha, tau]) - np.array([v_est, alpha_est, tau_est]))**2)
            self.assertAlmostEqual(squared_error, expected_squared_error, places=10)
            
            # For larger N, parameters should be recovered more accurately
            if N == 1000:
                self.assertLess(abs(v - v_est), 0.5)
                self.assertLess(abs(alpha - alpha_est), 0.5)
                self.assertLess(abs(tau - tau_est), 0.2)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values"""
        # Test cases that might cause numerical issues
        test_cases = [
            # Very small drift rate
            (0.001, 1.0, 0.3),
            # Very large drift rate
            (10.0, 1.0, 0.3),
            # Very small boundary
            (1.0, 0.01, 0.3),
            # Very large boundary
            (1.0, 10.0, 0.3),
            # Very small non-decision time
            (1.0, 1.0, 0.01),
            # Large non-decision time
            (1.0, 1.0, 0.9)
        ]
        
        for v, alpha, tau in test_cases:
            try:
                # Forward equations should not raise exceptions
                R_pred, M_pred, V_pred = forward_ez(v, alpha, tau)
                
                # Results should be finite
                self.assertTrue(np.isfinite(R_pred) and np.isfinite(M_pred) and np.isfinite(V_pred))
                
                # If values are valid, inverse should work too
                if 0 < R_pred < 1 and M_pred > 0 and V_pred > 0:
                    v_est, alpha_est, tau_est = inverse_ez(R_pred, M_pred, V_pred)
                    self.assertTrue(np.isfinite(v_est) and np.isfinite(alpha_est) and np.isfinite(tau_est))
            except Exception as e:
                self.fail(f"Failed with parameters v={v}, alpha={alpha}, tau={tau}: {e}")
    def test_bias_decreases_with_increasing_N(self):
        """Test that bias decreases as sample size N increases"""
        # Set true parameters
        v, alpha, tau = 1.0, 1.0, 0.3
        
        # Test with different sample sizes
        N_values = [10, 100, 1000]
        squared_errors = []
        
        np.random.seed(42)  # For reproducibility
        
        for N in N_values:
            total_squared_error = 0
            iterations = 50  # Reduced iterations for test speed
            
            for _ in range(iterations):
                _, _, _, _, squared_error = simulate_and_recover(v, alpha, tau, N)
                total_squared_error += squared_error
            
            avg_squared_error = total_squared_error / iterations
            squared_errors.append(avg_squared_error)
        
        # Verify that squared error decreases as N increases
        self.assertGreater(squared_errors[0], squared_errors[1])
        self.assertGreater(squared_errors[1], squared_errors[2])

    def test_simulate_observed_stats(self):
        """Test the simulation of observed summary statistics"""
        # Set predicted values
        R_pred = 0.75
        M_pred = 0.7
        V_pred = 0.05
        
        # Test with different sample sizes
        N_values = [10, 100, 1000]
        
        np.random.seed(42)  # For reproducibility
        
        for N in N_values:
            # Run multiple simulations to check distributions
            R_samples = []
            M_samples = []
            V_samples = []
            
            iterations = 100  # Number of simulations to check distribution
            
            for _ in range(iterations):
                R_obs, M_obs, V_obs = simulate_observed_stats(R_pred, M_pred, V_pred, N)
                R_samples.append(R_obs)
                M_samples.append(M_obs)
                V_samples.append(V_obs)
            
            # Check if mean of observed values is close to predicted values
            self.assertAlmostEqual(np.mean(R_samples), R_pred, places=1)
            self.assertAlmostEqual(np.mean(M_samples), M_pred, places=1)
            self.assertAlmostEqual(np.mean(V_samples), V_pred, places=1)
            
            # Check if variance decreases with increasing N (Central Limit Theorem)
            if N > 10:  # Skip the smallest N
                # Variance of R_obs should be approximately R_pred * (1 - R_pred) / N
                expected_var_R = R_pred * (1 - R_pred) / N
                self.assertAlmostEqual(np.var(R_samples), expected_var_R, places=2, 
                                    msg=f"R variance for N={N} doesn't match expected")
                
                # Variance of M_obs should be approximately V_pred / N
                expected_var_M = V_pred / N
                self.assertLess(abs(np.var(M_samples) - expected_var_M) / expected_var_M, 0.5, 
                            msg=f"M variance for N={N} is too far from expected")

if __name__ == '__main__':
    unittest.main()