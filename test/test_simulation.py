import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from simulate_and_recover import forward_ez, inverse_ez

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

if __name__ == '__main__':
    unittest.main()