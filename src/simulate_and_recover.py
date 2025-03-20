import sys
import os
import unittest
import numpy as np
import random
import scipy.stats as stats

# Add this function and its dependencies to your script
def forward_ez(v, alpha, tau):
    """
    Forward EZ diffusion model equations.
    
    Parameters:
    v (float): Drift rate
    alpha (float): Boundary separation
    tau (float): Non-decision time
    
    Returns:
    tuple: (R_pred, M_pred, V_pred) - predicted accuracy, mean RT, and variance of RT
    """
    # Substitute parameter names (v, alpha, tau) for (v, a, t) in the equations
    y = np.exp(-alpha * v)
    
    # Equation 1: Predicted accuracy
    R_pred = 1 / (1 + y)
    
    # Equation 2: Predicted mean RT
    M_pred = tau + (alpha / (2 * v)) * ((1 - y) / (1 + y))
    
    # Equation 3: Predicted variance of RT
    V_pred = (alpha / (2 * v**3)) * ((1 - 2*alpha*v*y - y**2) / (1 + y)**2)
    
    return R_pred, M_pred, V_pred

def inverse_ez(R_obs, M_obs, V_obs):
    """
    Inverse EZ diffusion model equations.
    
    Parameters:
    R_obs (float): Observed accuracy
    M_obs (float): Observed mean RT
    V_obs (float): Observed variance of RT
    
    Returns:
    tuple: (v_est, alpha_est, tau_est) - estimated drift rate, boundary separation, and non-decision time
    """
    # Intermediate calculation L
    L = np.log(R_obs / (1 - R_obs))
    
    # Equation 4: Estimated drift rate
    v_est = np.sign(R_obs - 0.5) * (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)**(1/4)
    
    # Equation 5: Estimated boundary separation
    alpha_est = L / v_est
    
    # Equation 6: Estimated non-decision time
    y_est = np.exp(-v_est * alpha_est)
    tau_est = M_obs - (alpha_est / (2 * v_est)) * ((1 - y_est) / (1 + y_est))
    
    return v_est, alpha_est, tau_est

def simulate_observed_stats(R_pred, M_pred, V_pred, N):
    """
    Simulate observed summary statistics from predicted ones.
    
    Parameters:
    R_pred (float): Predicted accuracy rate
    M_pred (float): Predicted mean RT
    V_pred (float): Predicted variance of RT
    N (int): Sample size
    
    Returns:
    tuple: (R_obs, M_obs, V_obs) - observed accuracy, mean RT, and variance of RT
    """
    # Equation 7: Simulating observed number of correct trials
    T_obs = stats.binom.rvs(n=N, p=R_pred)
    R_obs = T_obs / N
    
    # Equation 8: Simulating observed mean RT
    M_obs = stats.norm.rvs(loc=M_pred, scale=np.sqrt(V_pred / N))
    
    # Equation 9: Simulating observed variance of RT
    V_obs = stats.gamma.rvs(a=(N-1)/2, scale=2*V_pred/(N-1))
    
    return R_obs, M_obs, V_obs

def simulate_and_recover(v, alpha, tau, N):
    """
    Simulate data from true parameters and recover the parameters.
    
    Parameters:
    v (float): True drift rate
    alpha (float): True boundary separation
    tau (float): True non-decision time
    N (int): Sample size
    
    Returns:
    tuple: (v_est, alpha_est, tau_est, bias, squared_error) - estimated parameters and error metrics
    """
    # True parameters as tuple
    true_params = (v, alpha, tau)
    
    # Generate predicted summary statistics
    R_pred, M_pred, V_pred = forward_ez(v, alpha, tau)
    
    # Simulate observed summary statistics
    R_obs, M_obs, V_obs = simulate_observed_stats(R_pred, M_pred, V_pred, N)
    
    # In case of extreme values, cap R_obs away from 0 and 1
    R_obs = max(0.001, min(0.999, R_obs))
    
    # Recover parameters from observed statistics
    v_est, alpha_est, tau_est = inverse_ez(R_obs, M_obs, V_obs)
    est_params = (v_est, alpha_est, tau_est)
    
    # Calculate bias and squared error
    bias = np.array(true_params) - np.array(est_params)
    squared_error = np.sum(bias**2)
    
    return v_est, alpha_est, tau_est, bias, squared_error