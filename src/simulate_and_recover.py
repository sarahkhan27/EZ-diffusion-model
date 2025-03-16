# Acknowledging reference to and help from ChatGPT

import numpy as np
import math

def simulate_predicted_parameters(alpha, nu, tau):
    """ Simulate predicted parameters R_pred, M_pred, and V_pred """
    y = np.exp(-alpha * nu)
    R_pred = 1 / (y + 1)
    M_pred = tau + (alpha / (2 * nu)) * (1 - y / (y + 1))
    V_pred = (alpha / (2 * nu**3)) * (1 - 2 * alpha * nu * y - (y**2) / ((y + 1)**2))
    return R_pred, M_pred, V_pred

def simulate_observed_statistics(R_pred, M_pred, V_pred, N):
    """ Simulate observed statistics from the predicted parameters """
    # Simulate observed correct responses (Binomial distribution)
    R_obs = np.random.binomial(N, R_pred) / N

    # Simulate observed mean RT (Normal distribution)
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))

    # Simulate observed RT variance (Gamma distribution)
    V_obs = np.random.gamma(N - 1 / 2, 2 * V_pred / (N - 1))

    return R_obs, M_obs, V_obs

def estimate_parameters(R_obs, M_obs, V_obs):
    """ Estimate parameters from the observed data using the inverse EZ equations """
    L = np.log(R_obs / (1 - R_obs))

    # Estimate drift rate (nu)
    nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * ((R_obs**2) * L - R_obs * L + R_obs - 0.5) / V_obs)

    # Estimate boundary separation (alpha)
    alpha_est = L / nu_est

    # Estimate non-decision time (tau)
    tau_est = M_obs - (alpha_est / (2 * nu_est)) * (1 - np.exp(-nu_est * alpha_est) / (1 + np.exp(-nu_est * alpha_est)))

    return nu_est, alpha_est, tau_est

def compute_bias(true_params, estimated_params):
    """ Compute the bias between true and estimated parameters """
    return np.array(true_params) - np.array(estimated_params)

def compute_squared_error(true_params, estimated_params):
    """ Compute squared error between true and estimated parameters """
    return np.sum((np.array(true_params) - np.array(estimated_params)) ** 2)