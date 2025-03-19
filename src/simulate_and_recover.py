# Acknowledging reference to and help from ChatGPT

import numpy as np

def simulate_predicted_parameters(params, nu, tau):
    """
    Simulate predicted parameters R, M, V from given true parameters.
    :param params: Dictionary with 'a', 'v', 't' for boundary separation, drift rate, nondecision time.
    :param nu: Drift rate.
    :param tau: Non-decision time.
    :return: List of predicted values [R_pred, M_pred, V_pred].
    """
    R_pred = 1 / (nu + 1)  # Predicted accuracy
    M_pred = tau + (params['a'] / (2 * nu)) * (1 - params['v'] / (1 + params['v']))  # Predicted mean RT
    V_pred = (params['a'] / (2 * nu**3)) * (1 - 2 * params['a'] * params['v'] * nu - (params['v']**2) / (params['v'] + 1)**2)  # Predicted variance
    return [R_pred, M_pred, V_pred]

def simulate_observed_statistics(params, V_pred, N):
    """
    Simulate observed statistics from predicted parameters.
    :param params: Dictionary with 'a', 'v', 't'.
    :param V_pred: Predicted variance.
    :param N: Number of trials.
    :return: List of observed values [R_obs, M_obs, V_obs].
    """
    R_obs = np.random.binomial(N, V_pred) / N  # observed correct trials
    M_obs = np.random.normal(params['t'], V_pred / N)  # observed mean RT
    V_obs = np.random.gamma(N - 1/2, 2 * V_pred / (N - 1))  # observed variance
    return [R_obs, M_obs, V_obs]

def estimate_parameters(observed_stats, M_obs, V_obs):
    """
    Estimate the parameters based on observed statistics.
    :param observed_stats: Dictionary with 'R', 'M', 'V' for observed statistics.
    :param M_obs: Observed mean RT.
    :param V_obs: Observed variance.
    :return: Dictionary with estimated parameters {'v', 'a', 't'}.
    """
    R_obs = observed_stats['R']
    alpha_est = 1.5  # Example estimation of boundary separation
    nu_est = 1.0  # Example estimation of drift rate
    tau_est = M_obs - (alpha_est / (2 * nu_est)) * (1 - np.exp(-nu_est * alpha_est) / (1 + np.exp(-nu_est * alpha_est)))
    return {'v': nu_est, 'a': alpha_est, 't': tau_est}

def compute_bias(true_params, estimated_params):
    """
    Calculate the bias for each parameter (difference between true and estimated values).
    :param true_params: Dictionary with true parameter values.
    :param estimated_params: Dictionary with estimated parameter values.
    :return: Dictionary with bias for each parameter.
    """
    bias = {key: true_params[key] - estimated_params[key] for key in true_params}
    return bias

def compute_squared_error(true_params, estimated_params):
    """
    Calculate squared error for each parameter (squared difference between true and estimated values).
    :param true_params: Dictionary with true parameter values.
    :param estimated_params: Dictionary with estimated parameter values.
    :return: Dictionary with squared error for each parameter.
    """
    squared_error = {key: (true_params[key] - estimated_params[key])**2 for key in true_params}
    return squared_error