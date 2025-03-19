import numpy as np

# Function to simulate predicted statistics (R_pred, M_pred, V_pred)
def generate_predicted_statistics(alpha, v, tau):
    y = np.exp(-alpha * v)
    R_pred = 1 / (y + 1)
    M_pred = tau + (alpha / (2 * v)) * (1 - y / (1 + y))
    V_pred = (alpha / (2 * v ** 3)) * (1 - 2 * alpha * v * y - (y ** 2) / (y + 1) ** 2)
    return R_pred, M_pred, V_pred

# Function to simulate observed statistics (R_obs, M_obs, V_obs)
def generate_observed_statistics(R_pred, M_pred, V_pred, N):
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = np.random.gamma(N - 1 / 2, 2 * V_pred / (N - 1))
    return R_obs, M_obs, V_obs

# Function to recover estimated parameters (v_est, alpha_est, tau_est)
def recover_parameters(R_obs, M_obs, V_obs):
    # Avoid division by zero by adding a small epsilon when R_obs is 1
    epsilon = 1e-10
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)  # Ensure R_obs is within (0, 1)
    
    L = np.log(R_obs / (1 - R_obs))  # Log of the odds ratio

    # Safeguard against division by zero or very small v_est
    v_est = np.sign(R_obs - 1 / 2) * 4 * np.sqrt(L * ((R_obs ** 2) * L - (R_obs) * L + R_obs - 1 / 2) / V_obs)
    
    # Add a small epsilon to v_est if it is too close to zero to avoid division by zero
    v_est = np.where(np.abs(v_est) < epsilon, epsilon, v_est)

    alpha_est = L / v_est

    # Safeguard for tau_est if alpha_est or v_est is very small
    tau_est = M_obs - (alpha_est / (2 * v_est)) * (1 - np.exp(-v_est * alpha_est) / (1 + np.exp(-v_est * alpha_est)))

    return v_est, alpha_est, tau_est

# Function to calculate bias and squared error
def calculate_bias_and_squared_error(true_params, estimated_params):
    bias = np.array(true_params) - np.array(estimated_params)
    squared_error = np.sum(bias ** 2)
    return bias, squared_error

# Simulate-and-recover process
def simulate_and_recover(true_params, N, iterations=1000):
    bias_all = np.zeros((iterations, 3))  # To store bias for each parameter
    squared_error_all = np.zeros(iterations)  # To store squared error for each iteration

    for i in range(iterations):
        # Generate predicted statistics
        R_pred, M_pred, V_pred = generate_predicted_statistics(*true_params)

        # Generate observed statistics
        R_obs, M_obs, V_obs = generate_observed_statistics(R_pred, M_pred, V_pred, N)

        # Recover parameters from observed data
        estimated_params = recover_parameters(R_obs, M_obs, V_obs)

        # Calculate bias and squared error
        bias, squared_error = calculate_bias_and_squared_error(true_params, estimated_params)

        bias_all[i] = bias
        squared_error_all[i] = squared_error

    return bias_all, squared_error_all

# Main function to run the simulation and recovery for all sample sizes
def main():
    # True parameters (within specified range)
    true_params = [np.random.uniform(0.5, 2),  # alpha
                   np.random.uniform(0.5, 2),  # v (drift rate)
                   np.random.uniform(0.1, 0.5)]  # tau (non-decision time)

    # Sample sizes to test
    sample_sizes = [10, 40, 4000]
    
    # Results container
    all_bias = {}
    all_squared_error = {}

    # Run simulations for each sample size
    for N in sample_sizes:
        bias, squared_error = simulate_and_recover(true_params, N)
        all_bias[N] = bias
        all_squared_error[N] = squared_error

    # Output results for analysis
    for N in sample_sizes:
        print(f"Sample size: {N}")
        print(f"Average bias: {np.mean(all_bias[N], axis=0)}")
        print(f"Average squared error: {np.mean(all_squared_error[N])}")
        print()

if __name__ == "__main__":
    main()