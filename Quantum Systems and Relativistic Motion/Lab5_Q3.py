# Lab5_Q3.py
# Purpose:
#   - (a) Evaluates the integral (x**(-0.5)) / (1 + e**x) dx using the Mean Value Method.
#   - (b) Estimates the same integral using Importance Sampling to handle the singularity.
#   - (c) Generates histograms of the integral estimates for both methods.
#   - (d) Applies Importance Sampling with a normal distribution to evaluate a sharply peaked integral.
# Requires:
#   NumPy and Matplotlib.
# Output:
#   Mean integral estimates printed to the console and histograms visualizing the distribution of estimates.


import numpy as np
import matplotlib.pyplot as plt

### (a) Monte Carlo Integration Using the Mean Value Method

# Set the random seed for reproducibility
np.random.seed(123)

def f(x):
    """
    Function to integrate: f(x) = x^(-0.5) / (1 + exp(x)).
    Assumes x > 0 due to the x^(-0.5) term.
    """
    return x**(-0.5) / (1 + np.exp(x))

# Integration limits
a = 0.0  # Lower limit
b = 1.0  # Upper limit

# Number of sample points per estimate
N = 10000

# Number of repetitions for the Monte Carlo simulation
num_repeats = 1000

# List to store results of the mean value estimates
results_mean = []

# Variables to count out-of-domain samples (not used in this example but useful for debugging)
k = 0
k2 = 0

# Perform Monte Carlo integration
for i in range(num_repeats):
    # Generate uniform random samples between a and b
    x_samples = np.random.uniform(a, b, N)
    # Evaluate f(x) for the sampled points
    fx = f(x_samples)
    # Estimate the integral as the mean of f(x) scaled by the interval length
    estimate = (b - a) * np.mean(fx)
    # Append the result to the list
    results_mean.append(estimate)

# Calculate the mean of all the estimates
mean_result = np.mean(results_mean)

print(f"Mean of 1000 integral estimates (Mean Value Method): {mean_result:.6f}")

### (b) Monte Carlo Integration Using Importance Sampling

def g(x):
    """
    Function g(x) = f(x) / w(x), where w(x) is the importance sampling distribution.
    Here, w(x) is proportional to x^(-0.5), normalized over the interval [0, 1].
    """
    return 1 / (1 + np.exp(x))

# List to store results of the importance sampling estimates
results_importance = []

# Perform Monte Carlo integration with importance sampling
for i in range(num_repeats):
    # Generate uniform random samples between a and b, and transform them to follow the distribution x^2
    z = np.random.uniform(a, b, N)
    x_samples_b = z**2

    # Evaluate g(x) for the sampled points
    g_samples = g(x_samples_b)

    # Estimate the integral as the mean of g(x) scaled by the transformation factor
    estimate_b = 2 * np.mean(g_samples)  # Factor 2 accounts for the derivative of the transformation
    results_importance.append(estimate_b)

# Calculate the mean of all the estimates
mean_importance = np.mean(results_importance)

print(f"Mean of 1000 integral estimates (Importance Sampling): {mean_importance:.6f}")

### (c) Plot Histograms of the Estimates

# Histogram for the Mean Value Method
plt.figure(figsize=(10, 5))
plt.hist(results_mean, bins=100, alpha=0.7, color='blue')
plt.title('Histogram of Integral Estimates (Mean Value Method)')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Histogram for the Importance Sampling Method
plt.figure(figsize=(10, 5))
plt.hist(results_importance, bins=100, alpha=0.7, color='green')
plt.title('Histogram of Integral Estimates (Importance Sampling)')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

### (d) Monte Carlo Integration with Normal Distribution Sampling

def h(x):
    """
    Function to integrate: h(x) = exp(-2 * |x - 5|).
    The function is symmetric around x = 5 and decays exponentially with distance from 5.
    """
    return np.exp(-2 * np.abs(x - 5))

# List to store results of the integration
results_d = []

# Perform Monte Carlo integration using normal distribution sampling
for i in range(num_repeats):
    # Generate samples from a normal distribution centered at 5 with standard deviation 1
    x_d = np.random.normal(loc=5, scale=1, size=N)
    # Restrict samples to the interval [0, 10]
    x_d = x_d[(x_d >= 0) & (x_d <= 10)]

    # Evaluate h(x) and the weight function w(x)
    hx = h(x_d)
    wx = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_d - 5)**2)  # PDF of normal distribution

    # Calculate the integrand h(x) / w(x)
    integrand = hx / wx

    # Estimate the integral as the mean of the integrand
    estimate_d = np.mean(integrand)
    results_d.append(estimate_d)

# Calculate the mean of all the estimates
mean_result_d = np.mean(results_d)

print(f"Mean of 1000 integral estimates: {mean_result_d:.6f}")

# Plot histogram of the estimates
plt.figure(figsize=(10, 5))
plt.hist(results_d, bins=100, alpha=0.7, color='red')
plt.title('Histogram of Integral Estimates (Importance Sampling)')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
