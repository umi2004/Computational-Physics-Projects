"""
Purpose: This script calculates the period of a relativistic particle on a spring, showing the transition between classical and relativistic cases.
- Part (a) calculates the period for N = 8 and N = 16 using Gaussian quadrature and estimates the fractional error.
- Part (b) plots the integrand and weighted values for the period calculation.
- Part (c) plots the period T as a function of initial position x0 and compares it with the classical and relativistic limits.

External Functions: Requires 'gaussxw' for Gaussian quadrature integration.
Outputs:
- Part (a): Calculated period values and estimated fractional error printed to the console.
- Part (b): Plots of integrand and weighted values saved as '8_Integrand.png', '8_Weighted.png', '16_Integrand.png', and '16_Weighted.png'.
- Part (c): A plot of the period as a function of x0 saved as 'Period_Func.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

# Gaussian Quadrature weights and points
def gaussxw(N):
    a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x ** 2)
        dx = p1 / dp
        x -= dx
        delta = np.max(np.abs(dx))
    w = 2 * (N + 1) ** 2 / (N ** 2 * (1 - x ** 2) * dp ** 2)
    return x, w

# Gaussian Quadrature transformation to the interval [a, b]
def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * w
    return x_mapped, w_mapped

# Function to calculate the integrand for g
def g(x, x0, m, c, k):
    numerator = k * (x0 ** 2 - x ** 2) * (2 * m * c ** 2 + k * (x0 ** 2 - x ** 2) / 2)
    denominator = 2 * (m * c ** 2 + k * (x0 ** 2 - x ** 2) / 2) ** 2
    return c * np.sqrt(numerator / denominator)

# Function to compute the period T using Gaussian Quadrature
def compute_T(N, x0, m, c, k):
    x, w = gaussxwab(N, 0.0, x0)
    integrand = 1.0 / g(x, x0, m, c, k)
    integral = np.sum(w * integrand)
    T = 4 * integral
    return T

# Plot the integrand for a given N
def plot_integrand(N, x0, m, c, k):
    x, w = gaussxwab(N, 0.0, x0)
    integrand = 4.0 / g(x, x0, m, c, k)
    plt.plot(x, integrand, 'o-', label='Integrand $4/g(x_k)$')
    plt.title(f'Integrand Values for N = {N}')
    plt.xlabel('$x_k$ (m)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{N}_Integrand.png")
    print(f"Have been saved: {N}_Integrand.png \n")
    plt.cla()

# Plot the weighted values for a given N
def plot_weighted(N, x0, m, c, k):
    x, w = gaussxwab(N, 0.0, x0)
    integrand = 4.0 / g(x, x0, m, c, k)
    weighted_values = integrand * w
    plt.plot(x, weighted_values, 's-', label='Weighted $4w_k/g(x_k)$')
    plt.title(f'Weighted Values for N = {N}')
    plt.xlabel('$x_k$ (m)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{N}_Weighted.png")
    print(f"Have been saved: {N}_Weighted.png \n")
    plt.cla()

# Compute T as a function of x0
def compute_T_vs_x0(x0_values, m, c, k, N=100):
    T_values = []
    for x0 in x0_values:
        T = compute_T(N, x0, m, c, k)
        T_values.append(T)
    return np.array(T_values)

# Main function to run all parts of the code
def main():
    # Parameters
    m = 1.0        # kg
    c = 3.0e8      # m/s
    k = 12.0       # N/m
    x0 = 0.01      # m (1 cm)

    # Part (a): Calculate T for N = 8 and N = 16
    print("Q2 - Part (a)")
    T_N8 = compute_T(8, x0, m, c, k)
    T_N16 = compute_T(16, x0, m, c, k)
    fractional_error = abs(T_N16 - T_N8) / T_N16
    print(f"T (N=8): {T_N8:.10e} seconds")
    print(f"T (N=16): {T_N16:.10e} seconds")
    print(f"Estimated fractional error: {fractional_error:.10e}\n")

    # Part (b): Plot integrand and weighted values for N = 8 and N = 16
    print("Q2 - Part (b)")
    plot_integrand(8, x0, m, c, k)
    plot_weighted(8, x0, m, c, k)
    plot_integrand(16, x0, m, c, k)
    plot_weighted(16, x0, m, c, k)

    # Part (c): Compute T as a function of x0
    print("Q2 - Part (c)")
    x_c = c * np.sqrt(m / k)  # Characteristic length
    x0_values = np.linspace(1.0, 10 * x_c, 100)
    T_values = compute_T_vs_x0(x0_values, m, c, k, N=100)
    T_classical = 2 * np.pi * np.sqrt(m / k)
    T_relativistic = 4 * x0_values / c
    plt.plot(x0_values, T_values, label='Computed Period $T$')
    plt.axhline(T_classical, color='r', linestyle='--', label='Classical Period')
    plt.plot(x0_values, T_relativistic, 'k--', label='Relativistic Limit $T = 4 x_0 / c$')
    plt.title('Period $T$ as a Function of Amplitude $x_0$')
    plt.xlabel('Amplitude $x_0$ (m)')
    plt.ylabel('Period $T$ (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Period_Func.png")
    print(f"Have been saved: Period_Func.png \n")

if __name__ == "__main__":
    main()