"""
Purpose: This script uses numerical differentiation to estimate the derivative of a given function and compares the results with analytical derivatives.
- Part (a) calculates the numerical derivative of f(x) = exp(-x^2) at x = 0.5 using central difference approximation for varying values of h.
- Part (b) calculates the analytical derivative and compares it with numerical results to estimate the relative error.
- Part (c) repeats Part (a) and (b) using the forward difference approximation.
- Part (d) plots the relative error of the numerical derivatives for both central and forward difference methods.
- Part (f) calculates the first 5 derivatives of g(x) = exp(2x) at x = 0 using central difference approximation.

External Functions: None required.
Outputs:
- Parts (a), (b), (c), and (f): Numerical and analytical derivative values, and relative errors printed to the console.
- Part (d): A plot of relative error vs h for both methods saved as 'Period_Func.png'.
"""


import numpy as np
import matplotlib.pyplot as plt

# define function f(x) here
def f(x):
    return np.exp(-x ** 2)

# calculate the m-th derivative of f at x using recursion.
def delta(f, x, m, h):
    if m > 1:
        return (delta(f, x + h/2, m - 1, h) - delta(f, x - h/2, m - 1, h)) / h
    else:
        return (f(x + h/2) - f(x - h/2)) / h

# Function to evaluate f(x)
def f(x):
    return np.exp(-x ** 2)

# Function to calculate the central difference derivative
def derivative_cent(x, h):
    return (f(x + h / 2) - f(x - h / 2)) / h

# Function to calculate the forward difference derivative
def derivative_forward(x, h):
    return (f(x + h) - f(x)) / h

# Function to calculate higher-order central difference derivatives
def central_difference(f, x, h, n):
    if n == 0:
        return f(x)
    elif n == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        return delta(f, x, n, h)

def g(x):
    return np.exp(2 * x)

# Main function to run all parts of the code
def main():
    # Part (a): Calculate numerical derivatives using central difference
    print("Q3 - Part (a)")
    h_val = np.logspace(-16, 0, num=17)
    x0 = 0.5
    num_derivatives = derivative_cent(x0, h_val)
    print(f"Numerical derivative for Central difference at x={x0}: {num_derivatives}")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    # Part (b): Analytical derivative and relative errors
    print("Q3 - Part (b)")
    ana_derivatives = -2 * x0 * np.exp(-x0 ** 2)
    print(f"Analytical derivative at x={x0}: {ana_derivatives}")
    relative_errors = np.abs(num_derivatives - ana_derivatives) / np.abs(ana_derivatives)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(relative_errors)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"Minimum relative error for Central difference: {min(relative_errors)}\n")

    # Part (c): Calculate numerical derivatives using forward difference
    print("Q3 - Part (c)")
    num_derivatives_forward = derivative_forward(x0, h_val)
    print(f"Numerical derivative for Forward difference at x={x0}: {num_derivatives_forward}")
    relative_errors_forward = np.abs(num_derivatives_forward - ana_derivatives) / np.abs(ana_derivatives)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(relative_errors_forward)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"Minimum relative error for Forward difference: {min(relative_errors_forward)}\n")

    # Part (d): Plot relative errors
    print("Q3 - Part (d)")
    plt.figure(figsize=(10, 6))
    plt.loglog(h_val, relative_errors, label='Central Difference', marker='o')
    plt.loglog(h_val, relative_errors_forward, label='Forward Difference', marker='s')
    plt.xlabel('Step size $h$')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs. Step Size for $f(x) = e^{-x^2}$ at $x = 0.5$')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"Relative_Error_f.png")
    print(f"Have been saved: Period_Func.png \n")

    # Part (f): Higher-order derivatives for g(x) = exp(2 * x)
    print("Q3 - Part (f)")

    h = 1e-6
    x0 = 0.0
    max_n = 5
    for n in range(1, max_n + 1):
        derivative = delta(g, x0, n, h)
        analytical = 2 ** n
        relative_error = abs(derivative - analytical) / abs(analytical)
        print(f"{n}-th derivative at x={x0}:")
        print(f"Numerical = {derivative:.6f}, Analytical = {analytical:.6f}, Relative Error = {relative_error:.2e}")

if __name__ == "__main__":
    main()
