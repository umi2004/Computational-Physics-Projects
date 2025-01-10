"""
Purpose: This script calculates and plots properties of the Quantum Harmonic Oscillator (QHO).
- Part (a) defines a function to calculate the Hermite polynomials H(n, x) for the QHO.
- Part (b) generates and plots the wavefunctions for energy levels n = 0, 1, 2, and 3 over the range -4 <= x <= 4.
- Part (c) calculates the potential energy (⟨X^2⟩) for energy levels n = 0 to 10 using Gaussian quadrature.

External Functions: Requires 'gaussxw' for Gaussian quadrature integration.
Outputs: 
- Part (b): A plot of the QHO wavefunctions saved as 'Quantum_Harmonic_Oscillator_Wavefunctions.png'.
- Part (c): Potential energy values printed to the console.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp, factorial

# Function to calculate the Hermite Polynomial
def H(n, x):
    if n == 0:
        return 1.0  # Initial
    elif n == 1:
        return 2.0 * x  # Initial
    else:
        H_nm1 = 2.0 * x
        H_nm2 = 1.0
        for k in range(2, n + 1):  # Recursive calculation
            H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
            H_nm2 = H_nm1
            H_nm1 = H_n
        return H_n

# Function to calculate the Harmonic Oscillator Wavefunction psi
def psi(n, x):
    coeff = 1.0 / sqrt(2 ** n * factorial(n) * sqrt(pi))
    return coeff * np.exp(-x ** 2 / 2) * H(n, x)

# Gaussian Quadrature weights and points
def gaussxw(N):
    a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = np.cos(pi * a + 1 / (8 * N * N * np.tan(a)))
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w

# Gaussian Quadrature transformation to the interval [a, b]
def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

# Function to calculate the integrand for potential energy
def g(z, n):
    if np.abs(z) >= 1.0:
        return 0.0
    x = z / (1 - z ** 2)
    num = z ** 2 * (1 + z ** 2)
    deno = (1 - z ** 2) ** 4
    integrand = num / deno * abs(psi(n, x)) ** 2
    return integrand

# Calculating potential energy using Gaussian Quadrature
def potential_energy(n, N=100):
    z_n, weights = gaussxw(N)
    I = 0.0
    for k in range(N):
        z = z_n[k]
        w = weights[k]
        integrand = g(z, n)
        I += w * integrand
    return I

# Main function to run all parts of the code
def main():
    # Part (a): Print a comment
    print("Q1 - Part (a)")
    print("Calculating Hermite Polynomials H(n, x) for Quantum Harmonic Oscillator\n")

    # Part (b): Plot the wavefunctions for n = 0, 1, 2, 3
    print("Q1 - Part (b)")
    print("Plotting the Quantum Harmonic Oscillator Wavefunctions\n")
    x = np.linspace(-4, 4, 1000)
    for n in range(4):
        y = psi(n, x)
        plt.plot(x, y, label=f'$n = {n}$')
    plt.title("Quantum Harmonic Oscillator Wavefunctions")
    plt.xlabel("$x$")
    plt.ylabel("$\psi_n(x)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("Quantum_Harmonic_Oscillator_Wavefunctions.png")
    print("Have been saved: Quantum_Harmonic_Oscillator_Wavefunctions.png \n")

    # Part (c): Calculate potential energies for n = 0 to 10
    print("Q1 - Part (c)")
    print("Calculating Potential Energy ⟨X^2⟩ for Quantum Harmonic Oscillator States:\n")
    for n in range(11):
        X2_n = potential_energy(n, N=100)
        print(f"n = {n}, $⟨X^2⟩$ = {X2_n}")

if __name__ == "__main__":
    main()