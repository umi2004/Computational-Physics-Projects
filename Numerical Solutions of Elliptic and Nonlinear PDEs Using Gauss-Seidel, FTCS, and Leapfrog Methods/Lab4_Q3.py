# Lab4_Q3.py
# Purpose:This script uses the leapfrog method to solve the inviscid Burger's equation.
#          The equation models wave propagation in a nonlinear medium, with wave steepening effects.
# Requires:NumPy for numerical array handling and Matplotlib for visual output.
# Output:Plot of the wave profile, demonstrating nonlinear steepening and shock formation.

import numpy as np
import matplotlib.pyplot as plt

#Parameters
epsilon = 1. #constant
dx = 0.02 # grid spacing (m)
dt = 0.005 #time step (s)
Lx = 2*np.pi #length of spatial domain (m)
Tf = 2. #Time duration (s)

Nx = int(Lx / dx) + 1 #number of grid points (315)
Nt = int(Tf / dt) + 1 #number of time steps (401)


#Spatial grid 
x = np.linspace(0, Lx, Nx)

#Time grid
t = np.linspace(0, Tf, Nt)

#Initialize u 
u = np.zeros((Nt, Nx))

# Set initial condition 
u[0,:] = np.sin(x)

#beta parameter
beta = epsilon*dt/dx

#First time steps with EULER METHOD
for i in range(1, Nx-1):
    u[1,i] = u[0, i] - (beta/2)*(u[0, i+1]**2 - u[0, i-1]**2)

# Set boundary condition
u[0, 0] = 0.
u[0, -1] = 0.
u[1, 0] = 0.
u[1, -1] = 0.


#Leapfrog method
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        u[n + 1, i] = u[n - 1, i] - (beta/2) * (u[n, i + 1]**2 - u[n, i - 1]**2)

    u[n+1, 0] = 0.
    u[n+1, -1] = 0.


time_plot = [0., 0.5, 1., 1.5]
indices_plot = [int(time/dt) for time in time_plot]

#Graph
plt.figure(figsize=(10, 10))
for idx, time in zip(indices_plot, time_plot):
    plt.plot(x, u[idx, :], label=f't = {time:.1f} s')
plt.xlabel('x (m)')
plt.ylabel('u(x, t)')
plt.title('Solution of Burgers Equation at Different Times')
plt.legend()
plt.grid(True)
plt.show()