# Lab4_Q2.py
# Purpose:This script simulates the propagation of shallow water waves using the FTCS scheme.
#          It aims to model fluid flow by discretizing the shallow water equations.
# Requires:NumPy for numerical calculations, Matplotlib for visual representation.
# Output:Plot of wave propagation over time, demonstrating numerical stability and instabilities.

import numpy as np
import matplotlib.pyplot as plt

#Parameters
L = 1.0 # Domain Length (m)
J = 50  # Number of intervals
dx = L/J #Grid spacing (m)
x = np.linspace(0, L, J+1) # Grid points from x=0 to x=L

g = 9.81
H = 0.01
eta_b = np.zeros_like(x) #Flat bottom topography 

#Boundary condition
u_left = 0.0 #u(0,t)
u_right = 0.0 #u(L,t)

dt = 0.01 #time step (s)
t_end = 4.0
Nt = int(t_end / dt) #400 steps

#Initial conditions parameters
A = 0.002 # Amplitude of Gaussian (m)
mu = 0.5 # Mean of Gaussian (m)
sigma = 0.05 #Std. dev. (m)

#Initial Functions
u = np.zeros_like(x)

#Initial eta
gaussian = A * np.exp(-((x-mu)**2)/sigma**2)
gaussian_avg = np.mean(gaussian)

eta = H + gaussian - gaussian_avg

#Time stamp
plot_times = [0., 1., 4.]
plot_indices = [int(time/dt) for time in plot_times] #for each plot_times have different indices
eta_plots = []

# FTCS scheme loop to simulate wave propagation
for n in range(Nt+1):
    t = n*dt

    if n in plot_indices:
        eta_plots.append((t, eta.copy()))
    
    F1 = 0.5 * u**2 + g * eta
    F2 = u * (eta - eta_b)

    u_new = np.copy(u)
    eta_new = np.copy(eta)

    u_new[1:J] = u[1:J] - (dt/(2*dx)) * (F1[2:J+1] - F1[0:J-1])
    eta_new[1:J] = eta[1:J] - (dt/(2*dx)) * (F2[2:J+1] - F2[0:J-1])

    u_new[0] = u_left
    u_new[J] = u_right

    eta_new[0] = eta[0] - (dt/dx) * (F2[1] - F2[0])
    eta_new[J] = eta[J] - (dt/dx) * (F2[J] - F2[J-1])

    u = np.copy(u_new)
    eta = np.copy(eta_new)


# Plotting the wave profile
plt.figure(figsize=(10, 6))
for t, eta_t in eta_plots:
    plt.plot(x, eta_t, label=f't = {t:.1f} s')
plt.xlabel('x (m)')
plt.ylabel(r'$\eta$ (m)')
plt.title(r'$\eta$ vs x at different times')
plt.legend()
plt.grid(True)
plt.show()