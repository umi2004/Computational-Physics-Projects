# Lab4_Q1.py
# Purpose:This script uses the Gauss-Seidel iterative method to solve Laplace's equation 
#          for a parallel-plate capacitor. It calculates the electrostatic potential within 
#          a 2D region and visualizes it using contour and stream plots.
# Requires:NumPy for numerical computations and Matplotlib for visualization.
# Output:Contour plot of electrostatic potential, stream plot of electric field lines.

from pylab import imshow, gray, show
import matplotlib.pyplot as plt
import numpy as np

M = 100
V = 1.0
target = 1.e-6
omega1 = 0.1
omega5 = 0.5 #0.5

# Initialize the potential grid and set boundary conditions
def initializePhi(M):
    phi_initial = np.zeros([M+1, M+1], float)

    return phi_initial

#Gauss-Seidel iteration loop
def Laplace(target, M, phi):
    phiprime = np.empty([M+1, M+1], float)

    delta = 1.0

    # Iterate over internal grid points
    while delta > target:

        for i in range(M+1):
            for j in range(M+1):
                if i == 0 or i == M or j == 0 or j == M:
                    phiprime[i,j] = phi[i,j]
                elif j == 20 and i > 20 and i < 80:
                    phiprime[i,j] = V
                elif j == 80 and i > 20 and i < 80:
                    phiprime[i,j] = -V
                else:
                    phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4
        
        delta = np.max(abs(phi-phiprime))
        phi, phiprime = phiprime, phi

    return phi

#Gauss-Seidel iteration loop when omega = 0.1
def Laplace_omega1(target, M, phi):
    phiprime = np.empty([M+1, M+1], float)

    delta = 1.0

    # Iterate over internal grid points with over-relaxation
    while delta > target:

        for i in range(M+1):
            for j in range(M+1):
                if i == 0 or i == M or j == 0 or j == M:
                    phiprime[i,j] = phi[i,j]
                elif j == 20 and i > 20 and i < 80:
                    phiprime[i,j] = V
                elif j == 80 and i > 20 and i < 80:
                    phiprime[i,j] = -V
                else:
                    phiprime[i,j] = phi[i,j] + omega1 * ((phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - phi[i,j])

        
        delta = np.max(abs(phi-phiprime))
        phi, phiprime = phiprime, phi

    return phi

#Gauss-Seidel iteration loop when omega = 0.5
def Laplace_omega5(target, M, phi):
    phiprime = np.empty([M+1, M+1], float)

    delta = 1.0

    # Iterate over internal grid points with over-relaxation
    while delta > target:

        for i in range(M+1):
            for j in range(M+1):
                if i == 0 or i == M or j == 0 or j == M:
                    phiprime[i,j] = phi[i,j]
                elif j == 20 and i > 20 and i < 80:
                    phiprime[i,j] = V
                elif j == 80 and i > 20 and i < 80:
                    phiprime[i,j] = -V
                else:
                    phiprime[i,j] = phi[i,j] + omega5 * ((phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - phi[i,j])

        
        delta = np.max(abs(phi-phiprime))
        phi, phiprime = phiprime, phi

    return phi

# Plotting the potential field
phi0 = Laplace(target, M, initializePhi(M))

plt.imshow(phi0)
plt.gray()
plt.show()
plt.cla()

# Plotting the potential field when omega = 0.1
print(omega1)
phi1 = Laplace_omega1(target, M, initializePhi(M))

plt.imshow(phi1)
plt.gray()
plt.show()
plt.cla()

# Plotting the potential field when omega = 0.5
print(omega5)
phi5 = Laplace_omega5(target, M, initializePhi(M))

plt.imshow(phi5)
plt.gray()
plt.show()
plt.cla()

#generates a stream plot to visualize the electric field lines based on the given potential field
def StreamPlots(phi):
    M = phi.shape[0] - 1
    # Create x and y coordinates for the grid points
    x = np.linspace(0, M, M+1) 
    y = np.linspace(0, M, M+1)
    # Create coordinate matrices for vector field visualization
    X, Y = np.meshgrid(x, y)

    # Calculate the gradient of the potential field (phi) to determine electric field components
    dphi_dy, dphi_dx = np.gradient(phi)
    Ex = -dphi_dx
    Ey = -dphi_dy

    # Create a figure to plot the electric field lines
    fig = plt.figure(figsize=(5, 5))
    strm = plt.streamplot(X, Y, Ex, Ey, color=phi, linewidth=1, cmap='autumn', density=2)
    cbar = fig.colorbar(strm.lines)
    cbar.set_label('Potential $V$')
    plt.title('Electric field lines')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Call the StreamPlots function with phi0
StreamPlots(phi0)
