# Lab5_Q1.py
# Purpose:
#   - (a) Simulates a 2D random walk of a particle over 5000 steps and plots its trajectory.
#   - (b) Models Diffusion-Limited Aggregation (DLA) by adding particles until one anchors at the center.
# Requires:
#   NumPy and Matplotlib.
# Output:
#   Plots of x vs time, y vs time, trajectory, and the final DLA cluster.


import numpy as np
import matplotlib.pyplot as plt

# (a) Random walk simulation
# Set the random seed for reproducibility
np.random.seed(123)

# Initialize parameters
L = 101  # Grid size (number of cells along one axis)
N = 5000  # Number of steps for the random walk
dt = 1e-3  # Time step size in seconds
dxx = 1e-3  # Distance step size in meters

# Initialize the starting index at the center of the grid
x_index = L // 2  # Center index for x-axis
y_index = L // 2  # Center index for y-axis

# Lists to store positions and time values
x_pos = []  # x-coordinates in mm
y_pos = []  # y-coordinates in mm
t_val = []  # Time values in ms

# Calculate and store the initial position at time t = 0
x = (x_index - L//2) * dxx * 1e3  # Convert from grid index to mm
y = (y_index - L//2) * dxx * 1e3  # Convert from grid index to mm
x_pos.append(x)
y_pos.append(y)
t_val.append(0.0)

# Define possible movement directions
directions = ['up', 'down', 'left', 'right']

# Perform the random walk
for n in range(1, N+1):
    moved = False
    while not moved:
        # Randomly select a movement direction
        direction = np.random.choice(directions)
        if direction == 'up':
            dx, dy = 0, 1
        elif direction == 'down':
            dx, dy = 0, -1
        elif direction == 'left':
            dx, dy = -1, 0
        elif direction == 'right':
            dx, dy = 1, 0
        
        # Calculate the new position
        new_x = x_index + dx
        new_y = y_index + dy

        # Ensure the new position is within grid boundaries
        if 0 <= new_x < L and 0 <= new_y < L:
            x_index = new_x
            y_index = new_y
            moved = True
    
    # Update position and time values
    x = (x_index - L//2) * dxx * 1e3  # Convert from grid index to mm
    y = (y_index - L//2) * dxx * 1e3  # Convert from grid index to mm
    x_pos.append(x)
    y_pos.append(y)
    t_val.append(n * dt * 1e3)  # Convert time to ms

# Plot x-position vs. time
plt.figure(figsize=(8, 6))
plt.plot(t_val, x_pos)
plt.xlabel('Time (ms)')
plt.ylabel('x-position (mm)')
plt.title('x-position vs. Time')
plt.grid(True)
plt.show()

# Plot y-position vs. time
plt.figure(figsize=(8, 6))
plt.plot(t_val, y_pos)
plt.xlabel('Time (ms)')
plt.ylabel('y-position (mm)')
plt.title('y-position vs. Time')
plt.grid(True)
plt.show()

# Plot y-position vs. x-position
plt.figure(figsize=(10, 10))
plt.plot(x_pos, y_pos)
plt.xlabel('x-position (mm)')
plt.ylabel('y-position (mm)')
plt.title('y-position vs. x-position')
plt.axis('equal')  # Ensure equal scaling for x and y axes
plt.xlim(-50, 50)  # Set x-axis limits
plt.ylim(-50, 50)  # Set y-axis limits
plt.grid(True)
plt.show()

# (b) Diffusion-limited aggregation (DLA)

# Initialize the grid with boundaries
grid = np.zeros((L, L), dtype=int)  # Create a grid of zeros
grid[0, :] = 1  # Top boundary wall
grid[L-1, :] = 1  # Bottom boundary wall
grid[:, 0] = 1  # Left boundary wall
grid[:, L-1] = 1  # Right boundary wall

# List to store anchored particle positions
anchored_particles = []

# Variable to store all particle paths (optional, not visualized here)
particle_paths = []

# Simulation parameters
center_anchored = False  # Flag to stop simulation when center is anchored
particle_number = 0  # Counter for the number of particles

# Main loop for particle movement
while not center_anchored:
    particle_number += 1  # Increment particle counter

    # Initialize particle at the center of the grid
    x_index_b = L // 2
    y_index_b = L // 2

    # Check if the center cell is already occupied
    if grid[x_index_b, y_index_b] == 1:
        center_anchored = True
        break

    moving = False
    while not moving:
        # Randomly choose a direction for the particle to move
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up':
            dx, dy = 0, 1
        elif direction == 'down':
            dx, dy = 0, -1
        elif direction == 'left':
            dx, dy = -1, 0
        elif direction == 'right':
            dx, dy = 1, 0
        
        # Calculate the new position
        new_x_b = x_index_b + dx
        new_y_b = y_index_b + dy

        # Ensure the new position is within grid boundaries
        if 0 <= new_x_b < L and 0 <= new_y_b < L:
            x_index_b = new_x_b
            y_index_b = new_y_b

            # Check if the particle becomes anchored
            neighbors = [
                (x_index_b + 1, y_index_b),
                (x_index_b - 1, y_index_b),
                (x_index_b, y_index_b + 1),
                (x_index_b, y_index_b - 1),
            ]

            for nx, ny in neighbors:
                if 0 <= nx < L and 0 <= ny < L and grid[nx, ny] == 1:
                    grid[x_index_b, y_index_b] = 1
                    anchored_particles.append((x_index_b, y_index_b))
                    moving = True
                    break
            
            # If the particle reaches the boundary, anchor it
            if grid[x_index_b, y_index_b] == 1:
                anchored_particles.append((x_index_b, y_index_b))
                moving = True
        else:
            # Anchor particles if out of bounds
            grid[x_index_b, y_index_b] = 1
            anchored_particles.append((x_index_b, y_index_b))
            moving = True

# Convert anchored particle positions to mm for plotting
x_positions = [(x - L // 2) * dxx * 1e3 for x, y in anchored_particles]
y_positions = [(y - L // 2) * dxx * 1e3 for x, y in anchored_particles]

# Plot the DLA cluster of anchored particles
plt.figure(figsize=(10, 10))
plt.scatter(x_positions, y_positions, color='blue')  # Plot anchored particles
plt.xlabel('x-position (mm)')
plt.ylabel('y-position (mm)')
plt.title('DLA Cluster of Anchored Particles')
plt.grid(True)
plt.show()
