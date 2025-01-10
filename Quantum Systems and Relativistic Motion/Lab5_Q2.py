# Lab5_Q2.py
# Purpose:
#   - (a) Uses Simulated Annealing to find the global minimum of f(x, y) = x**2 - cos(4πx) + (y - 1)**2.
#   - (b) Extends the Simulated Annealing approach to optimize a more complex function with additional cosine terms.
# Requires:
#   NumPy and Matplotlib.
# Output:
#   Plots of x and y positions over time-steps and the estimated global minimum coordinates.


import numpy as np
import matplotlib.pyplot as plt

### (a) Simulated Annealing for Function Optimization

# Set the random seed for reproducibility
np.random.seed(124)

def f(x, y):
    """
    Function to find the global minimum of.
    Includes a cosine term to introduce local minima and a quadratic term for the global minimum.
    """
    return x**2 - np.cos(4 * np.pi * x) + (y - 1)**2

def draw_normal(sigma=1.0):
    """
    Generate random numbers from a normal distribution with mean 0 and standard deviation sigma.
    Uses polar coordinates to implement the Box-Muller transform.
    """
    theta = 2 * np.pi * np.random.random()  # Random angle
    z = np.random.random()  # Random radius factor
    r = np.sqrt(-2 * sigma**2 * np.log(1 - z))  # Radius
    return r * np.cos(theta), r * np.sin(theta)  # Convert to Cartesian coordinates

def get_temperature(T0, tau, t):
    """
    Exponential cooling schedule.
    T0: Initial temperature
    tau: Cooling time constant
    t: Current time-step
    """
    return T0 * np.exp(-t / tau)

def decide(new_val, old_val, temperature):
    """
    Decide whether to accept the new move.
    Always accept if the new value is better. Otherwise, accept with a probability
    proportional to the temperature (to allow exploration of the solution space).
    """
    if new_val < old_val:
        return True
    else:
        probability = np.exp(-(new_val - old_val) / temperature)
        return np.random.random() < probability

# Parameters for simulated annealing
T0 = 5.0       # Initial temperature
Tf = 1e-5      # Final temperature
tau = 30000    # Cooling time constant
sigma = 1.0    # Standard deviation for Gaussian steps
x0, y0 = 2.0, 2.0  # Initial position

# Lists to store the trajectory of the optimization
x_list = [x0]
y_list = [y0]
t_list = [0]
temperature_list = [T0]

# Initialize the current position and function value
t = 0  # Time-step counter
T = T0  # Initial temperature
current_x, current_y = x0, y0
current_f = f(current_x, current_y)

# Simulated annealing loop
while T > Tf:
    t += 1  # Increment time-step
    T = get_temperature(T0, tau, t)  # Update temperature
    dx, dy = draw_normal(sigma)  # Random step
    new_x = current_x + dx
    new_y = current_y + dy
    new_f = f(new_x, new_y)  # Evaluate function at new position
    if decide(new_f, current_f, T):  # Decide whether to accept the move
        current_x, current_y = new_x, new_y
        current_f = new_f
    # Record the current state
    x_list.append(current_x)
    y_list.append(current_y)
    t_list.append(t)
    temperature_list.append(T)

# Output the final result
print(f"The global minimum is estimated to be at (x, y) = ({current_x:.4f}, {current_y:.4f})")
print(f"f({current_x:.4f}, {current_y:.4f}) = {current_f:.4f}")

# Plot x and y as functions of time-step
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_list, x_list, '.', markersize=2)
plt.xlabel('Time-step')
plt.ylabel('x')
plt.title('x vs. Time-step')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_list, y_list, '.', markersize=2)
plt.xlabel('Time-step')
plt.ylabel('y')
plt.title('y vs. Time-step')
plt.grid(True)

plt.tight_layout()
plt.show()

### (b) Simulated Annealing with a Modified Function

def f_b(x, y):
    """
    Function to find the global minimum of.
    Includes multiple cosine terms for local minima and a quadratic term for the global minimum.
    """
    return np.cos(x) + np.cos(np.sqrt(2) * x) + np.cos(np.sqrt(3) * x) + (y - 1)**2

# Parameters for simulated annealing
T0_b = 1000.0
Tf_b = 1e-10
tau_b = 10000
sigma_b = 1.0
x0_b, y0_b = 10.0, 1.0  # Initial position

# Lists to store the trajectory
x_list_b = [x0_b]
y_list_b = [y0_b]
t_list_b = [0]
temperature_list_b = [T0_b]

# Initialize the current position and function value
t_b = 0  # Time-step counter
T_b = T0_b  # Initial temperature
current_x_b, current_y_b = x0_b, y0_b
current_f_b = f_b(current_x_b, current_y_b)

# Simulated annealing loop
while T_b > Tf_b:
    t_b += 1  # Increment time-step
    T_b = get_temperature(T0_b, tau_b, t_b)  # Update temperature
    dx_b, dy_b = draw_normal(sigma_b)  # Random step
    new_x = current_x_b + dx_b
    new_y = current_y_b + dy_b

    # Check boundary conditions
    if not (0 < new_x < 50 and -20 < new_y < 20):
        x_list_b.append(current_x_b)
        y_list_b.append(current_y_b)
        t_list_b.append(t_b)
        temperature_list.append(T_b)
        continue

    new_f = f_b(new_x, new_y)  # Evaluate function at new position
    if decide(new_f, current_f_b, T_b):  # Decide whether to accept the move
        current_x_b, current_y_b = new_x, new_y
        current_f_b = new_f

    # Record the current state
    x_list_b.append(current_x_b)
    y_list_b.append(current_y_b)
    t_list_b.append(t_b)
    temperature_list.append(T_b)

# Output the final result
print(f"The global minimum is estimated to be at (x, y) = ({current_x_b:.4f}, {current_y_b:.4f})")
print(f"f_b({current_x_b:.4f}, {current_y_b:.4f}) = {current_f_b:.4f}")

# Plot x and y as functions of time-step
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_list_b, x_list_b, '.', markersize=1)
plt.xlabel('Time-step')
plt.ylabel('x')
plt.title('x vs. Time-step')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_list_b, y_list_b, '.', markersize=1)
plt.xlabel('Time-step')
plt.ylabel('y')
plt.title('y vs. Time-step')
plt.grid(True)

plt.tight_layout()
plt.show()
