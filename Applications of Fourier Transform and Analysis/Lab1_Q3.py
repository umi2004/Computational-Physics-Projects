'''
We will numerically evaluate the integral given in question 3 using two methods. Trapezoidal and Simposons' rule. First, we compute the exact value and then, compare the results obtained by the two numerical methods for different numbers of slices. After that, we will determine how many slices are required to achieve a certain error for each method and implement a practical error estimation for a trapezoidal method with different slice numbers. Finally, we propose adapting the practical error estimation method for Simpson's rule. 

libraries used: Numpy for numerical operations, SciPy for integration, Time for finding the running time 

Output: numerical results for each method, and errors. Also, the accuracy is based on the number of slices. It also outputs a practical error estimate for the Trapezoidal Rule. 
'''

#import the numpy and scipy
import numpy as np
from scipy import integrate
import time

#Define the function for the integral 
def f(x):
  return 4/(1 + x**2)

#Define the Trapezoidal Rule Function based on textbook code: trapezoidal.py
def trapezoidal_rule(a, b, N):
    h = (b - a) / N
    s = 0.5 * f(a) + 0.5 * f(b)
    for k in range(1, N):
        s += f(a + k * h)
    return h * s


#Define the Simpson's Rule function
def simpsons_rule(a, b, N):
    # Ensure that N is even
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")

    h = (b - a) / N
    s = f(a) + f(b)
    
    # Apply Simpson's rule: adding 4 times odd terms and 2 times even terms
    for k in range(1, N, 2):  # Sum over odd indices
        s += 4 * f(a + k * h)
    for k in range(2, N-1, 2):  # Sum over even indices
        s += 2 * f(a + k * h)
    
    return (h / 3) * s


#estimating error by using while loop to identify the appropriate N size that is under specific error. 
def estimate_error(method, a, b, exact_value, target_error, use_scipy=False):
    #initial
    N = 4
    error = float('inf')
    
    start_time = time.time()
    
    while error >= target_error:
        #for simpsons
        #x_values = np.linspace(a, b, N+1)
        #y_values = f(x_values)

        if use_scipy:
            #result = integrate.simpson(y_values, x = x_values)
            result = method(a , b, N)
        else:
            result = method(a , b, N)

        error = np.abs(result - exact_value)
        #even number of N
        N*=2

    end_time = time.time()
    total_time = end_time - start_time
    return N//2, result, error, total_time



#Practical estimation of errors for Trapezoidal in part d
def trapezoidal_error_estimation(N1, N2, a, b):
    I1 = trapezoidal_rule(a, b, N1)
    I2 = trapezoidal_rule(a, b, N2)
    return (I2 - I1)/3




def main():


    print("\n")
    print("\n")
    print("Q3 a")
    #Q3 a
    # -------------------------------------------
    exact_value, err= integrate.quad(f, 0, 1)
    print(f"The exact value of the integral is {exact_value} or PI")

    
    print("\n")
    print("\n")
    print("Q3 b")
    #Q3 b
    # -------------------------------------------
    N = 4
    a, b = 0, 1 

    #Trapezoidal Rule
    trap_result = trapezoidal_rule(a, b, N)
    print(f"Trapezoidal Rule result with N=4: {trap_result}")

    #Simpsons Rule
    #x_values = np.linspace(a, b, N+1)
    #y_values = f(x_values)
    #simp_result = integrate.simpson(y_values, x = x_values)
    simp_result = simpsons_rule(a, b, N)
    print(f"Simpson's Rule result with N=4 (using scipy): {simp_result}")


    #Compare with exact values
    trap_error = np.abs(trap_result - exact_value)
    simp_error = np.abs(simp_result - exact_value)
    print(f"Trapezoidal error: {trap_error}")
    print(f"Simpson's error: {simp_error}")

    print("\n")
    print("\n")
    print("Q3 c")
    #Q3 c
    # -------------------------------------------    
    target_error = 1e-9

    #Estimate Slice for Trapezoidal method 
    trap_est_N, trap_est_result, trap_est_error, trap_time = estimate_error(trapezoidal_rule, a, b, exact_value, target_error)
    print(f"Trapezoidal rule requires N={trap_est_N} slices for error {trap_est_error} and took {trap_time} seconds")

     
    #Estimate Slice for Simpson's method
    simp_est_N, simp_est_result, simp_est_error, simp_time = estimate_error(simpsons_rule, a, b, exact_value, target_error, use_scipy=True)
    print(f"Simpson's rule requires N={simp_est_N} slices for error {simp_est_error} and took {simp_time} seconds")


    print("\n")
    print("\n")
    print("Q3 d")
    #Q3 d
    # -------------------------------------------    

    #error estimate from "practical estimation of error" from the textbook
    N1, N2 = 16, 32
    trap_prac_error = trapezoidal_error_estimation(N1, N2, a, b)

    print(f"Trapezoidal error estimate for N2={N2}: {trap_prac_error}")



if __name__ == "__main__":
    main()








































































































