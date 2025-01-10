''' 
We will explore the effects of roundoff error when calculating the polynomial p(u) = (1-u)^8 and compare it with its expanded form q(u) for values of u near 1. 

We will be going to plot p(u) and q(u) in the range 0.98 < u < 1.02 to compare which is noisier. plot the difference and generate a histogram of this difference. Then, calculate the fractional error as u approaches 1. Finally, we investigate the roundoff error in a product f = u**8 / ((u**4) * (u**4)) for u near 1.


Libraries used: numpy for numerical operations, matplotlib for plotting.

Output: This code produces several plots and calculates errors. It does not write any output files.
'''

#import the numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

#define p(u) = (1-u)^8
def p(u):
    return (1 - u)**8


#define q(u) = expanded form of (1-u)^8
def q(u):
    return 1 - 8*u + 28*u**2 - 56*u**3 + 70*u**4 - 56*u**5 + 28*u**6 - 8*u**7 + u**8


def main():
    
    print("\n")
    print("\n")
    print("Q2 a")
    #Q2 a
    # -------------------------------------------

    #create range of calculation with 500 steps
    u_values = np.linspace(0.98, 1.02, 500)
    p_values = p(u_values)
    q_values = q(u_values)

    #plot comparison of two different form of function near 1
    plt.figure(figsize=(10, 6))
    plt.plot(u_values, p_values, label="p(u) = (1 - u)^8", color="blue")
    plt.plot(u_values, q_values, label="q(u) = expanded form", color="red")
    plt.title("Comparison of p(u) and q(u) near u = 1")
    plt.xlabel("u")
    plt.ylabel("Polynomial value")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('p_and_q.png')
    print("plot is saved for Q2 a: p_and_q.png")

    print("\n")
    print("\n")
    print("Q2 b")
    #Q2 b
    # -------------------------------------------

    #create a difference values between p and q values
    diff_values = p_values - q_values

    #plotting for the p(u) - q(u)
    plt.figure(figsize=(10, 6))
    plt.plot(u_values, diff_values, label="p(u) - q(u)", color="blue")
    plt.title("Difference p(u) - q(u) near u = 1")
    plt.xlabel("u")
    plt.ylabel("Difference value")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('pq_difference_plot.png')
    print("plot is saved for Q2 b: pq_difference_plot.png")

    #create histogram of p(u) - q(u)
    plt.figure(figsize=(10, 6))
    plt.hist(diff_values, bins=50, color='purple')
    plt.title("Histogram of p(u) - q(u) near u = 1")
    plt.xlabel("Difference value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    plt.savefig('pq_difference_hist.png')
    print("plot is saved for Q2 b: pq_difference_hist.png")

    #Calculate the std of the distribution of p(u) - q(u)
    std_pq = np.std(diff_values)
    print(f"Standard deviation of p(u) - q(u): {std_pq}")

    #use equation (3): σ = C * sqrt(N) * sqrt(mean(x^2))
    C = 1.e-16
    N = 10
    x = np.mean(np.array([0, 1, 8, 28, 56, 70, 56, 28, 8, 1])**2)

    estimated_error = C * np.sqrt(N) * np.sqrt(x)
    print(f"Estimated error from equation (3): {estimated_error}")


    print("\n")
    print("\n")
    print("Q2 c")
    #Q2 c
    # -------------------------------------------  
    #Show using equation (4), where the error is around 100%.

    u = 1
    
    x_100 = np.array([(1-u)**8, 1, 8*u, 28*u**2, 56*u**3, 70*u**4, 56*u**5, 28*u**6, 8*u**7, u**8])
    
    mean_100 = np.mean(x_100)
    mean_square_100 = np.mean(x_100**2)
    estimated_error_100 =( (C * np.sqrt(mean_square_100)) / (np.sqrt(N) * mean_100) )* np.sum(x_100)

    u = (1 - estimated_error_100**(1/8))
    print(f"Fractional error: {u}")
    
    
    #Verify using plot out: abs(p-q)/abs(p) for u = 0.980 to upto 0.984
    u_values_frac = np.linspace(0.980, 0.9819, 500)
    p_values_frac = p(u_values_frac)
    q_values_frac = q(u_values_frac)
    diff_values_frac = p_values_frac - q_values_frac    
    fractional_error = np.abs(diff_values_frac) / np.abs(p_values_frac)
    
    # Plot 
    plt.figure(figsize=(10, 6))
    plt.plot(u_values_frac, fractional_error, label="Fractional error |p-q|/|p|", color="red")
    plt.title("Fractional error |p-q|/|p| near u = 1")
    plt.xlabel("u")
    plt.ylabel("Fractional error")
    plt.grid(True)
    plt.show()
    plt.savefig('fractional_error.png')
    print("plot is saved for Q2 c: fractional_error.png")




    print("\n")
    print("\n")
    print("Q2 d")
    #Q2 d
    # -------------------------------------------  

    #Investigate roundoff error in the product f = u**8/((u**4)*(u**4))
    f_values = u_values**8 / (u_values**4 * u_values**4)
    roundoff_error = f_values - 1

    #Textbook equation (4.5)
    estimated_error = C * f_values
    
    plt.figure(figsize=(10, 6))
    plt.plot(u_values, roundoff_error, label="Roundoff error in f = u^8/(u^4*u^4)", color="red")
    plt.title("Roundoff error in f =u^8/(u^4*u^4) near u=1")
    plt.xlabel("u")
    plt.ylabel("Roundoff error (f-1)")
    plt.grid(True)
    plt.show()
    plt.savefig('roundoff_error.png')
    print("plot is saved for Q2 d: roundoff_error.png")

    # Print the calculated and estimated errors for comparison
    print(f"Mean of roundoff error: {np.mean(roundoff_error)}")
    print(f"Mean estimated error: {np.mean(estimated_error)}")

if __name__ == "__main__":
    main()
































































































































