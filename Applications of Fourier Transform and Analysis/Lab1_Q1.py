'''
In this question, we explore the numerical issues with standard deviation calculations by comparing the one-pass and two-pass methods. The one-pass method uses the alternative formula for standard deviation, while the two-pass method follows the traditional formula where the mean is computed first.

We will start analyzing the relative error in standard deviation calculations between these two methods. Compare these errors with the numpy standard deviation function (numpy.std(array, ddof=1)) to determine which method is more accurate.

Then investigate how the relative errors behave for different datasets, including one-dimensional arrays and artificially generated sequences.

Finally, we will propose and test a simple workaround for the problems encountered with the one-pass method.

Libraries used: numpy for numerical operations

Output: This code produces plots, printed outputs of relative errors, and will help determine which method is more robust in standard deviation calculations. It does not write any output files.
'''

import numpy as np


#two-pass method
def two_pass_method(data):
    #mean for two-pass
    mean_two_pass = np.mean(data)
    #standard deviation for two-pass
    return np.sqrt(np.sum((data - mean_two_pass)**2)/(len(data) - 1)) 

#one-pass method
def one_pass_method(data):
    mean_two_pass = np.mean(data)
    sum_one_pass = np.sum(data**2)
    std_one_pass = np.sqrt((sum_one_pass - (len(data))*mean_two_pass**2)/(len(data) - 1))
    if std_one_pass <0:
        print("Warning! Nagtive value encountered")
    return std_one_pass
    
#true value
def numpy_std(data):
    return np.std(data, ddof=1)

#Calculate relative error
def relative_error(x , y):
    return (x - y)/y

#Compare the relative error from the true value: numpy_std
def compare_error(error_1 , error_2):
    if abs(error_2) == abs(error_1): 
        print("The relative errors of the two-pass method and the one-pass method are equal!!")
    elif abs(error_2) > abs(error_1):
        print("The two pass method is larger relative error than one pass method!!")
    else:
        print("The one pass method is larger relative error than two pass method!!")    

#one-pass method with fix the issue of numerical
def one_pass_fix(data):
    mean_data = np.mean(data)
    shifted_data = data - mean_data

    sum_shifted = np.sum(shifted_data**2)
    std_shifted = np.sqrt((sum_shifted) / (len(shifted_data) - 1))

    return std_shifted

def main():
    #load txt file (one-dimensional array)
    data = np.loadtxt('cdata.txt')

    #call each methods: two-pass method, one-pass method, true value method
    two_pass = two_pass_method(data)
    one_pass = one_pass_method(data)
    numpy_value = numpy_std(data)

    print("\n")
    print("\n")
    print("Q1 a-b")
    #Q1 a-b
    # -------------------------------------------
    #Calculate relative error
    error_two_pass = relative_error(two_pass , numpy_value)
    error_one_pass = relative_error(one_pass , numpy_value)

    #print the results for each method
    print("standard deviation for two pass method is:", two_pass)
    print("standard deviation for one pass method is:", one_pass)
    print("standard deviation of true value is:", numpy_value)
    
    #print the relative error from the true value: numpy_std
    print("relative error for two pass method is:", error_two_pass)
    print("relative error for one pass method is:", error_one_pass)

    #compare the relative error of two different methods
    compare_error(error_one_pass, error_two_pass)

    print("\n")
    print("\n")
    print("Q1 c")
    #Q1 c
    # -------------------------------------------

    #create two sequences with different means
    seq1 = np.random.normal(0., 1., 2000)
    seq2 = np.random.normal(1.e7 , 1. , 2000)

    #calculate standard deviations for every methods: <seq1> 
    seq1_two_pass = two_pass_method(seq1)
    seq1_one_pass = one_pass_method(seq1)
    seq1_numpy_std = numpy_std(seq1)
    
    #calculate standard deviations for every methods: <seq2> 
    seq2_two_pass = two_pass_method(seq2)
    seq2_one_pass = one_pass_method(seq2)
    seq2_numpy_std = numpy_std(seq2)

    #calculate relative errors: <seq1> 
    seq1_error_two_pass = relative_error(seq1_two_pass , seq1_numpy_std)
    seq1_error_one_pass = relative_error(seq1_one_pass , seq1_numpy_std)

    #calculate relative errors: <seq2> 
    seq2_error_two_pass = relative_error(seq2_two_pass , seq2_numpy_std)
    seq2_error_one_pass = relative_error(seq2_one_pass , seq2_numpy_std)

    print("Sequence 1 (mean=0. , sigma=1. , 2000 shots):")
    print(f"Two-pass relative error: {seq1_error_two_pass}")
    print(f"One-pass relative error: {seq1_error_one_pass}")
    compare_error(seq1_error_one_pass , seq1_error_two_pass)
    
    print("\n")
    
    print("Sequence 2 (mean=1.e7 , sigma=1. , 2000 shots):")
    print(f"Two-pass relative error: {seq2_error_two_pass}")
    print(f"One-pass relative error: {seq2_error_one_pass}")
    compare_error(seq2_error_one_pass , seq2_error_two_pass)


    print("\n")
    print("\n")
    print("Q1 d")
    #Q1 d
    # -------------------------------------------
    one_fix = one_pass_fix(data)
    error_fix_one_pass = relative_error(one_fix , numpy_value)
    
    print("standard deviation for fixed one pass method is:", error_fix_one_pass)
    compare_error(error_fix_one_pass, error_two_pass)

if __name__ == "__main__":
    main()































































