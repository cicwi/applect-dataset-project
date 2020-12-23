"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.
Code is available via LINKXXXX.
Reference paper: XXXXX
"""
import numpy as np
import cvxpy as cvx
import sys, getopt
import time

def find_sequence(sequence_probability, sequence_number, dataset_src):
    """ Searches for the best split between training and test subsets using MIQP solver ECOS_BB.
        
    Parameters
    ----------
    sequence_probability : float
        Percentage of defects to achieve in the test subset.
    sequence_number : int
        Number of apples to be included in the test subset.
    dataset_src : str
        Location of a csv file containing information about apple defects.
    
    Returns
    -------
    numpy array
        Testing set sequence that achieves the closest percentage of defects to the target.
    """
    data = np.loadtxt(dataset_src, delimiter = ",")
    # Matrix A shows how many defect pixels are present in every apple sample
    # Column 1 from the csv file is ignored since it contains apple labels
    A = np.copy(data)
    A = A[:,1:]
    A = A / A.sum(axis = 0)

    total_apple_num = A.shape[0]

    # Vector b corresponds to fractions of defects to be achieved
    b = np.ones((4)) * sequence_probability
    # W - indentity weight matrix that will be used in the contraint
    W = np.ones((total_apple_num))

    # Every apple is either included or excluded, therefore the solution vector contains only 1 and 0
    x = cvx.Variable(total_apple_num, boolean=True)
    # The L2-norm is minimized
    cost = cvx.sum_squares(A.T @ x - b)
    # The constraint is set to allow only solutions consisting of 20 apples
    prob = cvx.Problem(cvx.Minimize(cost), [W @ x == sequence_number, x >= 0])
    
    start_time = time.perf_counter()
    prob.solve(solver='ECOS_BB')
    end_time = time.perf_counter()
    print("Time = {}s".format(end_time - start_time))

    solution = np.copy(x.value)
    # Solution might contain values close to zero but not equal to zero
    solution[solution < 1e-10] = 0
    solution[solution == 1.0] = 1
    solution = solution.astype(np.bool)

    full_sequence = np.arange(1, total_apple_num+1, 1)
    test_sequence = full_sequence[solution]
    training_sequence = full_sequence[np.logical_not(solution)]

    print("Defect fractions of the sequence are ", np.dot(A.T, x.value))
    print("The norm of the residual is ", cvx.norm(A.T @ x - b, p=2).value)
    print("Test sequence is ", repr(test_sequence))
    print("Training sequence is ", repr(training_sequence))

    test_df = data[test_sequence - 1,:]
    np.savetxt("apple_defect_test.csv", test_df, delimiter=",", fmt="%d")
    training_df = data[training_sequence - 1,:]
    np.savetxt("apple_defect_test.csv", training_df, delimiter=",", fmt="%d")
    
    return test_sequence
    
if __name__ == "__main__":
    sequence_probability = 0.2
    sequence_number = 20
    dataset_src = "apple_defect_full.csv"
    
    opts, args = getopt.getopt(sys.argv[1:],"p:n:i:")
    for opt, arg in opts:
        if opt == "-p":
            sequence_probability = float(arg)
        elif opt == "-n":
            sequence_number = int(arg)
        elif opt == "-i":
            dataset_src = arg
    
    find_sequence(sequence_probability, sequence_number, dataset_src)
