"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.

Code is available via AppleCT Dataset Project; www.github.com/cicwi/applect-dataset-project

Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
Parallel-beam X-ray CT datasets of apples with internal defects and label balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346

Dataset available via Zenodo; 10.5281/zenodo.4212301
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, getopt

def empirical_search(sample_size, sequence_probability, sequence_number, dataset_src):
    """ Searches for the best split between training and test subsets trying different random sequences for a test split.
        
    Parameters
    ----------
    sample_size : int
        Number of random sequences to try.
    sequence_probability : float
        Percentage of defects to achieve in the test subset.
    sequence_number : int
        Number of apples to be included in the test subset.
    dataset_src : str
        Location of a csv file containing information about apple defects.
    
    Returns
    -------
    numpy array
        An array of test split sequences
        This array is sorted by the difference between an achieved defect percentage and the target defined by the sequence_probability.
    """
    apple_df = pd.read_csv(dataset_src)
    A = apple_df.to_numpy()[:,1:]
    A = A / A.sum(axis = 0)
    total_apple_num = A.shape[0]
    defect_num = A.shape[1]
    sort_defect = 3
    sequence = np.arange(total_apple_num)
    successful_seq = []
    
    for _ in tqdm(range(sample_size)):
        sequence = np.random.permutation(sequence)
        A_tmp = A[sequence, :]
        vector_below = A_tmp[:sequence_number-1,:].sum(axis=0)
        vector_above = A_tmp[:sequence_number,:].sum(axis=0)
        if (vector_below[sort_defect] < sequence_probability) and (vector_above[sort_defect] > sequence_probability):
            successful_seq.append(np.array(sorted(sequence[:sequence_number])) + 1)
        
    target = np.ones((defect_num)) * sequence_probability
    successful_seq.sort(key = lambda x: sequence_sort(x, target, dataset_src))
    successful_seq = np.array(successful_seq)
    
    np.savetxt("empirical.csv", successful_seq, delimiter=",")
    return successful_seq

def check_sequence(sequence, dataset_src):
    """Calculates defect percentages for a given split sequence of apples.
    Parameters
    ----------
    sequence_number : numpy array
        Sequence of apples to check.
    dataset_src : str
        Location of a csv file containing information about apple defects.
        
    Returns
    -------
    numpy array
        A vector containing defect percentages for bitterpit, holes, rot, and browning.
    """
    apple_df = pd.read_csv(dataset_src)
    # Matrix A shows how many defect pixels are present in every apple sample
    # Column 1 from the csv file is ignored since it contains apple labels
    A = apple_df.to_numpy()[:,1:]
    A = A / A.sum(axis = 0)
    total_apple_num = A.shape[0]
    
    # Apples are counted from 0
    sequence_tmp = sequence - 1
    test_vector = np.zeros((total_apple_num))
    test_vector[sequence_tmp] = 1
    
    defect_probability = np.dot(A.T, test_vector)
    return defect_probability

def sequence_sort(sequence, target, dataset_src):
    """Sorting function for an array of sequences.
    Parameters
    ----------
    sequence_number : numpy array
        Sequence of apples.
    target : numpy array
        Vector containg the required defect percentages fot a sequences.
    dataset_src : str
        Location of a csv file containing information about apple defects.
        
    Returns
    -------
    float
        L2-norm of the difference between a vector of defect probabilities for a given sequence and the target vector.
    """
    defect_probability = check_sequence(sequence, dataset_src)
    dif = defect_probability - target
    dif = np.power(dif, 2)
    l2_norm = dif.sum()
    return l2_norm
            
if __name__ == "__main__":
    np.random.seed(42)
    sample_size = 1000
    sequence_probability = 0.2
    sequence_number = 20
    dataset_src = "full.csv"
    
    opts, args = getopt.getopt(sys.argv[1:],"s:p:n:i:")
    for opt, arg in opts:
        if opt == "-s":
            sample_size = int(arg)
        elif opt == "-p":
            sequence_probability = float(arg)
        elif opt == "-n":
            sequence_number = int(arg)
        elif opt == "-i":
            dataset_src = arg
    
    empirical_search(sample_size, sequence_probability, sequence_number, dataset_src)
