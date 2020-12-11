"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.
Based on the msdnet example https://github.com/dmpelt/msdnet/blob/master/examples/train_regr.py
Code is available via LINKXXXX.
Reference paper: XXXXX
"""
import msdnet
from pathlib import Path
import numpy as np
import random

def train_nn(input_folder, target_folder, train_seq, validation_num, layers_num)
    """ Trains the network using the specified subset of data.
    By default, msdnet performs the training until the script execution is stopped manually.
    Training is run with 100 layers, L2 loss, and ADAM optimization algorithm.
        
    Parameters
    ----------
    input_folder : string
        Path to the input images
    target_folder : string
        Path to the target images
    train_seq : int numpy array
        Sequence of numbers of apples to train the network on
    validation_num : int
        Number of images that will be used as a validation set
    layers_num: int
        Number of network layers
    
    Returns
    -------
    None
    """
    
    # Full list of apple labels
    apple_numbers = np.array(['31101', '31102', '31103', '31104', '31105', '31106', '31107', '31108', '31109', '31110', '31111', '31112', '31113', '31114', '31115', '31116', '31117', '31118', '31119', '31120', '31121', '31122', '31201', '31202', '31203', '31204', '31205', '31206', '31207', '31208', '31209', '31210', '31211', '31212', '31213', '31214', '31215', '31216', '31217', '31218', '31219', '31220', '31221', '31222', '31301', '31302', '31303', '31304', '31305', '31306', '31307', '31308', '31309', '31310', '31311', '31312', '31313', '31314', '31315', '31316', '31317', '31318', '31319', '31320', '31321', '31322', '32101', '32102', '32103', '32104', '32105', '32106', '32107', '32108', '32109', '32110', '32111', '32112', '32113', '32114', '32115', '32116', '32117', '32118', '32119', '32120', '32121', '32122', '32201', '32202', '32203', '32204', '32205', '32206'])

    # Prepare a list of apple samples for training
    train_seq = train_seq - 1
    apple_mask = np.zeros((94), dtype = np.bool)
    train_mask = apple_mask
    train_mask[train_seq] = True
    train_apples = apple_numbers[train_mask]

    # Define dilations in [1,10] 
    dilations = msdnet.dilations.IncrementDilations(10)

    # Creat the network object with one input channel and one output channel
    n = msdnet.network.MSDNet(layers_num, dilations, 1, 1, gpu=True)

    n.initialize()

    # flsin and flstg contain slice names that will be used in training
    flsin = []
    flstg = []
    for number in train_apples:
        flsin.extend(Path(input_folder).glob('*_{}_*.tif'.format(number)))
        flstg.extend(Path(target_folder).glob('{}_*.tif'.format(number)))

    flsin = sorted(flsin)
    flstg = sorted(flstg)

    # Randomly shuffle a list of slices. Later we will use images at the beginning as a validation set.
    pairs = list(zip(flsin, flstg))
    random.shuffle(pairs)

    # Define training data as a list of datapoints
    dats = []
    for i in range(validation_num, len(flsin)):
        d = msdnet.data.ImageFileDataPoint(str(pairs[i][0]),str(pairs[i][1]))
        d_augm = msdnet.data.RotateAndFlipDataPoint(d)
        dats.append(d_augm)

    # Normalize input and output of network to zero mean and unit variance using training data images
    n.normalizeinout(dats)

    # Use image batches of a single image
    bprov = msdnet.data.BatchProvider(dats,1)

    # Define validation data
    datsv = []
    for i in range(validation_num):
        d = msdnet.data.ImageFileDataPoint(str(pairs[i][0]),str(pairs[i][1]))
        datsv.append(d)

    # Select L2 loss function
    l2loss = msdnet.loss.L2Loss()
    val = msdnet.validate.LossValidation(datsv, loss=l2loss)
    # Use Adam training algorithm
    t = msdnet.train.AdamAlgorithm(n, loss=l2loss)

    # Define loggers
    consolelog = msdnet.loggers.ConsoleLogger()
    filelog = msdnet.loggers.FileLogger('log_regr.txt')
    imagelog = msdnet.loggers.ImageLogger('log_regr', onlyifbetter=True)

    # Train the network until exection is stopped manually
    msdnet.train.train(n, t, val, bprov, 'regr_params.h5',loggers=[consolelog,filelog,imagelog], val_every=len(datsv), progress=True)
    # Every epoch when the validation performance is better than the previous best, network parameters are saved in the 'regr_params.h5' file
    
if __main__ == "__main__":
    # Replace with the location of input and target files
    input_folder = "/path/to/input/files"
    target_folder = "/path/to/target/files"
    
    train_seq = np.array([ 1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20,
    23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 48, 50, 
    51, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 78, 
    79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 91, 92, 93])
    
    train_nn(input_folder, target_folder, train_seq, 1000, 100)
    
