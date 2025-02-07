"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.

Based on the msdnet example https://github.com/dmpelt/msdnet/blob/master/examples/apply_regr.py
Code is available via AppleCT Dataset Project; www.github.com/cicwi/applect-dataset-project

Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
Parallel-beam X-ray CT datasets of apples with internal defects and label balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346

Dataset available via Zenodo; 10.5281/zenodo.4212301
"""
import msdnet
from pathlib import Path
from tqdm import tqdm
import numpy as np
import imageio
from skimage.metrics import structural_similarity
import sys, getopt

def psnr(image, ground_truth):
    """ Computes PSNR for a network output based on the known ground truth
        
    Parameters
    ----------
    image : float numpy array
        Network output to be compared with the ground truth
    ground_truth : float numpy array
        Ground truth image
    
    Returns
    -------
    float
        PSNR value in dB
    """
    mse = np.mean((image - ground_truth)**2)
    if mse == 0.:
        return float('inf')
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def ssim(image, ground_truth):
    """ Computes SSIM between a network output and a known ground truth
        
    Parameters
    ----------
    image : float numpy array
        Network output to be compared with the ground truth
    ground_truth : float numpy array
        Ground truth image
    
    Returns
    -------
    float
        SSIM value
    """
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)

def test_nn(input_folder, target_folder, gt_mask, test_seq, imsave_flag):
    """ Applies the trained network to the test set and compares the results with the ground truth
        
    Parameters
    ----------
    input_folder : string
        Path to the input images
    target_folder : string
        Path to the target images
    gt_mask: int numpy array
        Sequence of slices to use from the ground truth. Scattering is only computed for a subset of slices.
    test_seq : int numpy array
        Sequence of test images to be used
    imsave_flag : boolean
        If True, the network outputs and their comparison with the ground truth will be saved in the 'results' folder
    
    Returns
    -------
    list
        List of metrics describing the network accuracy: (average MSE, std of MSE, average SSIM, std of SSIM)
    """
    
    # Full list of apple labels
    apple_numbers = np.array(['31101', '31102', '31103', '31104', '31105', '31106', '31107', '31108', '31109', '31110', '31111', '31112', '31113', '31114', '31115', '31116', '31117', '31118', '31119', '31120', '31121', '31122', '31201', '31202', '31203', '31204', '31205', '31206', '31207', '31208', '31209', '31210', '31211', '31212', '31213', '31214', '31215', '31216', '31217', '31218', '31219', '31220', '31221', '31222', '31301', '31302', '31303', '31304', '31305', '31306', '31307', '31308', '31309', '31310', '31311', '31312', '31313', '31314', '31315', '31316', '31317', '31318', '31319', '31320', '31321', '31322', '32101', '32102', '32103', '32104', '32105', '32106', '32107', '32108', '32109', '32110', '32111', '32112', '32113', '32114', '32115', '32116', '32117', '32118', '32119', '32120', '32121', '32122', '32201', '32202', '32203', '32204', '32205', '32206'])

    # Test sequence array is used to select the labels to use in testing
    test_seq = test_seq - 1
    apple_mask = np.zeros((94), dtype = np.bool)
    test_mask = apple_mask
    test_mask[test_seq] = True
    test_apples = apple_numbers[test_mask]

    # flsin and flstg contain slice names that will be used in testing
    flsin = []
    flstg = []
    for number in test_apples:
        for gt_num in gt_mask:
            flsin.extend(Path(input_folder).glob('*_{}_{}.tif'.format(number, gt_num)))
            flstg.extend(Path(target_folder).glob('{}_{}.tif'.format(number, gt_num)))

    flsin = sorted(flsin)
    flstg = sorted(flstg)

    # Make folder for output
    outfolder = Path('results')
    outfolder.mkdir(exist_ok=True)
    tifffolder = Path('results/tiff')
    tifffolder.mkdir(exist_ok=True)
    compfolder = Path('results/comp')
    compfolder.mkdir(exist_ok=True)
    
    metrics = np.zeros((len(flsin), 4))

    # Load network from file
    n = msdnet.network.MSDNet.from_file('regr_params.h5', gpu=True)

    # Process all test images
    for i in tqdm(range(len(flsin))):
        # Create datapoint with only input image
        d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
        tg = msdnet.data.ImageFileDataPoint(str(flstg[i]))
        # Compute network output
        output = n.forward(d.input)
        image = d.getinputarray()[0]
        ground_truth = tg.getinputarray()[0]
        
        if imsave_flag:
            # Save network output to file
            imageio.imsave(tifffolder / 'regr_{:05d}.tiff'.format(i), output[0])
            # Create and save comparison image
            imageio.imsave(compfolder / 'comp_{:05d}.png'.format(i), msdnet.loggers.stitchimages([image, ground_truth, output[0]], scaleoutput=True))
        
        # Compute and save comparison metrics
        metrics[i,0] = psnr(image, ground_truth)
        metrics[i,1] = psnr(output[0], ground_truth)
        metrics[i,2] = ssim(image, ground_truth)
        metrics[i,3] = ssim(output[0], ground_truth)
        #print("{} - {}", i, flsin[i])
        #print(metrics[i,:])
    
    # Compute mean value and std for psnr and ssim
    psnr_before_mean = metrics[:,0].mean()
    psnr_before_std = metrics[:,0].std()
    psnr_after_mean = metrics[:,1].mean()
    psnr_after_std = metrics[:,1].std()
    ssim_before_mean = metrics[:,2].mean()
    ssim_before_std = metrics[:,2].std()
    ssim_after_mean = metrics[:,3].mean()
    ssim_after_std = metrics[:,3].std()
    return (psnr_before_mean, psnr_before_std, psnr_after_mean, psnr_after_std, ssim_before_mean, ssim_before_std, ssim_after_mean, ssim_after_std)
        
if __name__ == "__main__":
    # Replace with the position of input and target files
    input_folder = "/path/to/input/files"
    target_folder = "/path/to/target/files"
    
    gt_mask = np.concatenate((
                  np.arange(195, 215, 1),
                  np.arange(365, 405, 1),
                  np.arange(525, 545, 1)
              ), axis = 0)
    
    test_seq = np.array([6, 12, 15, 21, 22, 28, 35, 42, 46, 47, 49, 52, 53, 54, 67, 72, 77, 87, 90, 94])
    
    opts, args = getopt.getopt(sys.argv[1:],"i:t:v")
    for opt, arg in opts:
        if opt == "-i":
            input_folder = arg
        elif opt == "-t":
            target_folder = arg
        elif opt == "-v":
            imsave_flag = True
    
    res = test_nn(input_folder, target_folder, gt_mask, test_seq, False)
    print(res)
