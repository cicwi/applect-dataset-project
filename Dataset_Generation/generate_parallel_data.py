"""
Copyright (c) 2020 Poulami Somanya Ganguly
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.
Code is available via LINKXXXX.
Reference paper: XXXXX
"""
import numpy as np
import astra
import odl
import os
from skimage.transform import resize
import tifffile
import sys, getopt

# create parallel beam geometry in ODL

# number of projection angles
NUM_ANGLES = 50 

# shape of ground truth images
RECO_IM_SHAPE = (972,972)

# shape of resized ground truth images
IM_SHAPE = (1000,1000)

# backend used for projection
IMPL='astra_cuda'

# definition of projection and reconstruction spaces in ODL
MIN_PT = [-1.0, -1.0]
MAX_PT = [1.0, 1.0]
reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                          dtype=np.float64)

# definition of projection geometries
reco_geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=NUM_ANGLES)
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=NUM_ANGLES, det_shape=reco_geometry.detector.shape)

# definition of ray transform
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

def forward_projection(file):
    """ Generates parallel beam projection data from ground truth.
        
    Parameters
    ----------
    file : str
        Ground truth filename.
    
    Returns
    -------
    data       : numpy array
                 Noiseless parallel beam projection data.
    data_noisy : numpy array
                 Noisy parallel beam projection data
    """
    # read tif file
    image_org = tifffile.imread(file)
    
    # resize image
    image = resize(image_org, IM_SHAPE, order=1)
    
    # forward project
    data = ray_trafo(image)
    
    # add 5% gaussian noise
    data_noisy = data + odl.phantom.white_noise(ray_trafo.range, seed=None) * np.mean(data) * 0.05
    
    return data.asarray(), data_noisy.asarray()

if __name__ == "__main__":
    
    opts, args = getopt.getopt(sys.argv[1:],"s:d:n:")
    for opt, arg in opts:
        if opt == "-s":
            source_dir = arg
        elif opt == "-d":
            save_dir_data = arg
        elif opt == "-n":
            save_dir_data_noisy = arg
    
    os.makedirs(save_dir_data, exist_ok=True)
    os.makedirs(save_dir_data_noisy, exist_ok=True)
    
    
    files = [f for f in os.listdir(source_dir)]
    for f in files:
        d, d_noisy = forward_projection(source_dir+'/'+f)
        tifffile.imsave(save_dir_data+'/data_'+f, d.astype(np.float32))
        tifffile.imsave(save_dir_data_noisy+'/data_noisy_'+f, d_noisy.astype(np.float32))
        print('Saving proj of '+str(f))