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

MIN_PT = [-1.0, -1.0]
MAX_PT = [1.0, 1.0]

# ray transform and fbp operator for full data
reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=RECO_IM_SHAPE, dtype=np.float32)
reco_geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=NUM_ANGLES)
reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl=IMPL)
fbp_op = odl.tomo.analytic.fbp_op(reco_ray_trafo, filter_type='Shepp-Logan')

# specify missing wedge configs 1 and 2
angles = reco_geometry.angles
l1 = 10
u1 = 40
l2 = 15
u2 = 35
mw1 = angles[l1:u1]
mw2 = angles[np.hstack((np.arange(0,l2), np.arange(u2,50)))]


# operators for mw1
apart_mw1 = odl.nonuniform_partition(mw1)
reco_geometry_mw1 = odl.tomo.geometry.Parallel2dGeometry(apart_mw1, reco_geometry.det_partition)
reco_ray_trafo_mw1 = odl.tomo.RayTransform(reco_space, reco_geometry_mw1, impl=IMPL)
fbp_op_mw1 = odl.tomo.analytic.fbp_op(reco_ray_trafo_mw1)

# operators for mw2
apart_mw2 = odl.nonuniform_partition(mw2)
reco_geometry_mw2 = odl.tomo.geometry.Parallel2dGeometry(apart_mw2, reco_geometry.det_partition)
reco_ray_trafo_mw2 = odl.tomo.RayTransform(reco_space, reco_geometry_mw2, impl=IMPL)
fbp_op_mw2 = odl.tomo.analytic.fbp_op(reco_ray_trafo_mw2)


def fbp_reco(file):
    """ Generates fbo reconstructions from data.
        
    Parameters
    ----------
    file : str
        Data filename.
    
    Returns
    -------
    reco       : numpy array
                 Reconstruction of full data.
    reco_mw1   : numpy array
                 Reconstruction of mw1 data
    reco_mw2   : numpy array
                 Reconstruction of mw2 data
    """
    # read sinogram
    data = tifffile.imread(file)
    
    # create missing wedge sinograms
    data_mw1 = data[l1:u1]
    data_mw2 = data[np.hstack((np.arange(0,l2), np.arange(u2,50)))]
    
    # reconstruct
    reco = fbp_op(reco_ray_trafo.range.element(data))
    reco_mw1 = fbp_op_mw1(reco_ray_trafo_mw1.range.element(data_mw1))
    reco_mw2 = fbp_op_mw2(reco_ray_trafo_mw2.range.element(data_mw2))
    
    return reco, reco_mw1, reco_mw2

if __name__ == "__main__":
    
    opts, args = getopt.getopt(sys.argv[1:],"d:f:m:n:")
    for opt, arg in opts:
        if opt == "-d":
            source_dir = arg
        elif opt == "-f":
            save_dir_reco = arg
        elif opt == "-m":
            save_dir_reco_mw1 = arg
        elif opt == "-n":
            save_dir_reco_mw2 = arg
    
    os.makedirs(save_dir_reco, exist_ok=True)
    os.makedirs(save_dir_reco_mw1, exist_ok=True)
    os.makedirs(save_dir_reco_mw2, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir)]
    for f in files:
        r, r_mw1, r_mw2 = fbp_reco(source_dir+'/'+f)
        tifffile.imsave(save_dir_reco+'/reco_'+f, r.astype(np.float32))
        tifffile.imsave(save_dir_reco_mw1+'/reco_mw1_'+f, r_mw1.astype(np.float32))
        tifffile.imsave(save_dir_reco_mw2+'/reco_mw2_'+f, r_mw2.astype(np.float32))
