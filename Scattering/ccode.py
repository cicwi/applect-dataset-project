"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.

Code is available via AppleCT Dataset Project; www.github.com/cicwi/applect-dataset-project

Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
Parallel-beam X-ray CT datasets of apples with internal defects and label balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346

Dataset available via Zenodo; 10.5281/zenodo.4212301
"""
import ctypes
from pathlib import Path
import numpy as np

"""
This is Python wrapper to the C code used for scattering computation.
"""

# Location of the C code file
libpath = Path(__file__).parent / 'libccode_c.so'
lib = ctypes.CDLL(str(libpath))

# Necessary variable types
asint = ctypes.c_uint32
asfloat = ctypes.c_float
cfloatp = ctypes.POINTER(ctypes.c_float)
def asfloatp(arr):
    return arr.ctypes.data_as(cfloatp)

def simulate_scattering(height, width, px_size, blur_par):
    """Calls the "opt_array_scattering" function.
        
    Arguments:
    height -- image height
    width -- image width
    px_size -- pixel size in mm
    blur_par -- array of blur parameters for every pixel of the image
    """
    # Allocate an array where the results will be stored
    tmp = np.zeros((height, width), dtype = np.float32)
    
    height_c = asint(height)
    width_c = asint(width)
    px_c = asfloat(px_size)
    
    # Create an array for the parameter A with a form of the image
    A = np.zeros((height, width), dtype = np.float32)
    A[:, :] = blur_par[:, :, 0]
    A_c = asfloatp(A)
    
    # Create an array for the parameter B with a form of the image
    B = np.zeros((height, width), dtype = np.float32)
    B[:, :] = blur_par[:, :, 1]
    B_c = asfloatp(B)
    
    # Create an array for the parameter sigma1 with a form of the image
    sigma1 = np.zeros((height, width), dtype = np.float32)
    sigma1[:, :] = blur_par[:, :, 2]
    sigma1_c = asfloatp(sigma1)
    
    # Create an array for the parameter sigma2 with a form of the image
    sigma2 = np.zeros((height, width), dtype = np.float32)
    sigma2[:, :] = blur_par[:, :, 3]
    sigma2_c = asfloatp(sigma2)
    
    lib.opt_array_scattering(height_c, width_c, px_c, A_c, B_c, sigma1_c, sigma2_c, asfloatp(tmp))
    return tmp

def target_row_scattering(target_row_range, height, width, px_size, blur_par):
    """Calls the "target_row_scattering" function.
        
    Arguments:
    target_row_range -- [min, max] for a range of rows
    height -- image height
    width -- image width
    px_size -- pixel size in mm
    blur_par -- array of blur parameters for every pixel of the image
    """
    
    # Allocate an array where the results will be stored
    tmp = np.zeros((target_row_range[1] - target_row_range[0], width), dtype = np.float32)
    
    row_start_c = asint(target_row_range[0])
    row_end_c = asint(target_row_range[1])
    height_c = asint(height)
    width_c = asint(width)
    px_c = asfloat(px_size)
    
    # Create an array for the parameter A with a form of the image
    A = np.zeros((height, width), dtype = np.float32)
    A[:, :] = blur_par[:, :, 0]
    A_c = asfloatp(A)
    
    # Create an array for the parameter B with a form of the image
    B = np.zeros((height, width), dtype = np.float32)
    B[:, :] = blur_par[:, :, 1]
    B_c = asfloatp(B)
    
    # Create an array for the parameter sigma1 with a form of the image
    sigma1 = np.zeros((height, width), dtype = np.float32)
    sigma1[:, :] = blur_par[:, :, 2]
    sigma1_c = asfloatp(sigma1)
    
    # Create an array for the parameter sigma2 with a form of the image
    sigma2 = np.zeros((height, width), dtype = np.float32)
    sigma2[:, :] = blur_par[:, :, 3]
    sigma2_c = asfloatp(sigma2)
                
    lib.target_row_scattering(row_start_c, row_end_c, height_c, width_c, px_c, A_c, B_c, sigma1_c, sigma2_c, asfloatp(tmp))
    return tmp
