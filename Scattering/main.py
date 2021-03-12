"""
Copyright (c) 2020 Vladyslav Andriiashen
Centrum Wiskunde & Informatica, Amsterdam, the Netherlands.

Code is available via AppleCT Dataset Project; www.github.com/cicwi/applect-dataset-project

Referenced paper: S.B. Coban, V. Andriiashen, P.S. Ganguly, et al.
Parallel-beam X-ray CT datasets of apples with internal defects and label balancing for machine learning. 2020. www.arxiv.org/abs/2012.13346

Dataset available via Zenodo; 10.5281/zenodo.4212301
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import ccode as wrap
from pathlib import Path
from tifffile import TiffFile
from tifffile import TiffWriter
from tqdm import tqdm
import sys, getopt

class ScatteringFilter(object):
    def __init__(self, proj, data_fname = "data.csv", pixel_size = 0.15, alpha = 1.0):
        """Initialization of the scattering filter.
        
        Arguments:
        proj -- CT projection after flatfield correction and logarithm
        data_fname -- path to the file containing scattering properties, default is data.csv
        pixel_size -- pixel size in mm, default is 150 um
        alpha -- scattering intensity scaling factor, default is 1.0
        """
        
        self.log_proj = proj
        self.par = MCparameters(data_fname, alpha)
        self.pixel_size = pixel_size # in mm
    
    def simulate_scattering(self):
        """Compute scattering photon distribution.
        
        It is expected that flatfield values are uniformly distributed over the detector plane.
        Output values are fractions of the constant flatfield value.
        """
        res = np.zeros_like(self.log_proj)
        height = self.log_proj.shape[0]
        width = self.log_proj.shape[1]
        
        # Blur parameters are calculated for every pixel of the projection.
        blur_par = np.zeros((height, width, 4))
        blur_par[:,:,0] = self.par.get_A(self.log_proj)
        blur_par[:,:,1] = self.par.get_B(self.log_proj)
        blur_par[:,:,2] = self.par.get_sigma1(self.log_proj)
        blur_par[:,:,3] = self.par.get_sigma2(self.log_proj)
        
        # Call to the C function
        res = wrap.simulate_scattering(height, width, self.pixel_size, blur_par)

        return res
    
    def simulate_row_scattering(self, row_range):
        """Compute scattering for a set of rows in the image.
        Arguments:
        row_range -- [min, max] for a range of rows
        
        It is expected that flatfield values are uniformly distributed over the detector plane.
        Output values are fractions of the constant flatfield value.
        """
        height = self.log_proj.shape[0]
        width = self.log_proj.shape[1]
        res = np.zeros((row_range[1] - row_range[0], width))
        
        # Blur parameters are calculated for every pixel of the projection.
        # Scattering in a single row depends on the whole image.
        blur_par = np.zeros((height, width, 4))
        blur_par[:,:,0] = self.par.get_A(self.log_proj)
        blur_par[:,:,1] = self.par.get_B(self.log_proj)
        blur_par[:,:,2] = self.par.get_sigma1(self.log_proj)
        blur_par[:,:,3] = self.par.get_sigma2(self.log_proj)
        
        # Call to the C function
        res = wrap.target_row_scattering((row_range[0], row_range[1]), height, width, self.pixel_size, blur_par)
        
        return res

class MCparameters(object):
    def __init__(self, data_file, alpha):
        """Initialization of the object containing scattering parameters extracted from the Monte-Carlo simulation.
        
        Arguments:
        data_file -- path to the file containing scattering properties
        alpha -- scattering intensity scaling factor
        """
        self.data = pd.read_csv(data_file)
        self.alpha = alpha
    
    def get_A(self, inp):
        """Interpolate value of the parameter A for the pixel intensity inp."""
        xp = self.data["Attenuation"].to_numpy()
        fp = self.data["A"].to_numpy()
        return self.alpha * np.interp(inp, xp, fp)
    def get_B(self, inp):
        """Interpolate value of the parameter B for the pixel intensity inp."""
        xp = self.data["Attenuation"].to_numpy()
        fp = self.data["B"].to_numpy()
        return self.alpha * np.interp(inp, xp, fp)
    def get_sigma1(self, inp):
        """Interpolate value of the parameter sigma1 for the pixel intensity inp."""
        xp = self.data["Attenuation"].to_numpy()
        fp = self.data["Sigma1_mm"].to_numpy()
        return np.interp(inp, xp, fp)
    def get_sigma2(self, inp):
        """Interpolate value of the parameter sigma2 for the pixel intensity inp."""
        xp = self.data["Attenuation"].to_numpy()
        fp = self.data["Sigma2_mm"].to_numpy()
        return np.interp(inp, xp, fp)

def compute_full_projections(input_folder, output_folder):
    """Compute a set of scattering distibutions for a group of projections
    
    Arguments:
    input_folder - path to the projections saved in tif files.
    output_folder - path to the output: photon count divided by flatfield and resulting projections are saved.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    photon_folder = output_folder / "raw_data"
    log_folder = output_folder / "log_data"
    inputs = sorted(input_folder.glob("*.tif"))
    
    #if the input files are not logarithmed, set this to True.
    not_log = True
    
    for i in tqdm(range(len(inputs))):
        fname = inputs[i]
        # Read the tif file
        with TiffFile(fname) as f:
            image = f.asarray()
        # Compute -log if necessary
        if not_log:
            image = -np.log(image)
        # Initialize filter object
        filt = ScatteringFilter(image)
        # Calculate scattering
        scattering = filt.simulate_scattering()
        
        # Write a raw distribution of scattered photons
        phot_name = photon_folder / fname.parts[-1]
        with TiffWriter(phot_name, imagej=True) as f:
            f.save(scattering.astype(np.float32))
        
        # Convert the input image to the photon counts and add scattered photons.
        raw_photon_data = np.exp(-image)
        raw_photon_data += scattering
        res = -np.log(raw_photon_data)
        
        # Write images with a scattering effect
        res_name = log_folder / fname.parts[-1]
        with TiffWriter(res_name, imagej=True) as f:
            f.save(res.astype(np.float32))
                
def show_scattering_comparison(input_folder, output_folder):
    """Draws a comparison between images without scattering from input folder and projections with scatterin from output folder."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    photon_folder = output_folder / "raw_data"
    log_folder = output_folder / "log_data"
    inputs = sorted(input_folder.glob("*.tif"))
    
    #if the input files are not logarithmed, set this to True.
    not_log = True
    
    for i in range(len(inputs[:1])):
        fname = inputs[i]
        # Read image without scattering
        with TiffFile(fname) as f:
            image = f.asarray()
            
        # Compute -log if necessary
        if not_log:
            image = -np.log(image)
        
        # Read a distibution of scattered photons
        phot_name = photon_folder / fname.parts[-1]
        with TiffFile(phot_name) as f:
            scattering = f.asarray()
        
        # Convert image without scattering to the photon distribution
        raw_photon_data = np.exp(-image)
        
        fig=plt.figure(figsize = (9, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        # Plot original photon distribution
        plt.imshow(raw_photon_data)
        ax2 = fig.add_subplot(1, 3, 2)
        # Plot a distribution of scattered photons
        plt.imshow(scattering)
        ax3 = fig.add_subplot(1, 3, 3)
        # Plot a ratio between added scattering effect and original image
        plt.imshow(np.divide(scattering, raw_photon_data))
        ax1.title.set_text("Original projection")
        ax2.title.set_text("Simulated scattering")
        ax3.title.set_text("SNR")
        plt.tight_layout()
        plt.show()
                
def sino_processing(input_folder, output_folder, sample_name, value_scale, alpha, row_range):
    """Compute scattering for a certain range of rows for all slices of a given apple
    
    Arguments:
    input_folder -- path to the projections saved in tif files.
    output_folder -- path to the output: photon count divided by flatfield and resulting projections are saved.
    sample_name -- label of the apple of interest.
    value_scale -- arbitrary parameter that adjusts values in the codesprint projections in a way that they resemble typical apple absorption.
    alpha -- scattering intensity scaling parameter
    row_range -- [min, max] for a range of rows
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    photon_folder = output_folder / "raw_data"
    log_folder = output_folder / "log_data"
    name_mask = "*_{}_*.tif".format(sample_name)

    # Get image dimensions and read content to the numpy array
    inputs = sorted(input_folder.glob(name_mask))
    with TiffFile(inputs[0]) as f:
        im_shape = np.array(f.asarray()).shape
    print(im_shape)
    proj_volume = np.zeros((len(inputs), im_shape[0], im_shape[1]))
    print(proj_volume.shape)
    for i in range(len(inputs)):
        with TiffFile(inputs[i]) as f:
            proj_volume[i] = f.asarray()
            
    # scaling to get realistic absorption
    proj_volume *= value_scale
    
    # Create array for the scattering signal
    scat_photon_volume = np.zeros((row_range[1] - row_range[0], proj_volume.shape[1], proj_volume.shape[2]))
    res_volume = np.zeros_like(scat_photon_volume)
    
    # Calculate scattering for slices
    for i in tqdm(range(proj_volume.shape[1])):
        filt = ScatteringFilter(proj_volume[:,i,:], alpha = alpha)
        tmp = filt.simulate_row_scattering(row_range)
        scat_photon_volume[:,i,:] = tmp
    
    raw_photon_data = np.exp(-proj_volume[row_range[0]:row_range[1], :, :])
    raw_photon_data += scat_photon_volume
    res_volume = -np.log(raw_photon_data)
    
    # Write the results
    for i in range(row_range[0], row_range[1]):
        phot_name = photon_folder / "data_{}_{}.tif".format(sample_name, i)
        log_name = log_folder / "data_{}_{}.tif".format(sample_name, i)
        with TiffWriter(phot_name, imagej=True) as f:
            f.save(scat_photon_volume[i - row_range[0],:,:].astype(np.float32))
        with TiffWriter(log_name, imagej=True) as f:
            f.save(res_volume[i - row_range[0],:,:].astype(np.float32))

def get_name_masks(input_folder):
    """Get apple labels in the folder."""
    input_folder = Path(input_folder)
    inputs = sorted(input_folder.glob("*000.tif"))
    select = []
    for name in inputs:
        parts = str(name.parts[-1]).split("_")
        select.append(parts[-2])
    print(select)
    return select

if __name__ == "__main__":
    # Folder to read images without scattering
    input_folder = "/path/to/input/"
    # Folder to write images with scattering. It should contain "raw_data" and "log_data" subfolders
    output_folder = "/path/to/output"
    
    value_scale = 400
    alpha = 5.0
    
    row_range = [375, 385]
    
    opts, args = getopt.getopt(sys.argv[1:],"i:o:a:b:e:")
    for opt, arg in opts:
        if opt == "-i":
            input_folder = arg
        elif opt == "-o":
            output_folder = arg
        elif opt == "-a":
            alpha = float(arg)
        elif opt == "-b":
            row_range[0] = int(arg)
        elif opt == "-e":
            row_range[1] = int(arg)
    
    select = get_name_masks(input_folder)[:]
    for num in select:
        sino_processing(input_folder, output_folder, num, value_scale, alpha, row_range)
