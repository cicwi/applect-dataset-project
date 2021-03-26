# applect-data-project
This repository includes scripts supporting the [AppleCT Datasets](https://arxiv.org/abs/2012.13346) manuscript, henceforth referred to as *the submission*. 

## AppleCT Datasets
In the submission, we present three parallel-beam tomographic datasets of 94 apples with internal defects along with defect label information. The datasets are prepared for development and testing of data-driven, learning-based image reconstruction, segmentation and post-processing methods. 

The three versions are a noiseless simulation (*Dataset A*); simulation with added Gaussian noise (*Dataset B*), and with X-ray scattering (*Dataset C*). The datasets are based on real 3D X-ray CT data and their subsequent volume reconstructions. The ground truth images, generated based on the volume reconstructions, are also available through this project, see the links in **Dataset access**. 

Apples contain various defects, and if ignored, the data split may naturally introduce label bias. We reduce label bias by formulating the data split as an optimization problem, which we solve using two methods: a simple heuristic algorithm and through mixed integer quadratic programming. This ensures the datasets can be split into test, training or validation subsets with the label bias eliminated. We refer the readers to the submission for the technical details. 

The datasets are suitable for image reconstruction, segmentation, automatic defect detection, and studying the effects (as well as testing the elimination) of label bias in machine learning.


### Codes
Scripts are arranged into subfolders depending on their use.

* *Dataset_Generation*: This folder contains scripts for generating Datasets A and B.
* *Scattering*: This folder contains scripts for generating Dataset C.
* *bias_elimination*: This contains the two bias elimination algorithms introduced in the submission, both in Python and MATLAB syntax. The input files *apple_defects_full.csv* and *apple_defects_partial.csv* are available via the Zenodo page for the Datasets A-C (see link below). 
* *Technical_Validation*: This contains the scripts used to obtain the results in **Technical Validation** in the submission. 


## Dataset access
The datasets and the ground truth reconstructions are made available via Zenodo. 

* [Simulated Datasets A-C](https://zenodo.org/record/4212301)
* [Ground Truth Reconstructions 1 of 6](https://zenodo.org/record/4550729)
* [Ground Truth Reconstructions 2 of 6](https://zenodo.org/record/4575904)
* [Ground Truth Reconstructions 3 of 6](https://zenodo.org/record/4576078)
* [Ground Truth Reconstructions 4 of 6](https://zenodo.org/record/4576122)
* [Ground Truth Reconstructions 5 of 6](https://zenodo.org/record/4576202)
* [Ground Truth Reconstructions 6 of 6](https://zenodo.org/record/4576260)



## Contributions or suggestions 
External contributions to the codes are currently not allowed. However, we welcome any feedback or suggestions for making the scripts more user-friendly, please email the corresponding authors listed in the submission. 
