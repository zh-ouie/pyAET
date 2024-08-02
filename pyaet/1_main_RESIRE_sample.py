import os
import numpy as np
from pyaet.resire import RESIRE_Reconstructor
from pyaet.resire.reconstruct import reconstruct

# Add paths (equivalent to MATLAB's addpath)
import sys
sys.path.append('src/')
sys.path.append('src/splinterp/')

# Define file paths
pj_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_projections_amorphous.npy'
angle_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_angles_amorphous.npy'
results_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/output/sample_amorphous_res'

# Create an instance of the RESIRE_Reconstructor class
RESIRE = RESIRE_Reconstructor.RESIRE_Reconstructor()

# Set parameters
RESIRE.filename_Projections = pj_filename
RESIRE.filename_Angles = angle_filename
RESIRE.filename_Results = results_filename

RESIRE.set_parameters(
    oversamplingRatio=3,
    numIterations=2,
    monitor_R=True,
    monitorR_loopLength=1,
    griddingMethod=1,
    vector3=[1, 0, 0],
    use_parallel=1,
    save_temp=1,
    save_loopLength=1
)

# Read files
RESIRE.readFiles()

# Check and prepare data
RESIRE.CheckPrepareData()

# Run gridding
RESIRE.runGridding()

# Reconstruct
reconstruct(RESIRE)

# Clear calculation variables
# RESIRE.ClearCalcVariables()

# Get the reconstruction result
Reconstruction = RESIRE.reconstruction

# Save results
np.save('reconstruction_volume.npy', Reconstruction)
RESIRE.SaveResults()
print("done")