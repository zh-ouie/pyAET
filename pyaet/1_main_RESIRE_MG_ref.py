import os
import numpy as np
# from splinterp import RESIRE_Reconstructor

# Add paths (equivalent to MATLAB's addpath)
import sys
sys.path.append('src/')
sys.path.append('src/splinterp/')

# Define file paths
pj_filename = 'input/Projections.mat'
angle_filename = 'input/Angles.mat'
results_filename = 'output/RESIRE_experiment_result.mat'

# Create an instance of the RESIRE_Reconstructor class
RESIRE = RESIRE_Reconstructor()

# Set parameters
RESIRE.filename_Projections = pj_filename
RESIRE.filename_Angles = angle_filename
RESIRE.filename_Results = results_filename

RESIRE.set_parameters(
    oversamplingRatio=4,
    numIterations=200,
    monitor_R=True,
    monitorR_loopLength=20,
    griddingMethod=1,
    vector3=[1, 0, 0],
    use_parallel=1
)

# Read files (assuming you have appropriate functions for file reading)
RESIRE = readFiles(RESIRE)

# Check and prepare data (assuming you have appropriate functions for this)
RESIRE = CheckPrepareData(RESIRE)

# Run gridding (assuming you have appropriate functions for this)
RESIRE = runGridding(RESIRE)

# Reconstruct (assuming you have appropriate functions for this)
RESIRE = reconstruct(RESIRE)

# Clear calculation variables
RESIRE = ClearCalcVariables(RESIRE)

# Get the reconstruction result
Reconstruction = RESIRE.reconstruction

# Save results (assuming you have appropriate functions for saving)
SaveResults(RESIRE)
