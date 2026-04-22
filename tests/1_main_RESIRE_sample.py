import os
import numpy as np
from pyaet.resire_numpy import RESIRE_Reconstructor
from pyaet.resire_numpy.reconstruct import reconstruct

# Add paths (equivalent to MATLAB's addpath)
# import sys
# sys.path.append('src/')
# sys.path.append('src/splinterp/')

# Define file paths
pj_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_projections_amorphous.npy'
angle_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_angles_amorphous.npy'
results_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/output/sample_amorphous_res'

# Create an instance of the RESIRE_Reconstructor class
RESIRE = RESIRE_Reconstructor()

# Set parameters
RESIRE.filename_Projections = pj_filename
RESIRE.filename_Angles = angle_filename
RESIRE.filename_Results = results_filename

RESIRE.set_parameters(
    oversampling_ratio = 3,
    num_iterations = 2,
    monitor_R = True,
    monitorR_loopLength = 1,
    gridding_method = 1,
    vector3 = [1, 0, 0],
    use_parallel = True,
    save_temp = True,
    save_loopLength = 1
)

# Read files
RESIRE.read_files()

# Check and prepare data
RESIRE.check_prepare_data()

# Run gridding
RESIRE.run_gridding()

# Reconstruct
reconstruct(RESIRE)

# Clear calculation variables
# RESIRE.clear_calc_variables()

# Get the reconstruction result
Reconstruction = RESIRE.reconstruction

# Save results
np.save('reconstruction_volume.npy', Reconstruction)
RESIRE.save_results()
print("done")
