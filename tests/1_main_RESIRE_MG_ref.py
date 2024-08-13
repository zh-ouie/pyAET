import numpy as np
from pyaet.resire import RESIRE_Reconstructor
from pyaet.resire.reconstruct import reconstruct

# Add paths (equivalent to MATLAB's addpath)
# import sys
# sys.path.append('src/')
# sys.path.append('src/splinterp/')

# Define file paths
angle_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/1Angles.npy'
pj_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/1Projections.npy'
results_filename = 'output/RESIRE_experiment_result_full'

# Create an instance of the RESIRE_Reconstructor class
RESIRE = RESIRE_Reconstructor.RESIRE_Reconstructor()

# Set parameters
RESIRE.filename_Projections = pj_filename
RESIRE.filename_Angles = angle_filename
RESIRE.filename_Results = results_filename

RESIRE.set_parameters(
    oversampling_ratio = 4,
    num_iterations = 5, #200
    monitor_R = True,
    monitorR_loopLength = 2, #20
    gridding_method = 1,
    vector3 = [1, 0, 0],
    use_parallel = True
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
np.save('reconstruction_volume_full.npy', Reconstruction)
RESIRE.save_results()
