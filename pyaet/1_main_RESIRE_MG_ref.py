from RESIRE_Reconstructor.RESIRE_Reconstructor import RESIRE_Reconstructor as rr

# Add paths (equivalent to MATLAB's addpath)
import sys
sys.path.append('src/')
sys.path.append('src/splinterp/')

# Define file paths
pj_filename = 'input/Projections.mat'
angle_filename = 'input/Angles.mat'
results_filename = 'output/RESIRE_experiment_result.mat'

# Create an instance of the RESIRE_Reconstructor class
RESIRE = rr()
 
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
RESIRE = rr.readFiles(RESIRE)

# Check and prepare data (assuming you have appropriate functions for this)
RESIRE = rr.CheckPrepareData(RESIRE)

# Run gridding (assuming you have appropriate functions for this)
RESIRE = rr.runGridding(RESIRE)

# Reconstruct (assuming you have appropriate functions for this)
RESIRE = rr.reconstruct(RESIRE)

# Clear calculation variables
RESIRE = rr.ClearCalcVariables(RESIRE)

# Get the reconstruction result
Reconstruction = RESIRE.reconstruction

# Save results (assuming you have appropriate functions for saving)
rr.SaveResults(RESIRE)
