import os
import numpy as np
from pyaet.resire import RESIRE_Reconstructor
from pyaet.resire.reconstruct import reconstruct

# Add paths (equivalent to MATLAB's addpath)
# import sys
# sys.path.append('src/')
# sys.path.append('src/splinterp/')


def main_reconstruction(projections_file_path, angles_file_path, resire_param, output_fn):
    """
    Run main reconstruction.

    Args:
        projections_file_path (str): File path. Projections, in the shape of (300, 300, 55).
        angles_file_path (str): File path. Angles, in the shape of (55, 3).
        resire_param (dist): The dictionary to update RESIRE parameters.
        output_fn (str): The output python pickle filename and .npy filename, Ex: 'reconstruction_volume'. Write in the same folder of input projection file.
    """
    # Define file paths
    # pj_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_projections_amorphous.npy'
    # angle_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_angles_amorphous.npy'
    # output_fn = 'reconstruction_volume'

    # Create an instance of the RESIRE_Reconstructor class
    RESIRE = RESIRE_Reconstructor.RESIRE_Reconstructor()

    # Set parameters
    RESIRE.filename_Projections = projections_file_path
    RESIRE.filename_Angles = angles_file_path
    output_file_path = os.path.join(os.path.dirname(projections_file_path), output_fn)
    RESIRE.filename_Results = output_file_path

    # RESIRE.set_parameters(
    #     oversampling_ratio=3,
    #     num_iterations=2,
    #     monitor_R=True,
    #     monitorR_loopLength=1,
    #     gridding_method=1,
    #     vector3=[1, 0, 0],
    #     use_parallel=True,
    #     save_temp=True,
    #     save_loopLength=1
    # )

    RESIRE.set_parameters(resire_param)


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
    np.save(output_file_path+".npy", Reconstruction)
    RESIRE.save_results()
    print("reconstruction finished.")
    return

# pj_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_projections_amorphous.npy'
# angle_filename = '/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/sample_angles_amorphous.npy'
# output_fn = 'reconstruction_volume'
#
# resire_param = {
#     "oversampling_ratio": 3,
#     "num_iterations": 2,
#     "monitor_R": True,
#     "monitorR_loopLength": 2,
#     "gridding_method": 1,
#     "vector3": [1, 0, 0],
#     "use_parallel": True,
#     "save_temp": True,
#     "save_loopLength": 5
# }
#
# main_reconstruction(pj_filename, angle_filename, resire_param, output_fn)
