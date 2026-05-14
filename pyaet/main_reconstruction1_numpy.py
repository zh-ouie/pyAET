"""NumPy RESIRE reconstruction entry point.

This module is the NumPy counterpart of ``main_reconstruction1_torch.py``.
It runs the MATLAB-style RESIRE reconstruction pipeline with NumPy arrays and
the C++ splinterp extension that wraps the MATLAB MEX interpolation routines.
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pyaet.resire_numpy import RESIRE_Reconstructor
from pyaet.resire_numpy.reconstruct import reconstruct


def main_reconstruction(projections_file_path, angles_file_path, resire_param, output_fn):
    """Run NumPy/C++ RESIRE reconstruction.

    Args:
        projections_file_path (str): Path to projection data. The expected
            array shape is ``(height, width, num_projections)``.
        angles_file_path (str): Path to Euler angle data. The expected shape is
            ``(num_projections, 3)``; a single-angle column is expanded to
            MATLAB-compatible three-angle form by ``RESIRE_Reconstructor``.
        resire_param (dict): RESIRE parameters, such as
            ``oversampling_ratio``, ``num_iterations``, ``monitor_R``,
            ``gridding_method``, ``vector3``, and ``dtype``.
        output_fn (str): Output file stem. ``.npy`` and ``.pkl`` outputs are
            written next to ``projections_file_path``.

    Returns:
        None. Results are written to disk.
    """
    print("check input parameter:")
    print("projections_file_path:", projections_file_path)
    print("angles_file_path:", angles_file_path)
    print("resire_param:", resire_param)
    print("output_fn:", output_fn)
    print("PID:", os.getpid())
    print("SPLINTERP_NUM_THREADS:", os.environ.get("SPLINTERP_NUM_THREADS", "unset"))
    print("mode:", "numpy_cpp")

    start_time = time.time()
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024

    print(f"Start running - Memory: {start_memory:.1f} MB")
    print("==============")

    resire = RESIRE_Reconstructor()
    resire.filename_Projections = projections_file_path
    resire.filename_Angles = angles_file_path
    output_file_path = os.path.join(os.path.dirname(projections_file_path), output_fn)
    resire.filename_Results = output_file_path

    resire.set_parameters(resire_param)
    resire.read_files()
    resire.check_prepare_data()
    resire.run_gridding()
    reconstruct(resire)

    reconstruction = resire.reconstruction

    end_time = time.time()
    current_memory = process.memory_info().rss / 1024 / 1024
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time = end_time - start_time
    memory_used = current_memory - start_memory
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Memory usage: Current increase {memory_used:.2f} MB, Peak {peak/1024/1024:.2f} MB")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    np.save(output_file_path + ".npy", reconstruction)
    resire.save_results()

    print("reconstruction finished.")
    return


# Edit this block before running the script directly.
# These defaults are aligned with the formal MATLAB MG reconstruction settings.
# The repository does not package the raw MG Projections/Angles files, so set
# these paths to your local MG inputs before running.
PROJECTIONS_FILE_PATH = "/path/to/Projections.mat"
ANGLES_FILE_PATH = "/path/to/Angles.mat"
OUTPUT_FN = "MG_reconstruction_volume"

RESIRE_PARAM = {
    "oversampling_ratio": 4,
    "num_iterations": 200,
    "monitor_R": True,
    "monitorR_loopLength": 20,
    "gridding_method": 1,
    "vector3": [1, 0, 0],
    "use_parallel": True,
    "dtype": "float32",
}


if __name__ == "__main__":
    main_reconstruction(
        PROJECTIONS_FILE_PATH,
        ANGLES_FILE_PATH,
        RESIRE_PARAM,
        OUTPUT_FN,
    )


__all__ = [
    "RESIRE_Reconstructor",
    "reconstruct",
    "main_reconstruction",
    "PROJECTIONS_FILE_PATH",
    "ANGLES_FILE_PATH",
    "OUTPUT_FN",
    "RESIRE_PARAM",
]
