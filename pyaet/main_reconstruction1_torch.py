import csv
import os
import sys
import time

import numpy as np
import psutil
import tracemalloc
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pyaet.resire_torch import RESIRE_Reconstructor
from pyaet.resire_torch.reconstruct import reconstruct


def main_reconstruction(projections_file_path, angles_file_path, resire_param, output_fn):
    print("check input parameter:")
    print("projections_file_path:", projections_file_path)
    print("angles_file_path:", angles_file_path)
    print("resire_param:", resire_param)
    print("output_fn:", output_fn)
    print("PID:", os.getpid())
    print("SPLINTERP_NUM_THREADS:", os.environ.get("SPLINTERP_NUM_THREADS", "unset"))
    print("mode:", "torch_grad_update")

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

    extra_device = resire_param.get("gpu_grad_device")
    valid_resire_param = {k: v for k, v in resire_param.items() if k != "gpu_grad_device"}
    resire.set_parameters(valid_resire_param)
    if extra_device is not None:
        setattr(resire, "gpu_grad_device", extra_device)

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
    perf_stats = getattr(resire, "perf_stats", {})
    recon_perf = perf_stats.get("reconstruct_torch", {})
    if recon_perf:
        iter_csv = output_file_path + "_iter_timing.csv"
        with open(iter_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "grad_update_s", "grad_backend"])
            for idx, value in enumerate(recon_perf.get("iter_grad_update_s", []), start=1):
                writer.writerow([idx, f"{value:.9f}", recon_perf.get("grad_backend", "unknown")])

        descent_iter_csv = output_file_path + "_iter_descent_update_only.csv"
        with open(descent_iter_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "descent_update_only_s", "grad_backend"])
            for idx, value in enumerate(recon_perf.get("iter_descent_update_only_s", []), start=1):
                writer.writerow([idx, f"{value:.9f}", recon_perf.get("grad_backend", "unknown")])

        summary_csv = output_file_path + "_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "execution_time_s",
                "memory_increase_mb",
                "peak_tracemalloc_mb",
                "gridding_total_s",
                "reconstruct_total_s",
                "grad_backend",
                "grad_backend_s",
                "descent_update_only_s",
                "mex_function3_s",
                "mex_function2_backproj_s",
            ])
            writer.writerow([
                f"{execution_time:.9f}",
                f"{memory_used:.9f}",
                f"{peak/1024/1024:.9f}",
                f"{perf_stats.get('gridding_total_s', 0.0):.9f}",
                f"{recon_perf.get('reconstruct_total_s', 0.0):.9f}",
                recon_perf.get("grad_backend", "unknown"),
                f"{recon_perf.get('grad_backend_s', 0.0):.9f}",
                f"{recon_perf.get('descent_update_only_s', 0.0):.9f}",
                f"{recon_perf.get('mex_function3_s', 0.0):.9f}",
                f"{recon_perf.get('mex_function2_backproj_s', 0.0):.9f}",
            ])

    np.save(output_file_path + ".npy", reconstruction)
    resire.save_results()

    if perf_stats:
        print("Performance summary:")
        if "gridding_total_s" in perf_stats:
            print(f"  gridding_total_s = {perf_stats['gridding_total_s']:.3f}")
        interp_perf = perf_stats.get("interp_pj_realspace", {})
        if interp_perf:
            print(f"  gridding_mex_function2_calls = {interp_perf.get('mex_function2_gridding_calls', 0)}")
            print(f"  gridding_mex_function2_s = {interp_perf.get('mex_function2_gridding_s', 0.0):.3f}")
        if recon_perf:
            print(f"  reconstruct_total_s = {recon_perf.get('reconstruct_total_s', 0.0):.3f}")
            print(f"  grad_backend = {recon_perf.get('grad_backend', 'unknown')}")
            print(f"  grad_backend_s = {recon_perf.get('grad_backend_s', 0.0):.3f}")
            print(f"  descent_update_only_s = {recon_perf.get('descent_update_only_s', 0.0):.3f}")
            print(f"  mex_function3_calls = {recon_perf.get('mex_function3_calls', 0)}")
            print(f"  mex_function3_s = {recon_perf.get('mex_function3_s', 0.0):.3f}")
            print(f"  mex_function2_backproj_calls = {recon_perf.get('mex_function2_backproj_calls', 0)}")
            print(f"  mex_function2_backproj_s = {recon_perf.get('mex_function2_backproj_s', 0.0):.3f}")

    print("reconstruction finished.")
    return


# Edit this block before running the script directly.
# These defaults are aligned with the formal MATLAB MG reconstruction settings.
# The repository does not package the raw MG Projections/Angles files, so set
# these paths to your local MG inputs before running.
PROJECTIONS_FILE_PATH = "/path/to/Projections.mat"
ANGLES_FILE_PATH = "/path/to/Angles.mat"
OUTPUT_FN = "MG_reconstruction_volume_torch"

RESIRE_PARAM = {
    "oversampling_ratio": 4,
    "num_iterations": 200,
    "monitor_R": True,
    "monitorR_loopLength": 20,
    "gridding_method": 1,
    "vector3": [1, 0, 0],
    "use_parallel": True,
    "dtype": "float32",
    "gpu_grad_device": "cuda",
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
