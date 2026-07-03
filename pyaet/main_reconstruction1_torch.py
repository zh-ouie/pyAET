import csv
import os
import sys
import time
import argparse

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


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = RELEASE_ROOT / "data"
OUTPUT_ROOT = RELEASE_ROOT / "outputs"

# ========================= User settings =========================
# Edit this block for a normal run, then execute:
#     python pyaet/main_reconstruction1_torch.py
PROJECTIONS_FILE_PATH = DATA_ROOT / "1_Measured_data" / "Projections.mat"
ANGLES_FILE_PATH = DATA_ROOT / "1_Measured_data" / "Angles.mat"
OUTPUT_STEM = OUTPUT_ROOT / "step1" / "MG_reconstruction_volume"

# DEVICE:
#   "auto" -> CUDA if available, otherwise torch CPU
#   "cuda" -> require CUDA
#   "cpu"  -> force torch CPU
DEVICE = "auto"

OVERSAMPLING_RATIO = 4
NUM_ITERATIONS = 200
MONITOR_LOOP_LENGTH = 20
MONITOR_R = False

BACKPROJ_BACKEND = "grid_sample"  # "grid_sample" or "torch_loop"
BACKPROJ_CHUNK_SIZE = 5
PRECOMPUTE_BACKPROJ_GRID = None   # None chooses automatically by GPU memory
BACKPROJ_ROT_ON_DEMAND = None     # None chooses automatically
AVOID_FFTSHIFT_COPY = None        # None chooses automatically
INTERP3_CHUNK_SIZE = None         # None chooses automatically
SAVE_PICKLE = False
# ================================================================


def _resolve_device(requested):
    import torch

    requested = str(requested).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {requested!r}. Use auto, cpu, or cuda.")


def _cuda_total_memory_gb():
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _build_resire_param(args):
    device = _resolve_device(args.device)
    precompute_grid = args.precompute_backproj_grid
    if precompute_grid is None:
        precompute_grid = device == "cuda" and _cuda_total_memory_gb() >= 70.0
    backproj_rot_on_demand = args.backproj_rot_on_demand
    if backproj_rot_on_demand is None:
        backproj_rot_on_demand = device == "cuda" and not precompute_grid
    avoid_fftshift_copy = args.avoid_fftshift_copy
    if avoid_fftshift_copy is None:
        avoid_fftshift_copy = device == "cuda"
    interp3_chunk_size = args.interp3_chunk_size
    if interp3_chunk_size is None:
        interp3_chunk_size = 5 if device == "cuda" else 0
    return {
        "oversampling_ratio": args.oversampling_ratio,
        "num_iterations": args.iterations,
        "monitor_R": args.monitor_r,
        "monitorR_loopLength": args.monitor_loop_length,
        "gridding_method": 1,
        "vector3": [1, 0, 0],
        "use_parallel": True,
        "save_temp": False,
        "dtype": "float32",
        "gpu_grad_device": device,
        "gpu_clear_cache_each_iter": False,
        "gpu_sync_timing": False,
        "gpu_backproj_backend": args.backproj_backend,
        "gpu_backproj_chunk_size": args.backproj_chunk_size,
        "gpu_precompute_backproj_grid": precompute_grid,
        "gpu_backproj_rot_on_demand": backproj_rot_on_demand,
        "gpu_avoid_fftshift_copy": avoid_fftshift_copy,
        "gpu_interp3_chunk_size": interp3_chunk_size,
        "save_pickle": args.save_pickle,
    }


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
    output_path = Path(output_fn)
    if output_path.is_absolute():
        output_file_path = str(output_path)
    elif output_path.parent == Path("."):
        output_file_path = str(Path(projections_file_path).parent / output_path)
    else:
        output_file_path = str(output_path)
    resire.filename_Results = output_file_path

    extra_device = resire_param.get("gpu_grad_device")
    save_pickle = bool(resire_param.get("save_pickle", True))
    local_only_keys = {
        "gpu_grad_device",
        "save_pickle",
        "gpu_clear_cache_each_iter",
        "gpu_sync_timing",
        "gpu_backproj_backend",
        "gpu_backproj_chunk_size",
        "gpu_precompute_backproj_grid",
        "gpu_interp3_chunk_size",
        "gpu_backproj_rot_on_demand",
        "gpu_avoid_fftshift_copy",
    }
    valid_resire_param = {k: v for k, v in resire_param.items() if k not in local_only_keys}
    resire.set_parameters(valid_resire_param)
    if extra_device is not None:
        setattr(resire, "gpu_grad_device", extra_device)
    if "gpu_clear_cache_each_iter" in resire_param:
        setattr(resire, "gpu_clear_cache_each_iter", bool(resire_param["gpu_clear_cache_each_iter"]))
    if "gpu_sync_timing" in resire_param:
        setattr(resire, "gpu_sync_timing", bool(resire_param["gpu_sync_timing"]))
    if "gpu_backproj_backend" in resire_param:
        setattr(resire, "gpu_backproj_backend", resire_param["gpu_backproj_backend"])
    if "gpu_backproj_chunk_size" in resire_param:
        setattr(resire, "gpu_backproj_chunk_size", int(resire_param["gpu_backproj_chunk_size"]))
    if "gpu_precompute_backproj_grid" in resire_param:
        setattr(resire, "gpu_precompute_backproj_grid", bool(resire_param["gpu_precompute_backproj_grid"]))
    if "gpu_interp3_chunk_size" in resire_param:
        setattr(resire, "gpu_interp3_chunk_size", int(resire_param["gpu_interp3_chunk_size"]))
    if "gpu_backproj_rot_on_demand" in resire_param:
        setattr(resire, "gpu_backproj_rot_on_demand", bool(resire_param["gpu_backproj_rot_on_demand"]))
    if "gpu_avoid_fftshift_copy" in resire_param:
        setattr(resire, "gpu_avoid_fftshift_copy", bool(resire_param["gpu_avoid_fftshift_copy"]))

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
                "gridding_grid_sample_s",
                "gridding_grid_sample_precompute_s",
                "reconstruct_total_s",
                "grad_backend",
                "grad_backend_s",
                "descent_update_only_s",
                "mex_function3_s",
                "mex_function2_backproj_s",
                "torch_fft_s",
                "torch_interp3_s",
                "torch_ifft2_s",
                "torch_interp2_backproj_s",
                "grid_sample_backproj_s",
                "grid_sample_precompute_s",
                "grid_sample_chunk_size",
                "precompute_grid_sample_grid",
                "backproj_rot_on_demand",
                "avoid_fftshift_copy",
                "interp3_chunk_size",
            ])
            writer.writerow([
                f"{execution_time:.9f}",
                f"{memory_used:.9f}",
                f"{peak/1024/1024:.9f}",
                f"{perf_stats.get('gridding_total_s', 0.0):.9f}",
                f"{perf_stats.get('interp_pj_realspace_torch', {}).get('grid_sample_gridding_s', 0.0):.9f}",
                f"{perf_stats.get('interp_pj_realspace_torch', {}).get('grid_sample_precompute_s', 0.0):.9f}",
                f"{recon_perf.get('reconstruct_total_s', 0.0):.9f}",
                recon_perf.get("grad_backend", "unknown"),
                f"{recon_perf.get('grad_backend_s', 0.0):.9f}",
                f"{recon_perf.get('descent_update_only_s', 0.0):.9f}",
                f"{recon_perf.get('mex_function3_s', 0.0):.9f}",
                f"{recon_perf.get('mex_function2_backproj_s', 0.0):.9f}",
                f"{recon_perf.get('torch_fft_s', 0.0):.9f}",
                f"{recon_perf.get('torch_interp3_s', 0.0):.9f}",
                f"{recon_perf.get('torch_ifft2_s', 0.0):.9f}",
                f"{recon_perf.get('torch_interp2_backproj_s', 0.0):.9f}",
                f"{recon_perf.get('grid_sample_backproj_s', 0.0):.9f}",
                f"{recon_perf.get('grid_sample_precompute_s', 0.0):.9f}",
                f"{recon_perf.get('grid_sample_chunk_size', 0):.0f}",
                str(bool(recon_perf.get("precompute_grid_sample_grid", False))),
                str(bool(recon_perf.get("backproj_rot_on_demand", False))),
                str(bool(recon_perf.get("avoid_fftshift_copy", False))),
                f"{recon_perf.get('interp3_chunk_size', 0):.0f}",
            ])

    np.save(output_file_path + ".npy", reconstruction)
    if save_pickle:
        resire.save_results()

    if perf_stats:
        print("Performance summary:")
        if "gridding_total_s" in perf_stats:
            print(f"  gridding_total_s = {perf_stats['gridding_total_s']:.3f}")
        interp_torch_perf = perf_stats.get("interp_pj_realspace_torch", {})
        if interp_torch_perf:
            print(f"  gridding_torch_interp2_s = {interp_torch_perf.get('torch_interp2_gridding_s', 0.0):.3f}")
            print(f"  gridding_grid_sample_s = {interp_torch_perf.get('grid_sample_gridding_s', 0.0):.3f}")
            print(f"  gridding_grid_sample_precompute_s = {interp_torch_perf.get('grid_sample_precompute_s', 0.0):.3f}")
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
            print(f"  torch_fft_s = {recon_perf.get('torch_fft_s', 0.0):.3f}")
            print(f"  torch_interp3_s = {recon_perf.get('torch_interp3_s', 0.0):.3f}")
            print(f"  torch_ifft2_s = {recon_perf.get('torch_ifft2_s', 0.0):.3f}")
            print(f"  torch_interp2_backproj_s = {recon_perf.get('torch_interp2_backproj_s', 0.0):.3f}")
            print(f"  grid_sample_backproj_s = {recon_perf.get('grid_sample_backproj_s', 0.0):.3f}")
            print(f"  grid_sample_precompute_s = {recon_perf.get('grid_sample_precompute_s', 0.0):.3f}")
            print(f"  grid_sample_chunk_size = {recon_perf.get('grid_sample_chunk_size', 0)}")
            print(f"  precompute_grid_sample_grid = {recon_perf.get('precompute_grid_sample_grid', False)}")
            print(f"  backproj_rot_on_demand = {recon_perf.get('backproj_rot_on_demand', False)}")
            print(f"  avoid_fftshift_copy = {recon_perf.get('avoid_fftshift_copy', False)}")
            print(f"  interp3_chunk_size = {recon_perf.get('interp3_chunk_size', 0)}")

    print("reconstruction finished.")
    return


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Step1 reconstruction. Uses the same Torch code on CPU or CUDA."
    )
    parser.add_argument(
        "--projections",
        default=str(PROJECTIONS_FILE_PATH),
        help="Path to Projections.mat or .npy.",
    )
    parser.add_argument(
        "--angles",
        default=str(ANGLES_FILE_PATH),
        help="Path to Angles.mat or .npy.",
    )
    parser.add_argument(
        "--output-stem",
        default=str(OUTPUT_STEM),
        help="Output file stem without .npy.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument("--oversampling-ratio", type=int, default=OVERSAMPLING_RATIO)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--monitor-loop-length", type=int, default=MONITOR_LOOP_LENGTH)
    parser.add_argument("--monitor-r", action="store_true", default=MONITOR_R)
    parser.add_argument("--backproj-backend", choices=["grid_sample", "torch_loop"], default=BACKPROJ_BACKEND)
    parser.add_argument("--backproj-chunk-size", type=int, default=BACKPROJ_CHUNK_SIZE)
    parser.add_argument(
        "--precompute-backproj-grid",
        action=argparse.BooleanOptionalAction,
        default=PRECOMPUTE_BACKPROJ_GRID,
        help="Precompute backprojection grids on the selected torch device. Default: enabled on large-memory CUDA GPUs.",
    )
    parser.add_argument(
        "--backproj-rot-on-demand",
        action=argparse.BooleanOptionalAction,
        default=BACKPROJ_ROT_ON_DEMAND,
        help="Build backprojection rotation grids chunk by chunk. Default: enabled for CUDA when grids are not precomputed.",
    )
    parser.add_argument(
        "--avoid-fftshift-copy",
        action=argparse.BooleanOptionalAction,
        default=AVOID_FFTSHIFT_COPY,
        help="Avoid an extra fftshift copy during Step1 FFT. Default: enabled for CUDA.",
    )
    parser.add_argument(
        "--interp3-chunk-size",
        type=int,
        default=INTERP3_CHUNK_SIZE,
        help="Number of projections per 3D interpolation chunk. Default: 5 on CUDA, unchunked on CPU.",
    )
    parser.add_argument("--save-pickle", action="store_true", default=SAVE_PICKLE)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main_reconstruction(
        args.projections,
        args.angles,
        _build_resire_param(args),
        args.output_stem,
    )


__all__ = [
    "RESIRE_Reconstructor",
    "reconstruct",
    "main_reconstruction",
]
