from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat

# Allow direct execution: python pyaet/main_position_refinement4.py
if __package__ is None or __package__ == "":
    _PKG_ROOT = Path(__file__).resolve().parents[1]
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

from pyaet.src.cal_Bproj_2type_fast import (
    cal_Bproj_2type2,
    cal_Bproj_2type2_cuda_release_stats_paired,
)
from pyaet.src.gradient_B_2type_difB import gradient_B_2type_difB_torch
from pyaet.src.gradient_fixHB_XYZ import gradient_fixHB_XYZ_torch
from pyaet.src.io_helper import read_mat_file
from pyaet.src.matlab_trust_region_lsq import (
    least_squares_matlab_trust_region,
    least_squares_matlab_trust_region_stats,
)
from pyaet.src.my_paddzero import my_paddzero


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = RELEASE_ROOT / "data"
OUTPUT_ROOT = RELEASE_ROOT / "outputs"

HALF_WIDTH = 4
RESOLUTION_ANGSTROM = 0.347
Z_BY_TYPE = [28, 45, 78]
DEFAULT_OUTER_ITERATIONS = 10
INNER_GRADIENT_ITERATIONS = 10
MATLAB_FTOL = 1e-12
H1_BOUND_EPS = 1e-6

# ========================= User settings =========================
# Edit this block for a normal run, then execute:
#     python pyaet/main_position_refinement4.py
PROJECTIONS_FILE_PATH = DATA_ROOT / "1_Measured_data" / "Projections.mat"
ANGLES_FILE_PATH = DATA_ROOT / "1_Measured_data" / "Angles.mat"

# Default Step4 input is the MATLAB-aligned refinement input.
# To refine the output of Step2/Step3 instead, change these two paths to:
#     OUTPUT_ROOT / "step2" / "traced_model_inPixel.npy"
#     OUTPUT_ROOT / "step3" / "Local_classification_type.npy"
MODEL_FILE_PATH = DATA_ROOT / "5_Position_refinement" / "input" / "Local_classification_coord_OriOri.mat"
ATOMS_FILE_PATH = DATA_ROOT / "5_Position_refinement" / "input" / "Local_classification_type.mat"

OUTPUT_STEM = OUTPUT_ROOT / "step4" / "model_refined_res"

# DEVICE:
#   "auto" -> CUDA if available, otherwise CPU release-parity projector
#   "cuda" -> require CUDA
#   "cpu"  -> force CPU path
DEVICE = "auto"

NUM_OUTER_ITERATIONS = DEFAULT_OUTER_ITERATIONS
# ================================================================

_IGNORED_STEP4_ENV = {
    "PYAET_BPROJ_BACKEND",
    "PYAET_BPROJ_DEVICE",
    "PYAET_BPROJ_DTYPE",
    "PYAET_BPROJ_CUDA_ACCUM_DTYPE",
    "PYAET_BGRAD_BACKEND",
    "PYAET_BGRAD_DEVICE",
    "PYAET_BGRAD_DTYPE",
    "PYAET_XYZ_BACKEND",
    "PYAET_XYZ_DEVICE",
    "PYAET_XYZ_DTYPE",
    "PYAET_POSREF_EXACT_H1_BOUND",
    "PYAET_POSREF_H1_BOUND_EPS",
    "PYAET_POSREF_LSQ_BPROJ_BACKEND",
    "PYAET_POSREF_LSQ_BPROJ_DTYPE",
    "PYAET_POSREF_LSQ_EFFECTIVE_MAX_EVALS",
    "PYAET_POSREF_LSQ_DIFF_STEP",
    "PYAET_POSREF_LSQ_FD_WORKERS",
    "PYAET_POSREF_LSQ_FREE_ONLY",
    "PYAET_POSREF_LSQ_MAX_EVALS",
    "PYAET_POSREF_LSQ_PARALLEL_BASE",
    "PYAET_POSREF_LSQ_REL_STEP",
    "PYAET_POSREF_LSQ_SOLVER",
    "PYAET_POSREF_LSQ_USE_ANALYTIC_STATS",
    "PYAET_POSREF_LSQ_USE_PAIRED64_STATS",
    "PYAET_POSREF_LSQ_USE_PAIRED_STATS",
    "PYAET_POSREF_LSQ_USE_STATS",
    "PYAET_POSREF_SAVE_MAT",
    "PYAET_POSREF_SKIP_LSQ",
    "PYAET_POSREF_SKIP_PROJECTION_CHECK",
}


@dataclass(frozen=True)
class Step4Runtime:
    device: str
    lsq_bproj_backend: str
    use_cuda_paired_stats: bool

    @property
    def bgrad_function(self):
        if self.device == "cuda":
            from pyaet.src.gradient_B_2type_difB_cuda import gradient_B_2type_difB_cuda

            return gradient_B_2type_difB_cuda
        return gradient_B_2type_difB_torch

    @property
    def xyz_function(self):
        if self.device == "cuda":
            from pyaet.src.gradient_fixHB_XYZ_cuda import gradient_fixHB_XYZ_cuda

            return gradient_fixHB_XYZ_cuda
        return gradient_fixHB_XYZ_torch


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment error.
        raise ImportError(
            "pyAET Step4 requires PyTorch. Install torch before running position refinement."
        ) from exc
    return torch


def _clean_step4_environment() -> None:
    removed = [name for name in sorted(_IGNORED_STEP4_ENV) if name in os.environ]
    for name in removed:
        os.environ.pop(name, None)
    if removed:
        print("Ignoring Step4 experimental environment overrides: " + ", ".join(removed))


def _resolve_runtime(device: str) -> Step4Runtime:
    torch = _require_torch()
    requested = str(device).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported Step4 device: {device!r}. Use auto, cpu, or cuda.")

    if requested == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = requested

    if resolved == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return Step4Runtime(
            device="cuda",
            lsq_bproj_backend="cuda_release",
            use_cuda_paired_stats=True,
        )

    return Step4Runtime(
        device="cpu",
        lsq_bproj_backend="numba",
        use_cuda_paired_stats=False,
    )


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return np.asarray(read_mat_file(str(path)))
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    raise ValueError(f"Unsupported input file type: {path}")


def _normalise_model(model: np.ndarray) -> np.ndarray:
    model = np.asarray(model, dtype=np.float64)
    if model.shape[0] != 3 and model.shape[-1] == 3:
        model = model.T
    if model.shape[0] != 3:
        raise ValueError(f"Step4 model must have shape (3, N); got {model.shape}")
    return np.ascontiguousarray(model)


def _normalise_atoms(atoms: np.ndarray) -> np.ndarray:
    atoms = np.asarray(atoms)
    if atoms.ndim == 1:
        atoms = atoms.reshape(1, -1)
    return atoms


def _output_stem(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".npy", ".mat", ".npz"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_xdata(model: np.ndarray, atoms: np.ndarray, angles: np.ndarray, runtime: Step4Runtime) -> dict:
    xdata = {
        "Res": RESOLUTION_ANGSTROM,
        "Z_arr": Z_BY_TYPE,
        "half_width": HALF_WIDTH,
        "atoms": atoms,
        "model": model,
        "angles": angles,
        "bproj_backend": runtime.lsq_bproj_backend,
        "bproj_device": runtime.device,
        "bproj_cuda_accum_dtype": "float64",
        "bgrad_device": runtime.device,
        "xyz_device": runtime.device,
        "bgrad_torch_dtype": "float32",
        "xyz_torch_dtype": "float32",
    }
    return xdata


def _run_matlab_aligned_lsq(
    x0: np.ndarray,
    xdata: dict,
    projections: np.ndarray,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    runtime: Step4Runtime,
):
    yflat = projections.ravel(order="F")
    max_fun_evals = 100 * x0.size

    if runtime.use_cuda_paired_stats:
        def eval_stats(params, eval_lb, eval_ub):
            return cal_Bproj_2type2_cuda_release_stats_paired(params, xdata, eval_lb, eval_ub)

        return least_squares_matlab_trust_region_stats(
            eval_stats,
            x0,
            lb_vec,
            ub_vec,
            tol_fun=MATLAB_FTOL,
            max_fun_evals=max_fun_evals,
            soft_max_fun_evals=None,
            max_iter=400,
        )

    def residual(params):
        y_pred, _ = cal_Bproj_2type2(params, xdata)
        return y_pred.ravel(order="F") - yflat

    return least_squares_matlab_trust_region(
        residual,
        x0,
        lb_vec,
        ub_vec,
        tol_fun=MATLAB_FTOL,
        max_fun_evals=max_fun_evals,
        soft_max_fun_evals=None,
        max_iter=400,
    )


def main_position_refinement(
    projections_file_path: str,
    angles_file_path: str,
    model_file_path: str,
    atoms_file_path: str,
    num_iterations: int = DEFAULT_OUTER_ITERATIONS,
    output_fn: str = str(OUTPUT_ROOT / "step4" / "model_refined_res"),
    device: str = "auto",
):
    """
    Run the clean Step4 position-refinement path.

    The active path is MATLAB-aligned: ten outer iterations by default, full
    MaxFunEvals for the trust-region LSQ step, CUDA paired-stats projector on
    GPU, and the release-parity CPU projector when CUDA is unavailable.
    """

    _clean_step4_environment()
    runtime = _resolve_runtime(device)
    print(
        "Step4 clean runtime: "
        f"device={runtime.device}, "
        f"lsq_bproj_backend={runtime.lsq_bproj_backend}, "
        f"lsq_max_fun_evals=full600, "
        f"outer_iterations={num_iterations}"
    )

    projections = np.asarray(_load_array(projections_file_path), dtype=np.float64)
    angles = np.asarray(_load_array(angles_file_path), dtype=np.float64)
    model = _normalise_model(_load_array(model_file_path))
    atoms = _normalise_atoms(_load_array(atoms_file_path))
    output_file_path = str(_output_stem(output_fn))

    projections = np.maximum(projections, 0)
    projections = projections[1:, 1:, :]
    projections = my_paddzero(
        projections,
        np.array(projections.shape) + np.array([50, 50, 0]),
        dtype=np.float64,
    )

    xdata = _prepare_xdata(model, atoms, angles, runtime)
    para0 = np.array([[1, 1.36, 2.58], [13.3, 13.3, 13.3]], dtype=np.float64)
    lb = np.array([[1 - H1_BOUND_EPS, 1, 1], [5, 5, 5]], dtype=np.float64)
    ub = np.array([[1 + H1_BOUND_EPS, 2, 3], [15, 15, 15]], dtype=np.float64)
    model_refined = model.copy()

    hb_history = []
    hb_after_b_history = []
    errR_b_history = []
    errR_xyz_history = []
    lsq_nfev_history = []
    lsq_cost_history = []
    lsq_status_history = []

    for outer_idx in range(int(num_iterations)):
        outer_t0 = time.perf_counter()
        print(f"Iteration num: {outer_idx + 1}")

        x0 = para0.copy()
        x0[0, :] /= x0[0, 0]
        x0 = x0.reshape(-1, order="F")
        lb_vec = lb.reshape(-1, order="F")
        ub_vec = ub.reshape(-1, order="F")

        xdata["model"] = model_refined.copy()
        xdata["model_ori"] = model_refined.copy()
        xdata["projections"] = projections

        print(
            "LSQ step: "
            "solver=matlab_trust_region, "
            "max_evals=600, "
            f"backend={runtime.lsq_bproj_backend}, "
            f"cuda_paired_stats={int(runtime.use_cuda_paired_stats)}, "
            f"h1_bound_eps={H1_BOUND_EPS:g}"
        )
        lsq_t0 = time.perf_counter()
        lsq_result = _run_matlab_aligned_lsq(
            x0,
            xdata,
            projections,
            lb_vec,
            ub_vec,
            runtime,
        )
        lsq_s = time.perf_counter() - lsq_t0
        print(
            f"LSQ step time: {lsq_s:.3f} s, "
            f"nfev={lsq_result.nfev}, cost={lsq_result.cost:.6g}, status={lsq_result.status}"
        )

        para0 = lsq_result.x.reshape(para0.shape, order="F")
        lsq_nfev_history.append(lsq_result.nfev)
        lsq_cost_history.append(lsq_result.cost)
        lsq_status_history.append(lsq_result.status)

        print(f"H1 = {para0[0, 0]:.3f}, H2 = {para0[0, 1]:.3f}, H3 = {para0[0, 2]:.3f}")
        print(f"B1 = {para0[1, 0]:.3f}, B2 = {para0[1, 1]:.3f}, B3 = {para0[1, 2]:.3f}")
        hb_history.append(para0.copy())

        xdata["step_sz"] = 1
        xdata["iterations"] = INNER_GRADIENT_ITERATIONS
        bgrad_t0 = time.perf_counter()
        _, para0, errR = runtime.bgrad_function(para0, xdata, projections)
        print(f"B-gradient time: {time.perf_counter() - bgrad_t0:.3f} s")
        hb_after_b_history.append(para0.copy())
        errR_b_history.append(np.asarray(errR, dtype=np.float64).copy())

        xdata["step_sz"] = 1
        xdata["iterations"] = INNER_GRADIENT_ITERATIONS
        xyz_t0 = time.perf_counter()
        _, para, errR, _ = runtime.xyz_function(para0, xdata, projections)
        print(f"XYZ-gradient time: {time.perf_counter() - xyz_t0:.3f} s")
        errR_xyz_history.append(np.asarray(errR, dtype=np.float64).copy())
        model_refined = para[2:5, :]
        print(f"Outer iteration time: {time.perf_counter() - outer_t0:.3f} s")

    model_refined_res = np.asarray(model_refined, dtype=np.float64)
    np.save(output_file_path + ".npy", model_refined_res)
    savemat(output_file_path + ".mat", {"model_refined_res": model_refined_res})

    hb_hist_arr = np.stack(hb_history, axis=0) if hb_history else np.zeros((0, 2, 3), dtype=np.float64)
    hb_after_b_arr = (
        np.stack(hb_after_b_history, axis=0)
        if hb_after_b_history
        else np.zeros((0, 2, 3), dtype=np.float64)
    )
    errR_b_arr = np.array(errR_b_history, dtype=np.float64)
    errR_xyz_arr = np.array(errR_xyz_history, dtype=np.float64)
    lsq_nfev_arr = np.asarray(lsq_nfev_history, dtype=np.float64)
    lsq_cost_arr = np.asarray(lsq_cost_history, dtype=np.float64)
    lsq_status_arr = np.asarray(lsq_status_history, dtype=np.float64)

    np.savez(
        output_file_path + "_compare_stats.npz",
        hb_history=hb_hist_arr,
        hb_after_b_history=hb_after_b_arr,
        errR_b_history=errR_b_arr,
        errR_xyz_history=errR_xyz_arr,
        lsq_nfev_history=lsq_nfev_arr,
        lsq_cost_history=lsq_cost_arr,
        lsq_status_history=lsq_status_arr,
    )
    savemat(
        output_file_path + "_compare_stats.mat",
        {
            "hb_history": hb_hist_arr,
            "hb_after_b_history": hb_after_b_arr,
            "errR_b_history": errR_b_arr,
            "errR_xyz_history": errR_xyz_arr,
            "lsq_nfev_history": lsq_nfev_arr,
            "lsq_cost_history": lsq_cost_arr,
            "lsq_status_history": lsq_status_arr,
        },
    )
    print(f"position refinement finished. Saved: {output_file_path}.npy/.mat")
    print(f"compare stats saved: {output_file_path}_compare_stats.npz/.mat")
    return {
        "model_refined_res": model_refined_res,
        "hb_history": hb_hist_arr,
        "hb_after_b_history": hb_after_b_arr,
        "errR_b_history": errR_b_history,
        "errR_xyz_history": errR_xyz_history,
        "output_file_path": output_file_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean Step4 position refinement.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument(
        "--projections",
        default=str(PROJECTIONS_FILE_PATH),
        help="Path to projections file (.mat or .npy).",
    )
    parser.add_argument(
        "--angles",
        default=str(ANGLES_FILE_PATH),
        help="Path to angles file (.mat or .npy).",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_FILE_PATH),
        help="Path to Step4 coordinate model file (.mat or .npy), shape (3, N).",
    )
    parser.add_argument(
        "--atoms",
        default=str(ATOMS_FILE_PATH),
        help="Path to atom-type file (.mat or .npy).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_OUTER_ITERATIONS,
        help=f"Number of outer refinement iterations. Default: {DEFAULT_OUTER_ITERATIONS}.",
    )
    parser.add_argument(
        "--output-stem",
        default=str(OUTPUT_STEM),
        help="Output file stem without extension.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main_position_refinement(
        projections_file_path=args.projections,
        angles_file_path=args.angles,
        model_file_path=args.model,
        atoms_file_path=args.atoms,
        num_iterations=args.iterations,
        output_fn=args.output_stem,
        device=args.device,
    )
