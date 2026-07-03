import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pyaet.src.obtain_tight_support import (
    eroded_support_contains_points,
    obtain_tight_support_multi,
    prepare_tight_support_base,
)
from pyaet.src.my_round import my_round_num
from pyaet.src.initial_class_kmean import initial_class_kmean
from pyaet.src.io_helper import read_mat_file


RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = RELEASE_ROOT / "outputs"

# ========================= User settings =========================
# Edit this block for a normal run, then execute:
#     python pyaet/main_polynomial_tracing2.py
VOLUME_FILE_PATH = OUTPUT_ROOT / "step1" / "MG_reconstruction_volume.npy"
OUTPUT_STEM = OUTPUT_ROOT / "step2" / "traced_model_inPixel"

# DEVICE:
#   "auto" -> CUDA if available, otherwise torch CPU
#   "cuda" -> require CUDA
#   "cpu"  -> force torch CPU
DEVICE = "auto"

MAX_NUM_TH = 100000
MIN_DIST_ANGSTROM = 2.0
TORCH_CHUNK = 4096
# ================================================================


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def _runtime_thread_count():
    for name in ("PYAET_NUM_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if not value:
            continue
        try:
            threads = int(value)
        except ValueError:
            continue
        if threads > 0:
            return threads
    return None


def _configure_torch_threads(torch):
    threads = _runtime_thread_count()
    if threads is None:
        return
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(threads, 8)))
    except Exception:
        pass


def _build_fit_coeff():
    fit_coeff = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i + j + k <= 4:
                    if max([i, j, k]) == 4:
                        fit_coeff.append([i, j, k, -1])
                    else:
                        fit_coeff.append([i, j, k, 0])
    return np.array(fit_coeff)


def _term_indices(orders):
    return {tuple(int(v) for v in order): idx for idx, order in enumerate(orders)}


def _prepare_polynomial_grid(search_rad):
    crop_half_size = search_rad
    X, Y, Z = np.meshgrid(np.arange(-crop_half_size, crop_half_size + 1),
                         np.arange(-crop_half_size, crop_half_size + 1),
                         np.arange(-crop_half_size, crop_half_size + 1))
    X = np.transpose(X, (1, 0, 2))
    Y = np.transpose(Y, (1, 0, 2))
    Z = np.transpose(Z, (1, 0, 2))

    sphere_ind = np.where((X**2 + Y**2 + Z**2 <= (search_rad + 0.5)**2).flatten(order='F'))[0]
    sphere_subs = np.unravel_index(sphere_ind, X.shape, order='F')

    return {
        'X': X.flatten(order='F')[sphere_ind].astype(np.float64, copy=False),
        'Y': Y.flatten(order='F')[sphere_ind].astype(np.float64, copy=False),
        'Z': Z.flatten(order='F')[sphere_ind].astype(np.float64, copy=False),
        'sphere_ind': sphere_ind,
        'sphere_subs': tuple(np.asarray(s, dtype=np.intp) for s in sphere_subs),
    }


def _interp_centered_axes(shape):
    axes = []
    targets = []
    for length in shape:
        axis = np.arange(length) - my_round_num((length + 1) / 2) + 1
        target = np.arange(3 * axis[0], axis[-1] * 3 + 1) / 3
        axes.append(axis)
        targets.append(target[2:])
    return axes, targets


def _center_pad_slices(shape, padded_shape):
    slices = []
    for size, padded_size in zip(shape, padded_shape):
        if size % 2 == 0:
            start = (padded_size - size) // 2 if padded_size % 2 == 0 else (padded_size - size - 1) // 2
        else:
            start = (padded_size - size + 1) // 2 if padded_size % 2 == 0 else (padded_size - size) // 2
        slices.append(slice(start, start + size))
    return tuple(slices)


def _torch_device_from_env(torch):
    requested_device = os.environ.get("PYAET_TRACING_DEVICE")
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PYAET_TRACING_DEVICE requests CUDA, but torch.cuda.is_available() is False")
    return device


def _set_tracing_device(device):
    import torch

    _configure_torch_threads(torch)
    device = str(device).strip().lower()
    if device == "auto":
        os.environ.pop("PYAET_TRACING_DEVICE", None)
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        os.environ["PYAET_TRACING_DEVICE"] = "cuda"
        return "cuda"
    if device == "cpu":
        os.environ["PYAET_TRACING_DEVICE"] = "cpu"
        return "cpu"
    if device == "torch":
        os.environ.pop("PYAET_TRACING_DEVICE", None)
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unsupported tracing device: {device!r}. Use auto, cpu, or cuda.")


def _torch_sync_if_needed(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _spline_weight_matrix(length):
    axes, targets = _interp_centered_axes((length,))
    x = axes[0].astype(np.float64, copy=False)
    xi = targets[0].astype(np.float64, copy=False)
    return CubicSpline(
        x,
        np.eye(length, dtype=np.float64),
        axis=0,
        bc_type="not-a-knot",
        extrapolate=False,
    )(xi)


def _upsample3x_spline_torch(data, torch, device):
    data_t = torch.as_tensor(data, dtype=torch.float64, device=device)
    sx, sy, sz = (
        torch.as_tensor(_spline_weight_matrix(length), dtype=torch.float64, device=device)
        for length in data.shape
    )

    out = torch.tensordot(sx, data_t, dims=([1], [0]))
    del data_t, sx
    out = torch.einsum("jy,xyz->xjz", sy, out)
    del sy
    out = torch.einsum("kz,xyz->xyk", sz, out)
    del sz
    return out


def _pad_center_torch(vol, padded_shape, torch):
    padded = torch.zeros(tuple(int(v) for v in padded_shape), dtype=vol.dtype, device=vol.device)
    padded[_center_pad_slices(tuple(vol.shape), tuple(int(v) for v in padded_shape))] = vol
    return padded


def _pad_center_numpy(vol, padded_shape):
    padded = np.zeros(tuple(int(v) for v in padded_shape), dtype=vol.dtype)
    padded[_center_pad_slices(tuple(vol.shape), tuple(int(v) for v in padded_shape))] = vol
    return padded


def _torch_local_maxima_strel3(vol, max_num_th, threshold, torch):
    import torch.nn.functional as F

    neg_inf = torch.finfo(vol.dtype).min
    padded = F.pad(vol[None, None], (1, 1, 1, 1, 1, 1), value=neg_inf)[0, 0]
    dilated = padded[1:-1, 1:-1, 1:-1]
    dilated = torch.maximum(dilated, padded[:-2, 1:-1, 1:-1])
    dilated = torch.maximum(dilated, padded[2:, 1:-1, 1:-1])
    dilated = torch.maximum(dilated, padded[1:-1, :-2, 1:-1])
    dilated = torch.maximum(dilated, padded[1:-1, 2:, 1:-1])
    dilated = torch.maximum(dilated, padded[1:-1, 1:-1, :-2])
    dilated = torch.maximum(dilated, padded[1:-1, 1:-1, 2:])

    mask = (vol == dilated) & (vol > threshold)
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return np.empty((0, 3), dtype=int)
    vals = vol[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = min(int(max_num_th), vals.numel())
    _, order = torch.topk(vals, keep, largest=True, sorted=True)
    return coords[order].detach().cpu().numpy().astype(int, copy=False)


def _spatial_cell(pos, cell_size):
    return tuple(np.floor(pos / cell_size).astype(np.int64))


def _is_duplicate_position(candidate, spatial_hash, cell_size, min_dist):
    if not spatial_hash:
        return False
    base_cell = _spatial_cell(candidate, cell_size)
    min_dist_sq = min_dist * min_dist
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                for accepted in spatial_hash.get(cell, ()):
                    delta = accepted - candidate
                    if float(np.dot(delta, delta)) < min_dist_sq:
                        return True
    return False


def _add_spatial_position(candidate, spatial_hash, cell_size):
    spatial_hash.setdefault(_spatial_cell(candidate, cell_size), []).append(candidate.copy())


def _apply_min_distance_filter(raw_pos_arr, raw_exit_flag_arr, max_XYZ_base, min_dist):
    exit_flag_arr = raw_exit_flag_arr.copy()
    tot_pos_arr = np.zeros_like(raw_pos_arr, dtype=float)
    spatial_hash = {}
    cell_size = max(float(min_dist), np.finfo(float).eps)

    for i in range(len(raw_pos_arr)):
        if raw_exit_flag_arr[i] != 0:
            continue
        candidate = raw_pos_arr[i, :] + max_XYZ_base[i, :]
        if _is_duplicate_position(candidate, spatial_hash, cell_size, min_dist):
            exit_flag_arr[i] = -3
        else:
            tot_pos_arr[i, :] = candidate
            _add_spatial_position(candidate, spatial_hash, cell_size)
    return tot_pos_arr, exit_flag_arr


def _indices_inside_support(points, tight_support):
    if points.size == 0:
        return np.zeros(points.shape[1], dtype=bool)
    coords = np.round(points).astype(int) - 1
    valid = np.all((coords >= 0) & (coords < np.asarray(tight_support.shape)[:, None]), axis=0)
    inside = np.zeros(points.shape[1], dtype=bool)
    if np.any(valid):
        good = coords[:, valid]
        inside[valid] = tight_support[good[0], good[1], good[2]] == 1
    return inside


def _torch_solve_lstsq(torch, A, b, device, dtype):
    mode = os.environ.get("PYAET_TRACING_TORCH_SOLVER", "lstsq").strip().lower()
    if mode in ("normal", "normal_eq", "normal-equation", "normal_equation"):
        ata = torch.matmul(A.transpose(-2, -1), A)
        atb = torch.matmul(A.transpose(-2, -1), b.unsqueeze(-1))
        ridge = float(os.environ.get("PYAET_TRACING_RIDGE", "0"))
        if ridge:
            eye = torch.eye(ata.shape[-1], dtype=dtype, device=device)
            ata = ata + ridge * eye
        return torch.linalg.solve(ata, atb).squeeze(-1)

    if device.type == "cuda":
        return torch.linalg.lstsq(A, b.unsqueeze(-1), driver="gels").solution.squeeze(-1)
    return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)


def _trace_peaks_torch(
    FinalVol,
    max_XYZ,
    max_XYZ_base,
    min_dist,
    fit_coeff,
    orders,
    grid,
    max_iter,
    crit_iter,
    Q,
    alpha,
    crop_half_size,
):
    import torch

    _configure_torch_threads(torch)
    device = _torch_device_from_env(torch)
    dtype = torch.float64
    chunk_size = int(os.environ.get(
        "PYAET_TRACING_TORCH_CHUNK",
        "4096" if device.type == "cuda" else "256",
    ))

    print(
        "tracing solver = torch "
        f"(device={device}, dtype={str(dtype).replace('torch.', '')}, chunk={chunk_size})"
    )
    FinalVol_t = torch.as_tensor(FinalVol, dtype=dtype, device=device)
    sphere_i = torch.as_tensor(grid['sphere_subs'][0], dtype=torch.long, device=device)
    sphere_j = torch.as_tensor(grid['sphere_subs'][1], dtype=torch.long, device=device)
    sphere_k = torch.as_tensor(grid['sphere_subs'][2], dtype=torch.long, device=device)
    X_t = torch.as_tensor(grid['X'], dtype=dtype, device=device)
    Y_t = torch.as_tensor(grid['Y'], dtype=dtype, device=device)
    Z_t = torch.as_tensor(grid['Z'], dtype=dtype, device=device)
    powers_t = torch.arange(int(np.max(orders)) + 1, dtype=dtype, device=device)
    orders_t = torch.as_tensor(orders, dtype=torch.long, device=device)
    term_idx = _term_indices(orders)
    idx = {key: term_idx[key] for key in (
        (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (0, 1, 1), (1, 0, 1),
    )}

    n_peaks = len(max_XYZ)
    raw_pos_arr = np.zeros_like(max_XYZ, dtype=float)
    raw_exit_flag_arr = np.zeros(n_peaks, dtype=int)

    for start in range(0, n_peaks, chunk_size):
        end = min(start + chunk_size, n_peaks)
        base = torch.as_tensor(max_XYZ[start:end], dtype=torch.long, device=device)
        pos = torch.zeros((end - start, 3), dtype=dtype, device=device)
        flags = torch.zeros(end - start, dtype=torch.int16, device=device)
        active = torch.ones(end - start, dtype=torch.bool, device=device)
        consec = torch.zeros(end - start, dtype=torch.int16, device=device)

        for iter_num in range(1, max_iter + 2):
            active_idx = torch.nonzero(active, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                break
            forced_done = iter_num > max_iter
            if forced_done:
                flags[active_idx] = -4

            base_a = base[active_idx]
            pos_a = pos[active_idx]
            ix = base_a[:, 0:1] - crop_half_size + sphere_i[None, :]
            iy = base_a[:, 1:2] - crop_half_size + sphere_j[None, :]
            iz = base_a[:, 2:3] - crop_half_size + sphere_k[None, :]
            crop_values = FinalVol_t[ix, iy, iz]

            dx = X_t[None, :] - pos_a[:, 0:1]
            dy = Y_t[None, :] - pos_a[:, 1:2]
            dz = Z_t[None, :] - pos_a[:, 2:3]
            weight = torch.exp(-alpha * (dx * dx + dy * dy + dz * dz) / (crop_half_size ** 2))
            dx_pow = dx.unsqueeze(-1).pow(powers_t)
            dy_pow = dy.unsqueeze(-1).pow(powers_t)
            dz_pow = dz.unsqueeze(-1).pow(powers_t)
            design = (
                dx_pow.index_select(2, orders_t[:, 0])
                * dy_pow.index_select(2, orders_t[:, 1])
                * dz_pow.index_select(2, orders_t[:, 2])
            )
            A = design * weight.unsqueeze(-1)
            b = crop_values * weight
            coeff = _torch_solve_lstsq(torch, A, b, device, dtype)

            P_200 = coeff[:, idx[(2, 0, 0)]]
            P_020 = coeff[:, idx[(0, 2, 0)]]
            P_002 = coeff[:, idx[(0, 0, 2)]]
            P_100 = coeff[:, idx[(1, 0, 0)]]
            P_010 = coeff[:, idx[(0, 1, 0)]]
            P_001 = coeff[:, idx[(0, 0, 1)]]
            P_110 = coeff[:, idx[(1, 1, 0)]]
            P_011 = coeff[:, idx[(0, 1, 1)]]
            P_101 = coeff[:, idx[(1, 0, 1)]]

            J = -2 * (
                -4 * P_200 * P_020 * P_002
                + P_200 * P_011**2
                + P_020 * P_101**2
                + P_002 * P_110**2
                - P_110 * P_101 * P_011
            )
            valid = J < 0
            safe_J = torch.where(valid, J, torch.ones_like(J))
            dX = (
                -4 * P_100 * P_020 * P_002
                + 2 * P_110 * P_010 * P_002
                + 2 * P_101 * P_020 * P_001
                - P_110 * P_011 * P_001
                - P_101 * P_010 * P_011
                + P_100 * P_011**2
            ) / safe_J
            dY = (
                -4 * P_010 * P_200 * P_002
                + 2 * P_011 * P_001 * P_200
                + 2 * P_110 * P_100 * P_002
                - P_101 * P_011 * P_100
                - P_110 * P_001 * P_101
                + P_010 * P_101**2
            ) / safe_J
            dZ = (
                -4 * P_001 * P_200 * P_020
                + 2 * P_010 * P_011 * P_200
                + 2 * P_101 * P_100 * P_020
                - P_110 * P_011 * P_100
                - P_110 * P_010 * P_101
                + P_001 * P_110**2
            ) / safe_J

            shift = torch.stack((dX, dY, dZ), dim=1).clamp(min=-Q, max=Q)
            flags[active_idx[~valid]] = -1
            valid_idx = active_idx[valid]
            if valid_idx.numel():
                valid_shift = shift[valid]
                pos[valid_idx] = pos[valid_idx] + valid_shift
                boundary = torch.max(torch.abs(pos[valid_idx]), dim=1).values > crop_half_size
                flags[valid_idx[boundary]] = -2

                not_boundary = ~boundary
                nb_idx = valid_idx[not_boundary]
                nb_shift = valid_shift[not_boundary]
                if nb_idx.numel():
                    small = torch.max(torch.abs(nb_shift), dim=1).values < Q
                    success = small & (consec[nb_idx] == crit_iter - 1)
                    flags[nb_idx[success]] = 0

                    keep_small = small & ~success
                    consec[nb_idx[keep_small]] += 1
                    consec[nb_idx[~small]] = 0

            done = ~valid
            if valid_idx.numel():
                valid_done = torch.zeros(valid_idx.numel(), dtype=torch.bool, device=device)
                valid_done[boundary] = True
                if nb_idx.numel():
                    nb_done = success | torch.full_like(success, forced_done)
                    valid_done[not_boundary] = nb_done
                done[valid] = valid_done
            if forced_done:
                done = torch.ones_like(done)
            active[active_idx[done]] = False

        raw_pos_arr[start:end, :] = pos.detach().cpu().numpy()
        raw_exit_flag_arr[start:end] = flags.detach().cpu().numpy().astype(int)
        print(f"torch peak chunk {start}-{end - 1} finished")

    tot_pos_arr, exit_flag_arr = _apply_min_distance_filter(
        raw_pos_arr, raw_exit_flag_arr, max_XYZ_base, min_dist
    )
    if _env_flag("PYAET_TRACING_VERBOSE_PEAKS", True):
        for i in range(0, n_peaks, 500):
            print(f'peak {i}, flag {exit_flag_arr[i]}')
        if n_peaks and (n_peaks - 1) % 500 != 0:
            print(f'peak {n_peaks - 1}, flag {exit_flag_arr[n_peaks - 1]}')
    return raw_pos_arr, tot_pos_arr, exit_flag_arr


def _resolve_solver(solver):
    solver = solver or os.environ.get("PYAET_TRACING_SOLVER", "torch")
    solver = solver.strip().lower()
    if solver in ("torch", "cuda", "gpu"):
        return "torch"
    if solver == "auto":
        return "torch"
    raise ValueError(
        f"Unsupported tracing solver: {solver!r}. "
        "This clean package is Torch-only for Step2 tracing."
    )


def main_polynomial_tracing(Dsetvol_file_path, max_num_th, min_dist, output_fn, solver=None):
    """
    The main polynomial tracing function.

    Args:
        Dsetvol_file_path (str): File path. Reconstructed volume, in the shape of (300, 300, 300).
        max_num_th (int): The maximum atom numbers for tracing.
        min_dist (float): The minimum inter-atomic distance. In units of A.
        output_fn (str): The output .npy filename, Ex: 'initial_traced_model'.

    Returns:

    """
    # Add path for user-defined functions
    # addpath('src/')

    # Add the path to load the reconstruction volume (you can comment it and move
    # the reconstruction into an input folder)
    # addpath('../3_Final_reconstruction_volume/') ;

    # Read in files: reconstruction volume
    # Dsetvol = np.load(Dsetvol_file_path)

    # Dsetvol_full = np.load('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/MG_reconstruction_volume.npy')
    # Dsetvol = Dsetvol_full[199:230, 199:230, 199:230]

    trace_solver = _resolve_solver(solver)
    profile = _env_flag("PYAET_TRACING_PROFILE", False)
    t_start = time.perf_counter()

    if Dsetvol_file_path.endswith('.mat'):
        Dsetvol = read_mat_file(Dsetvol_file_path)
    else:
        Dsetvol = np.load(Dsetvol_file_path)
    Dsetvol_support = Dsetvol

    output_path = Path(output_fn)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path(Dsetvol_file_path).parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path = str(output_path)

    # Constants
    max_iter = 14
    crit_iter = 7
    Th = _env_float("PYAET_TRACING_THRESHOLD", 1.0)
    pixel_size_angstrom = _env_float("PYAET_PIXEL_SIZE_ANGSTROM", 0.347)
    Res = pixel_size_angstrom / 3
    # min_dist = 2 / Res
    min_dist = min_dist / Res
    search_rad = _env_int("PYAET_TRACING_SEARCH_RAD", 3)
    max_iter = _env_int("PYAET_TRACING_MAX_ITER", max_iter)
    crit_iter = _env_int("PYAET_TRACING_CRIT_ITER", crit_iter)

    import torch

    device = _torch_device_from_env(torch)

    t_upsample = time.perf_counter()
    Dsetvol_trace = _upsample3x_spline_torch(Dsetvol, torch, device)
    FinalVol_cpu = None
    FinalVol = _pad_center_torch(Dsetvol_trace, np.array(Dsetvol_trace.shape) + 20, torch)
    del Dsetvol_trace
    _torch_sync_if_needed(torch, device)
    if profile:
        print("  tracing_upsample = spline")
        print(f"  fast_upsample_s = {time.perf_counter() - t_upsample:.3f}")

    # Get polynomial power array
    fit_coeff = _build_fit_coeff()

    # Get local maxima from the reconstruction volume on the selected torch device.
    t_peaks = time.perf_counter()
    max_XYZ = _torch_local_maxima_strel3(FinalVol, max_num_th, Th, torch)
    if profile:
        _torch_sync_if_needed(torch, FinalVol.device)
        print(f"  fast_peaks_s = {time.perf_counter() - t_peaks:.3f}")

    print('numpeak =', len(max_XYZ))
    max_XYZ_base = max_XYZ + 1

    # Initialize parameters
    Q = 0.5
    alpha = 1
    crop_half_size = search_rad
    grid = _prepare_polynomial_grid(search_rad)

    orders = fit_coeff[:, :3]

    # Perform the main tracing loop
    print(f"tracing solver = {trace_solver}")
    t_trace_start = time.perf_counter()
    pos_arr, tot_pos_arr, exit_flag_arr = _trace_peaks_torch(
        FinalVol, max_XYZ, max_XYZ_base, min_dist, fit_coeff, orders, grid,
        max_iter, crit_iter, Q, alpha, crop_half_size,
    )
    if profile:
        print(f"  tracing_core_s = {time.perf_counter() - t_trace_start:.3f}")

    atom_pos = tot_pos_arr[exit_flag_arr == 0, :].T
    atom_pos_all = atom_pos / 3 - 2
    debug_pre_support = os.environ.get("PYAET_DEBUG_SAVE_PRE_SUPPORT")
    if debug_pre_support:
        debug_path = (
            debug_pre_support
            if debug_pre_support.endswith(".npy")
            else output_file_path + "_pre_support_atom_pos_all.npy"
        )
        np.save(debug_path, atom_pos_all)
        print(f"  debug_pre_support_atom_pos_all = {debug_path} shape={atom_pos_all.shape}")
    if _env_flag("PYAET_TRACING_SKIP_POST", False):
        np.save(output_file_path+".npy", atom_pos_all)
        if profile:
            print(f"  tracing_total_s = {time.perf_counter() - t_start:.3f}")
        print("tracing finished.")
        return

    if FinalVol_cpu is not None:
        FinalVol = FinalVol_cpu
    elif hasattr(FinalVol, "detach"):
        FinalVol = FinalVol.detach().cpu().numpy()

    # Do raw classification and get all candidates for manual tracing
    FinalVol_single = FinalVol.astype(np.single)

    classify_info = {
        'num_species': 3,
        'half_size': 3,
        'plot_half_size': 1,
        'O_Ratio': 1,
        'SPHyn': True,
        'PLOT_YN': False,
        'separate_part': 120,
    }
    b1 = np.where(np.logical_or(atom_pos[0, :] < 15, atom_pos[0, :] > FinalVol_single.shape[0] - 15))[0]
    b2 = np.where(np.logical_or(atom_pos[1, :] < 15, atom_pos[1, :] > FinalVol_single.shape[1] - 15))[0]
    b3 = np.where(np.logical_or(atom_pos[2, :] < 15, atom_pos[2, :] > FinalVol_single.shape[2] - 15))[0]

    bT = np.union1d(np.union1d(b1, b2), b3)
    atom_pos = np.delete(atom_pos, bT, axis=1)
    t_classify = time.perf_counter()
    temp_model, temp_atomtype = initial_class_kmean(
        FinalVol_single, atom_pos, classify_info)
    if profile:
        print(f"  classify_s = {time.perf_counter() - t_classify:.3f}")

    atom_pos_o = temp_model / 3 - 2
    debug_classified = os.environ.get("PYAET_DEBUG_SAVE_CLASSIFIED")
    if debug_classified:
        debug_classified_path = (
            debug_classified
            if debug_classified.endswith(".npy")
            else output_file_path + "_classified_atom_pos_o.npy"
        )
        np.save(debug_classified_path, atom_pos_o)
        print(f"  debug_classified_atom_pos_o = {debug_classified_path} shape={atom_pos_o.shape}")

    # Calculate support from reconstruction and get the atoms inside
    support_para = {
        'th_dis_r_afterav': _env_float("PYAET_SUPPORT_TH_DIS_R_AFTERAV", 0.9125),
        'dilate_size': _env_int("PYAET_SUPPORT_DILATE_SIZE", 15),
        'erode_size': _env_int("PYAET_SUPPORT_ERODE1", 13),
        'bw_size': _env_int("PYAET_SUPPORT_BW_SIZE", 50000),
    }
    support_erode2 = _env_int("PYAET_SUPPORT_ERODE2", 18)
    print(
        "support parameters: "
        f"th_dis_r_afterav={support_para['th_dis_r_afterav']}, "
        f"bw_size={support_para['bw_size']}, "
        f"dilate_size={support_para['dilate_size']}, "
        f"erode1={support_para['erode_size']}, "
        f"erode2={support_erode2}"
    )

    # Implement the functions obtain_tight_support and my_paddzero similarly
    t_support = time.perf_counter()
    support_point_mode = _env_flag("PYAET_SUPPORT_POINT_QUERY", True)
    if support_point_mode:
        tight_support_base = prepare_tight_support_base(Dsetvol_support, support_para)
        tight_support2 = None
    else:
        tight_support1, tight_support2 = obtain_tight_support_multi(
            Dsetvol_support, support_para, (support_para['erode_size'], support_erode2)
        )
    if profile:
        print(f"  support_s = {time.perf_counter() - t_support:.3f}")
    if support_point_mode:
        print(f"  support_base_voxels = {int(np.count_nonzero(tight_support_base))}")
    else:
        print(f"  tight_support1_voxels = {int(np.count_nonzero(tight_support1))}")
        print(f"  tight_support2_voxels = {int(np.count_nonzero(tight_support2))}")

    t_filter = time.perf_counter()
    # Exclude atoms near the boundary first
    bdl_1 = 8
    bdl_2 = Dsetvol_support.shape[0] - 8
    ind_out1 = np.logical_or.reduce((atom_pos_o[0, :] <= bdl_1, atom_pos_o[1, :] <= bdl_1, atom_pos_o[2, :] <= bdl_1))
    ind_out2 = np.logical_or.reduce((atom_pos_o[0, :] >= bdl_2, atom_pos_o[1, :] >= bdl_2, atom_pos_o[2, :] >= bdl_2))
    atom_pos_o = np.delete(atom_pos_o, ind_out1 | ind_out2, axis=1)

    # Add the missing atoms inside tighter support
    if support_point_mode:
        inside_support1 = eroded_support_contains_points(
            tight_support_base, atom_pos_o, support_para['erode_size']
        )
        print(f"  tight_support1_points = {int(np.count_nonzero(inside_support1))}")
    else:
        inside_support1 = _indices_inside_support(atom_pos_o, tight_support1)
    temp_pos_arr1_arr = atom_pos_o[:, inside_support1].T

    # Exclude traced atoms outside the looser support
    if temp_pos_arr1_arr.size:
        near_tight = cKDTree(temp_pos_arr1_arr).query_ball_point(atom_pos_all.T, r=1e-4)
        keep_arr = np.fromiter((len(hit) > 0 for hit in near_tight), dtype=bool, count=atom_pos_all.shape[1])
    else:
        keep_arr = np.zeros(atom_pos_all.shape[1], dtype=bool)
    if np.any(~keep_arr):
        if support_point_mode:
            keep_arr[~keep_arr] = eroded_support_contains_points(
                tight_support_base, atom_pos_all[:, ~keep_arr], support_erode2
            )
        else:
            keep_arr[~keep_arr] = _indices_inside_support(atom_pos_all[:, ~keep_arr], tight_support2)

    temp_pos_arr2 = atom_pos_all[:, keep_arr]
    np.save(output_file_path+".npy", temp_pos_arr2)
    if profile:
        print(f"  support_filter_s = {time.perf_counter() - t_filter:.3f}")
        print(f"  tracing_total_s = {time.perf_counter() - t_start:.3f}")
    print("tracing finished.")
    return

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Step2 polynomial tracing. Uses the same Torch code on CPU or CUDA."
    )
    parser.add_argument(
        "volume_pos",
        nargs="?",
        help="Legacy positional reconstruction volume path.",
    )
    parser.add_argument("max_num_th_pos", nargs="?", type=int, help="Legacy positional max atom number.")
    parser.add_argument("min_dist_pos", nargs="?", type=float, help="Legacy positional min distance in Angstrom.")
    parser.add_argument("output_pos", nargs="?", help="Legacy positional output file stem.")
    parser.add_argument("legacy_device", nargs="?", help="Legacy positional device/solver token.")
    parser.add_argument(
        "--volume",
        default=str(VOLUME_FILE_PATH),
        help="Path to Step1 reconstruction .mat/.npy.",
    )
    parser.add_argument("--max-num-th", type=int, default=MAX_NUM_TH)
    parser.add_argument("--min-dist", type=float, default=MIN_DIST_ANGSTROM)
    parser.add_argument(
        "--output-stem",
        default=str(OUTPUT_STEM),
        help="Output file stem without .npy.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument("--torch-chunk", type=int, default=TORCH_CHUNK)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = args.device
    if args.legacy_device and args.device == "auto":
        legacy = args.legacy_device.strip().lower()
        if legacy in {"cpu", "cuda", "gpu", "auto"}:
            device = "cuda" if legacy == "gpu" else legacy
    _set_tracing_device(device)
    os.environ.setdefault("PYAET_TRACING_TORCH_SOLVER", "normal")
    os.environ.setdefault("PYAET_TRACING_TORCH_CHUNK", str(args.torch_chunk))
    os.environ.setdefault("PYAET_TRACING_SOLVER", "torch")
    os.environ.setdefault("PYAET_SUPPORT_POINT_QUERY", "1")
    os.environ.setdefault("PYAET_SUPPORT_EDT_DILATE", "1")
    os.environ.setdefault("PYAET_SUPPORT_EDT_BACKEND", "edt")

    main_polynomial_tracing(
        args.volume_pos or args.volume,
        args.max_num_th_pos if args.max_num_th_pos is not None else args.max_num_th,
        args.min_dist_pos if args.min_dist_pos is not None else args.min_dist,
        args.output_pos or args.output_stem,
        "torch",
    )
