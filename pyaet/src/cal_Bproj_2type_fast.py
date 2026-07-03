import os
import math

import numpy as np

from pyaet.src.my_ifft import my_ifft
from pyaet.src.my_fft import my_fft
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.make_fixed_fa_man import make_fixed_fa_man

try:
    from numba import cuda, float32, float64, int32, njit, prange
except Exception as exc:  # pragma: no cover - exercised when optional dependency is absent.
    njit = None
    cuda = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised when optional dependency is absent.
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


if njit is not None:

    @njit(cache=True)
    def _matlab_round_scalar(x):
        if x >= 0:
            return int(np.floor(x + np.float32(0.5)))
        return int(np.ceil(x - np.float32(0.5)))

    @njit(cache=True, parallel=True)
    def _accumulate_projected_atoms_plane(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h,
        b,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    ):
        projs = np.zeros((n1, n2, num_pj), dtype=np.float64)
        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1

        # Match MATLAB loop order: projection -> atom type -> atoms of that type.
        # Each type is accumulated into its own plane first, then type planes are
        # added to the projection in fixed type order, matching sum(Grad, 4).
        for i in prange(num_pj):
            for j in range(atom_type_num):
                plane = np.zeros((n1, n2), dtype=np.float64)
                hj = h[j]
                bj = b[j]
                for type_pos in range(type_counts[j]):
                    k = type_indices[j, type_pos]

                    x_cen = float(X_rot[i, k])
                    y_cen = float(Y_rot[i, k])
                    z_cen = float(Z_rot[i, k])
                    x_round = _matlab_round_scalar(x_cen)
                    y_round = _matlab_round_scalar(y_cen)
                    z_round = _matlab_round_scalar(z_cen)
                    x_round_f = np.float32(x_round)
                    y_round_f = np.float32(y_round)
                    z_round_f = np.float32(z_round)

                    z_sum = np.float32(0.0)
                    z_delta0 = z_round_f - z_cen
                    for dz in range(-half_width, half_width + 1):
                        z_delta = np.float32(dz) + z_delta0
                        z_sum += np.float32(np.exp(-np.float32((z_delta * z_delta) * bj)))

                    x_delta0 = x_round_f - x_cen
                    y_delta0 = y_round_f - y_cen
                    base_x = x_round + center_x
                    base_y = y_round + center_y

                    for dx in range(-half_width, half_width + 1):
                        xx = base_x + dx
                        x_delta = np.float32(dx) + x_delta0
                        x_l2 = np.float32(x_delta * x_delta)
                        for dy in range(-half_width, half_width + 1):
                            yy = base_y + dy
                            y_delta = np.float32(dy) + y_delta0
                            l2_xy = np.float32(x_l2 + y_delta * y_delta)
                            exp_xy = np.float32(np.exp(-np.float32(l2_xy * bj)))
                            patch = np.float32(hj * np.float32(exp_xy * z_sum))
                            plane[xx, yy] = np.float32(np.float32(plane[xx, yy]) + patch)

                for xx in range(n1):
                    for yy in range(n2):
                        projs[xx, yy, i] += plane[xx, yy]

        return projs


if cuda is not None:

    @cuda.jit(device=True)
    def _matlab_round_cuda(x):
        if x >= 0.0:
            return int32(math.floor(x + 0.5))
        return int32(math.ceil(x - 0.5))

    @cuda.jit
    def _accumulate_projected_atoms_type_planes_cuda(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h,
        b,
        type_planes,
        n1,
        n2,
        half_width,
        atom_type_num,
        max_type_count,
    ):
        idx = cuda.grid(1)
        num_pj = X_rot.shape[0]
        total = num_pj * atom_type_num * max_type_count
        if idx >= total:
            return

        type_pos = idx % max_type_count
        tmp = idx // max_type_count
        atom_type = tmp % atom_type_num
        proj_idx = tmp // atom_type_num
        if type_pos >= type_counts[atom_type]:
            return

        atom_idx = type_indices[atom_type, type_pos]
        if atom_idx < 0:
            return

        x_cen = X_rot[proj_idx, atom_idx]
        y_cen = Y_rot[proj_idx, atom_idx]
        z_cen = Z_rot[proj_idx, atom_idx]
        x_round = _matlab_round_cuda(x_cen)
        y_round = _matlab_round_cuda(y_cen)
        z_round = _matlab_round_cuda(z_cen)

        hj = h[atom_type]
        bj = b[atom_type]
        patch_n = 2 * half_width + 1
        z_sum = float32(0.0)
        z_delta0 = float32(z_round) - z_cen
        for dz_i in range(32):
            if dz_i >= patch_n:
                break
            dz_int = dz_i - half_width
            z_delta = float32(dz_int) + z_delta0
            z_l2 = float32(z_delta * z_delta)
            z_arg = float32(z_l2 * bj)
            z_sum += float32(math.exp(-z_arg))

        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1
        x_delta0 = float32(x_round) - x_cen
        y_delta0 = float32(y_round) - y_cen
        base_x = x_round + center_x
        base_y = y_round + center_y

        for dx_i in range(32):
            if dx_i >= patch_n:
                break
            dx_int = dx_i - half_width
            xx = base_x + dx_int
            x_delta = float32(dx_int) + x_delta0
            x_l2 = float32(x_delta * x_delta)
            for dy_i in range(32):
                if dy_i >= patch_n:
                    break
                dy_int = dy_i - half_width
                yy = base_y + dy_int
                if 0 <= xx < n1 and 0 <= yy < n2:
                    y_delta = float32(dy_int) + y_delta0
                    l2_xy = float32(x_l2 + y_delta * y_delta)
                    exp_xy = float32(math.exp(-float32(l2_xy * bj)))
                    patch = float32(hj * float32(exp_xy * z_sum))
                    pix = xx * n2 + yy
                    cuda.atomic.add(type_planes, (proj_idx, atom_type, pix), patch)


    @cuda.jit
    def _accumulate_projected_atoms_type_planes_and_diffs_cuda(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h_base,
        b_base,
        h_points,
        b_points,
        type_planes,
        diff_planes,
        n1,
        n2,
        half_width,
        atom_type_num,
        max_type_count,
    ):
        idx = cuda.grid(1)
        num_pj = X_rot.shape[0]
        total = num_pj * atom_type_num * max_type_count
        if idx >= total:
            return

        type_pos = idx % max_type_count
        tmp = idx // max_type_count
        atom_type = tmp % atom_type_num
        proj_idx = tmp // atom_type_num
        if type_pos >= type_counts[atom_type]:
            return

        atom_idx = type_indices[atom_type, type_pos]
        if atom_idx < 0:
            return

        x_cen = X_rot[proj_idx, atom_idx]
        y_cen = Y_rot[proj_idx, atom_idx]
        z_cen = Z_rot[proj_idx, atom_idx]
        x_round = _matlab_round_cuda(x_cen)
        y_round = _matlab_round_cuda(y_cen)
        z_round = _matlab_round_cuda(z_cen)

        hj = h_base[atom_type]
        bj = b_base[atom_type]
        patch_n = 2 * half_width + 1

        z_delta0 = float32(z_round) - z_cen
        z_sum = float32(0.0)
        for dz_i in range(32):
            if dz_i >= patch_n:
                break
            dz_int = dz_i - half_width
            z_delta = float32(dz_int) + z_delta0
            z_l2 = float32(z_delta * z_delta)
            z_arg = float32(z_l2 * bj)
            z_sum += float32(math.exp(-z_arg))

        z_sums_p = cuda.local.array(8, dtype=float32)
        num_points = h_points.shape[0]
        for point_idx in range(8):
            if point_idx >= num_points:
                break
            bjp = b_points[point_idx, atom_type]
            z_sum_p = float32(0.0)
            for dz_i in range(32):
                if dz_i >= patch_n:
                    break
                dz_int = dz_i - half_width
                z_delta = float32(dz_int) + z_delta0
                z_l2 = float32(z_delta * z_delta)
                z_arg = float32(z_l2 * bjp)
                z_sum_p += float32(math.exp(-z_arg))
            z_sums_p[point_idx] = z_sum_p

        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1
        x_delta0 = float32(x_round) - x_cen
        y_delta0 = float32(y_round) - y_cen
        base_x = x_round + center_x
        base_y = y_round + center_y

        for dx_i in range(32):
            if dx_i >= patch_n:
                break
            dx_int = dx_i - half_width
            xx = base_x + dx_int
            x_delta = float32(dx_int) + x_delta0
            x_l2 = float32(x_delta * x_delta)
            for dy_i in range(32):
                if dy_i >= patch_n:
                    break
                dy_int = dy_i - half_width
                yy = base_y + dy_int
                if 0 <= xx < n1 and 0 <= yy < n2:
                    y_delta = float32(dy_int) + y_delta0
                    l2_xy = float32(x_l2 + y_delta * y_delta)
                    exp_xy = float32(math.exp(-float32(l2_xy * bj)))
                    patch = float32(hj * float32(exp_xy * z_sum))
                    pix = xx * n2 + yy
                    cuda.atomic.add(type_planes, (proj_idx, atom_type, pix), patch)

                    for point_idx in range(8):
                        if point_idx >= num_points:
                            break
                        hjp = h_points[point_idx, atom_type]
                        bjp = b_points[point_idx, atom_type]
                        exp_xy_p = float32(math.exp(-float32(l2_xy * bjp)))
                        patch_p = float32(hjp * float32(exp_xy_p * z_sums_p[point_idx]))
                        cuda.atomic.add(diff_planes, (point_idx, proj_idx, pix), float(patch_p) - float(patch))


    @cuda.jit
    def _accumulate_projected_atoms_type_planes_and_diffs64_cuda(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h_base,
        b_base,
        h_points,
        b_points,
        type_planes,
        diff_planes,
        n1,
        n2,
        half_width,
        atom_type_num,
        max_type_count,
    ):
        idx = cuda.grid(1)
        num_pj = X_rot.shape[0]
        total = num_pj * atom_type_num * max_type_count
        if idx >= total:
            return

        type_pos = idx % max_type_count
        tmp = idx // max_type_count
        atom_type = tmp % atom_type_num
        proj_idx = tmp // atom_type_num
        if type_pos >= type_counts[atom_type]:
            return

        atom_idx = type_indices[atom_type, type_pos]
        if atom_idx < 0:
            return

        x_cen = float64(X_rot[proj_idx, atom_idx])
        y_cen = float64(Y_rot[proj_idx, atom_idx])
        z_cen = float64(Z_rot[proj_idx, atom_idx])
        x_round = _matlab_round_cuda(x_cen)
        y_round = _matlab_round_cuda(y_cen)
        z_round = _matlab_round_cuda(z_cen)

        hj = h_base[atom_type]
        bj = b_base[atom_type]
        patch_n = 2 * half_width + 1

        z_delta0 = float64(x_round)  # dummy initialization for CUDA type inference
        z_delta0 = float64(z_round) - z_cen
        z_sum = float64(0.0)
        for dz_i in range(32):
            if dz_i >= patch_n:
                break
            dz_int = dz_i - half_width
            z_delta = float64(dz_int) + z_delta0
            z_l2 = z_delta * z_delta
            z_sum += math.exp(-(z_l2 * bj))

        z_sums_p = cuda.local.array(8, dtype=float64)
        num_points = h_points.shape[0]
        for point_idx in range(8):
            if point_idx >= num_points:
                break
            bjp = b_points[point_idx, atom_type]
            z_sum_p = float64(0.0)
            for dz_i in range(32):
                if dz_i >= patch_n:
                    break
                dz_int = dz_i - half_width
                z_delta = float64(dz_int) + z_delta0
                z_l2 = z_delta * z_delta
                z_sum_p += math.exp(-(z_l2 * bjp))
            z_sums_p[point_idx] = z_sum_p

        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1
        x_delta0 = float64(x_round) - x_cen
        y_delta0 = float64(y_round) - y_cen
        base_x = x_round + center_x
        base_y = y_round + center_y

        for dx_i in range(32):
            if dx_i >= patch_n:
                break
            dx_int = dx_i - half_width
            xx = base_x + dx_int
            x_delta = float64(dx_int) + x_delta0
            x_l2 = x_delta * x_delta
            for dy_i in range(32):
                if dy_i >= patch_n:
                    break
                dy_int = dy_i - half_width
                yy = base_y + dy_int
                if 0 <= xx < n1 and 0 <= yy < n2:
                    y_delta = float64(dy_int) + y_delta0
                    l2_xy = x_l2 + y_delta * y_delta
                    exp_xy = math.exp(-(l2_xy * bj))
                    patch = hj * exp_xy * z_sum
                    pix = xx * n2 + yy
                    cuda.atomic.add(type_planes, (proj_idx, atom_type, pix), patch)

                    for point_idx in range(8):
                        if point_idx >= num_points:
                            break
                        hjp = h_points[point_idx, atom_type]
                        bjp = b_points[point_idx, atom_type]
                        exp_xy_p = math.exp(-(l2_xy * bjp))
                        patch_p = hjp * exp_xy_p * z_sums_p[point_idx]
                        cuda.atomic.add(diff_planes, (point_idx, proj_idx, pix), patch_p - patch)


    @cuda.jit
    def _accumulate_projected_atoms_type_planes_grads_cuda(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h,
        b,
        type_planes,
        grad_h_planes,
        grad_b_planes,
        n1,
        n2,
        half_width,
        atom_type_num,
        max_type_count,
    ):
        idx = cuda.grid(1)
        num_pj = X_rot.shape[0]
        total = num_pj * atom_type_num * max_type_count
        if idx >= total:
            return

        type_pos = idx % max_type_count
        tmp = idx // max_type_count
        atom_type = tmp % atom_type_num
        proj_idx = tmp // atom_type_num
        if type_pos >= type_counts[atom_type]:
            return

        atom_idx = type_indices[atom_type, type_pos]
        if atom_idx < 0:
            return

        x_cen = X_rot[proj_idx, atom_idx]
        y_cen = Y_rot[proj_idx, atom_idx]
        z_cen = Z_rot[proj_idx, atom_idx]
        x_round = _matlab_round_cuda(x_cen)
        y_round = _matlab_round_cuda(y_cen)
        z_round = _matlab_round_cuda(z_cen)

        hj = h[atom_type]
        bj = b[atom_type]
        patch_n = 2 * half_width + 1
        z_sum = float32(0.0)
        z_l2_sum = float32(0.0)
        z_delta0 = float32(z_round) - z_cen
        for dz_i in range(32):
            if dz_i >= patch_n:
                break
            dz_int = dz_i - half_width
            z_delta = float32(dz_int) + z_delta0
            z_l2 = float32(z_delta * z_delta)
            z_arg = float32(z_l2 * bj)
            exp_z = float32(math.exp(-z_arg))
            z_sum += exp_z
            z_l2_sum += float32(z_l2 * exp_z)

        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1
        x_delta0 = float32(x_round) - x_cen
        y_delta0 = float32(y_round) - y_cen
        base_x = x_round + center_x
        base_y = y_round + center_y

        for dx_i in range(32):
            if dx_i >= patch_n:
                break
            dx_int = dx_i - half_width
            xx = base_x + dx_int
            x_delta = float32(dx_int) + x_delta0
            x_l2 = float32(x_delta * x_delta)
            for dy_i in range(32):
                if dy_i >= patch_n:
                    break
                dy_int = dy_i - half_width
                yy = base_y + dy_int
                if 0 <= xx < n1 and 0 <= yy < n2:
                    y_delta = float32(dy_int) + y_delta0
                    l2_xy = float32(x_l2 + y_delta * y_delta)
                    exp_xy = float32(math.exp(-float32(l2_xy * bj)))
                    pj = float32(exp_xy * z_sum)
                    patch = float32(hj * pj)
                    grad_b = float32(patch * l2_xy + hj * float32(exp_xy * z_l2_sum))
                    pix = xx * n2 + yy
                    cuda.atomic.add(type_planes, (proj_idx, atom_type, pix), patch)
                    cuda.atomic.add(grad_h_planes, (proj_idx, atom_type, pix), pj)
                    cuda.atomic.add(grad_b_planes, (proj_idx, atom_type, pix), grad_b)


def _require_numba():
    if njit is None:
        raise ImportError(
            "Step4 refinement requires numba for the release projector."
        ) from _NUMBA_IMPORT_ERROR


def _matlab_round_torch(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.int64)


def _select_backend(xdata):
    backend = xdata.get("bproj_backend", None)
    if backend is None:
        backend = os.environ.get("PYAET_BPROJ_BACKEND", "numba")
    return str(backend).strip().lower()


def _select_torch_device(xdata):
    device = xdata.get("bproj_device", None)
    if device is None:
        device = os.environ.get("PYAET_BPROJ_DEVICE", None)
    if device is None:
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _torch_dtype_from_xdata(xdata):
    dtype_name = str(xdata.get("bproj_torch_dtype", os.environ.get("PYAET_BPROJ_DTYPE", "float32"))).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported PYAET_BPROJ_DTYPE: {dtype_name!r}")


def _cuda_accum_dtype_from_xdata(xdata):
    dtype_name = str(
        xdata.get("bproj_cuda_accum_dtype", os.environ.get("PYAET_BPROJ_CUDA_ACCUM_DTYPE", "float64"))
    ).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported PYAET_BPROJ_CUDA_ACCUM_DTYPE: {dtype_name!r}")


def _finite_difference_rel_step():
    value = os.environ.get("PYAET_POSREF_LSQ_REL_STEP")
    if value is None:
        value = os.environ.get("PYAET_POSREF_LSQ_DIFF_STEP")
    if value is None or value.strip() == "":
        return float(np.sqrt(np.finfo(float).eps))
    return float(value)


def _env_truthy(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_torch_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, atom, device, dtype):
    cache = xdata.setdefault("_cal_bproj_torch_cache", {})
    torch_key = (cache_key, str(device), str(dtype), int(atom.size))
    if cache.get("key") != torch_key:
        atom0 = atom.astype(np.int64, copy=False) - 1
        cache.clear()
        cache.update(
            {
                "key": torch_key,
                "fixed_fa": torch.as_tensor(fixed_fa, dtype=dtype, device=device),
                "X_rot": torch.as_tensor(X_rot, dtype=dtype, device=device),
                "Y_rot": torch.as_tensor(Y_rot, dtype=dtype, device=device),
                "Z_rot": torch.as_tensor(Z_rot, dtype=dtype, device=device),
                "atom0": torch.as_tensor(atom0, dtype=torch.long, device=device),
            }
        )
    return cache


def _get_torch_mixed_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, atom, device, atom_type_num):
    cache = xdata.setdefault("_cal_bproj_torch_mixed_cache", {})
    torch_key = (cache_key, str(device), "mixed", int(atom.size), int(atom_type_num))
    if cache.get("key") != torch_key:
        atom0 = atom.astype(np.int64, copy=False) - 1
        atom0_t = torch.as_tensor(atom0, dtype=torch.long, device=device)
        cache.clear()
        cache.update(
            {
                "key": torch_key,
                "fixed_fa": torch.as_tensor(fixed_fa, dtype=torch.float64, device=device),
                "X_rot": torch.as_tensor(X_rot, dtype=torch.float32, device=device),
                "Y_rot": torch.as_tensor(Y_rot, dtype=torch.float32, device=device),
                "Z_rot": torch.as_tensor(Z_rot, dtype=torch.float32, device=device),
                "atom0": atom0_t,
                "type_indices": [torch.nonzero(atom0_t == j, as_tuple=False).flatten() for j in range(atom_type_num)],
            }
        )
    return cache


def _get_cuda_release_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, type_indices, type_counts, device):
    cache = xdata.setdefault("_cal_bproj_cuda_release_cache", {})
    cuda_key = (cache_key, str(device), "cuda_release")
    if cache.get("key") != cuda_key:
        cache.clear()
        cache.update(
            {
                "key": cuda_key,
                "fixed_fa": torch.as_tensor(fixed_fa, dtype=torch.float64, device=device),
                "X_rot": torch.as_tensor(X_rot, dtype=torch.float32, device=device),
                "Y_rot": torch.as_tensor(Y_rot, dtype=torch.float32, device=device),
                "Z_rot": torch.as_tensor(Z_rot, dtype=torch.float32, device=device),
                "type_indices": torch.as_tensor(type_indices, dtype=torch.int32, device=device),
                "type_counts": torch.as_tensor(type_counts, dtype=torch.int32, device=device),
            }
        )
    return cache


def _get_cuda_release_ydata(cache, ydata, device):
    y_key = (id(ydata), tuple(ydata.shape), str(np.asarray(ydata).dtype), str(device))
    if cache.get("ydata_key") != y_key:
        cache["ydata_key"] = y_key
        cache["ydata"] = torch.as_tensor(ydata, dtype=torch.float64, device=device)
    return cache["ydata"]


def _accumulate_projected_atoms_torch(X_rot, Y_rot, Z_rot, atom0, h, b, n1, n2, num_pj, half_width):
    dtype = X_rot.dtype
    device = X_rot.device
    num_atom = X_rot.shape[1]
    n_patch = 2 * half_width + 1
    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1

    h_atom = h[atom0]
    b_atom = b[atom0]
    offsets = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)
    dx_grid, dy_grid = torch.meshgrid(offsets, offsets, indexing="ij")
    dx_flat = dx_grid.reshape(-1)
    dy_flat = dy_grid.reshape(-1)
    pj_offsets = torch.arange(num_pj, device=device, dtype=torch.long).view(num_pj, 1, 1)

    projs_flat = torch.zeros((num_pj, n1 * n2), dtype=dtype, device=device)

    for i0 in range(num_pj):
        x_cen = X_rot[i0]
        y_cen = Y_rot[i0]
        z_cen = Z_rot[i0]

        x_round = _matlab_round_torch(x_cen)
        y_round = _matlab_round_torch(y_cen)
        z_round = _matlab_round_torch(z_cen)

        x_round_f = x_round.to(dtype)
        y_round_f = y_round.to(dtype)
        z_round_f = z_round.to(dtype)

        z_delta = offsets[:, None] + (z_round_f - z_cen)[None, :]
        z_sum = torch.exp(-(z_delta * z_delta) * b_atom[None, :]).sum(dim=0)

        x_delta = dx_flat[:, None] + (x_round_f - x_cen)[None, :]
        y_delta = dy_flat[:, None] + (y_round_f - y_cen)[None, :]
        l2_xy = x_delta * x_delta + y_delta * y_delta
        patch = (h_atom[None, :] * torch.exp(-l2_xy * b_atom[None, :]) * z_sum[None, :]).to(dtype)

        xx = x_round[None, :] + center_x + dx_flat.to(torch.long)[:, None]
        yy = y_round[None, :] + center_y + dy_flat.to(torch.long)[:, None]
        valid = (xx >= 0) & (xx < n1) & (yy >= 0) & (yy < n2)
        flat_idx = xx * n2 + yy
        projs_flat[i0].scatter_add_(0, flat_idx[valid], patch[valid])

    return projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()


def _accumulate_projected_atoms_torch_mixed(
    X_rot,
    Y_rot,
    Z_rot,
    type_indices,
    h_np,
    b_np,
    n1,
    n2,
    num_pj,
    half_width,
):
    device = X_rot.device
    n_patch = 2 * half_width + 1
    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1

    offsets32 = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=device)
    dx_grid, dy_grid = torch.meshgrid(offsets32, offsets32, indexing="ij")
    dx_flat32 = dx_grid.reshape(-1)
    dy_flat32 = dy_grid.reshape(-1)
    dx_flat_i = dx_flat32.to(torch.long)
    dy_flat_i = dy_flat32.to(torch.long)

    projs_flat64 = torch.zeros((num_pj, n1 * n2), dtype=torch.float64, device=device)
    h64 = torch.as_tensor(h_np, dtype=torch.float64, device=device)
    b64 = torch.as_tensor(b_np, dtype=torch.float64, device=device)

    for i0 in range(num_pj):
        x_cen_all = X_rot[i0]
        y_cen_all = Y_rot[i0]
        z_cen_all = Z_rot[i0]
        for atom_type, idx in enumerate(type_indices):
            if idx.numel() == 0:
                continue
            x_cen = x_cen_all.index_select(0, idx)
            y_cen = y_cen_all.index_select(0, idx)
            z_cen = z_cen_all.index_select(0, idx)

            x_round = _matlab_round_torch(x_cen)
            y_round = _matlab_round_torch(y_cen)
            z_round = _matlab_round_torch(z_cen)
            x_round_f = x_round.to(torch.float32)
            y_round_f = y_round.to(torch.float32)
            z_round_f = z_round.to(torch.float32)

            bj = b64[atom_type]
            hj = h64[atom_type]

            z_delta = offsets32[:, None] + (z_round_f - z_cen)[None, :]
            z_l2 = z_delta * z_delta
            z_arg = (z_l2.to(torch.float64) * bj).to(torch.float32)
            z_sum = torch.exp(-z_arg).sum(dim=0, dtype=torch.float32)

            x_delta = dx_flat32[:, None] + (x_round_f - x_cen)[None, :]
            y_delta = dy_flat32[:, None] + (y_round_f - y_cen)[None, :]
            l2_xy = x_delta * x_delta + y_delta * y_delta
            xy_arg = (l2_xy.to(torch.float64) * bj).to(torch.float32)
            exp_xy = torch.exp(-xy_arg)
            patch = (hj * (exp_xy * z_sum[None, :]).to(torch.float64)).to(torch.float32)

            xx = x_round[None, :] + center_x + dx_flat_i[:, None]
            yy = y_round[None, :] + center_y + dy_flat_i[:, None]
            valid = (xx >= 0) & (xx < n1) & (yy >= 0) & (yy < n2)
            flat_idx = xx * n2 + yy

            plane32 = torch.zeros((n1 * n2,), dtype=torch.float32, device=device)
            plane32.scatter_add_(0, flat_idx[valid], patch[valid])
            projs_flat64[i0] += plane32.to(torch.float64)

    return projs_flat64.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()


def _apply_fixed_fa_torch(projs, fixed_fa):
    # Input/output are (N1, N2, num_pj), matching the NumPy path.
    proj_batch = projs.permute(2, 0, 1).contiguous()
    kspace = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(proj_batch, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    kspace = kspace * fixed_fa[None, :, :]
    out = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    ).real
    return out.permute(1, 2, 0).contiguous()


def _cal_Bproj_2type_torch(para, xdata, ydata, fixed_fa, X_rot, Y_rot, Z_rot, atom, cache_key, n1, n2, num_pj, half_width, atom_type_num):
    if torch is None:
        raise ImportError("Torch backend requested but torch is not importable.") from _TORCH_IMPORT_ERROR

    device = _select_torch_device(xdata)
    dtype = _torch_dtype_from_xdata(xdata)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA backend requested, but CUDA is not available.")

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order="F")
    h_np = para[0, :] / para[0, 0]
    b_np = (np.pi * xdata["Res"]) ** 2 / para[1, :]

    cache = _get_torch_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, atom, device, dtype)
    h = torch.as_tensor(h_np, dtype=dtype, device=device)
    b = torch.as_tensor(b_np, dtype=dtype, device=device)

    projs = _accumulate_projected_atoms_torch(
        cache["X_rot"],
        cache["Y_rot"],
        cache["Z_rot"],
        cache["atom0"],
        h,
        b,
        n1,
        n2,
        num_pj,
        int(half_width),
    )
    projs = _apply_fixed_fa_torch(projs, cache["fixed_fa"])

    y_t = torch.as_tensor(ydata, dtype=dtype, device=device)
    denom = torch.sum(projs * projs)
    k = torch.sum(projs * y_t) / denom
    projs = projs * k

    param = np.vstack([float(k.detach().cpu()) * h_np, (np.pi * xdata["Res"]) ** 2 / b_np])
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param


def _cal_Bproj_2type_torch_mixed(para, xdata, ydata, fixed_fa, X_rot, Y_rot, Z_rot, atom, cache_key, n1, n2, num_pj, half_width, atom_type_num):
    if torch is None:
        raise ImportError("Torch mixed backend requested but torch is not importable.") from _TORCH_IMPORT_ERROR

    device = _select_torch_device(xdata)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA backend requested, but CUDA is not available.")

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order="F")
    h_np = para[0, :] / para[0, 0]
    b_np = (np.pi * xdata["Res"]) ** 2 / para[1, :]

    cache = _get_torch_mixed_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, atom, device, atom_type_num)
    projs = _accumulate_projected_atoms_torch_mixed(
        cache["X_rot"],
        cache["Y_rot"],
        cache["Z_rot"],
        cache["type_indices"],
        h_np,
        b_np,
        n1,
        n2,
        num_pj,
        int(half_width),
    )
    projs = _apply_fixed_fa_torch(projs, cache["fixed_fa"])

    y_t = _get_cuda_release_ydata(cache, ydata, device)
    denom = torch.sum(projs * projs)
    k = torch.sum(projs * y_t) / denom
    projs = projs * k

    param = np.vstack([float(k.detach().cpu()) * h_np, (np.pi * xdata["Res"]) ** 2 / b_np])
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param


def _cuda_release_project_tensor(
    para,
    xdata,
    ydata,
    fixed_fa,
    X_rot,
    Y_rot,
    Z_rot,
    type_indices,
    type_counts,
    cache_key,
    n1,
    n2,
    num_pj,
    half_width,
    atom_type_num,
):
    if torch is None or cuda is None:
        raise ImportError("CUDA release backend requires torch and numba cuda.") from _NUMBA_IMPORT_ERROR

    device = _select_torch_device(xdata)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA release backend requested, but torch CUDA is not available.")

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order="F")
    h_np64 = para[0, :] / para[0, 0]
    b_np64 = (np.pi * xdata["Res"]) ** 2 / para[1, :]
    h = torch.as_tensor(h_np64, dtype=torch.float64, device=device)
    b = torch.as_tensor(b_np64, dtype=torch.float64, device=device)

    cache = _get_cuda_release_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, type_indices, type_counts, device)
    n_pix = n1 * n2
    accum_dtype = _cuda_accum_dtype_from_xdata(xdata)
    type_planes = torch.zeros((num_pj, atom_type_num, n_pix), dtype=accum_dtype, device=device)

    max_type_count = int(type_indices.shape[1])
    total = num_pj * atom_type_num * max_type_count
    threads = int(os.environ.get("PYAET_BPROJ_CUDA_THREADS", "128"))
    blocks = (total + threads - 1) // threads
    _accumulate_projected_atoms_type_planes_cuda[blocks, threads](
        cuda.as_cuda_array(cache["X_rot"]),
        cuda.as_cuda_array(cache["Y_rot"]),
        cuda.as_cuda_array(cache["Z_rot"]),
        cuda.as_cuda_array(cache["type_indices"]),
        cuda.as_cuda_array(cache["type_counts"]),
        cuda.as_cuda_array(h),
        cuda.as_cuda_array(b),
        cuda.as_cuda_array(type_planes),
        n1,
        n2,
        int(half_width),
        atom_type_num,
        max_type_count,
    )
    cuda.synchronize()

    projs_flat = torch.zeros((num_pj, n_pix), dtype=torch.float64, device=device)
    for atom_type in range(atom_type_num):
        projs_flat += type_planes[:, atom_type, :].to(torch.float64)
    projs = projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
    projs = _apply_fixed_fa_torch(projs, cache["fixed_fa"])

    y_t = torch.as_tensor(ydata, dtype=torch.float64, device=device)
    denom = torch.sum(projs * projs)
    k = torch.sum(projs * y_t) / denom
    projs = projs * k

    param = np.vstack([float(k.detach().cpu()) * h_np64, para[1, :]])
    return projs, param, cache, device


def _cal_Bproj_2type_cuda_release(
    para,
    xdata,
    ydata,
    fixed_fa,
    X_rot,
    Y_rot,
    Z_rot,
    type_indices,
    type_counts,
    cache_key,
    n1,
    n2,
    num_pj,
    half_width,
    atom_type_num,
):
    projs, param, _, _ = _cuda_release_project_tensor(
        para,
        xdata,
        ydata,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        cache_key,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    )
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param


def cal_Bproj_2type(para, xdata, ydata, fit_flag=True):
    _require_numba()
    if fit_flag:
        para = np.abs(para)

    Z_arr = xdata["Z_arr"]
    Res = xdata["Res"]
    half_width = xdata["half_width"]
    model = np.asarray(xdata["model"], dtype=np.float64)
    angles = np.asarray(xdata["angles"], dtype=np.float64)
    atom = np.asarray(xdata["atoms"]).ravel(order="F").astype(np.int64)
    num_atom = atom.size
    atom_type_num = len(np.unique(atom))

    n1, n2, _ = ydata.shape
    num_pj = angles.shape[0]

    model_scaled = model / Res

    # Cache static quantities across LSQ residual calls. During an H/B fit,
    # model and angles are fixed; only para changes.
    cache = xdata.setdefault("_cal_bproj_fast_cache", {})
    cache_key = (id(xdata["model"]), id(xdata["angles"]), n1, n2, num_pj, num_atom, float(Res))
    if cache.get("key") == cache_key:
        fixed_fa = cache["fixed_fa"]
        X_rot = cache["X_rot"]
        Y_rot = cache["Y_rot"]
        Z_rot = cache["Z_rot"]
        type_indices = cache["type_indices"]
        type_counts = cache["type_counts"]
    else:
        fixed_fa = np.asarray(make_fixed_fa_man([n1, n2], Res, Z_arr).reshape(n1, n2))
        type_counts = np.array([np.count_nonzero(atom == (j + 1)) for j in range(atom_type_num)], dtype=np.int64)
        max_type_count = int(np.max(type_counts)) if atom_type_num else 0
        type_indices = np.empty((atom_type_num, max_type_count), dtype=np.int64)
        type_indices.fill(-1)
        for j in range(atom_type_num):
            idx = np.nonzero(atom == (j + 1))[0].astype(np.int64)
            type_indices[j, : idx.size] = idx

        # MATLAB stores rotated coordinates as single precision before splatting.
        coord_dtype = np.float32
        X_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
        Y_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
        Z_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)

        for i in range(num_pj):
            R1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
            R2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
            R3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
            R = (np.dot(np.dot(R1, R2), R3)).T
            rot_coords = np.dot(R, model_scaled)
            X_rot[i, :] = rot_coords[0, :]
            Y_rot[i, :] = rot_coords[1, :]
            Z_rot[i, :] = rot_coords[2, :]

        cache.clear()
        cache.update({
            "key": cache_key,
            "fixed_fa": fixed_fa,
            "X_rot": X_rot,
            "Y_rot": Y_rot,
            "Z_rot": Z_rot,
            "type_indices": type_indices,
            "type_counts": type_counts,
        })

    backend = _select_backend(xdata)
    if backend in {"cuda_release", "gpu_release", "numba_cuda_release"}:
        return _cal_Bproj_2type_cuda_release(
            para,
            xdata,
            ydata,
            fixed_fa,
            X_rot,
            Y_rot,
            Z_rot,
            type_indices,
            type_counts,
            cache_key,
            n1,
            n2,
            num_pj,
            int(half_width),
            atom_type_num,
        )
    if backend in {"torch_mixed", "cuda_mixed", "gpu_mixed"}:
        return _cal_Bproj_2type_torch_mixed(
            para,
            xdata,
            ydata,
            fixed_fa,
            X_rot,
            Y_rot,
            Z_rot,
            atom,
            cache_key,
            n1,
            n2,
            num_pj,
            int(half_width),
            atom_type_num,
        )
    if backend in {"torch", "cuda", "gpu"}:
        return _cal_Bproj_2type_torch(
            para,
            xdata,
            ydata,
            fixed_fa,
            X_rot,
            Y_rot,
            Z_rot,
            atom,
            cache_key,
            n1,
            n2,
            num_pj,
            int(half_width),
            atom_type_num,
        )
    if backend not in {"numba", "cpu"}:
        raise ValueError(f"Unsupported bproj_backend: {backend!r}")

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order="F")
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]

    projs = _accumulate_projected_atoms_plane(
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        h.astype(np.float64),
        b.astype(np.float64),
        n1,
        n2,
        num_pj,
        int(half_width),
        atom_type_num,
    )

    for i in range(num_pj):
        projs[:, :, i] = np.real(my_ifft(my_fft(projs[:, :, i]) * fixed_fa))

    projs_flat = projs.ravel(order="F")
    ydata_flat = ydata.ravel(order="F")
    k = np.sum(projs_flat * ydata_flat) / np.sum(projs_flat ** 2)
    projs = projs * k

    param = np.vstack([k * h, (np.pi * Res) ** 2 / b])
    return projs, param


def _prepare_cal_bproj_static(xdata, ydata):
    Z_arr = xdata["Z_arr"]
    Res = xdata["Res"]
    half_width = xdata["half_width"]
    model = np.asarray(xdata["model"], dtype=np.float64)
    angles = np.asarray(xdata["angles"], dtype=np.float64)
    atom = np.asarray(xdata["atoms"]).ravel(order="F").astype(np.int64)
    num_atom = atom.size
    atom_type_num = len(np.unique(atom))

    n1, n2, _ = ydata.shape
    num_pj = angles.shape[0]
    model_scaled = model / Res

    cache = xdata.setdefault("_cal_bproj_fast_cache", {})
    cache_key = (id(xdata["model"]), id(xdata["angles"]), n1, n2, num_pj, num_atom, float(Res))
    if cache.get("key") == cache_key:
        return (
            cache_key,
            cache["fixed_fa"],
            cache["X_rot"],
            cache["Y_rot"],
            cache["Z_rot"],
            cache["type_indices"],
            cache["type_counts"],
            n1,
            n2,
            num_pj,
            int(half_width),
            atom_type_num,
        )

    fixed_fa = np.asarray(make_fixed_fa_man([n1, n2], Res, Z_arr).reshape(n1, n2))
    type_counts = np.array([np.count_nonzero(atom == (j + 1)) for j in range(atom_type_num)], dtype=np.int64)
    max_type_count = int(np.max(type_counts)) if atom_type_num else 0
    type_indices = np.empty((atom_type_num, max_type_count), dtype=np.int64)
    type_indices.fill(-1)
    for j in range(atom_type_num):
        idx = np.nonzero(atom == (j + 1))[0].astype(np.int64)
        type_indices[j, : idx.size] = idx

    coord_dtype = np.float32
    X_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
    Y_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
    Z_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)

    for i in range(num_pj):
        R1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        R2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        R3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        R = (np.dot(np.dot(R1, R2), R3)).T
        rot_coords = np.dot(R, model_scaled)
        X_rot[i, :] = rot_coords[0, :]
        Y_rot[i, :] = rot_coords[1, :]
        Z_rot[i, :] = rot_coords[2, :]

    cache.clear()
    cache.update({
        "key": cache_key,
        "fixed_fa": fixed_fa,
        "X_rot": X_rot,
        "Y_rot": Y_rot,
        "Z_rot": Z_rot,
        "type_indices": type_indices,
        "type_counts": type_counts,
    })
    return (
        cache_key,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        n1,
        n2,
        num_pj,
        int(half_width),
        atom_type_num,
    )


def cal_Bproj_2type2_cuda_release_stats(para, xdata, lb, ub, free_indices=None):
    ydata = xdata["projections"]
    (
        cache_key,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    ) = _prepare_cal_bproj_static(xdata, ydata)

    x = np.asarray(para, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if free_indices is None:
        free_indices = np.arange(x.size)
    else:
        free_indices = np.asarray(free_indices, dtype=np.int64)

    rel_step = _finite_difference_rel_step()
    points = []
    steps = []
    for i in free_indices:
        h = rel_step * max(abs(x[i]), 1.0)
        if x[i] + h > ub[i]:
            h = -h
        if x[i] + h < lb[i]:
            h = -h
        xp = x.copy()
        xp[i] += h
        points.append(xp)
        steps.append(h)

    f0_proj, _, cache, device = _cuda_release_project_tensor(
        x,
        xdata,
        ydata,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        cache_key,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    )
    y_t = _get_cuda_release_ydata(cache, ydata, device)
    f0 = f0_proj - y_t
    val_t = torch.sum(f0 * f0)

    cols = []
    g_vals = []
    for xp, h in zip(points, steps):
        fp_proj, _, _, _ = _cuda_release_project_tensor(
            xp,
            xdata,
            ydata,
            fixed_fa,
            X_rot,
            Y_rot,
            Z_rot,
            type_indices,
            type_counts,
            cache_key,
            n1,
            n2,
            num_pj,
            half_width,
            atom_type_num,
        )
        col = (fp_proj - f0_proj) / h
        cols.append(col)
        g_vals.append(torch.sum(col * f0))

    if cols:
        jac_cols = torch.stack(cols, dim=0).reshape(len(cols), -1)
        g_t = torch.stack(g_vals)
        ata_t = jac_cols @ jac_cols.T
    else:
        g_t = torch.zeros((0,), dtype=torch.float64, device=device)
        ata_t = torch.zeros((0, 0), dtype=torch.float64, device=device)

    val = float(val_t.detach().cpu())
    g = g_t.detach().cpu().numpy().astype(np.float64, copy=False)
    ata = ata_t.detach().cpu().numpy().astype(np.float64, copy=False)
    return val, g, ata, 1 + len(points)


def cal_Bproj_2type2_cuda_release_stats_paired(para, xdata, lb, ub, free_indices=None):
    ydata = xdata["projections"]
    (
        cache_key,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    ) = _prepare_cal_bproj_static(xdata, ydata)

    if torch is None or cuda is None:
        raise ImportError("CUDA release stats backend requires torch and numba cuda.") from _NUMBA_IMPORT_ERROR

    device = _select_torch_device(xdata)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA release stats backend requested, but torch CUDA is not available.")

    x = np.asarray(para, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if free_indices is None:
        free_indices = np.arange(x.size)
    else:
        free_indices = np.asarray(free_indices, dtype=np.int64)

    rel_step = _finite_difference_rel_step()
    points = []
    steps = []
    for i in free_indices:
        h = rel_step * max(abs(x[i]), 1.0)
        if x[i] + h > ub[i]:
            h = -h
        if x[i] + h < lb[i]:
            h = -h
        xp = x.copy()
        xp[i] += h
        points.append(xp)
        steps.append(h)

    if len(points) > 8:
        raise ValueError("Paired CUDA stats supports up to 8 finite-difference points.")

    para_base = np.reshape(x, [2, atom_type_num], order="F")
    h_base_np = para_base[0, :] / para_base[0, 0]
    b_base_np = (np.pi * xdata["Res"]) ** 2 / para_base[1, :]
    h_points_np = np.empty((len(points), atom_type_num), dtype=np.float64)
    b_points_np = np.empty((len(points), atom_type_num), dtype=np.float64)
    for row, xp in enumerate(points):
        para_point = np.reshape(xp, [2, atom_type_num], order="F")
        h_points_np[row, :] = para_point[0, :] / para_point[0, 0]
        b_points_np[row, :] = (np.pi * xdata["Res"]) ** 2 / para_point[1, :]

    cache = _get_cuda_release_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, type_indices, type_counts, device)
    n_pix = n1 * n2
    accum_dtype = _cuda_accum_dtype_from_xdata(xdata)
    type_planes = torch.zeros((num_pj, atom_type_num, n_pix), dtype=accum_dtype, device=device)
    diff_planes = torch.zeros((len(points), num_pj, n_pix), dtype=torch.float64, device=device)

    h_base = torch.as_tensor(h_base_np, dtype=torch.float64, device=device)
    b_base = torch.as_tensor(b_base_np, dtype=torch.float64, device=device)
    h_points = torch.as_tensor(h_points_np, dtype=torch.float64, device=device)
    b_points = torch.as_tensor(b_points_np, dtype=torch.float64, device=device)

    max_type_count = int(type_indices.shape[1])
    total = num_pj * atom_type_num * max_type_count
    threads = int(os.environ.get("PYAET_BPROJ_CUDA_THREADS", "128"))
    blocks = (total + threads - 1) // threads
    _accumulate_projected_atoms_type_planes_and_diffs_cuda[blocks, threads](
        cuda.as_cuda_array(cache["X_rot"]),
        cuda.as_cuda_array(cache["Y_rot"]),
        cuda.as_cuda_array(cache["Z_rot"]),
        cuda.as_cuda_array(cache["type_indices"]),
        cuda.as_cuda_array(cache["type_counts"]),
        cuda.as_cuda_array(h_base),
        cuda.as_cuda_array(b_base),
        cuda.as_cuda_array(h_points),
        cuda.as_cuda_array(b_points),
        cuda.as_cuda_array(type_planes),
        cuda.as_cuda_array(diff_planes),
        n1,
        n2,
        int(half_width),
        atom_type_num,
        max_type_count,
    )
    cuda.synchronize()

    projs_flat = torch.zeros((num_pj, n_pix), dtype=torch.float64, device=device)
    for atom_type in range(atom_type_num):
        projs_flat += type_planes[:, atom_type, :].to(torch.float64)
    projs = projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
    projs = _apply_fixed_fa_torch(projs, cache["fixed_fa"])

    y_t = _get_cuda_release_ydata(cache, ydata, device)
    denom = torch.sum(projs * projs)
    numer = torch.sum(projs * y_t)
    k = numer / denom
    f0_proj = projs * k
    f0 = f0_proj - y_t
    val_t = torch.sum(f0 * f0)

    cols = []
    g_vals = []
    for point_idx, h_step in enumerate(steps):
        diff_pre = diff_planes[point_idx].reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
        diff_proj = _apply_fixed_fa_torch(diff_pre, cache["fixed_fa"])
        fp_unscaled = projs + diff_proj
        fp_denom = torch.sum(fp_unscaled * fp_unscaled)
        fp_k = torch.sum(fp_unscaled * y_t) / fp_denom
        fp_proj = fp_unscaled * fp_k
        col = (fp_proj - f0_proj) / h_step
        cols.append(col)
        g_vals.append(torch.sum(col * f0))

    if cols:
        jac_cols = torch.stack(cols, dim=0).reshape(len(cols), -1)
        g_t = torch.stack(g_vals)
        ata_t = jac_cols @ jac_cols.T
    else:
        g_t = torch.zeros((0,), dtype=torch.float64, device=device)
        ata_t = torch.zeros((0, 0), dtype=torch.float64, device=device)

    val = float(val_t.detach().cpu())
    g = g_t.detach().cpu().numpy().astype(np.float64, copy=False)
    ata = ata_t.detach().cpu().numpy().astype(np.float64, copy=False)
    return val, g, ata, 1 + len(points)


def cal_Bproj_2type2_cuda_release_stats_paired64(para, xdata, lb, ub, free_indices=None):
    ydata = xdata["projections"]
    (
        cache_key,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    ) = _prepare_cal_bproj_static(xdata, ydata)

    if torch is None or cuda is None:
        raise ImportError("CUDA release paired64 stats backend requires torch and numba cuda.") from _NUMBA_IMPORT_ERROR

    device = _select_torch_device(xdata)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA release paired64 stats backend requested, but torch CUDA is not available.")

    x = np.asarray(para, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if free_indices is None:
        free_indices = np.arange(x.size)
    else:
        free_indices = np.asarray(free_indices, dtype=np.int64)

    rel_step = _finite_difference_rel_step()
    points = []
    steps = []
    for i in free_indices:
        h = rel_step * max(abs(x[i]), 1.0)
        if x[i] + h > ub[i]:
            h = -h
        if x[i] + h < lb[i]:
            h = -h
        xp = x.copy()
        xp[i] += h
        points.append(xp)
        steps.append(h)

    if len(points) > 8:
        raise ValueError("Paired64 CUDA stats supports up to 8 finite-difference points.")

    para_base = np.reshape(x, [2, atom_type_num], order="F")
    h_base_np = para_base[0, :] / para_base[0, 0]
    b_base_np = (np.pi * xdata["Res"]) ** 2 / para_base[1, :]
    h_points_np = np.empty((len(points), atom_type_num), dtype=np.float64)
    b_points_np = np.empty((len(points), atom_type_num), dtype=np.float64)
    for row, xp in enumerate(points):
        para_point = np.reshape(xp, [2, atom_type_num], order="F")
        h_points_np[row, :] = para_point[0, :] / para_point[0, 0]
        b_points_np[row, :] = (np.pi * xdata["Res"]) ** 2 / para_point[1, :]

    cache = _get_cuda_release_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, type_indices, type_counts, device)
    n_pix = n1 * n2
    type_planes = torch.zeros((num_pj, atom_type_num, n_pix), dtype=torch.float64, device=device)
    diff_planes = torch.zeros((len(points), num_pj, n_pix), dtype=torch.float64, device=device)

    h_base = torch.as_tensor(h_base_np, dtype=torch.float64, device=device)
    b_base = torch.as_tensor(b_base_np, dtype=torch.float64, device=device)
    h_points = torch.as_tensor(h_points_np, dtype=torch.float64, device=device)
    b_points = torch.as_tensor(b_points_np, dtype=torch.float64, device=device)

    max_type_count = int(type_indices.shape[1])
    total = num_pj * atom_type_num * max_type_count
    threads = int(os.environ.get("PYAET_BPROJ_CUDA_THREADS", "128"))
    blocks = (total + threads - 1) // threads
    _accumulate_projected_atoms_type_planes_and_diffs64_cuda[blocks, threads](
        cuda.as_cuda_array(cache["X_rot"]),
        cuda.as_cuda_array(cache["Y_rot"]),
        cuda.as_cuda_array(cache["Z_rot"]),
        cuda.as_cuda_array(cache["type_indices"]),
        cuda.as_cuda_array(cache["type_counts"]),
        cuda.as_cuda_array(h_base),
        cuda.as_cuda_array(b_base),
        cuda.as_cuda_array(h_points),
        cuda.as_cuda_array(b_points),
        cuda.as_cuda_array(type_planes),
        cuda.as_cuda_array(diff_planes),
        n1,
        n2,
        int(half_width),
        atom_type_num,
        max_type_count,
    )
    cuda.synchronize()

    projs_flat = torch.zeros((num_pj, n_pix), dtype=torch.float64, device=device)
    for atom_type in range(atom_type_num):
        projs_flat += type_planes[:, atom_type, :]
    projs = projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
    projs = _apply_fixed_fa_torch(projs, cache["fixed_fa"])

    y_t = _get_cuda_release_ydata(cache, ydata, device)
    denom = torch.sum(projs * projs)
    numer = torch.sum(projs * y_t)
    k = numer / denom
    f0_proj = projs * k
    f0 = f0_proj - y_t
    val_t = torch.sum(f0 * f0)

    cols = []
    g_vals = []
    for point_idx, h_step in enumerate(steps):
        diff_pre = diff_planes[point_idx].reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
        diff_proj = _apply_fixed_fa_torch(diff_pre, cache["fixed_fa"])
        fp_unscaled = projs + diff_proj
        fp_denom = torch.sum(fp_unscaled * fp_unscaled)
        fp_k = torch.sum(fp_unscaled * y_t) / fp_denom
        fp_proj = fp_unscaled * fp_k
        col = (fp_proj - f0_proj) / h_step
        cols.append(col)
        g_vals.append(torch.sum(col * f0))

    if cols:
        jac_cols = torch.stack(cols, dim=0).reshape(len(cols), -1)
        g_t = torch.stack(g_vals)
        ata_t = jac_cols @ jac_cols.T
    else:
        g_t = torch.zeros((0,), dtype=torch.float64, device=device)
        ata_t = torch.zeros((0, 0), dtype=torch.float64, device=device)

    val = float(val_t.detach().cpu())
    g = g_t.detach().cpu().numpy().astype(np.float64, copy=False)
    ata = ata_t.detach().cpu().numpy().astype(np.float64, copy=False)
    return val, g, ata, 1 + len(points)


def cal_Bproj_2type2_cuda_release_stats_analytic(para, xdata, lb, ub, free_indices=None):
    ydata = xdata["projections"]
    (
        cache_key,
        fixed_fa,
        X_rot,
        Y_rot,
        Z_rot,
        type_indices,
        type_counts,
        n1,
        n2,
        num_pj,
        half_width,
        atom_type_num,
    ) = _prepare_cal_bproj_static(xdata, ydata)

    if torch is None or cuda is None:
        raise ImportError("CUDA release analytic stats backend requires torch and numba cuda.") from _NUMBA_IMPORT_ERROR

    device = _select_torch_device(xdata)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA release analytic stats backend requested, but torch CUDA is not available.")

    x = np.asarray(para, dtype=np.float64)
    if free_indices is None:
        free_indices = np.arange(x.size)
    else:
        free_indices = np.asarray(free_indices, dtype=np.int64)

    para_mat = np.reshape(x, [2, atom_type_num], order="F")
    raw_h_np = para_mat[0, :]
    h_np = raw_h_np / raw_h_np[0]
    bparam_np = para_mat[1, :]
    c_res = (np.pi * xdata["Res"]) ** 2
    b_np = c_res / bparam_np

    cache = _get_cuda_release_cache(xdata, cache_key, fixed_fa, X_rot, Y_rot, Z_rot, type_indices, type_counts, device)
    n_pix = n1 * n2
    accum_dtype = _cuda_accum_dtype_from_xdata(xdata)
    type_planes = torch.zeros((num_pj, atom_type_num, n_pix), dtype=accum_dtype, device=device)
    grad_h_planes = torch.zeros_like(type_planes)
    grad_b_planes = torch.zeros_like(type_planes)

    h = torch.as_tensor(h_np, dtype=torch.float64, device=device)
    b = torch.as_tensor(b_np, dtype=torch.float64, device=device)

    max_type_count = int(type_indices.shape[1])
    total = num_pj * atom_type_num * max_type_count
    threads = int(os.environ.get("PYAET_BPROJ_CUDA_THREADS", "128"))
    blocks = (total + threads - 1) // threads
    _accumulate_projected_atoms_type_planes_grads_cuda[blocks, threads](
        cuda.as_cuda_array(cache["X_rot"]),
        cuda.as_cuda_array(cache["Y_rot"]),
        cuda.as_cuda_array(cache["Z_rot"]),
        cuda.as_cuda_array(cache["type_indices"]),
        cuda.as_cuda_array(cache["type_counts"]),
        cuda.as_cuda_array(h),
        cuda.as_cuda_array(b),
        cuda.as_cuda_array(type_planes),
        cuda.as_cuda_array(grad_h_planes),
        cuda.as_cuda_array(grad_b_planes),
        n1,
        n2,
        int(half_width),
        atom_type_num,
        max_type_count,
    )
    cuda.synchronize()

    type_planes64 = type_planes.to(torch.float64)
    grad_h64 = grad_h_planes.to(torch.float64)
    grad_b64 = grad_b_planes.to(torch.float64)
    unscaled_flat = torch.sum(type_planes64, dim=1)
    unscaled = unscaled_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
    projs_unscaled = _apply_fixed_fa_torch(unscaled, cache["fixed_fa"])

    y_t = _get_cuda_release_ydata(cache, ydata, device)
    denom = torch.sum(projs_unscaled * projs_unscaled)
    numer = torch.sum(projs_unscaled * y_t)
    k_scale = numer / denom
    f0_proj = projs_unscaled * k_scale
    f0 = f0_proj - y_t
    val_t = torch.sum(f0 * f0)

    derivative_tensors = []
    for raw_idx in free_indices:
        atom_type = int(raw_idx // 2)
        row = int(raw_idx % 2)
        if row == 0:
            coeffs = np.zeros(atom_type_num, dtype=np.float64)
            if atom_type == 0:
                coeffs[1:] = -raw_h_np[1:] / (raw_h_np[0] * raw_h_np[0])
            else:
                coeffs[atom_type] = 1.0 / raw_h_np[0]
            deriv_flat = torch.zeros_like(unscaled_flat)
            for j, coeff in enumerate(coeffs):
                if coeff != 0.0:
                    deriv_flat = deriv_flat + float(coeff) * grad_h64[:, j, :]
        else:
            coeff = c_res / (bparam_np[atom_type] * bparam_np[atom_type])
            deriv_flat = float(coeff) * grad_b64[:, atom_type, :]

        deriv = deriv_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
        deriv_filtered = _apply_fixed_fa_torch(deriv, cache["fixed_fa"])
        derivative_tensors.append(deriv_filtered)

    cols = []
    g_vals = []
    for deriv_filtered in derivative_tensors:
        d_denom = 2.0 * torch.sum(projs_unscaled * deriv_filtered)
        d_numer = torch.sum(deriv_filtered * y_t)
        dk = (d_numer * denom - numer * d_denom) / (denom * denom)
        col = k_scale * deriv_filtered + dk * projs_unscaled
        cols.append(col)
        g_vals.append(torch.sum(col * f0))

    if cols:
        jac_cols = torch.stack(cols, dim=0).reshape(len(cols), -1)
        g_t = torch.stack(g_vals)
        ata_t = jac_cols @ jac_cols.T
    else:
        g_t = torch.zeros((0,), dtype=torch.float64, device=device)
        ata_t = torch.zeros((0, 0), dtype=torch.float64, device=device)

    val = float(val_t.detach().cpu())
    g = g_t.detach().cpu().numpy().astype(np.float64, copy=False)
    ata = ata_t.detach().cpu().numpy().astype(np.float64, copy=False)
    return val, g, ata, 1


def cal_Bproj_2type2(para, xdata, fit_flag=True):
    ydata = xdata["projections"]
    return cal_Bproj_2type(para, xdata, ydata, fit_flag=fit_flag)
