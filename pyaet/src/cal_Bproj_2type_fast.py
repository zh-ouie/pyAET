import time

import numpy as np

from pyaet.src.my_ifft import my_ifft
from pyaet.src.my_fft import my_fft
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.make_fixed_fa_man import make_fixed_fa_man

try:
    from numba import njit, prange
except Exception as exc:  # pragma: no cover - exercised when optional dependency is absent.
    njit = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


# Release settings for the Numba projector.
STEP4_FAST_ACCUM_MODE = "plane"  # "plane" or "channel"
STEP4_TIMING = False


if njit is not None:

    @njit(cache=True)
    def _matlab_round_scalar(x):
        if x >= 0:
            return int(np.floor(x + np.float32(0.5)))
        return int(np.ceil(x - np.float32(0.5)))

    @njit(cache=True, parallel=True)
    def _accumulate_projected_atoms(
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
        grad = np.zeros((n1, n2, num_pj, atom_type_num), dtype=np.float64)
        center_x = (n1 + 1) // 2 - 1
        center_y = (n2 + 1) // 2 - 1

        # Match MATLAB loop order: projection -> atom type -> atoms of that type.
        for i in prange(num_pj):
            for j in range(atom_type_num):
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
                            # MATLAB accumulates in a per-type Grad channel.
                            # Grad(double) + patch(single) returns single
                            # before assignment back to double.
                            grad[xx, yy, i, j] = np.float32(np.float32(grad[xx, yy, i, j]) + patch)

        return grad

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


def _require_numba():
    if njit is None:
        raise ImportError(
            "Step4 refinement requires numba for the release projector."
        ) from _NUMBA_IMPORT_ERROR


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

    timings = {}
    t0 = time.perf_counter()
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
    timings["static_s"] = time.perf_counter() - t0

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order="F")
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]

    t1 = time.perf_counter()
    if STEP4_FAST_ACCUM_MODE == "plane":
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
    else:
        grad = _accumulate_projected_atoms(
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
        projs = np.sum(grad, axis=3)
    timings["splat_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    for i in range(num_pj):
        projs[:, :, i] = np.real(my_ifft(my_fft(projs[:, :, i]) * fixed_fa))
    timings["fft_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    projs_flat = projs.ravel(order="F")
    ydata_flat = ydata.ravel(order="F")
    k = np.sum(projs_flat * ydata_flat) / np.sum(projs_flat ** 2)
    projs = projs * k
    timings["scale_s"] = time.perf_counter() - t3
    timings["total_inner_s"] = sum(timings.values())
    if STEP4_TIMING:
        print(
            "cal_Bproj_fast timing: "
            + ", ".join(f"{name}={value:.3f}" for name, value in timings.items()),
            flush=True,
        )

    param = np.vstack([k * h, (np.pi * Res) ** 2 / b])
    return projs, param


def cal_Bproj_2type2(para, xdata, fit_flag=True):
    ydata = xdata["projections"]
    return cal_Bproj_2type(para, xdata, ydata, fit_flag=fit_flag)
