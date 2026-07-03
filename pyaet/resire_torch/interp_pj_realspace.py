import time

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "pyAET reconstruction gridding requires PyTorch. Install torch before running Step1."
    ) from exc

from pyaet.resire_torch.reconstruct import (
    _empty_device_cache,
    _grid_sample_backproj_grid_chunks,
    _grid_sample_backproj_sum,
    _resolve_device,
    _sync_device,
    _torch_dtype_from_numpy,
)
from pyaet.resire_torch.torch_interp import interp2 as torch_interp2
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.my_round import my_round_num


def interp_pj_realspace(obj):
    """Prepare RESIRE interpolation grids with torch interpolation.

    The rotation-coordinate setup stays in NumPy because it is cheap compared
    with interpolation and keeps exact parity with the reference path. The measured
    projection interpolation itself runs through torch on CUDA when available,
    otherwise on torch CPU.
    """
    device = _resolve_device(getattr(obj, "gpu_grad_device", None))

    gridding_start = time.perf_counter()
    projections = np.asarray(obj.InputProjections, dtype=np.float64, order="F")
    num_pj = obj.num_projs
    dimx = obj.dim1
    dimy = obj.dim2

    phi_angles = obj.InputAngles[:, 0]
    theta_angles = obj.InputAngles[:, 1]
    psi_angles = obj.InputAngles[:, 2]

    vec1 = obj.vector1
    vec2 = obj.vector2
    vec3 = obj.vector3

    dtype = np.dtype(obj.dtype)
    torch_dtype = _torch_dtype_from_numpy(dtype)
    n1_oversampled = obj.n1_oversampled
    n2_oversampled = obj.n2_oversampled
    use_grid_sample_gridding = (
        device == "cuda"
        and str(getattr(obj, "gpu_backproj_backend", "")) == "grid_sample"
    )
    grid_sample_chunk_size = int(getattr(obj, "gpu_backproj_chunk_size", 5))
    precompute_grid_sample_grid = bool(getattr(obj, "gpu_precompute_backproj_grid", False))

    ncy = my_round_num((dimy + 1) / 2)
    ncx = my_round_num((dimx + 1) / 2)

    k1 = np.arange(-np.ceil((dimx - 1) / 2), np.floor((dimx - 1) / 2) + 1, dtype=np.float64)
    k2 = np.arange(-np.ceil((dimy - 1) / 2), np.floor((dimy - 1) / 2) + 1, dtype=np.float64)
    k3 = k1
    YY, XX, ZZ = np.meshgrid(k1, k2, k3)
    XX = XX.flatten(order="F")
    YY = YY.flatten(order="F")
    ZZ = ZZ.flatten(order="F")
    xyz = np.vstack((XX, YY, ZZ))

    if use_grid_sample_gridding:
        rot_pjs_t = None
    else:
        rot_pjs_t = torch.empty((dimy, dimx, dimx, num_pj), dtype=torch_dtype, device=device)

    ncy_big = my_round_num((n2_oversampled + 1) / 2)
    ncx_big = my_round_num((n1_oversampled + 1) / 2)
    Y, X, Z = np.meshgrid(
        np.arange(1, n2_oversampled + 1) - ncy_big,
        np.arange(1, n1_oversampled + 1) - ncx_big,
        0,
    )
    Y = Y.flatten(order="F")
    X = X.flatten(order="F")
    Z = Z.flatten(order="F")
    xyz_big = np.vstack((X, Y, Z))

    xj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")
    yj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")
    zj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")

    Rot_x = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype, order="F")
    Rot_y = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype, order="F")

    interp_s = 0.0
    for k in range(num_pj):
        phi = phi_angles[k]
        theta = theta_angles[k]
        psi = psi_angles[k]
        R1 = matrix_quaternion_rot(vec1, phi)
        R2 = matrix_quaternion_rot(vec2, theta)
        R3 = matrix_quaternion_rot(vec3, psi)
        R = (R1 @ R2 @ R3).T

        rot_coords = R[0:2, :] @ xyz
        rot_x64 = np.reshape(rot_coords[0, :], (dimy, dimx, dimy), order="F") + ncx
        rot_y64 = np.reshape(rot_coords[1, :], (dimy, dimx, dimy), order="F") + ncy
        Rot_x[:, :, :, k] = rot_x64.astype(dtype, copy=False)
        Rot_y[:, :, :, k] = rot_y64.astype(dtype, copy=False)

        if not use_grid_sample_gridding:
            pj_t = torch.as_tensor(np.asfortranarray(projections[:, :, k]), dtype=torch_dtype, device=device)
            rot_x_t = torch.as_tensor(np.ascontiguousarray(rot_x64), dtype=torch_dtype, device=device)
            rot_y_t = torch.as_tensor(np.ascontiguousarray(rot_y64), dtype=torch_dtype, device=device)
            t0 = time.perf_counter()
            rot_pjs_t[:, :, :, k] = torch_interp2(pj_t, rot_x_t, rot_y_t)
            _sync_device(device)
            interp_s += time.perf_counter() - t0

        rot_coords = R.T @ xyz_big
        xj[:, :, k] = np.reshape(rot_coords[0, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big
        yj[:, :, k] = np.reshape(rot_coords[1, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big
        zj[:, :, k] = np.reshape(rot_coords[2, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big

    grid_sample_gridding_s = 0.0
    grid_sample_precompute_s = 0.0
    if use_grid_sample_gridding:
        projections_t = torch.as_tensor(projections, dtype=torch_dtype, device=device)
        rot_x_all_t = torch.as_tensor(Rot_x.astype(np.float64, copy=False), dtype=torch_dtype, device=device)
        rot_y_all_t = torch.as_tensor(Rot_y.astype(np.float64, copy=False), dtype=torch_dtype, device=device)
        grid_chunks = None
        vol_shape = None
        if precompute_grid_sample_grid:
            t0 = time.perf_counter()
            vol_shape, grid_chunks = _grid_sample_backproj_grid_chunks(
                rot_x_all_t,
                rot_y_all_t,
                proj_h=int(dimx),
                proj_w=int(dimy),
                chunk_size=grid_sample_chunk_size,
            )
            _sync_device(device)
            grid_sample_precompute_s = time.perf_counter() - t0
            obj._gpu_grid_sample_vol_shape = vol_shape
            obj._gpu_grid_sample_grid_chunks = grid_chunks

        t0 = time.perf_counter()
        sum_rot_pjs_t = _grid_sample_backproj_sum(
            projections_t,
            rot_x_all_t if grid_chunks is None else None,
            rot_y_all_t if grid_chunks is None else None,
            chunk_size=grid_sample_chunk_size,
            grid_chunks=grid_chunks,
            vol_shape=vol_shape,
        )
        _sync_device(device)
        grid_sample_gridding_s = time.perf_counter() - t0
        obj._gpu_sum_rot_pjs_t = sum_rot_pjs_t
        obj.sum_rot_pjs = np.asfortranarray(sum_rot_pjs_t.detach().cpu().numpy().astype(dtype, copy=False))
        del projections_t, rot_x_all_t, rot_y_all_t
        _empty_device_cache(device)
    else:
        obj.sum_rot_pjs = np.asfortranarray(torch.sum(rot_pjs_t, dim=3).detach().cpu().numpy().astype(dtype, copy=False))
    obj.xj = xj
    obj.yj = yj
    obj.zj = zj
    obj.Rot_x = Rot_x
    obj.Rot_y = Rot_y

    perf_stats = getattr(obj, "perf_stats", None)
    if perf_stats is None:
        perf_stats = {}
        obj.perf_stats = perf_stats
    perf_stats["gridding_total_s"] = time.perf_counter() - gridding_start
    perf_stats["interp_pj_realspace_torch"] = {
        "backend": f"torch:{device}",
        "torch_interp2_gridding_calls": int(num_pj),
        "torch_interp2_gridding_s": interp_s,
        "grid_sample_gridding_s": grid_sample_gridding_s,
        "grid_sample_precompute_s": grid_sample_precompute_s,
        "grid_sample_chunk_size": grid_sample_chunk_size if use_grid_sample_gridding else 0,
    }
    return obj


__all__ = ["interp_pj_realspace"]
