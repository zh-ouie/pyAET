from pyaet.fft_backend import fftn as shared_fftn, fftshift as shared_fftshift, ifftshift as shared_ifftshift, rfftn as shared_rfftn
import os
import time
import gc

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "pyAET reconstruction requires PyTorch. Install torch before running Step1."
    ) from exc

from pyaet.src.my_volume_index import my_volumn_index
from pyaet.resire_torch.torch_interp import interp2 as torch_interp2
from pyaet.resire_torch.torch_interp import interp3 as torch_interp3


def _torch_dtype_from_numpy(dtype: np.dtype):
    if dtype == np.dtype(np.float32):
        return torch.float32
    if dtype == np.dtype(np.float64):
        return torch.float64
    raise TypeError(f"Unsupported dtype for torch backend: {dtype}")


def _resolve_device(preferred_device):
    requested = "auto" if preferred_device in (None, "") else str(preferred_device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested in {"cuda", "gpu"} or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA was requested for Step1, but CUDA is unavailable.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported Step1 torch device: {preferred_device!r}. Use auto, cuda, or cpu.")


def _sync_device(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _sync_for_timing(device: str, enabled: bool):
    if enabled:
        _sync_device(device)


def _empty_device_cache(device: str):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _to_torch(array: np.ndarray, device: str, torch_dtype):
    return torch.as_tensor(np.asarray(array), dtype=torch_dtype, device=device)


def _as_torch_f_contig(array: np.ndarray, device: str, torch_dtype):
    return torch.as_tensor(np.asfortranarray(array), dtype=torch_dtype, device=device)


def _torch_fftshift(x, dim=None):
    return shared_fftshift(x, axes=dim)


def _torch_ifftshift(x, dim=None):
    return shared_ifftshift(x, axes=dim)


def _torch_my_fft(img_t):
    shifted_t = _torch_ifftshift(img_t)
    fft_t = shared_fftn(shifted_t, backend="torch_matlab_single")
    del shifted_t
    return _torch_fftshift(fft_t)


def _torch_my_fft_unshifted_output(img_t):
    return shared_fftn(shared_ifftshift(img_t), backend="torch_matlab_single")


def _torch_interp3_fftshifted(data, x, y, z, origin_offset: int = 1):
    if data.ndim != 3:
        raise ValueError("data must be 3D")

    from pyaet.resire_torch.torch_interp import _coord_tensor, _interp_indices

    device = data.device
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    x = _coord_tensor(x, device=device, dtype=real_dtype)
    y = _coord_tensor(y, device=device, dtype=real_dtype)
    z = _coord_tensor(z, device=device, dtype=real_dtype)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have the same shape")

    nrows, ncols, nlayers = data.shape
    x1, x2, x1_safe, x2_safe, valid_x, x0 = _interp_indices(x, nrows, origin_offset)
    y1, y2, y1_safe, y2_safe, valid_y, y0 = _interp_indices(y, ncols, origin_offset)
    z1, z2, z1_safe, z2_safe, valid_z, z0 = _interp_indices(z, nlayers, origin_offset)
    valid = valid_x & valid_y & valid_z

    sx = nrows // 2
    sy = ncols // 2
    sz = nlayers // 2
    x1_data = torch.remainder(x1_safe + sx, nrows)
    x2_data = torch.remainder(x2_safe + sx, nrows)
    y1_data = torch.remainder(y1_safe + sy, ncols)
    y2_data = torch.remainder(y2_safe + sy, ncols)
    z1_data = torch.remainder(z1_safe + sz, nlayers)
    z2_data = torch.remainder(z2_safe + sz, nlayers)

    f111 = data[x1_data, y1_data, z1_data]
    f121 = data[x1_data, y2_data, z1_data]
    f211 = data[x2_data, y1_data, z1_data]
    f221 = data[x2_data, y2_data, z1_data]

    f112 = data[x1_data, y1_data, z2_data]
    f122 = data[x1_data, y2_data, z2_data]
    f212 = data[x2_data, y1_data, z2_data]
    f222 = data[x2_data, y2_data, z2_data]

    wx1 = x2.to(real_dtype) - x0
    wx2 = x0 - x1.to(real_dtype)
    wy1 = y2.to(real_dtype) - y0
    wy2 = y0 - y1.to(real_dtype)
    wz1 = z2.to(real_dtype) - z0
    wz2 = z0 - z1.to(real_dtype)

    a1 = f111 * wx1 + f211 * wx2
    b1 = f121 * wx1 + f221 * wx2
    lower = a1 * wy1 + b1 * wy2

    a2 = f112 * wx1 + f212 * wx2
    b2 = f122 * wx1 + f222 * wx2
    upper = a2 * wy1 + b2 * wy2

    out = lower * wz1 + upper * wz2
    return torch.where(valid, out, torch.zeros((), device=device, dtype=out.dtype))


def _center_slices(shape, crop_size):
    if len(crop_size) == 1:
        crop_size = crop_size * len(shape)
    slices = []
    for size, crop in zip(shape, crop_size):
        center = int(np.floor((size + 1) / 2 + 0.5))
        crop_center = int(np.floor((crop + 1) / 2 + 0.5))
        start = center - crop_center
        slices.append(slice(start, start + crop))
    return tuple(slices)


def _torch_cropped_out(array_t, crop_size):
    return array_t[_center_slices(tuple(array_t.shape), list(crop_size))]


def _grid_sample_backproj_grid(rx, ry, *, proj_h: int, proj_w: int):
    rx0 = rx - 1.0
    ry0 = ry - 1.0
    valid = (
        torch.isfinite(rx0)
        & torch.isfinite(ry0)
        & (rx0 >= 0)
        & (rx0 <= (proj_h - 1))
        & (ry0 >= 0)
        & (ry0 <= (proj_w - 1))
    )
    grid_x = 2.0 * ry0 / (proj_w - 1) - 1.0
    grid_y = 2.0 * rx0 / (proj_h - 1) - 1.0
    outside = torch.full((), 2.0, dtype=grid_x.dtype, device=grid_x.device)
    grid_x = torch.where(valid, grid_x, outside)
    grid_y = torch.where(valid, grid_y, outside)
    vol_shape = tuple(rx.shape[1:])
    return torch.stack((grid_x, grid_y), dim=-1).reshape(
        rx.shape[0],
        vol_shape[0],
        vol_shape[1] * vol_shape[2],
        2,
    ).contiguous()


def _grid_sample_backproj_grid_chunks(rot_x_t, rot_y_t, *, proj_h: int, proj_w: int, chunk_size: int):
    vol_shape = tuple(rot_x_t.shape[:3])
    num_pj = int(rot_x_t.shape[3])
    chunk_size = max(1, int(chunk_size))
    chunks = []

    for start in range(0, num_pj, chunk_size):
        stop = min(start + chunk_size, num_pj)
        rx = rot_x_t[:, :, :, start:stop].permute(3, 0, 1, 2).contiguous()
        ry = rot_y_t[:, :, :, start:stop].permute(3, 0, 1, 2).contiguous()
        grid = _grid_sample_backproj_grid(rx, ry, proj_h=proj_h, proj_w=proj_w)
        chunks.append((start, stop, grid))

    return vol_shape, chunks


def _grid_sample_backproj_sum(
    pj_cal_t,
    rot_x_t=None,
    rot_y_t=None,
    *,
    chunk_size: int,
    grid_chunks=None,
    vol_shape=None,
):
    import torch.nn.functional as F

    proj_h, proj_w, num_pj = pj_cal_t.shape
    images = pj_cal_t.permute(2, 0, 1).contiguous().unsqueeze(1)
    if grid_chunks is None:
        vol_shape = rot_x_t.shape[:3]
        chunk_size = max(1, int(chunk_size))
    else:
        if vol_shape is None:
            raise ValueError("vol_shape is required when grid_chunks are precomputed.")
    grad = torch.zeros(vol_shape, dtype=pj_cal_t.dtype, device=pj_cal_t.device)

    if grid_chunks is None:
        for start in range(0, num_pj, chunk_size):
            stop = min(start + chunk_size, num_pj)
            rx = rot_x_t[:, :, :, start:stop].permute(3, 0, 1, 2).contiguous()
            ry = rot_y_t[:, :, :, start:stop].permute(3, 0, 1, 2).contiguous()
            grid = _grid_sample_backproj_grid(rx, ry, proj_h=int(proj_h), proj_w=int(proj_w))
            sampled = F.grid_sample(
                images[start:stop],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.reshape(stop - start, *vol_shape)
            grad = grad + sampled.sum(dim=0)
    else:
        for start, stop, grid in grid_chunks:
            sampled = F.grid_sample(
                images[start:stop],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.reshape(stop - start, *vol_shape)
            grad = grad + sampled.sum(dim=0)

    return grad


def _grid_sample_backproj_sum_numpy_rot(
    pj_cal_t,
    rot_x,
    rot_y,
    *,
    device: str,
    torch_dtype,
    chunk_size: int,
):
    import torch.nn.functional as F

    proj_h, proj_w, num_pj = pj_cal_t.shape
    images = pj_cal_t.permute(2, 0, 1).contiguous().unsqueeze(1)
    vol_shape = tuple(rot_x.shape[:3])
    chunk_size = max(1, int(chunk_size))
    grad = torch.zeros(vol_shape, dtype=pj_cal_t.dtype, device=pj_cal_t.device)

    for start in range(0, num_pj, chunk_size):
        stop = min(start + chunk_size, num_pj)
        rx_np = np.ascontiguousarray(np.moveaxis(rot_x[:, :, :, start:stop], 3, 0))
        ry_np = np.ascontiguousarray(np.moveaxis(rot_y[:, :, :, start:stop], 3, 0))
        rx = torch.as_tensor(rx_np, dtype=torch_dtype, device=device)
        ry = torch.as_tensor(ry_np, dtype=torch_dtype, device=device)
        grid = _grid_sample_backproj_grid(rx, ry, proj_h=int(proj_h), proj_w=int(proj_w))
        sampled = F.grid_sample(
            images[start:stop],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(stop - start, *vol_shape)
        grad = grad + sampled.sum(dim=0)
        del rx, ry, grid, sampled

    return grad

def reconstruct(obj):
    """Run RESIRE reconstruction with torch FFT/interpolation.

    The torch path keeps the reconstruction volume, FFT results, projection
    interpolation, back-projection interpolation, gradient accumulation, and
    descent update on one torch device. It only copies the final reconstruction
    back to NumPy for saving and downstream compatibility.
    """
    reconstruct_start = time.perf_counter()
    dtype = np.dtype(obj.dtype)
    iterations = obj.num_iterations

    perf_stats = getattr(obj, "perf_stats", None)
    if perf_stats is None:
        perf_stats = {}
        obj.perf_stats = perf_stats
    perf = perf_stats.setdefault("reconstruct_torch", {})
    perf["iterations"] = int(iterations)
    perf["torch_fft_s"] = 0.0
    perf["torch_interp3_s"] = 0.0
    perf["torch_ifft2_s"] = 0.0
    perf["torch_interp2_backproj_s"] = 0.0
    perf["grid_sample_backproj_s"] = 0.0
    perf["grid_sample_precompute_s"] = 0.0
    perf["grad_backend_s"] = 0.0
    perf["iter_grad_update_s"] = []
    perf["descent_update_only_s"] = 0.0
    perf["iter_descent_update_only_s"] = []
    perf["sync_timing"] = bool(getattr(obj, "gpu_sync_timing", False))
    perf["interp3_chunk_size"] = 0
    perf["grid_sample_chunk_size"] = 0
    perf["precompute_grid_sample_grid"] = False

    device = _resolve_device(getattr(obj, "gpu_grad_device", None))

    num_pj = obj.num_projs
    step_size = obj.step_size
    dimx = obj.dim1
    dimy = obj.dim2
    dt = step_size / num_pj / dimx
    torch_dtype = _torch_dtype_from_numpy(dtype)
    sync_timing = bool(getattr(obj, "gpu_sync_timing", False))
    backproj_backend = str(getattr(obj, "gpu_backproj_backend", "torch_loop"))
    use_grid_sample_backproj = device in {"cuda", "cpu"} and backproj_backend == "grid_sample"
    grid_sample_chunk_size = int(getattr(obj, "gpu_backproj_chunk_size", 5))
    precompute_grid_sample_grid = bool(getattr(obj, "gpu_precompute_backproj_grid", False))
    interp3_chunk_size = int(getattr(obj, "gpu_interp3_chunk_size", 0) or 0)
    backproj_rot_on_demand = bool(getattr(obj, "gpu_backproj_rot_on_demand", False))
    avoid_fftshift_copy = bool(getattr(obj, "gpu_avoid_fftshift_copy", False))
    # CUDA uses the validated half-spectrum kernel when Triton is installed.
    # CPU and installations without Triton retain full-spectrum interpolation.
    spectral_interpolator = None
    if device == "cuda" and avoid_fftshift_copy:
        from functools import partial
        from pyaet.resire_torch.triton_spectral import interp3_spectral, triton
        if triton is not None:
            spectral_interpolator = partial(interp3_spectral,
                full_shape=(obj.n2_oversampled, obj.n1_oversampled, obj.n2_oversampled))
    project_interp = spectral_interpolator or (
        _torch_interp3_fftshifted if avoid_fftshift_copy else torch_interp3)
    perf["fft_precision"] = "matlab_single"
    perf["spectral_acceleration"] = "rfft" if spectral_interpolator else "full"
    print("FFT: matlab_single; spectrum:", perf["spectral_acceleration"])
    perf["interp3_chunk_size"] = int(interp3_chunk_size)
    perf["grid_sample_chunk_size"] = int(grid_sample_chunk_size)
    perf["precompute_grid_sample_grid"] = bool(precompute_grid_sample_grid)
    perf["backproj_rot_on_demand"] = bool(backproj_rot_on_demand)
    perf["avoid_fftshift_copy"] = bool(avoid_fftshift_copy)

    projections_t = _to_torch(obj.InputProjections.astype(np.float64, copy=False), device, torch_dtype)
    xj_t = _to_torch(obj.xj.astype(np.float64, copy=False), device, torch_dtype)
    yj_t = _to_torch(obj.yj.astype(np.float64, copy=False), device, torch_dtype)
    zj_t = _to_torch(obj.zj.astype(np.float64, copy=False), device, torch_dtype)
    cached_sum_rot_pjs_t = getattr(obj, "_gpu_sum_rot_pjs_t", None)
    if cached_sum_rot_pjs_t is not None and str(cached_sum_rot_pjs_t.device) == device:
        sum_rot_pjs_t = cached_sum_rot_pjs_t.to(dtype=torch_dtype)
    else:
        sum_rot_pjs_t = _as_torch_f_contig(obj.sum_rot_pjs.astype(dtype, copy=False), device, torch_dtype)

    grid_sample_vol_shape = getattr(obj, "_gpu_grid_sample_vol_shape", None)
    grid_sample_grid_chunks = getattr(obj, "_gpu_grid_sample_grid_chunks", None)
    need_rot_tensors = (
        not backproj_rot_on_demand
        and not (use_grid_sample_backproj and grid_sample_grid_chunks is not None)
    )
    if need_rot_tensors:
        Rot_x_t = _to_torch(obj.Rot_x.astype(np.float64, copy=False), device, torch_dtype)
        Rot_y_t = _to_torch(obj.Rot_y.astype(np.float64, copy=False), device, torch_dtype)

    if use_grid_sample_backproj and precompute_grid_sample_grid and grid_sample_grid_chunks is None:
        t0 = time.perf_counter()
        grid_sample_vol_shape, grid_sample_grid_chunks = _grid_sample_backproj_grid_chunks(
            Rot_x_t,
            Rot_y_t,
            proj_h=int(dimx),
            proj_w=int(dimy),
            chunk_size=grid_sample_chunk_size,
        )
        _sync_for_timing(device, sync_timing)
        perf["grid_sample_precompute_s"] = time.perf_counter() - t0
        del Rot_x_t, Rot_y_t
        _empty_device_cache(device)

    rec_t = torch.zeros((dimy, dimx, dimy), dtype=torch_dtype, device=device)
    rec_big_t = torch.zeros((obj.n2_oversampled, obj.n1_oversampled, obj.n2_oversampled), dtype=torch_dtype, device=device)
    ind_V = my_volumn_index(tuple(rec_big_t.shape), tuple(rec_t.shape))
    vol_slices = (
        slice(ind_V[0, 0] - 1, ind_V[0, 1]),
        slice(ind_V[1, 0] - 1, ind_V[1, 1]),
        slice(ind_V[2, 0] - 1, ind_V[2, 1]),
    )
    if obj.initial_model == 1:
        rec_t = _as_torch_f_contig(np.asarray(obj.Support, dtype=dtype), device, torch_dtype)
        rec_big_t[vol_slices] = rec_t

    print(f"RESIRE: Reconstructing with torch {device} FFT/interpolation path... \n\n")
    backend_label = f"torch:{device}:fft_interp_update"
    if use_grid_sample_backproj:
        backend_label += f":grid_sample_backproj{grid_sample_chunk_size}"
        if grid_sample_grid_chunks is not None:
            backend_label += ":pregrid"
        elif backproj_rot_on_demand:
            backend_label += ":rot_on_demand"
    if interp3_chunk_size > 0 and interp3_chunk_size < num_pj:
        backend_label += f":interp3chunk{interp3_chunk_size}"
    if avoid_fftshift_copy:
        backend_label += ":unshifted_fft"
    perf["grad_backend"] = backend_label
    print(f"Gradient/update backend: {backend_label}")

    if obj.monitor_R:
        monitorR_loopLength = obj.monitorR_loopLength
        errR = 0.0
        Rarr_record = np.zeros((num_pj, iterations // monitorR_loopLength))
        Rarr2_record = np.zeros((num_pj, iterations // monitorR_loopLength))

    for iter in range(iterations):
        print(f"iteration {iter}")

        t0 = time.perf_counter()
        recK_t = (shared_rfftn(shared_ifftshift(rec_big_t)) if spectral_interpolator else
                  (_torch_my_fft_unshifted_output(rec_big_t) if avoid_fftshift_copy else _torch_my_fft(rec_big_t)))
        _sync_for_timing(device, sync_timing)
        perf["torch_fft_s"] += time.perf_counter() - t0

        if interp3_chunk_size > 0 and interp3_chunk_size < num_pj:
            pj_cal_t = torch.empty((dimx, dimy, num_pj), dtype=torch_dtype, device=device)
            for start in range(0, num_pj, interp3_chunk_size):
                stop = min(start + interp3_chunk_size, num_pj)
                t_interp = time.perf_counter()
                pj_chunk_t = project_interp(recK_t, xj_t[:, :, start:stop],
                                            yj_t[:, :, start:stop], zj_t[:, :, start:stop])
                _sync_for_timing(device, sync_timing)
                perf["torch_interp3_s"] += time.perf_counter() - t_interp

                t_ifft = time.perf_counter()
                pj_chunk_t = shared_ifftshift(pj_chunk_t, axes=(0, 1))
                pj_chunk_t = shared_fftn(pj_chunk_t, axes=(0, 1), inverse=True, backend="torch_matlab_single")
                pj_chunk_t = shared_fftshift(pj_chunk_t, axes=(0, 1)).real
                pj_cal_t[:, :, start:stop] = _torch_cropped_out(
                    pj_chunk_t,
                    [dimx, dimy, stop - start],
                )
                _sync_for_timing(device, sync_timing)
                perf["torch_ifft2_s"] += time.perf_counter() - t_ifft
                del pj_chunk_t
            ifft_done = True
        else:
            t0 = time.perf_counter()
            pj_cal_t = project_interp(recK_t, xj_t, yj_t, zj_t)
            _sync_for_timing(device, sync_timing)
            perf["torch_interp3_s"] += time.perf_counter() - t0
            ifft_done = False

        if not ifft_done:
            t0 = time.perf_counter()
            pj_cal_t = shared_ifftshift(pj_cal_t, axes=(0, 1))
            pj_cal_t = shared_fftn(pj_cal_t, axes=(0, 1), inverse=True, backend="torch_matlab_single")
            pj_cal_t = shared_fftshift(pj_cal_t, axes=(0, 1)).real
            pj_cal_t = _torch_cropped_out(pj_cal_t, [dimx, dimy, num_pj])
            _sync_for_timing(device, sync_timing)
            perf["torch_ifft2_s"] += time.perf_counter() - t0

        if obj.monitor_R and (iter + 1) % monitorR_loopLength == 0:
            diff_t = pj_cal_t - projections_t
            Rarr_t = torch.sum(torch.abs(diff_t), dim=(0, 1)) / torch.sum(torch.abs(projections_t), dim=(0, 1))
            Rarr2_t = torch.linalg.vector_norm(diff_t, dim=(0, 1)) / torch.linalg.vector_norm(projections_t, dim=(0, 1))
            errR = float(torch.mean(Rarr_t).detach().cpu().item())
            errR2 = float(torch.mean(Rarr2_t).detach().cpu().item())
            print(f"RESIRE: Iteration {iter + 1}. Rfactor = {errR:.4f}, R2factor = {errR2:.4f}")
            print()
            record_index = (iter + 1) // monitorR_loopLength - 1
            Rarr_record[:, record_index] = Rarr_t.detach().cpu().numpy()
            Rarr2_record[:, record_index] = Rarr2_t.detach().cpu().numpy()
        else:
            print(f"RESIRE: Iteration {iter + 1}")

        grad_t0 = time.perf_counter()
        if use_grid_sample_backproj:
            t1 = time.perf_counter()
            if backproj_rot_on_demand and grid_sample_grid_chunks is None:
                backproj_sum_t = _grid_sample_backproj_sum_numpy_rot(
                    pj_cal_t,
                    obj.Rot_x,
                    obj.Rot_y,
                    device=device,
                    torch_dtype=torch_dtype,
                    chunk_size=grid_sample_chunk_size,
                )
            else:
                backproj_sum_t = _grid_sample_backproj_sum(
                    pj_cal_t,
                    Rot_x_t if "Rot_x_t" in locals() else None,
                    Rot_y_t if "Rot_y_t" in locals() else None,
                    chunk_size=grid_sample_chunk_size,
                    grid_chunks=grid_sample_grid_chunks,
                    vol_shape=grid_sample_vol_shape,
                )
            _sync_for_timing(device, sync_timing)
            backproj_s = time.perf_counter() - t1
            perf["grid_sample_backproj_s"] += backproj_s

            descent_t0 = time.perf_counter()
            grad_t = backproj_sum_t - sum_rot_pjs_t
            rec_t = torch.clamp(rec_t - dt * grad_t, min=0)
            _sync_for_timing(device, sync_timing)
            descent_update_s = time.perf_counter() - descent_t0
        else:
            grad_t = -sum_rot_pjs_t.clone()
            backproj_s = 0.0
            for k in range(num_pj):
                t1 = time.perf_counter()
                rot_pj_cal_t = torch_interp2(pj_cal_t[:, :, k], Rot_x_t[:, :, :, k], Rot_y_t[:, :, :, k])
                _sync_for_timing(device, sync_timing)
                backproj_s += time.perf_counter() - t1
                grad_t = grad_t + rot_pj_cal_t.to(dtype=torch_dtype)

            perf["torch_interp2_backproj_s"] += backproj_s

            descent_t0 = time.perf_counter()
            rec_t = torch.clamp(rec_t - dt * grad_t, min=0)
            _sync_for_timing(device, sync_timing)
            descent_update_s = time.perf_counter() - descent_t0
        grad_update_s = time.perf_counter() - grad_t0
        perf["grad_backend_s"] += grad_update_s
        perf["iter_grad_update_s"].append(grad_update_s)
        perf["descent_update_only_s"] += descent_update_s
        perf["iter_descent_update_only_s"].append(descent_update_s)

        if obj.save_temp and iter % obj.save_loopLength == 0:
            dir_path = os.path.dirname(obj.saveFilename)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            np.save(obj.saveFilename + str(iter) + ".npy", rec_t.detach().cpu().numpy().astype(dtype, copy=False))

        rec_big_t[vol_slices] = rec_t
        del recK_t, pj_cal_t
        if "grad_t" in locals():
            del grad_t
        if "backproj_sum_t" in locals():
            del backproj_sum_t
        if "rot_pj_cal_t" in locals():
            del rot_pj_cal_t
        if "diff_t" in locals():
            del diff_t, Rarr_t, Rarr2_t
        if getattr(obj, "gpu_clear_cache_each_iter", False):
            _empty_device_cache(device)

    if obj.monitor_R:
        obj.errR = errR
        obj.Rarr_record = Rarr_record
        obj.Rarr2_record = Rarr2_record
    perf["reconstruct_total_s"] = time.perf_counter() - reconstruct_start
    obj.reconstruction = np.asfortranarray(rec_t.detach().cpu().numpy().astype(dtype, copy=False))
