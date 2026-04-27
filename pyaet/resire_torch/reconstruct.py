import os
import time

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pyaet.src.cropped_out import cropped_out
from pyaet.src.my_fft import my_fft
from pyaet.src.my_volume_index import my_volumn_index
from pyaet.splinterp_cpp import mex_function2, mex_function3


def _torch_dtype_from_numpy(dtype: np.dtype):
    if dtype == np.dtype(np.float32):
        return torch.float32
    if dtype == np.dtype(np.float64):
        return torch.float64
    raise TypeError(f"Unsupported dtype for torch backend: {dtype}")


def _resolve_device(preferred_device):
    if torch is None:
        return "cpu"
    if preferred_device:
        if preferred_device == "cuda" and not torch.cuda.is_available():
            print("Torch backend requested cuda, but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return preferred_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_torch(array: np.ndarray, device: str, torch_dtype):
    return torch.as_tensor(np.asarray(array), dtype=torch_dtype, device=device)


def reconstruct(obj):
    """Run RESIRE reconstruction with torch-assisted gradient/update steps.

    Fourier-space interpolation uses C++ ``mex_function3`` and rotated
    back-projection interpolation uses C++ ``mex_function2``. This path does
    not fall back to the Python ``interp2_bilinear`` helper.
    """
    reconstruct_start = time.perf_counter()
    projections = obj.InputProjections.astype(np.float64, copy=False)
    num_pj = obj.num_projs
    step_size = obj.step_size
    dimx = obj.dim1
    dimy = obj.dim2
    dtype = np.dtype(obj.dtype)
    iterations = obj.num_iterations

    xj = np.asfortranarray(obj.xj.astype(np.float64, copy=False))
    yj = np.asfortranarray(obj.yj.astype(np.float64, copy=False))
    zj = np.asfortranarray(obj.zj.astype(np.float64, copy=False))
    xj_c = np.ascontiguousarray(xj)
    yj_c = np.ascontiguousarray(yj)
    zj_c = np.ascontiguousarray(zj)

    sum_rot_pjs = np.asarray(obj.sum_rot_pjs, dtype=dtype, order="F")
    Rot_x = np.asfortranarray(obj.Rot_x.astype(np.float64, copy=False))
    Rot_y = np.asfortranarray(obj.Rot_y.astype(np.float64, copy=False))
    dt = step_size / num_pj / dimx

    rec = np.zeros((dimy, dimx, dimy), dtype=dtype, order="F")
    rec_big = np.zeros((obj.n2_oversampled, obj.n1_oversampled, obj.n2_oversampled), dtype=dtype, order="F")
    ind_V = my_volumn_index(rec_big.shape, rec.shape)
    if obj.initial_model == 1:
        rec = obj.Support
        rec_big[ind_V[0, 0] - 1:ind_V[0, 1], ind_V[1, 0] - 1:ind_V[1, 1], ind_V[2, 0] - 1:ind_V[2, 1]] = rec
    print("RESIRE: Reconstructing with torch grad/update path... \n\n")

    if obj.monitor_R:
        monitorR_loopLength = obj.monitorR_loopLength
        errR_arr = np.zeros(iterations // monitorR_loopLength)
        Rarr_record = np.zeros((num_pj, iterations // monitorR_loopLength))
        Rarr2_record = np.zeros((num_pj, iterations // monitorR_loopLength))

    perf = getattr(obj, "perf_stats", {})
    perf = perf.setdefault("reconstruct_torch", {})
    perf["iterations"] = int(iterations)
    perf["mex_function3_calls"] = 0
    perf["mex_function3_s"] = 0.0
    perf["mex_function2_backproj_calls"] = 0
    perf["mex_function2_backproj_s"] = 0.0
    perf["grad_backend_s"] = 0.0
    perf["iter_grad_update_s"] = []
    perf["descent_update_only_s"] = 0.0
    perf["iter_descent_update_only_s"] = []

    backend_label = "numpy"
    device = "cpu"
    if torch is not None:
        device = _resolve_device(getattr(obj, "gpu_grad_device", None))
        torch_dtype = _torch_dtype_from_numpy(dtype)
        base_grad_t = _to_torch(sum_rot_pjs, device, torch_dtype)
        rec_t = _to_torch(rec, device, torch_dtype)
        backend_label = f"torch:{device}"
    else:
        base_grad_t = None
        rec_t = None
        torch_dtype = None

    perf["grad_backend"] = backend_label
    print(f"Gradient/update backend: {backend_label}")

    for iter in range(iterations):
        print(f"iteration {iter}")

        recK = np.asfortranarray(my_fft(rec_big).astype(np.complex128, copy=False))
        t0 = time.perf_counter()
        pj_cal = mex_function3(recK, xj_c, yj_c, zj_c)
        perf["mex_function3_calls"] += 1
        perf["mex_function3_s"] += time.perf_counter() - t0
        pj_cal = np.fft.ifftshift(pj_cal, axes=(0, 1))
        pj_cal = np.fft.ifft2(pj_cal, axes=(0, 1))
        pj_cal = np.fft.fftshift(pj_cal, axes=(0, 1))
        pj_cal = pj_cal.real
        pj_cal = cropped_out(pj_cal, [dimx, dimy, num_pj])

        if obj.monitor_R and (iter + 1) % monitorR_loopLength == 0:
            Rarr = np.zeros(num_pj)
            Rarr2 = np.zeros(num_pj)
            for i in range(num_pj):
                pj = projections[:, :, i]
                proj_i = pj_cal[:, :, i]
                Rarr[i] = np.sum(np.abs(proj_i - pj)) / np.sum(np.abs(pj))
                Rarr2[i] = np.linalg.norm(proj_i - pj, "fro") / np.linalg.norm(pj, "fro")
            errR = np.mean(Rarr)
            errR2 = np.mean(Rarr2)
            print(f"RESIRE: Iteration {iter + 1}. Rfactor = {errR:.4f}, R2factor = {errR2:.4f}")
            print()
            record_index = (iter + 1) // monitorR_loopLength - 1
            errR_arr[record_index] = errR
            Rarr_record[:, record_index] = Rarr
            Rarr2_record[:, record_index] = Rarr2
        else:
            print(f"RESIRE: Iteration {iter + 1}")

        grad_t0 = time.perf_counter()
        if rec_t is None:
            grad = np.asfortranarray((-sum_rot_pjs).astype(dtype, copy=False))
        else:
            grad_t = -base_grad_t.clone()

        for k in range(num_pj):
            pj_cal_k = np.asfortranarray(pj_cal[:, :, k])
            t1 = time.perf_counter()
            rot_pj_cal = mex_function2(
                pj_cal_k,
                np.ascontiguousarray(Rot_x[:, :, :, k]),
                np.ascontiguousarray(Rot_y[:, :, :, k]),
            )
            perf["mex_function2_backproj_s"] += time.perf_counter() - t1
            perf["mex_function2_backproj_calls"] += 1
            rot_pj_cal = np.asfortranarray(np.asarray(rot_pj_cal, dtype=dtype))
            if rec_t is None:
                grad = grad + rot_pj_cal
            else:
                grad_t = grad_t + _to_torch(rot_pj_cal, device, torch_dtype)

        if rec_t is None:
            descent_t0 = time.perf_counter()
            rec = np.asfortranarray(np.maximum(0, rec - dt * grad).astype(dtype, copy=False))
            descent_update_s = time.perf_counter() - descent_t0
        else:
            descent_t0 = time.perf_counter()
            rec_t = torch.clamp(rec_t - dt * grad_t, min=0)
            descent_update_s = time.perf_counter() - descent_t0
            rec = np.asfortranarray(rec_t.detach().cpu().numpy().astype(dtype, copy=False))

        grad_update_s = time.perf_counter() - grad_t0
        perf["grad_backend_s"] += grad_update_s
        perf["iter_grad_update_s"].append(grad_update_s)
        perf["descent_update_only_s"] += descent_update_s
        perf["iter_descent_update_only_s"].append(descent_update_s)

        if obj.save_temp and iter % obj.save_loopLength == 0:
            dir_path = os.path.dirname(obj.saveFilename)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            temp_filename = obj.saveFilename + str(iter) + ".npy"
            np.save(temp_filename, rec)

        rec_big[ind_V[0, 0] - 1:ind_V[0, 1], ind_V[1, 0] - 1:ind_V[1, 1], ind_V[2, 0] - 1:ind_V[2, 1]] = rec
        rec_big = np.asfortranarray(rec_big.astype(dtype, copy=False))

    if obj.monitor_R:
        obj.errR = errR
        obj.Rarr_record = Rarr_record
        obj.Rarr2_record = Rarr2_record
    perf["reconstruct_total_s"] = time.perf_counter() - reconstruct_start
    obj.reconstruction = rec
