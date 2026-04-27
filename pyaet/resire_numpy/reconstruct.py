import os

import numpy as np

from pyaet.src.cropped_out import cropped_out
from pyaet.src.my_fft import my_fft
from pyaet.src.my_volume_index import my_volumn_index
from pyaet.splinterp_cpp import mex_function2, mex_function3


def reconstruct(obj):
    """Run iterative RESIRE reconstruction with NumPy and C++ splinterp.

    This function mirrors the MATLAB RESIRE reconstruction loop. Fourier-space
    interpolation uses ``mex_function3`` and rotated back-projection
    interpolation uses ``mex_function2``. It intentionally does not fall back
    to ``interp2_bilinear``.
    """
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

    print("RESIRE: Reconstructing... \n\n")

    if obj.monitor_R:
        monitorR_loopLength = obj.monitorR_loopLength
        errR_arr = np.zeros(iterations // monitorR_loopLength)
        Rarr_record = np.zeros((num_pj, iterations // monitorR_loopLength))
        Rarr2_record = np.zeros((num_pj, iterations // monitorR_loopLength))

    for iter in range(iterations):
        print(f"iteration {iter}")
        recK = np.asfortranarray(my_fft(rec_big).astype(np.complex128, copy=False))

        pj_cal = mex_function3(recK, xj_c, yj_c, zj_c)
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

        grad = np.asfortranarray((-sum_rot_pjs).astype(dtype, copy=False))
        for k in range(num_pj):
            pj_cal_k = np.asfortranarray(pj_cal[:, :, k])
            rot_pj_cal = mex_function2(
                pj_cal_k,
                np.ascontiguousarray(Rot_x[:, :, :, k]),
                np.ascontiguousarray(Rot_y[:, :, :, k]),
            )
            grad = grad + np.asfortranarray(np.asarray(rot_pj_cal, dtype=dtype))

        rec = np.asfortranarray(np.maximum(0, rec - dt * grad).astype(dtype, copy=False))

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
    obj.reconstruction = rec
