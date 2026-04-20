import numpy as np
import os
#from scipy.interpolate import map_coordinates
from pyaet.src.my_volume_index import my_volumn_index
from pyaet.src.my_fft import my_fft
from pyaet.src.cropped_out import cropped_out
from pyaet.splinterp.splinterp3 import mex_function3
from pyaet.splinterp.splinterp2 import mex_function2
from pyaet.splinterp_cpp import mex_function3
from pyaet.splinterp_cpp import mex_function2

def reconstruct(obj):
    projections = obj.InputProjections
    num_pj = obj.num_projs
    step_size = obj.step_size
    dimx = obj.dim1
    dimy = obj.dim2
    dtype = obj.dtype
    Rot_x = obj.Rot_x
    Rot_y = obj.Rot_y
    iterations = obj.num_iterations

    xj = obj.xj
    yj = obj.yj
    zj = obj.zj

    sum_rot_pjs = obj.sum_rot_pjs

    rec = np.zeros((dimy, dimx, dimy), dtype=dtype)
    rec_big = np.zeros((obj.n2_oversampled, obj.n1_oversampled, obj.n2_oversampled), dtype=dtype)
    ind_V = my_volumn_index(rec_big.shape, rec.shape)
    if obj.initial_model == 1:
        rec = obj.Support
        rec_big[ind_V[0, 0]-1:ind_V[0, 1], ind_V[1, 0]-1:ind_V[1, 1], ind_V[2, 0]-1:ind_V[2, 1]] = rec
    dt = (step_size / num_pj / dimx)

    print('RESIRE: Reconstructing... \n\n')

    if obj.monitor_R:
        monitorR_loopLength = obj.monitorR_loopLength
        errR_arr = np.zeros(iterations // monitorR_loopLength)
        Rarr_record = np.zeros((num_pj, iterations // monitorR_loopLength))
        Rarr2_record = np.zeros((num_pj, iterations // monitorR_loopLength))

    # flag for using parallel calculation
    if obj.use_parallel:
        parforArg = True
    else:
        parforArg = False

    for iter in range(iterations):
        print(f"iteration {iter}")
        recK = my_fft(rec_big)

        # compute rotated projections via Fourier Slice Theorem
        # pj_cal = np.zeros((dimy, dimx, num_pj), dtype=dtype)
        # for k in range(num_pj):
        #     pj_cal[:, :, k] = map_coordinates(recK, [yj[:, :, k], xj[:, :, k], zj[:, :, k]], order=1)
        pj_cal = mex_function3(recK, xj, yj, zj)
        # pj_cal = np.real(my_fftshift(my_ifft2(my_ifftshift(pj_cal))))
        pj_cal = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pj_cal))))

        # np.savez('test_cropped_out.npz', pj_cal=pj_cal, dimx=dimx, dimy=dimy, num_pj=num_pj)

        pj_cal = cropped_out(pj_cal, [dimx, dimy, num_pj])

        # compute R factor
        if obj.monitor_R and iter % monitorR_loopLength == 0:
            Rarr = np.zeros(num_pj)
            Rarr2 = np.zeros(num_pj)
            for i in range(num_pj):
                pj = projections[:, :, i]
                proj_i = pj_cal[:, :, i]
                Rarr[i] = np.sum(np.abs(proj_i - pj)) / np.sum(np.abs(pj))
                Rarr2[i] = np.linalg.norm(proj_i - pj, 'fro') / np.linalg.norm(pj, 'fro')
            errR = np.mean(Rarr)
            errR2 = np.mean(Rarr2)
            print(f'RESIRE: Iteration {iter}. Rfactor = {errR:.4f}, R2factor = {errR2:.4f}')
            print()
            errR_arr[iter // monitorR_loopLength-1] = errR
            Rarr_record[:, iter // monitorR_loopLength-1] = Rarr
            Rarr2_record[:, iter // monitorR_loopLength-1] = Rarr2
        else:
            print(f'RESIRE: Iteration {iter}')

        # compute gradient & apply gradient descent
        grad = -sum_rot_pjs
        for k in range(num_pj):
            # rot_pj_cal = map_coordinates(pj_cal[:, :, k], [Rot_y[:, :, :, k], Rot_x[:, :, :, k]], order=1)
            rot_pj_cal = mex_function2(pj_cal[:, :, k], Rot_x[:, :, :, k], Rot_y[:, :, :, k])
            grad = grad + rot_pj_cal
        rec = rec - dt * grad
        rec = np.maximum(0, rec)

        # flag for saving temporary reconstruction
        if obj.save_temp and iter % obj.save_loopLength == 0:

            # check if folder exists, otherwise, create the folder.
            dir_path = os.path.dirname(obj.saveFilename)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            temp_filename = obj.saveFilename + str(iter) + '.npy'
            np.save(temp_filename, rec)

        rec_big[ind_V[0, 0]-1:ind_V[0, 1], ind_V[1, 0]-1:ind_V[1, 1], ind_V[2, 0]-1:ind_V[2, 1]] = rec

    if obj.monitor_R:
        obj.errR = errR
        obj.Rarr_record = Rarr_record
        obj.Rarr2_record = Rarr2_record
    obj.reconstruction = rec
