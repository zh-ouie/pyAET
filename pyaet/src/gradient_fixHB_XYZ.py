import numpy as np
from pyaet.src.my_ifft import my_ifft
from pyaet.src.my_fft import my_fft
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.make_fixed_fa_man import make_fixed_fa_man


def gradient_fixHB_XYZ(para, xdata, ydata):
    print('\nHB gradient algorithm')
    errR = []
    model_arr = []

    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    half_width = xdata['half_width']
    iterations = xdata['iterations']
    step_sz = xdata['step_sz']
    model = xdata['model']
    model_ori = xdata['model_ori']
    angles = xdata['angles']
    atom = xdata['atoms'][0] #todo: check

    num_atom = atom.size
    atom_type_num = len(np.unique(atom))

    N1, N2, num_pj = ydata.shape

    fixed_fa = make_fixed_fa_man([N1, N2], Res, Z_arr).reshape(N1, N2)
    model = model / Res
    model_ori = model_ori / Res

    dtype = np.float32
    Y_crop, X_crop = np.meshgrid(np.arange(-half_width, half_width + 1), np.arange(-half_width, half_width + 1))
    Z_crop = np.arange(-half_width, half_width + 1)

    para = np.reshape(para, [2, atom_type_num])
    h = np.zeros(num_atom, dtype=dtype)
    b = np.zeros(num_atom, dtype=dtype)
    # h = np.zeros((1, 1, num_atom), dtype=dtype)
    # b = np.zeros((1, 1, num_atom), dtype=dtype)

    if para.shape[1] == atom_type_num:
        for k in range(atom_type_num):
            h[atom == k+1] = para[0, k] #todo: check. why error?
            b[atom == k+1] = para[1, k]
    elif para.shape[1] == num_atom:
        h[:] = para[0, :]
        b[:] = para[1, :]
    else:
        print('error')
        return

    b = (Res * np.pi) ** 2 / b

    N_s = 2 * half_width + 1
    index = np.zeros((2, num_pj, num_atom), dtype=np.int32)

    grad_h_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_b_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_x_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_y_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_z_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)

    h = h.astype(np.float32)
    b = b.astype(np.float32)
    X_crop = X_crop.astype(np.float32)
    Y_crop = Y_crop.astype(np.float32)
    Z_crop = Z_crop.astype(np.float32)

    X_ori = model_ori[0, :].reshape((1, 1, num_atom))
    Y_ori = model_ori[1, :].reshape((1, 1, num_atom))
    Z_ori = model_ori[2, :].reshape((1, 1, num_atom))

    # X_ori = model_ori[0, :]
    # Y_ori = model_ori[1, :]
    # Z_ori = model_ori[2, :]

    X = model[0, :].reshape((1, 1, num_atom))
    Y = model[1, :].reshape((1, 1, num_atom))
    Z = model[2, :].reshape((1, 1, num_atom))

    # X = model[0, :]
    # Y = model[1, :]
    # Z = model[2, :]

    scale = 1 / Res

    for iter in range(iterations):
        projs = np.zeros((N1, N2, num_pj), dtype=np.float32)

        for i in range(num_pj):
            RM1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
            RM2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
            RM3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
            R = np.dot(np.dot(RM1, RM2), RM3)
            model_rot = np.dot(R.T, np.vstack((X.flatten(), Y.flatten(), Z.flatten())))

            X_cen = model_rot[0, :].reshape((1, 1, num_atom))
            Y_cen = model_rot[1, :].reshape((1, 1, num_atom))
            Z_cen = model_rot[2, :].reshape((1, 1, num_atom))

            # X_cen = model_rot[0, :]
            # Y_cen = model_rot[1, :]
            # Z_cen = model_rot[2, :]

            X_round = np.round(X_cen).astype(int)
            Y_round = np.round(Y_cen).astype(int)
            Z_round = np.round(Z_cen).astype(int)

            Dx = X_crop.reshape(X_crop.shape[0], X_crop.shape[1], 1) + (X_round - X_cen) #todo: stuck! bsxfun plus can handle different size.
            Dy = Y_crop.reshape(Y_crop.shape[0], Y_crop.shape[1], 1) + (Y_round - Y_cen)
            Dz = Z_crop.reshape(1, Z_crop.shape[0], 1) + (Z_round - Z_cen)

            l2_xy = Dx ** 2 + Dy ** 2
            l2_z = Dz ** 2

            l2_xy_b = l2_xy * b
            l2_z_b = l2_z * b
            exp_l2_z_b = np.exp(-l2_z_b)
            exp_l2_xy_b = np.exp(-l2_xy_b)

            pj_j = exp_l2_xy_b * (np.sum(exp_l2_z_b, axis=1).reshape(1, 1, -1))
            pj_j = pj_j * h
            pj_j_b = pj_j * b

            grad_exp = exp_l2_xy_b * (np.sum(l2_z * np.exp(-l2_z_b), axis=1).reshape(1, 1, -1))
            bj_j = h * grad_exp + pj_j * l2_xy

            R2_Dx = ((R[0, 0] * Dx + R[0, 1] * Dy) * pj_j_b)
            R2_Dy = ((R[1, 0] * Dx + R[1, 1] * Dy) * pj_j_b)
            R2_Dz = ((R[2, 0] * Dx + R[2, 1] * Dy) * pj_j_b)

            Dz_exp_l2_z_b = Dz * exp_l2_z_b
            sum_Dz_exp = ((np.sum(Dz_exp_l2_z_b, axis=1).reshape(1,1,-1)) * exp_l2_xy_b)

            sum_Dz_hb = sum_Dz_exp * (h * b)

            xj_j = R2_Dx + R[0, 2] * sum_Dz_hb
            yj_j = R2_Dy + R[1, 2] * sum_Dz_hb
            zj_j = R2_Dz + R[2, 2] * sum_Dz_hb

            for k in range(num_atom):
                indx = X_round[0,0,k] + np.arange(-half_width, half_width + 1) + (N1 + 1) // 2 -1
                indy = Y_round[0,0,k] + np.arange(-half_width, half_width + 1) + (N2 + 1) // 2 -1

                indx = indx.astype(int)
                indy = indy.astype(int)

                projs[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] += pj_j[:, :, k]
                index[:, i, k] = [X_round[0,0,k], Y_round[0,0,k]]

                grad_h_set[:, :, i, k] = pj_j[:, :, k]
                grad_b_set[:, :, i, k] = bj_j[:, :, k]

                grad_x_set[:, :, i, k] = xj_j[:, :, k]
                grad_y_set[:, :, i, k] = yj_j[:, :, k]
                grad_z_set[:, :, i, k] = zj_j[:, :, k]

        for i in range(num_pj):
            projs[:, :, i] = np.real( my_ifft( my_fft(projs[:, :, i]) * fixed_fa ) ) #todo: check

        res = projs - ydata
        errR.append(np.sum(np.abs(res.flatten())) / np.sum(np.abs(ydata.flatten())))
        print(f'{iter+1}.f = {errR[-1]:.5f}')

        for k in range(num_atom):
            grad_x_k = 0
            grad_y_k = 0
            grad_z_k = 0
            grad_h_k = 0
            grad_b_k = 0
            for i in range(num_pj):
                indx = index[0, i, k] + np.arange(-half_width, half_width + 1) + (N1 + 1) // 2 -1
                indy = index[1, i, k] + np.arange(-half_width, half_width + 1) + (N2 + 1) // 2 -1

                indx = indx.astype(int)
                indy = indy.astype(int)

                grad_x_k += np.sum((res[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] * grad_x_set[:, :, i, k]).flatten())
                grad_y_k += np.sum((res[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] * grad_y_set[:, :, i, k]).flatten())
                grad_z_k += np.sum((res[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] * grad_z_set[:, :, i, k]).flatten())
                grad_h_k += np.sum((res[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] * grad_h_set[:, :, i, k]).flatten())
                grad_b_k += np.sum((res[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i] * grad_b_set[:, :, i, k]).flatten()) #todo: check

            dt = step_sz / np.mean(h) ** 2 / np.mean(b) ** 2 / half_width ** 2 / num_pj / (N1 * N2)
            X[0, 0, k] -= dt * grad_x_k
            Y[0, 0, k] -= dt * grad_y_k
            Z[0, 0, k] -= dt * grad_z_k

        diff_X = X - X_ori
        diff_Y = Y - Y_ori
        diff_Z = Z - Z_ori
        diff_norm = np.sqrt(diff_X ** 2 + diff_Y ** 2 + diff_Z ** 2)
        index_3 = (diff_norm > scale) #todo: check
        X[index_3] = X_ori[index_3] + scale * diff_X[index_3] / diff_norm[index_3]
        Y[index_3] = Y_ori[index_3] + scale * diff_Y[index_3] / diff_norm[index_3]
        Z[index_3] = Z_ori[index_3] + scale * diff_Z[index_3] / diff_norm[index_3]

        h = np.maximum(h, 0)
        b = np.maximum(b, 0)

        # model_arr.append([X[0, 0, :], Y[0, 0, :], Z[0, 0, :]] * Res)
        model_arr.append((np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T * Res).T)  #todo: change model_arr from list into 3D matrix. So that model_arr(:,:,iter)= new_value.

    # model = np.array([X[0, 0, :], Y[0, 0, :], Z[0, 0, :]]) * Res
    model = (np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T * Res).T
    param = np.vstack([h.flatten(), (Res * np.pi) ** 2 / b, model])

    return projs, param, errR, model_arr