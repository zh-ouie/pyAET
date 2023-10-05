import numpy as np

def gradient_fixHB_XYZ(para, xdata, ydata):
    print('\nHB gradient algorithm')
    errR = []
    model_arr = []
    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    halfWidth = xdata['halfWidth']
    iterations = xdata['iterations']
    step_sz = xdata['step_sz']
    model = xdata['model']
    model_ori = xdata['model_ori']
    angles = xdata['angles']
    atom = xdata['atoms']
    num_atom = len(atom)
    num_atom_type = len(np.unique(atom))
    N1, N2, num_pj = ydata.shape

    fixedfa = make_fixedfa_man([N1, N2], Res, Z_arr).reshape((N1, N2))
    model = model / Res
    model_ori = model_ori / Res

    h = np.zeros(num_atom, dtype=np.float32)
    b = np.zeros(num_atom, dtype=np.float32)

    if para.shape[1] == num_atom_type:
        for k in range(num_atom_type):
            h[atom == k] = para[0, k]
            b[atom == k] = para[1, k]
    elif para.shape[1] == num_atom:
        h[:] = para[0, :]
        b[:] = para[1, :]
    else:
        print('error')
        return

    b = (Res * np.pi) ** 2 / b

    N_s = 2 * halfWidth + 1
    index = np.zeros((2, num_pj, num_atom), dtype=np.int32)

    grad_h_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_b_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_x_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_y_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)
    grad_z_set = np.zeros((N_s, N_s, num_pj, num_atom), dtype=np.float32)

    X_crop, Y_crop = np.meshgrid(np.arange(-halfWidth, halfWidth + 1), np.arange(-halfWidth, halfWidth + 1))
    Z_crop = np.arange(-halfWidth, halfWidth + 1)

    X_crop = X_crop.astype(np.float32)
    Y_crop = Y_crop.astype(np.float32)
    Z_crop = Z_crop.astype(np.float32)

    X_ori = model_ori[0, :].reshape((1, 1, num_atom))
    Y_ori = model_ori[1, :].reshape((1, 1, num_atom))
    Z_ori = model_ori[2, :].reshape((1, 1, num_atom))
    X = model[0, :].reshape((1, 1, num_atom))
    Y = model[1, :].reshape((1, 1, num_atom))
    Z = model[2, :].reshape((1, 1, num_atom))
    scale = 1 / Res

    for iter in range(iterations):
        Projs = np.zeros((N1, N2, num_pj), dtype=np.float32)

        for i in range(num_pj):
            RM1 = MatrixQuaternionRot([0, 0, 1], angles[i, 0])
            RM2 = MatrixQuaternionRot([0, 1, 0], angles[i, 1])
            RM3 = MatrixQuaternionRot([1, 0, 0], angles[i, 2])
            R = np.dot(np.dot(RM1, RM2), RM3)
            model_rot = np.dot(R.T, np.vstack((X.flatten(), Y.flatten(), Z.flatten())))

            X_cen = model_rot[0, :].reshape((1, 1, num_atom))
            Y_cen = model_rot[1, :].reshape((1, 1, num_atom))
            Z_cen = model_rot[2, :].reshape((1, 1, num_atom))

            X_round = np.round(X_cen)
            Y_round = np.round(Y_cen)
            Z_round = np.round(Z_cen)

            Dx = X_crop + X_round - X_cen
            Dy = Y_crop + Y_round - Y_cen
            Dz = Z_crop + Z_round - Z_cen

            l2_xy = Dx ** 2 + Dy ** 2
            l2_z = Dz ** 2

            l2_xy_b = l2_xy * b
            l2_z_b = l2_z * b
            exp_l2_z_b = np.exp(-l2_z_b)
            exp_l2_xy_b = np.exp(-l2_xy_b)

            pj_j = exp_l2_xy_b * np.sum(exp_l2_z_b, axis=None) * h
            pj_j_b = pj_j * b
            grad_exp = exp_l2_xy_b * np.sum(l2_z * np.exp(-l2_z_b), axis=None) * h
            bj_j = h * grad_exp + pj_j * l2_xy

            R2_Dx = ((R[0, 0] * Dx + R[0, 1] * Dy) * pj_j_b)
            R2_Dy = ((R[1, 0] * Dx + R[1, 1] * Dy) * pj_j_b)
            R2_Dz = ((R[2, 0] * Dx + R[2, 1] * Dy) * pj_j_b)

            Dz_exp_l2_z_b = Dz * exp_l2_z_b
            sum_Dz_exp = (np.sum(Dz_exp_l2_z_b, axis=None) * exp_l2_xy_b)

            sum_Dz_hb = (sum_Dz_exp * h * b)

            xj_j = R2_Dx + R[0, 2] * sum_Dz_hb
            yj_j = R2_Dy + R[1, 2] * sum_Dz_hb
            zj_j = R2_Dz + R[2, 2] * sum_Dz_hb

            for k in range(num_atom):
                indx = X_round[0, 0, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N1 + 1) / 2))
                indy = Y_round[0, 0, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N2 + 1) / 2))
                Projs[indx, indy, i] = Projs[indx, indy, i] + pj_j[:, :, k]
                index[:, i, k] = [X_round[0, 0, k], Y_round[0, 0, k]]
                grad_h_set[:, :, i, k] = pj_j[:, :, k]
                grad_b_set[:, :, i, k] = bj_j[:, :, k]
                grad_x_set[:, :, i, k] = xj_j[:, :, k]
                grad_y_set[:, :, i, k] = yj_j[:, :, k]
                grad_z_set[:, :, i, k] = zj_j[:, :, k]

        for i in range(num_pj):
            Projs[:, :, i] = np.real(my_ifft(my_fft(Projs[:, :, i]) * fixedfa))

        res = Projs - ydata
        errR.append(np.sum(np.abs(Projs - ydata)) / np.sum(np.abs(ydata)))
        print(f'{iter+1}.f = {errR[-1]:.5f}')

        for i in range(num_pj):
            res[:, :, i] = np.real(my_ifft(my_fft(res[:, :, i]) * np.conj(fixedfa)))

        for k in range(num_atom):
            grad_x_k = 0
            grad_y_k = 0
            grad_z_k = 0
            grad_h_k = 0
            grad_b_k = 0
            for i in range(num_pj):
                indx = index[0, i, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N1 + 1) / 2))
                indy = index[1, i, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N2 + 1) / 2))
                grad_x_k += np.sum(np.sum(res[indx, indy, i] * grad_x_set[:, :, i, k]))
                grad_y_k += np.sum(np.sum(res[indx, indy, i] * grad_y_set[:, :, i, k]))
                grad_z_k += np.sum(np.sum(res[indx, indy, i] * grad_z_set[:, :, i, k]))
                grad_h_k += np.sum(np.sum(res[indx, indy, i] * grad_h_set[:, :, i, k]))
                grad_b_k += np.sum(np.sum(res[indx, indy, i] * grad_b_set[:, :, i, k]))

            dt = step_sz / np.mean(h) ** 2 / np.mean(b) ** 2 / halfWidth ** 2 / num_pj / (N1 * N2)
            X[0, 0, k] -= dt * grad_x_k
            Y[0, 0, k] -= dt * grad_y_k
            Z[0, 0, k] -= dt * grad_z_k

        diff_X = X - X_ori
        diff_Y = Y - Y_ori
        diff_Z = Z - Z_ori
        diff_norm = np.sqrt(diff_X ** 2 + diff_Y ** 2 + diff_Z ** 2)
        index_3 = diff_norm > scale
        X[index_3] = X_ori[index_3] + scale * diff_X[index_3] / diff_norm[index_3]
        Y[index_3] = Y_ori[index_3] + scale * diff_Y[index_3] / diff_norm[index_3]
        Z[index_3] = Z_ori[index_3] + scale * diff_Z[index_3] / diff_norm[index_3]

        h = np.maximum(h, 0)
        b = np.maximum(b, 0)

        model_arr.append([X[0, 0, :], Y[0, 0, :], Z[0, 0, :]] * Res)

    model = np.array([X[0, 0, :], Y[0, 0, :], Z[0, 0, :]]) * Res
    params = [h, (Res * np.pi) ** 2 / b, model]
    return Projs, params, errR, model_arr