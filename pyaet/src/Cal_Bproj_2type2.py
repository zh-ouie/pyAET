import numpy as np
from src.my_ifft import my_ifft
from src.my_fft import my_fft
from src.make_fixedfa_man import make_fixedfa_man
from src.MatrixQuaternionRot import MatrixQuaternionRot

def Cal_Bproj_2type2(para, xdata):
    para = np.abs(para)

    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    halfWidth = xdata['halfWidth']
    model = xdata['model']
    angles = xdata['angles']
    atom = xdata['atoms']
    num_atom = len(atom)
    atom_type_num = len(np.unique(atom))
    ydata = xdata['projections']

    N1, N2, _ = ydata.shape
    num_pj = angles.shape[0]

    fixedfa = np.reshape(make_fixedfa_man([N1, N2], Res, Z_arr), [N1, N2])
    model = model / Res

    dtype = 'single'
    X_rot = np.zeros((num_pj, num_atom), dtype=dtype)
    Y_rot = np.zeros((num_pj, num_atom), dtype=dtype)
    Z_rot = np.zeros((num_pj, num_atom), dtype=dtype)

    for i in range(num_pj):
        R1 = MatrixQuaternionRot([0, 0, 1], angles[i, 0])
        R2 = MatrixQuaternionRot([0, 1, 0], angles[i, 1])
        R3 = MatrixQuaternionRot([1, 0, 0], angles[i, 2])
        R = (np.dot(np.dot(R1, R2), R3)).T

        rotCoords = np.dot(R, model)
        X_rot[i, :] = rotCoords[0, :]
        Y_rot[i, :] = rotCoords[1, :]
        Z_rot[i, :] = rotCoords[2, :]

    X_crop, Y_crop = np.meshgrid(np.arange(-halfWidth, halfWidth + 1), np.arange(-halfWidth, halfWidth + 1))
    Z_crop = -halfWidth + np.arange(2 * halfWidth + 1)

    para = np.reshape(para, [2, atom_type_num])
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]

    num_atom_type = np.zeros(atom_type_num, dtype=int)
    for j in range(atom_type_num):
        num_atom_type[j] = np.count_nonzero(atom == (j + 1))

    Grad = np.zeros((N1, N2, num_pj, 3), dtype=dtype)

    for i in range(num_pj):
        for j in range(atom_type_num):
            atom_type_j = atom == (j + 1)

            X_cen = np.reshape(X_rot[i, atom_type_j], (1, 1, num_atom_type[j]))
            Y_cen = np.reshape(Y_rot[i, atom_type_j], (1, 1, num_atom_type[j]))
            Z_cen = np.reshape(Z_rot[i, atom_type_j], (1, 1, num_atom_type[j]))

            X_round = np.round(X_cen)
            Y_round = np.round(Y_cen)
            Z_round = np.round(Z_cen)

            l2_xy = (np.add.outer(X_crop, X_round - X_cen) ** 2 +
                     np.add.outer(Y_crop, Y_round - Y_cen) ** 2)
            l2_z = np.add.outer(Z_crop, Z_round - Z_cen) ** 2

            pj_j = (np.exp(-l2_xy * b[j]) *
                    np.sum(np.exp(-l2_z * b[j])))
            pj_j_h = h[j] * pj_j

            for k in range(num_atom_type[j]):
                indx = X_round[0, 0, k] + (-halfWidth + np.arange(2 * halfWidth + 1))
                indy = Y_round[0, 0, k] + (-halfWidth + np.arange(2 * halfWidth + 1))

                Grad[indx, indy, i, j] += pj_j_h[:, :, k]

    Projs = np.sum(Grad, axis=3)

    for i in range(num_pj):
        Projs[:, :, i] = my_ifft(my_fft(Projs[:, :, i]) * fixedfa)

    k = np.sum(Projs * ydata) / np.sum(Projs ** 2)
    Projs = Projs * k

    param = np.concatenate((k * h, (np.pi * Res) ** 2 / b))

    return Projs, param
