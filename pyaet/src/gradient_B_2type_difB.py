import numpy as np
from pyaet.src.my_ifft import my_ifft
from pyaet.src.my_fft import my_fft
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.make_fixed_fa_man import make_fixed_fa_man

def gradient_B_2type_difB(para, xdata, ydata):
    print('\nHB gradient algorithm')
    errR = []

    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    half_width = xdata['half_width']
    iterations = xdata['iterations']
    step_sz = xdata['step_sz']
    
    model = xdata['model']
    angles = xdata['angles']
    atom = xdata['atoms']
    num_atom = atom.size
    atom_type_num = len(np.unique(atom))
    
    N1, N2, num_pj = ydata.shape
    N_s = 2 * half_width + 1
    
    fixed_fa = make_fixed_fa_man([N1, N2], Res, Z_arr).reshape(N1, N2)
    max_fa = np.max(np.abs(fixed_fa))
    model = model / Res

    dtype = np.float32
    X_rot = np.zeros((num_pj, num_atom), dtype=dtype)
    Y_rot = np.zeros((num_pj, num_atom), dtype=dtype)
    Z_rot = np.zeros((num_pj, num_atom), dtype=dtype)
    
    for i in range(num_pj):
        R1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        R2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        R3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        R = np.dot(np.dot(R1, R2), R3).T
        
        rot_coords = np.dot(R, model)
        X_rot[i, :] = rot_coords[0, :]
        Y_rot[i, :] = rot_coords[1, :]
        Z_rot[i, :] = rot_coords[2, :]
    
    X_crop, Y_crop = np.meshgrid(np.arange(-half_width, half_width + 1), np.arange(-half_width, half_width + 1))
    Z_crop = np.arange(-half_width, half_width + 1)

    para = np.reshape(para, [2, atom_type_num])
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]
    
    num_atom_type = np.zeros(atom_type_num, dtype=int)
    for j in range(atom_type_num):
        num_atom_type[j] = np.count_nonzero(atom == (j + 1))
    
    t = (step_sz / max_fa ** 2 / num_pj / N1 ** 2) / num_atom_type
    
    for iter in range(iterations):
        grad = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        grad_h = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        grad_b = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        
        for i in range(num_pj):
            for j in range(atom_type_num):
                atom_type_j = (atom == (j + 1))
                atom_type_j = atom_type_j[0]
                
                # X_cen = np.reshape(X_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])
                # Y_cen = np.reshape(Y_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])
                # Z_cen = np.reshape(Z_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])

                X_cen = X_rot[i, atom_type_j]
                Y_cen = Y_rot[i, atom_type_j]
                Z_cen = Z_rot[i, atom_type_j]

                X_round = np.round(X_cen).astype(int)
                Y_round = np.round(Y_cen).astype(int)
                Z_round = np.round(Z_cen).astype(int)

                l2_xy = (np.add.outer(X_crop, X_round - X_cen) ** 2 +
                         np.add.outer(Y_crop, Y_round - Y_cen) ** 2)
                l2_z = np.add.outer(Z_crop, Z_round - Z_cen) ** 2

                # todo: some problem
                pj_j = np.exp(-l2_xy * b[j]) * np.sum(np.exp(-l2_z * b[j]), axis=0).reshape(1, 1, -1)

                pj_j_h = h[j] * pj_j

                bj_j = pj_j_h * l2_xy + h[j] * np.exp(-l2_xy * b[j]) * np.sum(l2_z * np.exp(-l2_z * b[j]), axis=0).reshape(1, 1, -1)
                
                for k in range(num_atom_type[j ]):
                    indx = X_round[k] + np.arange(-half_width, half_width + 1) + (N1 + 1) // 2 - 1
                    indy = Y_round[k] + np.arange(-half_width, half_width + 1) + (N2 + 1) // 2 - 1

                    indx = indx.astype(int)
                    indy = indy.astype(int)

                    grad[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i, j] += pj_j_h[:, :, k]
                    grad_h[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i, j] += pj_j[:, :, k]
                    grad_b[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i, j] += bj_j[:, :, k]
        
        projs = np.sum(grad, axis=3)
        for i in range(num_pj):
            projs[:, :, i] = np.real( my_ifft( my_fft(projs[:, :, i]) * fixed_fa ) ) #todo: check

        k = np.sum(projs.flatten() * ydata.flatten()) / np.sum(projs.flatten() ** 2)
        projs = projs * k
        grad_b = grad_b * k
        
        res = projs - ydata
        errR.append(np.sum(np.abs(res.flatten())) / np.sum(np.abs(ydata.flatten())))
        print(f'{iter + 1}.f = {errR[-1]:.5f}')
        
        for j in range(atom_type_num):
            b[j] = b[j] + (t[j] / N_s ** 6) * np.sum((res * grad_b[:, :, :, j]).flatten())
            h[j] = h[j] - (t[j]) * np.sum((res * grad_h[:, :, :, j]).flatten())
        
        h = np.maximum(h, 0)
        h = h / h[0]
        b = np.maximum(b, 0)

    param = np.vstack([k * h, (np.pi * Res) ** 2 / b])

    # return projs, param.flatten(), errR
    return projs, param, errR