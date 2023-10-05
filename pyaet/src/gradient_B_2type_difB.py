import numpy as np

def gradient_B_2type_difB(para, xdata, ydata):
    errR = []
    print('\nHB gradient algorithm')
    
    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    halfWidth = xdata['halfWidth']
    iterations = xdata['iterations']
    step_sz = xdata['step_sz']
    
    model = xdata['model']
    angles = xdata['angles']
    atom = xdata['atoms']
    num_atom = len(atom)
    atom_type_num = len(np.unique(atom))
    
    N1, N2, num_pj = ydata.shape
    N_s = 2 * halfWidth + 1
    
    fixedfa = np.reshape(make_fixedfa_man([N1, N2], Res, Z_arr), [N1, N2])
    max_fa = np.max(np.abs(fixedfa))
    model = model / Res
    
    X_rot = np.zeros((num_pj, num_atom), dtype=np.float32)
    Y_rot = np.zeros((num_pj, num_atom), dtype=np.float32)
    Z_rot = np.zeros((num_pj, num_atom), dtype=np.float32)
    
    for i in range(num_pj):
        R1 = MatrixQuaternionRot([0, 0, 1], angles[i, 0])
        R2 = MatrixQuaternionRot([0, 1, 0], angles[i, 1])
        R3 = MatrixQuaternionRot([1, 0, 0], angles[i, 2])
        R = np.dot(np.dot(R1, R2), R3).T
        
        rotCoords = np.dot(R, model)
        X_rot[i, :] = rotCoords[0, :]
        Y_rot[i, :] = rotCoords[1, :]
        Z_rot[i, :] = rotCoords[2, :]
    
    X_crop, Y_crop = np.meshgrid(np.arange(-halfWidth, halfWidth + 1), np.arange(-halfWidth, halfWidth + 1))
    Z_crop = np.arange(-halfWidth, halfWidth + 1)
    
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]
    
    num_atom_type = np.zeros(atom_type_num)
    for j in range(1, atom_type_num + 1):
        num_atom_type[j - 1] = np.count_nonzero(atom == j)
    
    t = (step_sz / max_fa ** 2 / num_pj / N1 ** 2) / num_atom_type
    
    for iter in range(iterations):
        Grad = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        grad_h = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        grad_b = np.zeros((N1, N2, num_pj, 3), dtype=np.float32)
        
        for i in range(num_pj):
            for j in range(1, atom_type_num + 1):
                atom_type_j = (atom == j)
                
                X_cen = np.reshape(X_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])
                Y_cen = np.reshape(Y_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])
                Z_cen = np.reshape(Z_rot[i, atom_type_j], [1, 1, num_atom_type[j - 1]])
                
                X_round = np.round(X_cen)
                Y_round = np.round(Y_cen)
                Z_round = np.round(Z_cen)
                
                l2_xy = np.add.outer(X_crop, X_round - X_cen) ** 2 + np.add.outer(Y_crop, Y_round - Y_cen) ** 2
                l2_z = np.add.outer(Z_crop, Z_round - Z_cen) ** 2
                
                pj_j = np.exp(-l2_xy * b[j - 1]) * np.sum(np.exp(-l2_z * b[j - 1]))
                pj_j_h = h[j - 1] * pj_j
                bj_j = pj_j_h * l2_xy + h[j - 1] * np.exp(-l2_xy * b[j - 1]) * np.sum(l2_z * np.exp(-l2_z * b[j - 1]))
                
                for k in range(num_atom_type[j - 1]):
                    indx = X_round[0, 0, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N1 + 1) / 2))
                    indy = Y_round[0, 0, k] + np.arange(-halfWidth, halfWidth + 1) + int(round((N2 + 1) / 2))
                    
                    Grad[indx, indy, i, j - 1] += pj_j_h[:, :, k]
                    grad_h[indx, indy, i, j - 1] += pj_j[:, :, k]
                    grad_b[indx, indy, i, j - 1] += bj_j[:, :, k]
        
        Projs = np.sum(Grad, axis=3)
        for i in range(num_pj):
            Projs[:, :, i] = my_ifft(my_fft(Projs[:, :, i]) * fixedfa)
        
        k = np.sum(Projs * ydata) / np.sum(Projs ** 2)
        Projs = Projs * k
        grad_b = grad_b * k
        
        res = Projs - ydata
        errR.append(np.sum(np.abs(Projs - ydata)) / np.sum(np.abs(ydata)))
        print(f'{iter + 1}.f = {errR[-1]:.5f}')
        
        for j in range(1, atom_type_num + 1):
            b[j - 1] = b[j - 1] + (t[j - 1] / N_s ** 6) * np.sum(np.sum(np.sum(res * grad_b[:, :, :, j - 1])))
            h[j - 1] = h[j - 1] - (t[j - 1]) * np.sum(np.sum(np.sum(res * grad_h[:, :, :, j - 1])))
        
        h = np.maximum(h, 0)
        h = h / h[0]
        b = np.maximum(b, 0)
    
    param = np.concatenate((k * h, (np.pi * Res) ** 2 / b))
    return Projs, param, errR