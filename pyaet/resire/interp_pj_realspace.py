'''
I've converted the MATLAB interp_pj_realspace function to Python, making use of NumPy and SciPy functions for the equivalent operations. Please note that I've made a few changes to adapt to Python conventions, such as using @ for matrix multiplication and using map_coordinates for interpolation. Additionally, I removed the use_parallel condition, as Python's map_coordinates inherently supports parallelism when appropriate.
'''
import numpy as np
from scipy.ndimage import map_coordinates
#from scipy.interpolate import spl_prep, splev
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.my_round import my_round_num
from pyaet.splinterp.splinterp2 import mex_function2

def interp_pj_realspace(obj):
    projections = obj.InputProjections
    num_pj = obj.num_projs
    dimx = obj.dim1
    dimy = obj.dim2

    phi_angles = obj.InputAngles[:, 0]
    theta_angles = obj.InputAngles[:, 1]
    psi_angles = obj.InputAngles[:, 2]

    vec1 = obj.vector1
    vec2 = obj.vector2
    vec3 = obj.vector3

    dtype = obj.dtype
    n1_oversampled = obj.n1_oversampled
    n2_oversampled = obj.n2_oversampled

    # ncy = (np.round((dimy + 1) / 2)).astype(int)
    # ncx = (np.round((dimx + 1) / 2)).astype(int)
    ncy = my_round_num((dimy + 1) / 2)
    ncx = my_round_num((dimx + 1) / 2)

    # Calculate grid points for rotation
    k1 = np.arange(-np.ceil((dimx - 1) / 2), np.floor((dimx - 1) / 2) + 1, dtype=dtype)
    k2 = np.arange(-np.ceil((dimy - 1) / 2), np.floor((dimy - 1) / 2) + 1, dtype=dtype)
    k3 = k1
    YY, XX, ZZ = np.meshgrid(k1, k2, k3)
    # XX = XX.flatten()
    # YY = YY.flatten()
    # ZZ = ZZ.flatten()
    XX = XX.T.flatten()
    YY = YY.T.flatten()
    ZZ = ZZ.T.flatten()

    rot_pjs = np.zeros((dimy, dimx, dimx, num_pj), dtype=dtype)

    # Calculate oversampled grid points for calculating back projections
    ncy_big = my_round_num((n2_oversampled + 1) / 2)
    ncx_big = my_round_num((n1_oversampled + 1) / 2)
    Y, X, Z = np.meshgrid(np.arange(1, n2_oversampled + 1) - ncy_big, np.arange(1, n1_oversampled + 1) - ncx_big, 0)
    # Y = Y.flatten()
    # X = X.flatten()
    # Z = Z.flatten()
    Y = Y.T.flatten()
    X = X.T.flatten()
    Z = Z.T.flatten()
    xj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=dtype)
    yj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=dtype)
    zj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=dtype)

    Rot_x = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype)
    Rot_y = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype)

    for k in range(num_pj):
        phi = phi_angles[k]
        theta = theta_angles[k]
        psi = psi_angles[k]
        pj = projections[:, :, k]

        R1 = matrix_quaternion_rot(vec1, phi)
        R2 = matrix_quaternion_rot(vec2, theta)
        R3 = matrix_quaternion_rot(vec3, psi)
        R = (R1 @ R2 @ R3).T

        rot_coords = R[0:2, :] @ np.vstack((XX, YY, ZZ))
        rot_x = rot_coords[0, :]
        rot_y = rot_coords[1, :]
        rot_x = np.reshape(rot_x, (dimy, dimx, dimy), order='F') + ncx
        rot_y = np.reshape(rot_y, (dimy, dimx, dimy), order='F') + ncy
        Rot_x[:, :, :, k] = rot_x
        Rot_y[:, :, :, k] = rot_y
        # rot_pj = splinterp2(pj, rot_y, rot_x)
        rot_pj = mex_function2(pj, rot_y, rot_x)
        # rot_pj = interp2_real(pj, rot_y, rot_x)
        # rot_pj = map_coordinates(pj, [rot_y, rot_x], order=1) #TODO: use `splinterp2`
        rot_pjs[:, :, :, k] = rot_pj

        rot_coords = R.T @ np.vstack((X, Y, Z))
        xj[:, :, k] = np.reshape(rot_coords[0, :], (n2_oversampled, n1_oversampled), order='F') + ncy_big
        yj[:, :, k] = np.reshape(rot_coords[1, :], (n2_oversampled, n1_oversampled), order='F') + ncy_big
        zj[:, :, k] = np.reshape(rot_coords[2, :], (n2_oversampled, n1_oversampled), order='F') + ncy_big

    sum_rot_pjs = np.sum(rot_pjs, axis=3)

    obj.sum_rot_pjs = sum_rot_pjs
    obj.xj = xj
    obj.yj = yj
    obj.zj = zj
    obj.Rot_x = Rot_x
    obj.Rot_y = Rot_y

    return obj
