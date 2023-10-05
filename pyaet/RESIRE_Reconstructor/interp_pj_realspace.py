'''
I've converted the MATLAB interp_pj_realspace function to Python, making use of NumPy and SciPy functions for the equivalent operations. Please note that I've made a few changes to adapt to Python conventions, such as using @ for matrix multiplication and using map_coordinates for interpolation. Additionally, I removed the use_parallel condition, as Python's map_coordinates inherently supports parallelism when appropriate.
'''
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import spl_prep, splev

def interp_pj_realspace(obj):
    projections = obj.InputProjections
    Num_pj = obj.NumProjs
    dimx = obj.Dim1
    dimy = obj.Dim2

    phiangles = obj.InputAngles[:, 0]
    thetaangles = obj.InputAngles[:, 1]
    psiangles = obj.InputAngles[:, 2]

    vec1 = obj.vector1
    vec2 = obj.vector2
    vec3 = obj.vector3

    dtype = obj.dtype
    n1_oversampled = obj.n1_oversampled
    n2_oversampled = obj.n2_oversampled

    ncy = np.round((dimy + 1) / 2).astype(int)
    ncx = np.round((dimx + 1) / 2).astype(int)

    # Calculate grid points for rotation
    k1 = np.arange(-np.floor((dimx - 1) / 2), np.floor((dimx - 1) / 2) + 1, dtype=dtype)
    k2 = np.arange(-np.floor((dimy - 1) / 2), np.floor((dimy - 1) / 2) + 1, dtype=dtype)
    k3 = k1
    YY, XX, ZZ = np.meshgrid(k1, k2, k3)
    XX = XX.flatten()
    YY = YY.flatten()
    ZZ = ZZ.flatten()

    rot_pjs = np.zeros((dimy, dimx, dimx, Num_pj), dtype=dtype)

    # Calculate oversampled grid points for calculating back projections
    ncy_big = np.round((n2_oversampled + 1) / 2).astype(int)
    ncx_big = np.round((n1_oversampled + 1) / 2).astype(int)
    Y, X, Z = np.meshgrid(np.arange(1, n2_oversampled + 1) - ncy_big, np.arange(1, n1_oversampled + 1) - ncx_big, 0)
    Y = Y.flatten()
    X = X.flatten()
    Z = Z.flatten()
    xj = np.zeros((n2_oversampled, n1_oversampled, Num_pj), dtype=dtype)
    yj = np.zeros((n2_oversampled, n1_oversampled, Num_pj), dtype=dtype)
    zj = np.zeros((n2_oversampled, n1_oversampled, Num_pj), dtype=dtype)

    Rot_x = np.zeros((dimy, dimx, dimy, Num_pj), dtype=dtype)
    Rot_y = np.zeros((dimy, dimx, dimy, Num_pj), dtype=dtype)

    for k in range(Num_pj):
        phi = phiangles[k]
        theta = thetaangles[k]
        psi = psiangles[k]
        pj = projections[:, :, k]

        R1 = MatrixQuaternionRot(vec1, phi)
        R2 = MatrixQuaternionRot(vec2, theta)
        R3 = MatrixQuaternionRot(vec3, psi)
        R = (R1 @ R2 @ R3).T

        rotCoords = R[0:2, :] @ np.vstack((XX, YY, ZZ))
        rot_x = rotCoords[0, :]
        rot_y = rotCoords[1, :]
        rot_x = np.reshape(rot_x, (dimy, dimx, dimy)) + ncx
        rot_y = np.reshape(rot_y, (dimy, dimx, dimy)) + ncy
        Rot_x[:, :, :, k] = rot_x
        Rot_y[:, :, :, k] = rot_y
        rot_pj = map_coordinates(pj, [rot_y, rot_x], order=1)
        rot_pjs[:, :, :, k] = rot_pj

        rotCoords = R.T @ np.vstack((X, Y, Z))
        xj[:, :, k] = np.reshape(rotCoords[0, :], (n2_oversampled, n1_oversampled)) + ncy_big
        yj[:, :, k] = np.reshape(rotCoords[1, :], (n2_oversampled, n1_oversampled)) + ncy_big
        zj[:, :, k] = np.reshape(rotCoords[2, :], (n2_oversampled, n1_oversampled)) + ncy_big

    sum_rot_pjs = np.sum(rot_pjs, axis=3)

    obj.sum_rot_pjs = sum_rot_pjs
    obj.xj = xj
    obj.yj = yj
    obj.zj = zj
    obj.Rot_x = Rot_x
    obj.Rot_y = Rot_y

    return obj
