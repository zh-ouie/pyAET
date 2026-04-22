import numpy as np

from pyaet.src.interp2_numpy import interp2_bilinear
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.my_round import my_round_num
from pyaet.splinterp_cpp import mex_function2


def interp_pj_realspace(obj):
    projections = np.asarray(obj.InputProjections, dtype=np.float64, order="F")
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

    ncy = my_round_num((dimy + 1) / 2)
    ncx = my_round_num((dimx + 1) / 2)

    k1 = np.arange(-np.ceil((dimx - 1) / 2), np.floor((dimx - 1) / 2) + 1, dtype=np.float64)
    k2 = np.arange(-np.ceil((dimy - 1) / 2), np.floor((dimy - 1) / 2) + 1, dtype=np.float64)
    k3 = k1
    YY, XX, ZZ = np.meshgrid(k1, k2, k3)
    XX = XX.flatten(order="F")
    YY = YY.flatten(order="F")
    ZZ = ZZ.flatten(order="F")

    rot_pjs = np.zeros((dimy, dimx, dimx, num_pj), dtype=dtype, order="F")

    ncy_big = my_round_num((n2_oversampled + 1) / 2)
    ncx_big = my_round_num((n1_oversampled + 1) / 2)
    Y, X, Z = np.meshgrid(
        np.arange(1, n2_oversampled + 1) - ncy_big,
        np.arange(1, n1_oversampled + 1) - ncx_big,
        0,
    )
    Y = Y.flatten(order="F")
    X = X.flatten(order="F")
    Z = Z.flatten(order="F")
    xj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")
    yj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")
    zj = np.zeros((n2_oversampled, n1_oversampled, num_pj), dtype=np.float64, order="F")

    Rot_x = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype, order="F")
    Rot_y = np.zeros((dimy, dimx, dimy, num_pj), dtype=dtype, order="F")

    for k in range(num_pj):
        phi = phi_angles[k]
        theta = theta_angles[k]
        psi = psi_angles[k]
        pj = np.asfortranarray(projections[:, :, k], dtype=np.float64)

        R1 = matrix_quaternion_rot(vec1, phi)
        R2 = matrix_quaternion_rot(vec2, theta)
        R3 = matrix_quaternion_rot(vec3, psi)
        R = (R1 @ R2 @ R3).T

        rot_coords = R[0:2, :] @ np.vstack((XX, YY, ZZ))
        rot_x64 = np.reshape(rot_coords[0, :], (dimy, dimx, dimy), order="F") + ncx
        rot_y64 = np.reshape(rot_coords[1, :], (dimy, dimx, dimy), order="F") + ncy
        Rot_x[:, :, :, k] = rot_x64.astype(dtype, copy=False)
        Rot_y[:, :, :, k] = rot_y64.astype(dtype, copy=False)

        try:
            rot_pj = mex_function2(
                pj,
                np.ascontiguousarray(rot_x64),
                np.ascontiguousarray(rot_y64),
            )
        except Exception:
            rot_pj = interp2_bilinear(pj, rot_x64, rot_y64, origin_offset=1)
        rot_pjs[:, :, :, k] = np.asarray(rot_pj, dtype=dtype, order="F")

        rot_coords = R.T @ np.vstack((X, Y, Z))
        xj[:, :, k] = np.reshape(rot_coords[0, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big
        yj[:, :, k] = np.reshape(rot_coords[1, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big
        zj[:, :, k] = np.reshape(rot_coords[2, :], (n2_oversampled, n1_oversampled), order="F") + ncy_big

    obj.sum_rot_pjs = np.sum(rot_pjs, axis=3, dtype=rot_pjs.dtype)
    obj.xj = xj
    obj.yj = yj
    obj.zj = zj
    obj.Rot_x = Rot_x
    obj.Rot_y = Rot_y
    return obj
