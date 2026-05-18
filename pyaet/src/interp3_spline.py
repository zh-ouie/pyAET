import numpy as np
from scipy.interpolate import CubicSpline


def interp3_spline(data, xx, yy, zz, xxi, yyi, zzi):
    data = np.asarray(data, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    zz = np.asarray(zz, dtype=np.float64)
    xxi = np.asarray(xxi, dtype=np.float64)
    yyi = np.asarray(yyi, dtype=np.float64)
    zzi = np.asarray(zzi, dtype=np.float64)

    cs_x = CubicSpline(xx, data, axis=0, bc_type='not-a-knot', extrapolate=False)
    data_x = cs_x(xxi)
    cs_y = CubicSpline(yy, data_x, axis=1, bc_type='not-a-knot', extrapolate=False)
    data_xy = cs_y(yyi)
    cs_z = CubicSpline(zz, data_xy, axis=2, bc_type='not-a-knot', extrapolate=False)
    data_xyz = cs_z(zzi)
    return np.nan_to_num(data_xyz, nan=0.0)
