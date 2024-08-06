import numpy as np
from scipy.interpolate import interpn
from pyaet.splinterp_v5.pyyyyaet.splinterp3 import mexFunction3

def get_box_intensity(rec, curr_model, halfSize, O_Ratio, SPHyn, interp_type):
    Num_atom = curr_model.shape[1]

    # obtain global intensity histogram
    ds = 1 / O_Ratio

    XX, YY, ZZ = np.meshgrid(np.arange(-halfSize, halfSize + ds, ds),
                            np.arange(-halfSize, halfSize + ds, ds),
                            np.arange(-halfSize, halfSize + ds, ds))
    XX=np.transpose(XX, (1,0,2))
    YY=np.transpose(YY, (1,0,2))
    ZZ=np.transpose(ZZ, (1,0,2))

    if SPHyn:
        useInd = np.where(((XX**2 + YY**2 + ZZ**2) <= (halfSize + 0.5*ds)**2).flatten())[0]
    else:
        useInd = np.arange(len(XX))

    # generate points coordinates
    XX_use = XX.flatten(order='F')[useInd]
    YY_use = YY.flatten(order='F')[useInd]
    ZZ_use = ZZ.flatten(order='F')[useInd]

    # XX_use = XX.ravel()[useInd]
    # YY_use = YY.ravel()[useInd]
    # ZZ_use = ZZ.ravel()[useInd]

    y_set = np.zeros((len(YY_use), Num_atom))
    x_set = np.zeros((len(YY_use), Num_atom))
    z_set = np.zeros((len(YY_use), Num_atom))

    # interpolations for points
    for k in range(Num_atom):
        y_set[:, k] = YY_use + curr_model[1, k]
        x_set[:, k] = XX_use + curr_model[0, k]
        z_set[:, k] = ZZ_use + curr_model[2, k]

    points = mexFunction3(rec, x_set, y_set, z_set)

    return points


def get_box_intensity_old(rec, curr_model, halfSize, O_Ratio, SPHyn, interp_type):
    """
    Calculate the intensity values for points within a box around atomic positions.

    Parameters:
    - rec (numpy.ndarray): Reconstruction volume.
    - curr_model (numpy.ndarray): Current atomic positions as [X, Y, Z].
    - half_size (float): Half-size of the box.
    - O_Ratio (float): Oversampling ratio.
    - SPHyn (bool): Flag to use spherical region.
    - interp_type (str): Interpolation type ('linear' or other).

    Returns:
    - points (numpy.ndarray): Intensity values for points within the box.
    """
    # global points
    Num_atom = curr_model.shape[1]

    # Obtain global intensity histogram
    ds = 1 / O_Ratio

    XX = np.linspace(-halfSize, halfSize + int(ds), 2 * halfSize + int(ds))
    YY = np.linspace(-halfSize, halfSize + int(ds), 2 * halfSize + int(ds))
    ZZ = np.linspace(-halfSize, halfSize + int(ds), 2 * halfSize + int(ds))

    if SPHyn:
        useInd = (XX ** 2 + YY ** 2 + ZZ ** 2) <= (halfSize + 0.5 * ds) ** 2
    else:
        useInd = np.ones_like(XX, dtype=bool)
    indx = np.where(useInd)[0]
    rec = rec[indx[0] - 1:indx[-1], indx[0] - 1:indx[-1], indx[0] - 1:indx[-1]]

    # Generate points coordinates
    YY = YY[useInd]
    XX = XX[useInd]
    ZZ = ZZ[useInd]
    y_set = np.zeros((len(YY), Num_atom))
    x_set = np.zeros((len(XX), Num_atom))
    z_set = np.zeros((len(ZZ), Num_atom))

    # Interpolations for points
    for k in range(Num_atom):
        y_set[:, k] = YY + curr_model[1, k]
        x_set[:, k] = XX + curr_model[0, k]
        z_set[:, k] = ZZ + curr_model[2, k]

    points1 = (XX, YY, ZZ)
    points2 = np.array([x_set, y_set, z_set]).T
    points = interpn(points1, rec, points2, method=interp_type, bounds_error=False, fill_value=0)

    return points.T
