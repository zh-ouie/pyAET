import numpy as np
from scipy.interpolate import interpn

def get_box_intensity(rec, curr_model, halfSize, O_Ratio, SPHyn, interp_type):
    """
    Calculate the intensity values for points within a box around atomic positions.

    Parameters:
    - rec (numpy.ndarray): Reconstruction volume.
    - curr_model (numpy.ndarray): Current atomic positions as [X, Y, Z].
    - halfSize (float): Half-size of the box.
    - O_Ratio (float): Oversampling ratio.
    - SPHyn (bool): Flag to use spherical region.
    - interp_type (str): Interpolation type ('linear' or other).

    Returns:
    - points (numpy.ndarray): Intensity values for points within the box.
    """
    global points
    Num_atom = curr_model.shape[1]

    # Obtain global intensity histogram
    ds = 1 / O_Ratio
    
    XX=np.linspace(-halfSize, halfSize + int(ds),2*halfSize + int(ds))
    YY=np.linspace(-halfSize, halfSize + int(ds),2*halfSize + int(ds))
    ZZ=np.linspace(-halfSize, halfSize + int(ds),2*halfSize + int(ds))

    if SPHyn:
        useInd = (XX**2 + YY**2 + ZZ**2) <= (halfSize + 0.5 * ds)**2
    else:
        useInd = np.ones_like(XX, dtype=bool)
    indx=np.where(useInd)[0]
    rec = rec[indx[0]-1:indx[-1],indx[0]-1:indx[-1],indx[0]-1:indx[-1]]

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

    points1=(XX, YY, ZZ)
    points2=np.array([x_set,y_set, z_set]).T
    points = interpn(points1,rec,points2, method = interp_type,bounds_error=False,fill_value=0)

    return points.T