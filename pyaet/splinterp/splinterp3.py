import numpy as np


def interp3(data, x, y, z, origin_offset):
    """
    Perform 3D linear interpolation.

    Args:
        data (numpy.ndarray): 3D array containing data values.
        x (numpy.ndarray): 2D array of target x-coordinates.
        y (numpy.ndarray): 2D array of target y-coordinates.
        z (numpy.ndarray): 2D array of target z-coordinates.
        origin_offset (int): Offset for the origin. Defaults to 1.

    Returns:
        None (results are stored in the 'result_r' array).
    """
    # data = datar.real
    sh = x.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    nrows = data.shape[0]
    ncols = data.shape[1]
    nlayers = data.shape[2]
    N_max = max(len(x), len(y), len(z))
    N = min(len(x), len(y), len(z))
    result_r = np.zeros_like(x)

    for i in range(N):
        x_1 = int(np.floor(x[i]) - origin_offset)
        y_1 = int(np.floor(y[i]) - origin_offset)
        z_1 = int(np.floor(z[i]) - origin_offset)

        if (x[i] - origin_offset) == (nrows - 1):
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_1 -= 1
        if (z[i] - origin_offset) == (nlayers - 1):
            z_1 -= 1

        if (x_1 < 0) or (x_1 + 1 > (nrows - 1)) or (y_1 < 0) or (y_1 + 1 > (ncols - 1)) or (z_1 < 0) or (z_1 + 1 > (nlayers - 1)):
            result_r[i] = 0
        else:
            layer_size = ncols * nrows

            f_11_1 = data[z_1, x_1, y_1]
            f_12_1 = data[z_1, x_1, y_1 + 1]
            f_21_1 = data[z_1, x_1 + 1, y_1]
            f_22_1 = data[z_1, x_1 + 1, y_1 + 1]

            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_1 + 1 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a_1 = f_11_1 * w_x1 + f_21_1 * w_x2
            b_1 = f_12_1 * w_x1 + f_22_1 * w_x2
            F_1 = a_1 * w_y1 + b_1 * w_y2

            f_11_2 = data[z_1 + 1, x_1, y_1]
            f_12_2 = data[z_1 + 1, x_1, y_1 + 1]
            f_21_2 = data[z_1 + 1, x_1 + 1, y_1]
            f_22_2 = data[z_1 + 1, x_1 + 1, y_1 + 1]

            a_2 = f_11_2 * w_x1 + f_21_2 * w_x2
            b_2 = f_12_2 * w_x1 + f_22_2 * w_x2
            F_2 = a_2 * w_y1 + b_2 * w_y2

            w_z1 = z_1 + 1 - (z[i] - origin_offset)
            w_z2 = (z[i] - origin_offset) - z_1

            result_r[i] = F_1 * w_z1 + F_2 * w_z2
    result_r = result_r.reshape(sh)

    return result_r


def mex_function3(data, x, y, z, origin_offset = 1):
    if np.isreal(data).all():
        result = interp3(data, x, y, z, origin_offset)
    else:
        result = interp3(data.real, x, y, z, origin_offset) + interp3(data.imag, x, y, z, origin_offset)*1j

    return result
