import numpy as np


def interp2(data, x, y, origin_offset):
    """
    Perform 2D linear interpolation.

    Args:
        data (numpy.ndarray): 2D array containing data values.
        x (numpy.ndarray): 2D array of target x-coordinates.
        y (numpy.ndarray): 2D array of target y-coordinates.
        origin_offset (int): Offset for the origin. Defaults to 1.

    Returns:
        None (results are stored in the 'result_r' array).
    """
    # data = datar.real
    sh = x.shape
    x = x.flatten()
    y = y.flatten()
    N = min(len(x),len(y))
    N_max = max(len(x),len(y))
    nrows = data.shape[0]
    ncols = data.shape[1]
    result_r = np.zeros_like(x)
    for i in range(N):
        x_1 = int(np.floor(x[i]) - origin_offset)
        y_1 = int(np.floor(y[i]) - origin_offset)

        if (x[i] - origin_offset) == (nrows - 1):
            x_1 -= 1
        if (y[i] - origin_offset) == (ncols - 1):
            y_1 -= 1

        if (x_1 < 0) or (x_1 + 1 > (nrows - 1)) or (y_1 < 0) or (y_1 + 1 > (ncols - 1)):
            result_r[i] = 0
        else:
            f_11 = data[x_1, y_1]
            f_12 = data[x_1, y_1 + 1]
            f_21 = data[x_1 + 1, y_1]
            f_22 = data[x_1 + 1, y_1 + 1]

            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            w_x2 = (x[i] - origin_offset) - x_1
            w_y1 = y_1 + 1 - (y[i] - origin_offset)
            w_y2 = (y[i] - origin_offset) - y_1

            a = f_11 * w_x1 + f_21 * w_x2
            b = f_12 * w_x1 + f_22 * w_x2
            result_r[i] = a * w_y1 + b * w_y2
    result_r = result_r.reshape(sh)
    return result_r

def mex_function2(data, x, y, origin_offset = 1):
    sh = x.shape
    if np.isreal(data).all():
        result = interp2(data, x, y, origin_offset)
    else:
        result = interp2(data.real, x, y, origin_offset) + interp2(data.imag, x, y, origin_offset)*1j
    return result
