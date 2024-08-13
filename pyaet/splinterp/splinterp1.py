import numpy as np


def interp1(data, x, origin_offset):
    """
    Perform 1D linear interpolation.

    Args:
        data (numpy.ndarray): 1D array containing data values.
        x (numpy.ndarray): 1D array of target x-coordinates.
        origin_offset (int): Offset for the origin. Defaults to 1.

    Returns:
        None (results are stored in the 'result_r' array).
    """
    # datar = data.real
    sh = x.shape
    x = x.flatten()
    N = x.shape[0]
    nrows = data.shape[0]
    result_r = np.zeros_like(x)

    for i in range(N):
        x_1 = int(np.floor(x[i]) - origin_offset)
        if (x_1 - origin_offset) == (nrows - 1):
            x_1 = x_1 - 1
        if (x_1 < 0) or (x_1 + 1 > nrows - 1):
            result_r[i] = 0
        else:
            f_1 = data[x_1]
            f_2 = data[x_1 + 1]
            w_x1 = x_1 + 1 - (x[i] - origin_offset)
            result_r[i] = (f_1 - f_2) * w_x1 + f_2
    result_r = result_r.reshape(sh)
    return result_r

def mex_function1(data, x, origin_offset = 1):
    if np.isreal(data).all():
        result = interp1(data, x, origin_offset)
    else:
        result = interp1(data.real, x, origin_offset) + interp1(data.imag, x, origin_offset)*1j

    return result
