import numpy as np

def make_fixedfa_man(sizeX, Res, Z_arr):
    """
    Calculate a fixedfa (fixed form factor) for given parameters.

    Parameters:
    sizeX (tuple): Size of the final volume as a tuple (sizeX, sizeY, sizeZ).
    Res (float): Resolution parameter.
    Z_arr (list): List of atomic numbers.

    Returns:
    ndarray: The fixedfa array.

    Example:
    >>> sizeX = (256, 256, 256)
    >>> Res = 1.0
    >>> Z_arr = [6, 8, 14]
    >>> fixedfa = make_fixedfa_man(sizeX, Res, Z_arr)
    """
    finalvol_summed = np.zeros(sizeX)

    kx = np.arange(1, size(finalvol_summed, 0) + 1)
    ky = np.arange(1, size(finalvol_summed, 1) + 1)

    MultF_X = 1 / (len(kx) * Res)
    MultF_Y = 1 / (len(ky) * Res)

    CentPos = np.round((size(finalvol_summed) + 1) / 2)
    KX, KY = np.meshgrid((kx - CentPos[0]) * MultF_X, (ky - CentPos[1]) * MultF_Y)
    q2 = KX**2 + KY**2

    fixedfa_arr = np.zeros((len(Z_arr), q2.size))
    for i in range(len(Z_arr)):
        fixedfa_arr[i, :] = fatom_vector(np.sqrt(q2), Z_arr[i])

    fixedfa = np.mean(fixedfa_arr, axis=0)
    return fixedfa
