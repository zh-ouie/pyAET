import numpy as np
from pyaet.src.fatom_vector import fatom_vector

def make_fixed_fa_man(sizeX, Res, Z_arr):
    """
    Calculate a fixed_fa (fixed form factor) for given parameters.

    Parameters:
    sizeX (tuple): Size of the final volume as a tuple (sizeX, sizeY, sizeZ).
    Res (float): Resolution parameter.
    Z_arr (list): List of atomic numbers.

    Returns:
    ndarray: The fixed_fa array.

    Example:
    sizeX = (256, 256, 256)
    Res = 1.0
    Z_arr = [6, 8, 14]
    fixed_fa = make_fixed_fa_man(sizeX, Res, Z_arr)
    """
    finalvol_summed = np.zeros(sizeX, dtype=float)

    kx = np.arange(1, finalvol_summed.shape[0]+1)
    ky = np.arange(1, finalvol_summed.shape[1]+1)

    mult_f_x = 1 / (len(kx) * Res)
    mult_f_y = 1 / (len(ky) * Res)

    cent_pos = np.round((np.array(finalvol_summed.shape) + 1) / 2).astype(int)
    KYY, KXX = np.meshgrid((kx - cent_pos[0]) * mult_f_x, (ky - cent_pos[1]) * mult_f_y)
    q2 = KXX**2 + KYY**2

    fixed_fa_err = np.zeros((len(Z_arr), q2.size), dtype=float)
    for i in range(len(Z_arr)):
        fixed_fa_err[i, :] = fatom_vector(np.sqrt(q2), Z_arr[i])

    fixed_fa = np.mean(fixed_fa_err, axis=0)
    return fixed_fa
