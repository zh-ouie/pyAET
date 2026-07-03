import numpy as np
from pyaet.src.fparameters import fparameters

def fatom_vector(q, Z):
    """
    Calculate the atomic form factor for a given set of wave vectors and atomic number.

    Args:
        q (numpy.ndarray): Array of wave vectors.
        Z (int): Atomic number.

    Returns:
        numpy.ndarray: Array of atomic form factors.
    """
    # Retrieve atomic form factor parameters for the given atomic number Z
    fpara = fparameters(Z)
    a = np.array([fpara[0], fpara[2], fpara[4]])
    b = np.array([fpara[1], fpara[3], fpara[5]])
    c = np.array([fpara[6], fpara[8], fpara[10]])
    d = np.array([fpara[7], fpara[9], fpara[11]])

    q2 = np.ravel(q, order='F') ** 2
    suml = np.sum(a[:, None] / (q2[None, :] + b[:, None]), axis=0)
    sumg = np.sum(c[:, None] * np.exp(-q2[None, :] * d[:, None]), axis=0)
    return suml + sumg
