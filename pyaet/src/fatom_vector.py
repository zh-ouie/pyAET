import numpy as np

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
    a = [fpara[0], fpara[2], fpara[4]]
    b = [fpara[1], fpara[3], fpara[5]]
    c = [fpara[6], fpara[8], fpara[10]]
    d = [fpara[7], fpara[9], fpara[11]]

    num = len(q)
    v = np.zeros(num)

    for hh in range(num):
        # Lorenzians
        suml = np.sum(a / ((q[hh]**2) + b))
        # Gaussians
        sumg = np.sum(c * np.exp(-(q[hh]**2) * d))
        v[hh] = suml + sumg

    return v
