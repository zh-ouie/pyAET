import numpy as np
import scipy.ndimage as ndi

def strel3d(sesize):
    """
    Create a 3D spherical structuring element for morphological operations.

    Parameters:
    - sesize (int): The desired diameter size of the sphere (any positive integer).

    Returns:
    - se (ndarray): The structuring element as a binary 3D numpy array.

    Example:
    se = strel3d(1)
    se = strel3d(2)
    se = strel3d(5)
    """

    sw = (sesize - 1) // 2
    ses2 = (sesize + 1) // 2  # Use integer division to handle odd diameters
    x, y, z = np.meshgrid(np.arange(-sw, sw + 1), np.arange(-sw, sw + 1), np.arange(-sw, sw + 1))
    m = np.sqrt(x**2 + y**2 + z**2)
    b = (m <= m[ses2, ses2, sesize])
    se = ndi.generate_binary_structure(3, 1)
    se[~b] = 0
    return se
