import numpy as np

def my_ifftn(X):
    """
    Calculate the inverse N-dimensional Fast Fourier Transform (IFFT) of the input array.

    Parameters:
    X (ndarray): The input N-dimensional array.

    Returns:
    ndarray: The result of the inverse N-dimensional Fourier Transform.

    Example:
    ifft_result = my_ifftn(fft_result)
    """
    Y = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(X)))
    return Y
