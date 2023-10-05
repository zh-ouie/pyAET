import numpy as np

def My_FFTN(X):
    """
    Calculate the N-dimensional Fast Fourier Transform (FFT) of the input array.

    Parameters:
    X (ndarray): The input N-dimensional array.

    Returns:
    ndarray: The result of the N-dimensional Fourier Transform.

    Example:
    >>> fft_result = My_FFTN(input_array)
    """
    Y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(X)))
    return Y
