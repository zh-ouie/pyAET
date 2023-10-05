import numpy as np
from scipy.fft import fftshift, fftn, ifftshift

def my_fft(img):
    """
    Helper function to calculate the forward FFT (Fast Fourier Transform).

    Args:
        img (ndarray): The input image or data in the spatial domain.

    Returns:
        ndarray: The Fourier-transformed data in the frequency domain with the center shifted.

    Notes:
        This function computes the forward FFT and shifts the zero frequency component
        to the center of the output.
    """
    kout = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    # img_shifted = ifftshift(img)
    # kout = fftshift(fftn(img_shifted))
    return kout
