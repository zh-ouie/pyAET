import numpy as np
from scipy.fft import fftshift, fftn, ifftshift

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fftn as fftn_fftw

    pyfftw.interfaces.cache.enable()
    def _fftn(x):
        return fftn_fftw(x, threads=1, planner_effort="FFTW_MEASURE")
except Exception:
    _fftn = fftn

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
    img = np.asfortranarray(img)
    if not np.iscomplexobj(img) and not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float64, copy=False)
    kout = fftshift(_fftn(ifftshift(img)))
    # img_shifted = ifftshift(img)
    # kout = fftshift(fftn(img_shifted))
    return kout
