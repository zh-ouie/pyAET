import os

import numpy as np
from scipy.fft import fftshift, fftn, ifftshift


def _fft_threads():
    for name in ("PYAET_FFT_THREADS", "PYAET_NUM_THREADS", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


def _fftn_scipy(x):
    return fftn(x, workers=_fft_threads())


def _fftn_pyfftw(x):
    try:
        import pyfftw
        from pyfftw.interfaces.numpy_fft import fftn as fftn_fftw
    except Exception:
        return _fftn_scipy(x)

    pyfftw.interfaces.cache.enable()
    planner = os.environ.get("PYAET_FFTW_PLANNER", "FFTW_ESTIMATE")
    return fftn_fftw(x, threads=_fft_threads(), planner_effort=planner)


def _fftn(x):
    if os.environ.get("PYAET_USE_PYFFTW", "0") == "1":
        return _fftn_pyfftw(x)
    return _fftn_scipy(x)

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
