"""Compatibility wrapper for the shared FFT interface."""
from pyaet.fft_backend import centered_ifftn

def my_ifft(k):
    return centered_ifftn(k, backend="numpy")
