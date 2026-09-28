"""Compatibility wrapper for the shared FFT interface."""
from pyaet.fft_backend import legacy_forward_fftn

def my_fft(img):
    return legacy_forward_fftn(img)
