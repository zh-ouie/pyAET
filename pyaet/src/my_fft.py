"""Compatibility wrapper for the shared centred FFT."""
from pyaet.fft_backend import centered_fftn

def my_fft(img):
    return centered_fftn(img)
