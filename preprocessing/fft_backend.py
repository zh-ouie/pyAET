"""Compatibility imports for the shared FFT implementation in PyAET.

The numerical implementation belongs to pyaet.fft_backend. This module keeps
existing preprocessing imports working without maintaining a second copy.
"""
from pyaet.fft_backend import *  # noqa: F401,F403
from pyaet.fft_backend import __all__
