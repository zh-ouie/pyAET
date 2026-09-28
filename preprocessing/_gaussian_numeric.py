"""Legacy double-precision dark-current Gaussian arithmetic.

The exponential below is adapted from https://netlib.org/fdlibm/e_exp.c,
restricted to the Gaussian kernel's [-2, 0] argument range.

Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
"""
import ctypes
from functools import lru_cache
import math
import struct
import numpy as np


def _kernel_exp(x):
    if not -2. <= x <= 0.:
        raise ValueError('Gaussian exponential argument outside [-2, 0]')
    high = struct.unpack('>Q', struct.pack('>d', x))[0] >> 32
    high &= 0x7fffffff
    k = 0
    if high > 0x3fd62e42:
        if high < 0x3ff0a2b2:
            hi = x + 6.93147180369123816490e-1
            lo = -1.90821492927058770002e-10
            k = -1
        else:
            k = int(1.44269504088896338700*x - .5)
            hi = x-k*6.93147180369123816490e-1
            lo = k*1.90821492927058770002e-10
        x = hi-lo
    elif high < 0x3e300000:
        return 1.+x
    t = x*x
    c = x-t*(1.66666666666666019037e-1+t*(-2.77777777770155933842e-3+
        t*(6.61375632143793436117e-5+t*(-1.65339022054652515390e-6+
        t*4.13813679705723846039e-8))))
    if k == 0:
        return 1.-((x*c)/(c-2.)-x)
    return math.ldexp(1.-((lo-(x*c)/(2.-c))-hi), k)


@lru_cache(maxsize=1)
def dark_kernel():
    from ._noise_numeric import ordered_sum
    h = np.array([_kernel_exp(-float(i*i)/36./2.) for i in range(-12,13)])
    h /= ordered_sum(h)
    h.setflags(write=False)
    return h


_libm = ctypes.CDLL(None)
_fma = _libm.fma
_fma.argtypes = [ctypes.c_double]*3
_fma.restype = ctypes.c_double


def _convolve_columns(image, h, horizontal):
    height = image.shape[0]-len(h)+1
    result = np.zeros((height, image.shape[1]), dtype=np.float64)
    tail = height % 8
    for i in range(height):
        for j in range(image.shape[1]):
            total = 0.
            for k in range(len(h)-1, -1, -1):
                separate = i >= height-tail and (
                    not horizontal or k >= len(h)-(8-tail))
                if separate:
                    total = total + image[i+k,j]*h[k]
                else:
                    total = _fma(image[i+k,j], h[k], total)
            result[i,j] = total
    return result


try:
    from numba import njit
except ImportError:
    pass
else:
    # The explicit libm call fuses only the validated SIMD-path operations.
    # The scalar tail must not inherit contraction from fastmath flags.
    _convolve_columns = njit(_convolve_columns)


def dark_gaussian_filter(frames):
    """Original imgaussfilt(..., 6), radius 12, replicate padding.

    Accept a double 2-D image or a stack of images in the last dimension.
    Numba accelerates the loops when installed; the Python fallback retains
    the same arithmetic. These rules were checked on MATLAB R2025b/arm64.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        return dark_gaussian_filter(frames[:,:,None])[:,:,0]
    if frames.ndim != 3:
        raise ValueError('Expected a 2-D image or a 3-D frame stack')
    h = dark_kernel()
    result = np.empty_like(frames)
    for k in range(frames.shape[2]):
        padded = np.pad(frames[:,:,k], ((12,12),(12,12)), mode='edge')
        intermediate = _convolve_columns(padded, h, False)
        result[:,:,k] = _convolve_columns(intermediate.T, h, True).T
    return result
