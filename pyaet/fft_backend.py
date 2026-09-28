"""Shared FFT conventions for preprocessing, RESIRE, and GPU refinement.

The public functions in this module keep the MATLAB/RESIRE centering rule in
one place::

    fftshift(fftn(ifftshift(x)))

CPU arrays use FFTW exclusively, in single or double precision.
Missing FFTW libraries raise an installation error; there is no fallback.
PyTorch tensors stay on their current device and use ``torch.fft``.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
from threading import RLock
from typing import Iterable

import numpy as np
from scipy import fft as scipy_fft

try:  # PyTorch is optional for the CPU-only release path.
    import torch
except Exception:  # pragma: no cover - exercised in CPU-only installations.
    torch = None


_fftw_lock = RLock()


def _normalise_axes(ndim: int, axes: Iterable[int] | None) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    axes = tuple(axes)
    if any(not isinstance(axis, (int, np.integer)) or not -ndim <= axis < ndim for axis in axes):
        raise ValueError("FFT axis out of range or not an integer")
    result = tuple(int(axis) % ndim for axis in axes)
    if len(set(result)) != len(result):
        raise ValueError("FFT axes must be unique")
    return result


def is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


@lru_cache(maxsize=8)
def _fftw_library(single: bool):
    """Load FFTW explicitly; never silently switch numerical engines."""
    from ctypes.util import find_library
    variable = "AET_FFTW_SINGLE_LIBRARY" if single else "AET_FFTW_DOUBLE_LIBRARY"
    path = os.environ.get(variable)
    if single and not path:
        path = os.environ.get("AET_REGISTRATION_FFTW_LIBRARY")
    name = "fftw3f" if single else "fftw3"
    path = path or find_library(name)
    if not path:
        raise RuntimeError(f"FFTW is required for CPU FFTs. Install FFTW or set {variable}.")
    lib = ct.CDLL(path)
    prefix = "fftwf" if single else "fftw"
    plan = getattr(lib, prefix + "_plan_many_dft")
    plan.restype = ct.c_void_p
    plan.argtypes = [ct.c_int, ct.POINTER(ct.c_int), ct.c_int,
                     ct.c_void_p, ct.POINTER(ct.c_int), ct.c_int, ct.c_int,
                     ct.c_void_p, ct.POINTER(ct.c_int), ct.c_int, ct.c_int,
                     ct.c_int, ct.c_uint]
    execute = getattr(lib, prefix + "_execute")
    execute.argtypes = [ct.c_void_p]
    execute.restype = None
    destroy = getattr(lib, prefix + "_destroy_plan")
    destroy.argtypes = [ct.c_void_p]
    destroy.restype = None
    return plan, execute, destroy


def _fftw_axis_transform(array, axis, inverse):
    single = array.dtype in (np.dtype("float32"), np.dtype("complex64"))
    dtype = np.complex64 if single else np.complex128
    work = np.array(np.moveaxis(array, axis, 0), dtype=dtype, order="F")
    if not work.size:
        raise ValueError("FFT input dimensions must be nonempty")
    output = np.empty_like(work, order="F")
    length = ct.c_int(work.shape[0])
    plan_fn, execute, destroy = _fftw_library(single)
    # FFTW planning/destruction share global state. Execution uses private
    # arrays and plans and can run concurrently across independent angles.
    with _fftw_lock:
        plan = plan_fn(1, ct.byref(length), work.size // length.value,
                       work.ctypes.data, None, 1, length.value,
                       output.ctypes.data, None, 1, length.value,
                       1 if inverse else -1, 64)  # FFTW_ESTIMATE
    if not plan:
        raise RuntimeError("FFTW could not plan the transform")
    try:
        execute(plan)
    finally:
        with _fftw_lock:
            destroy(plan)
    if inverse:
        output /= np.float32(length.value) if single else float(length.value)
    return np.moveaxis(output, 0, axis)


def ordered_fft2(array, *, inverse=False):
    """Two-dimensional FFTW transform, axis 0 followed by axis 1."""
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError("ordered_fft2 expects a two-dimensional array")
    return fftn(value, inverse=inverse)


def fftn(array, *, axes=None, inverse=False):
    """Uncentred FFT: FFTW for arrays, Torch for tensors.

    Axes are evaluated in the supplied order (ascending by default).
    Forward transforms are unscaled; each inverse axis divides by its length.
    CPU float32/complex64 precision is preserved. Torch single inputs retain
    the reconstruction policy: double-precision transform rounded to complex64.
    """
    dims = _normalise_axes(array.ndim, axes) if is_torch_tensor(array) else None
    if is_torch_tensor(array):
        function = torch.fft.ifftn if inverse else torch.fft.fftn
        if array.dtype in (torch.float32, torch.complex64):
            dtype = torch.float64 if array.dtype == torch.float32 else torch.complex128
            return function(array.to(dtype), dim=dims, norm="backward").to(torch.complex64)
        return function(array, dim=dims, norm="backward")
    value = np.asarray(array)
    dims = _normalise_axes(value.ndim, axes)
    if value.dtype not in (np.dtype("float32"), np.dtype("complex64"),
                           np.dtype("float64"), np.dtype("complex128")):
        value = value.astype(np.complex128 if np.iscomplexobj(value) else np.float64)
    for axis in dims:
        value = _fftw_axis_transform(value, axis, inverse)
    return value


def fft_convolve_full(a, b):
    """Full linear convolution using the shared CPU FFTW backend."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != b.ndim:
        raise ValueError("Convolution inputs must have equal dimensionality")
    shape = tuple(x + y - 1 for x, y in zip(a.shape, b.shape))
    # Length selection does not perform a transform.
    sizes = tuple(scipy_fft.next_fast_len(n, real=True) for n in shape)
    dtype = np.result_type(a.dtype, b.dtype, np.float64)
    ap, bp = np.zeros(sizes, dtype=dtype), np.zeros(sizes, dtype=dtype)
    ap[tuple(slice(0,n) for n in a.shape)] = a
    bp[tuple(slice(0,n) for n in b.shape)] = b
    result = fftn(fftn(ap) * fftn(bp), inverse=True)
    result = result[tuple(slice(0,n) for n in shape)]
    return result if np.iscomplexobj(a) or np.iscomplexobj(b) else result.real


def rfftn(array, *, axes=None):
    """Torch half spectrum with the same precision and normalization as fftn."""
    if not is_torch_tensor(array):
        raise TypeError("rfftn is reserved for Torch reconstruction tensors")
    if array.is_complex():
        raise ValueError("rfftn requires real input")
    dims = _normalise_axes(array.ndim, axes)
    if array.dtype == torch.float32:
        return torch.fft.rfftn(array.to(torch.float64), dim=dims, norm="backward").to(torch.complex64)
    return torch.fft.rfftn(array, dim=dims, norm="backward")


def fftshift(array, *, axes: Iterable[int] | None = None):
    if is_torch_tensor(array):
        dims = _normalise_axes(array.ndim, axes)
        return torch.fft.fftshift(array, dim=dims)
    return scipy_fft.fftshift(np.asarray(array), axes=axes)


def ifftshift(array, *, axes: Iterable[int] | None = None):
    if is_torch_tensor(array):
        dims = _normalise_axes(array.ndim, axes)
        return torch.fft.ifftshift(array, dim=dims)
    return scipy_fft.ifftshift(np.asarray(array), axes=axes)


def centered_fftn(array, *, axes: Iterable[int] | None = None):
    """MATLAB/RESIRE centred forward transform."""
    dims = _normalise_axes(array.ndim, axes)
    return fftshift(fftn(ifftshift(array, axes=dims), axes=dims), axes=dims)


def centered_ifftn(array, *, axes: Iterable[int] | None = None):
    """MATLAB/RESIRE centred inverse transform."""
    dims = _normalise_axes(array.ndim, axes)
    return fftshift(
        fftn(ifftshift(array, axes=dims), axes=dims, inverse=True),
        axes=dims,
    )


__all__ = [
    "centered_fftn",
    "centered_ifftn",
    "fftn",
    "fft_convolve_full",
    "fftshift",
    "ifftshift",
    "is_torch_tensor",
    "ordered_fft2",
    "rfftn",
]
