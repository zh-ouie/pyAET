"""Shared FFT conventions for preprocessing, RESIRE, and GPU refinement.

The public functions in this module keep the MATLAB/RESIRE centering rule in
one place::

    fftshift(fftn(ifftshift(x)))

NumPy/SciPy arrays use an ordered CPU transform.  When
``AET_REGISTRATION_FFTW_LIBRARY`` points to the validated single-precision
FFTW build, two-dimensional float32/complex64 transforms use that library.
PyTorch tensors stay on their current device and use ``torch.fft``.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
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
    result = tuple(int(axis) % ndim for axis in axes)
    if len(set(result)) != len(result):
        raise ValueError("FFT axes must be unique")
    return result


def is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


def _torch_precision_profile(backend):
    """One supported Torch precision policy; CPU selectors remain internal."""
    if backend not in {"auto", "matlab_single", "torch_matlab_single"}:
        raise ValueError("Torch FFT uses the validated matlab_single policy")
    return "matlab_single"


@lru_cache(maxsize=2)
def _fftw_library(path: str):
    lib = ct.CDLL(str(Path(path).resolve(strict=True)))
    lib.fftwf_plan_many_dft.restype = ct.c_void_p
    lib.fftwf_plan_many_dft.argtypes = [
        ct.c_int,
        ct.POINTER(ct.c_int),
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_uint,
    ]
    lib.fftwf_execute.argtypes = [ct.c_void_p]
    lib.fftwf_execute.restype = None
    lib.fftwf_destroy_plan.argtypes = [ct.c_void_p]
    lib.fftwf_destroy_plan.restype = None
    return lib


def _fftw_axis_transform(array: np.ndarray, axis: int, inverse: bool, lib) -> np.ndarray:
    if array.ndim != 2 or axis not in (0, 1):
        raise ValueError("The validated FFTW path supports 2-D axis transforms")
    work = np.array(array if axis == 0 else array.T, dtype=np.complex64, order="F")
    output = np.empty_like(work)
    length = ct.c_int(work.shape[0])
    with _fftw_lock:
        plan = lib.fftwf_plan_many_dft(
            1,
            ct.byref(length),
            work.shape[1],
            work.ctypes.data,
            None,
            1,
            length.value,
            output.ctypes.data,
            None,
            1,
            length.value,
            1 if inverse else -1,
            64,  # FFTW_ESTIMATE
        )
        if not plan:
            raise RuntimeError("FFTW could not plan the transform")
        try:
            lib.fftwf_execute(plan)
        finally:
            lib.fftwf_destroy_plan(plan)
    if inverse:
        output /= np.float32(length.value)
    return output if axis == 0 else output.T


def ordered_fft2(array, *, inverse: bool = False):
    """Transform axis 0 and then axis 1, matching MATLAB dimension order.

    This is the uncentred primitive used by the parity-sensitive frame
    registration code.  Double precision uses SciPy.  Single precision uses
    the validated external FFTW build when configured, otherwise SciPy.
    """
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError("ordered_fft2 expects a two-dimensional array")
    library_path = os.environ.get("AET_REGISTRATION_FFTW_LIBRARY")
    if library_path and value.dtype in (np.dtype("float32"), np.dtype("complex64")):
        lib = _fftw_library(library_path)
        return _fftw_axis_transform(
            _fftw_axis_transform(value, 0, inverse, lib), 1, inverse, lib
        )
    transform = scipy_fft.ifft if inverse else scipy_fft.fft
    return transform(transform(value, axis=0), axis=1)


def fftn(array, *, axes: Iterable[int] | None = None, inverse: bool = False,
         backend: str = "auto"):
    """Run an uncentred N-D transform through the selected backend."""
    if is_torch_tensor(array):
        dims = _normalise_axes(array.ndim, axes)
        function = torch.fft.ifftn if inverse else torch.fft.fftn
        profile = _torch_precision_profile(backend)
        if profile == "matlab_single" and array.dtype in (
            torch.float32,
            torch.complex64,
        ):
            promoted_dtype = torch.float64 if array.dtype == torch.float32 else torch.complex128
            return function(array.to(promoted_dtype), dim=dims).to(torch.complex64)
        return function(array, dim=dims)

    value = np.asarray(array)
    dims = _normalise_axes(value.ndim, axes)
    selected = "matlab" if backend == "auto" else backend.lower()
    if selected in ("matlab", "ordered"):
        if value.ndim == 2 and dims == (0, 1):
            return ordered_fft2(value, inverse=inverse)
        function = scipy_fft.ifft if inverse else scipy_fft.fft
        output = value
        for axis in dims:
            output = function(output, axis=axis)
        return output
    if selected == "numpy":
        function = np.fft.ifftn if inverse else np.fft.fftn
        return function(value, axes=dims)
    raise ValueError(f"Unknown FFT backend: {backend!r}")


def legacy_forward_fftn(array):
    """Preserve the July my_fft backend, threading and dtype conventions."""
    value = np.asfortranarray(array)
    if not np.iscomplexobj(value) and not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64, copy=False)
    threads = 1
    for name in ("PYAET_FFT_THREADS", "PYAET_NUM_THREADS", "OMP_NUM_THREADS"):
        try:
            if os.environ.get(name):
                threads = max(1, int(os.environ[name]))
                break
        except ValueError:
            pass
    shifted = scipy_fft.ifftshift(value)
    if os.environ.get("PYAET_USE_PYFFTW", "0") == "1":
        try:
            import pyfftw
            from pyfftw.interfaces.numpy_fft import fftn as transform
        except Exception:
            result = scipy_fft.fftn(shifted, workers=threads)
        else:
            pyfftw.interfaces.cache.enable()
            result = transform(shifted, threads=threads,
                               planner_effort=os.environ.get("PYAET_FFTW_PLANNER", "FFTW_ESTIMATE"))
    else:
        result = scipy_fft.fftn(shifted, workers=threads)
    return scipy_fft.fftshift(result)


def rfftn(array, *, axes: Iterable[int] | None = None, backend: str = "auto"):
    """Uncentred real FFT with the same high-precision promotion as ``fftn``.

    Returns the nonredundant Hermitian half spectrum for Torch reconstruction.
    """
    if is_torch_tensor(array):
        if array.is_complex():
            raise ValueError("rfftn requires real input")
        dims = _normalise_axes(array.ndim, axes)
        profile = _torch_precision_profile(backend)
        if profile == "matlab_single":
            if array.dtype == torch.float32:
                return torch.fft.rfftn(array.to(torch.float64), dim=dims).to(torch.complex64)
        return torch.fft.rfftn(array, dim=dims)
    raise TypeError("rfftn is reserved for Torch reconstruction tensors")


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


def centered_fftn(array, *, axes: Iterable[int] | None = None, backend: str = "auto"):
    """MATLAB/RESIRE centred forward transform."""
    dims = _normalise_axes(array.ndim, axes)
    return fftshift(fftn(ifftshift(array, axes=dims), axes=dims, backend=backend), axes=dims)


def centered_ifftn(array, *, axes: Iterable[int] | None = None, backend: str = "auto"):
    """MATLAB/RESIRE centred inverse transform."""
    dims = _normalise_axes(array.ndim, axes)
    return fftshift(
        fftn(ifftshift(array, axes=dims), axes=dims, inverse=True, backend=backend),
        axes=dims,
    )


__all__ = [
    "centered_fftn",
    "centered_ifftn",
    "fftn",
    "fftshift",
    "ifftshift",
    "is_torch_tensor",
    "ordered_fft2",
    "rfftn",
]
