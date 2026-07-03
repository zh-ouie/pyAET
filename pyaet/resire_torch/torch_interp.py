"""Torch interpolation kernels used by the GPU reconstruction path.

The functions here mirror the boundary behavior of ``splinterp_cpp``:
coordinates are one-based by default, points outside the valid interpolation
stencil return zero, and exact hits on the last grid point are handled by
shifting the two-point stencil back by one cell.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _coord_tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _safe_floor_index(coord: torch.Tensor, origin_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(coord)
    safe_coord = torch.where(finite, coord, torch.zeros((), device=coord.device, dtype=coord.dtype))
    idx = torch.floor(safe_coord).to(torch.long) - int(origin_offset)
    return idx, finite


def _interp_indices(coord: torch.Tensor, size: int, origin_offset: int):
    idx1, finite = _safe_floor_index(coord, origin_offset)
    idx2 = idx1 + 1
    coord0 = coord - int(origin_offset)

    at_last = finite & (coord0 == (size - 1))
    idx1 = torch.where(at_last, idx1 - 1, idx1)
    idx2 = torch.where(at_last, idx2 - 1, idx2)

    valid = finite & (idx1 >= 0) & (idx2 <= (size - 1))
    idx1_safe = idx1.clamp(0, max(size - 1, 0))
    idx2_safe = idx2.clamp(0, max(size - 1, 0))
    return idx1, idx2, idx1_safe, idx2_safe, valid, coord0


def interp2(data, x, y, origin_offset: int = 1) -> torch.Tensor:
    """Bilinear interpolation with ``splinterp_cpp.interp2_F`` semantics."""
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D")

    device = data.device
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    x = _coord_tensor(x, device=device, dtype=real_dtype)
    y = _coord_tensor(y, device=device, dtype=real_dtype)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    nrows, ncols = data.shape
    x1, x2, x1_safe, x2_safe, valid_x, x0 = _interp_indices(x, nrows, origin_offset)
    y1, y2, y1_safe, y2_safe, valid_y, y0 = _interp_indices(y, ncols, origin_offset)
    valid = valid_x & valid_y

    f11 = data[x1_safe, y1_safe]
    f12 = data[x1_safe, y2_safe]
    f21 = data[x2_safe, y1_safe]
    f22 = data[x2_safe, y2_safe]

    wx1 = (x2.to(real_dtype) - x0)
    wx2 = (x0 - x1.to(real_dtype))
    wy1 = (y2.to(real_dtype) - y0)
    wy2 = (y0 - y1.to(real_dtype))

    a = f11 * wx1 + f21 * wx2
    b = f12 * wx1 + f22 * wx2
    out = a * wy1 + b * wy2
    return torch.where(valid, out, torch.zeros((), device=device, dtype=out.dtype))


def interp3(data, x, y, z, origin_offset: int = 1) -> torch.Tensor:
    """Trilinear interpolation with ``splinterp_cpp.interp3_F`` semantics."""
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.ndim != 3:
        raise ValueError("data must be 3D")

    device = data.device
    real_dtype = data.real.dtype if data.is_complex() else data.dtype
    x = _coord_tensor(x, device=device, dtype=real_dtype)
    y = _coord_tensor(y, device=device, dtype=real_dtype)
    z = _coord_tensor(z, device=device, dtype=real_dtype)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have the same shape")

    nrows, ncols, nlayers = data.shape
    x1, x2, x1_safe, x2_safe, valid_x, x0 = _interp_indices(x, nrows, origin_offset)
    y1, y2, y1_safe, y2_safe, valid_y, y0 = _interp_indices(y, ncols, origin_offset)
    z1, z2, z1_safe, z2_safe, valid_z, z0 = _interp_indices(z, nlayers, origin_offset)
    valid = valid_x & valid_y & valid_z

    f111 = data[x1_safe, y1_safe, z1_safe]
    f121 = data[x1_safe, y2_safe, z1_safe]
    f211 = data[x2_safe, y1_safe, z1_safe]
    f221 = data[x2_safe, y2_safe, z1_safe]

    f112 = data[x1_safe, y1_safe, z2_safe]
    f122 = data[x1_safe, y2_safe, z2_safe]
    f212 = data[x2_safe, y1_safe, z2_safe]
    f222 = data[x2_safe, y2_safe, z2_safe]

    wx1 = (x2.to(real_dtype) - x0)
    wx2 = (x0 - x1.to(real_dtype))
    wy1 = (y2.to(real_dtype) - y0)
    wy2 = (y0 - y1.to(real_dtype))
    wz1 = (z2.to(real_dtype) - z0)
    wz2 = (z0 - z1.to(real_dtype))

    a1 = f111 * wx1 + f211 * wx2
    b1 = f121 * wx1 + f221 * wx2
    lower = a1 * wy1 + b1 * wy2

    a2 = f112 * wx1 + f212 * wx2
    b2 = f122 * wx1 + f222 * wx2
    upper = a2 * wy1 + b2 * wy2

    out = lower * wz1 + upper * wz2
    return torch.where(valid, out, torch.zeros((), device=device, dtype=out.dtype))


__all__ = ["interp2", "interp3"]
