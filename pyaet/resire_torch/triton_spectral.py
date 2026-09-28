"""Optional fused Fourier-volume interpolation for CUDA reconstruction.

The arithmetic follows the torch reference with floating-point contraction
disabled. Half spectra use Hermitian lookup; FFT computation is kept in the
shared FFT backend. No backprojection or gradient reduction is changed.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _index(c, N: tl.constexpr, ORIGIN: tl.constexpr):
        finite = (c == c) & (tl.abs(c) != float('inf'))
        safe = tl.where(finite, c, 0.)
        lo = tl.floor(safe).to(tl.int64) - ORIGIN
        at_last = finite & (c - ORIGIN == N - 1)
        lo = tl.where(at_last, lo - 1, lo)
        hi = lo + 1
        valid = finite & (lo >= 0) & (hi < N)
        # Match the unshifted torch interpolation reference.
        a = (tl.minimum(tl.maximum(lo, 0), N - 1) + N // 2) % N
        b = (tl.minimum(tl.maximum(hi, 0), N - 1) + N // 2) % N
        wlo = hi.to(c.dtype) - (c - ORIGIN)
        whi = (c - ORIGIN) - lo.to(c.dtype)
        return a, b, wlo, whi, valid

    @triton.jit
    def _fetch(D, x, y, z, valid, NX: tl.constexpr, NY: tl.constexpr,
               NZ: tl.constexpr, DS0: tl.constexpr, DS1: tl.constexpr,
               DS2: tl.constexpr, HALF: tl.constexpr):
        if HALF:
            conjugate = z > NZ // 2
            x = tl.where(conjugate, (NX - x) % NX, x)
            y = tl.where(conjugate, (NY - y) % NY, y)
            z = tl.where(conjugate, NZ - z, z)
        offset = 2 * (x * DS0 + y * DS1 + z * DS2)
        re = tl.load(D + offset, mask=valid, other=0.)
        im = tl.load(D + offset + 1, mask=valid, other=0.)
        if HALF:
            im = tl.where(conjugate, -im, im)
        return re, im

    @triton.jit
    def _interp(D, X, Y, Z, O, COUNT: tl.constexpr,
                Q1: tl.constexpr, Q2: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
                YS0: tl.constexpr, YS1: tl.constexpr, YS2: tl.constexpr,
                ZS0: tl.constexpr, ZS1: tl.constexpr, ZS2: tl.constexpr,
                NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr,
                HALF: tl.constexpr, ORIGIN: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < COUNT
        q0 = i // (Q1 * Q2)
        q1 = (i // Q2) % Q1
        q2 = i % Q2
        x = tl.load(X + q0*XS0 + q1*XS1 + q2*XS2, mask, other=0.)
        y = tl.load(Y + q0*YS0 + q1*YS1 + q2*YS2, mask, other=0.)
        z = tl.load(Z + q0*ZS0 + q1*ZS1 + q2*ZS2, mask, other=0.)
        xa, xb, wx, vx, valid_x = _index(x, NX, ORIGIN)
        ya, yb, wy, vy, valid_y = _index(y, NY, ORIGIN)
        za, zb, wz, vz, valid_z = _index(z, NZ, ORIGIN)
        valid = mask & valid_x & valid_y & valid_z
        r111,i111 = _fetch(D,xa,ya,za,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r211,i211 = _fetch(D,xb,ya,za,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r121,i121 = _fetch(D,xa,yb,za,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r221,i221 = _fetch(D,xb,yb,za,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r112,i112 = _fetch(D,xa,ya,zb,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r212,i212 = _fetch(D,xb,ya,zb,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r122,i122 = _fetch(D,xa,yb,zb,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        r222,i222 = _fetch(D,xb,yb,zb,valid,NX,NY,NZ,DS0,DS1,DS2,HALF)
        lr = (r111*wx+r211*vx)*wy + (r121*wx+r221*vx)*vy
        li = (i111*wx+i211*vx)*wy + (i121*wx+i221*vx)*vy
        ur = (r112*wx+r212*vx)*wy + (r122*wx+r222*vx)*vy
        ui = (i112*wx+i212*vx)*wy + (i122*wx+i222*vx)*vy
        tl.store(O+2*i, tl.where(valid, lr*wz+ur*vz, 0.), mask)
        tl.store(O+2*i+1, tl.where(valid, li*wz+ui*vz, 0.), mask)


def interp3_spectral(data, x, y, z, *, full_shape=None, origin_offset=1):
    """Interpolate an unshifted full or Hermitian half spectrum on CUDA.

    Coordinates index the shifted full spectrum, exactly as in the torch
    reference. ``full_shape`` is required for a half spectrum, including odd
    last-axis lengths. Coordinate tensors may be strided projection chunks.
    """
    if triton is None:
        raise RuntimeError("gpu_spectral_acceleration requires Triton")
    if not data.is_cuda or data.ndim != 3 or not data.is_complex():
        raise ValueError("data must be a three-dimensional CUDA complex tensor")
    if any(t.device != data.device or t.dtype != data.real.dtype for t in (x,y,z)):
        raise ValueError("Coordinates must share the spectrum device and real dtype")
    if x.ndim != 3 or x.shape != y.shape or x.shape != z.shape:
        raise ValueError("Coordinates must have matching three-dimensional shapes")
    half = full_shape is not None
    shape = tuple(full_shape) if half else tuple(data.shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("full_shape must contain three dimensions of at least two")
    expected = (*shape[:2], shape[2]//2+1) if half else shape
    if tuple(data.shape) != expected:
        raise ValueError("Invalid full_shape for the supplied spectrum")
    output = torch.empty(x.shape, dtype=data.dtype, device=data.device)
    if output.numel():
        _interp[(triton.cdiv(output.numel(),256),)](
            torch.view_as_real(data), x, y, z, torch.view_as_real(output),
            output.numel(), x.shape[1], x.shape[2],
            *x.stride(), *y.stride(), *z.stride(), *shape, *data.stride(),
            half, int(origin_offset), 256, num_warps=4, enable_fp_fusion=False)
    return output
