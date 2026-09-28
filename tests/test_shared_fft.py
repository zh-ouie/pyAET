"""Shared FFTW/Torch transform conventions and precision checks."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from pyaet import fft_backend as fft

@pytest.mark.parametrize('shape', [(5,7), (6,8), (5,6,7)])
@pytest.mark.parametrize('dtype', [np.float32,np.float64,np.complex64,np.complex128])
def test_cpu_conventions(shape, dtype):
    rng=np.random.default_rng(19)
    x=rng.normal(size=shape).astype(dtype)
    if np.iscomplexobj(x): x += 1j*rng.normal(size=shape)
    tolerance=3e-6 if x.dtype in (np.float32,np.complex64) else 3e-13
    for axes in [None,(0,),(-1,),tuple(reversed(range(x.ndim)))]:
        expected=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x,axes=axes),axes=axes),axes=axes)
        actual=fft.centered_fftn(x,axes=axes)
        np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=tolerance)
        np.testing.assert_allclose(fft.centered_ifftn(actual,axes=axes),x,atol=tolerance,rtol=tolerance)


def test_parallel_execution():
    arrays=[np.random.default_rng(i).normal(size=(31,34)) for i in range(12)]
    serial=[fft.fftn(a) for a in arrays]
    with ThreadPoolExecutor(4) as pool: parallel=list(pool.map(fft.fftn,arrays))
    for a,b in zip(serial,parallel):np.testing.assert_array_equal(a,b)


def test_convolution():
    from scipy.signal import convolve
    rng=np.random.default_rng(3)
    a,b=rng.normal(size=(7,8)),rng.normal(size=(3,4))
    np.testing.assert_allclose(fft.fft_convolve_full(a,b),convolve(a,b,method='direct'),atol=1e-13)


def test_axes_and_missing_library(monkeypatch):
    with pytest.raises(ValueError):fft.fftn(np.ones((2,3)),axes=(2,))
    with pytest.raises(ValueError):fft.fftn(np.ones((2,3)),axes=(0,0))
    fft._fftw_library.cache_clear()
    monkeypatch.setenv('AET_FFTW_DOUBLE_LIBRARY','/missing/fftw.so')
    try:
        with pytest.raises(OSError):fft.fftn(np.ones((2,3)))
    finally:fft._fftw_library.cache_clear()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_torch_precision(device):
    torch=pytest.importorskip('torch')
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x=torch.arange(210,dtype=torch.float32,device=device).reshape(5,6,7)
    assert torch.equal(fft.fftn(x),torch.fft.fftn(x.double()).to(torch.complex64))
    assert torch.equal(fft.rfftn(x),torch.fft.rfftn(x.double()).to(torch.complex64))
    torch.testing.assert_close(fft.centered_ifftn(fft.centered_fftn(x)).real,x,atol=3e-5,rtol=1e-6)
    impulse=torch.zeros_like(x);impulse[2,3,3]=1
    torch.testing.assert_close(fft.centered_fftn(impulse),torch.ones_like(x,dtype=torch.complex64))

@pytest.mark.parametrize('shape',[(5,7),(6,8),(5,6,7)])
def test_cpu_torch_agreement(shape):
    torch=pytest.importorskip('torch')
    rng=np.random.default_rng(23)
    x=rng.normal(size=shape)+1j*rng.normal(size=shape)
    for inverse in (False,True):
        np.testing.assert_allclose(fft.fftn(x,inverse=inverse),
                                   fft.fftn(torch.from_numpy(x),inverse=inverse).numpy(),
                                   atol=3e-13,rtol=3e-13)


def test_no_alternative_cpu_fft_calls():
    """Prevent direct FFT calls from bypassing the shared implementation."""
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in list((root/'preprocessing').rglob('*.py'))+[root/'pyaet/fft_backend.py']:
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                if node.func.attr in ('fft','ifft','fft2','ifft2','fftn','ifftn','rfft','rfftn','irfftn'):
                    assert ast.unparse(node.func).startswith('torch.fft.'), (path,node.lineno)
