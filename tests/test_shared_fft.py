"""Small-array checks for the shared FFT policy and NumPy compatibility."""
import numpy as np
import pytest
from pyaet.fft_backend import fftn, centered_fftn

@pytest.mark.parametrize("shape", [(5, 7), (6, 8), (5, 6, 7)])
def test_numpy_compatibility(shape):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(centered_fftn(x, backend="numpy"),
                                 np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x))))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_precision(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.arange(210, dtype=torch.float32, device=device).reshape(5, 6, 7)
    expected = torch.fft.fftn(x.double()).to(torch.complex64)
    assert torch.equal(fftn(x), expected)
    with pytest.raises(ValueError):
        fftn(x, backend="torch_native")
