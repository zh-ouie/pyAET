"""Table 1 denoising step 1: experimental Gaussian-Poisson noise estimation."""
from dataclasses import dataclass
import numpy as np
from .pipeline import estimate_noise, fit_dark_current


@dataclass
class NoiseEstimate:
    corrected_frames: np.ndarray
    dark: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    dark_fit: dict

    @property
    def summed(self):
        return self.corrected_frames.sum(-1)


def estimate_projection_noise(frames, angles, dark_floor=1000):
    """Estimate per-view parameters; the legacy denoiser later uses their means."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or not np.isfinite(frames).all():
        raise ValueError('frames must be finite (row, column, view, frame)')
    corrected, dark, alpha, sigma = estimate_noise(frames, dark_floor)
    if not all(np.isfinite(value).all() for value in (corrected, dark, alpha, sigma)):
        raise ValueError('Noise estimation produced non-finite values; check repeated frames and background')
    if np.any(alpha <= 0) or np.any(sigma < 0):
        raise ValueError('Noise estimation requires positive alpha and non-negative sigma')
    return NoiseEstimate(corrected, dark, alpha, sigma, fit_dark_current(angles, dark))


__all__ = ['NoiseEstimate', 'estimate_projection_noise', 'estimate_noise', 'fit_dark_current']
