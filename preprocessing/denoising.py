"""Table 1 steps 2-6, preserving the supplied MATLAB wrapper's scaling."""
from dataclasses import dataclass
import numpy as np
from .pipeline import generalized_anscombe_forward, generalized_anscombe_inverse_exact


@dataclass(frozen=True)
class VSTScale:
    minimum: float
    value_range: float
    noise_std: float


def prepare_bm3d_input(image, alpha, sigma, factor=1.0):
    """Forward VST and legacy [0.15, 0.85] scaling inside the [0, 1] range."""
    transformed = generalized_anscombe_forward(image, sigma, alpha, 0)
    minimum, maximum = float(transformed.min()), float(transformed.max())
    value_range = max(maximum-minimum, np.finfo(float).eps)
    scale = VSTScale(minimum, value_range, 0.96/value_range*0.7*factor)
    return (transformed-minimum)/value_range*0.7+0.15, scale


def restore_bm3d_output(filtered, original, alpha, sigma, scale, vectors=None):
    """Undo scaling, exact inverse VST, then legacy intensity least-squares fit."""
    filtered = np.asarray(filtered, dtype=float)
    filtered = (filtered-0.15)/0.7*scale.value_range+scale.minimum
    estimate = generalized_anscombe_inverse_exact(filtered, sigma, alpha, 0, vectors)/alpha
    denominator = float(np.sum(estimate*estimate))
    multiplier = float(np.sum(original*estimate)/denominator) if denominator else 1.0
    return estimate*multiplier


def denoise_projections(projections, alpha=None, sigma=None, *, backend='readable', factor=1.0):
    """Complete VST/BM3D/inverse pipeline; readable selects the parity backend."""
    from .pipeline import denoise
    return denoise(projections, alpha, sigma, factor, backend)


__all__ = ['VSTScale', 'prepare_bm3d_input', 'restore_bm3d_output',
           'denoise_projections', 'generalized_anscombe_forward',
           'generalized_anscombe_inverse_exact']
