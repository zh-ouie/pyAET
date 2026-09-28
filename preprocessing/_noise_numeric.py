"""Double-precision reductions following MATLAB R2025b summation order.

Long-vector reductions match
MATLAB's single-thread order; a multithreaded MATLAB reference may retain
roundoff differences.
"""
import numpy as np


def ordered_sum(values, axis=0):
    values = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    count = len(values)
    if not count:
        return np.zeros(values.shape[1:], dtype=np.float64)
    if count >= 89000:
        # MATLAB's single-thread large-vector path sums 4096-element blocks.
        partials = [ordered_sum(values[start:start+4096], axis=0)
                    for start in range(0, count, 4096)]
        return np.cumsum(partials, axis=0)[-1]
    if count < 44:
        return np.cumsum(values, axis=0)[-1]
    end = count // 4 * 4
    lanes = np.cumsum(values[:end].reshape(-1, 4, *values.shape[1:]), axis=0)[-1].copy()
    # MATLAB adds the trailing elements to their lanes before reducing lanes.
    lanes[:count-end] += values[end:]
    return np.cumsum(lanes, axis=0)[-1]


def ordered_mean(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    return ordered_sum(values, axis=axis) / values.shape[axis]


def ordered_variance(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mean = np.expand_dims(ordered_mean(values, axis=axis), axis=axis)
    return ordered_sum((values-mean)**2, axis=axis) / max(values.shape[axis]-1, 1)


def alpha_objective(samples, alpha):
    scaled = samples / alpha
    mean = ordered_mean(scaled, axis=1)
    variance = ordered_variance(scaled, axis=1)
    return ordered_sum(variance * mean) / ordered_sum(mean * mean) - 1.
