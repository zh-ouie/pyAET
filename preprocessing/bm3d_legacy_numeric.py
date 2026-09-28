"""Ordered float32 operations used by the legacy normal-noise BM3D profile.

Kept separate from the mathematical reference to make numerical conventions
explicit. No legacy binary is loaded by this implementation.
"""
import numpy as np


def haar_forward(a):
    a = np.asarray(a, dtype=np.float32)
    size = len(a)
    result = np.empty_like(a)
    current = a.copy()
    length = size
    scale_size = 2
    while length > 1:
        even, odd = current[::2], current[1::2]
        result[length // 2:length] = (even - odd) * np.float32(1 / np.sqrt(scale_size))
        current = even + odd
        length //= 2
        scale_size *= 2
    result[0] = current[0] * np.float32(1 / np.sqrt(size))
    return result


def haar_inverse(a):
    size = len(a)
    if size == 1:
        return a.copy()
    current = np.stack((a[0] + a[1], a[0] - a[1])) * np.float32(1 / np.sqrt(size))
    length = 2
    while length < size:
        detail = a[length:2 * length] * np.float32(np.sqrt(length / size))
        result = np.empty((length * 2, *a.shape[1:]), dtype=np.float32)
        result[::2] = current + detail
        result[1::2] = current - detail
        current = result
        length *= 2
    return current


def inverse_dct(spectrum, forward_matrix):
    matrix = forward_matrix.T
    size = len(matrix)
    intermediate = matrix[:, 0, None] * spectrum[..., 0, None, :]
    for k in range(1, size):
        intermediate += matrix[:, k, None] * spectrum[..., k, None, :]
    result = intermediate[..., 0, None] * matrix[:, 0]
    for k in range(1, size):
        result += intermediate[..., k, None] * matrix[:, k]
    return result


def hard_filter(vectors, block, sigma, threshold, matrix):
    means = haar_forward(vectors[:, 0])
    spectrum = haar_forward(vectors[:, 1:].reshape(-1, block, block))
    cutoff = np.float32(sigma) * np.float32(threshold)
    keep = np.abs(spectrum) >= cutoff
    spectrum *= keep
    mean_keep = np.abs(means) >= (cutoff / np.float32(block))
    if len(vectors) >= 4:
        mean_keep[0] = True
    means *= mean_keep
    count = 1 + int(keep.sum()) + int(mean_keep.sum()) - int(len(vectors) >= 4)
    pixels = inverse_dct(haar_inverse(spectrum), matrix)
    pixels += haar_inverse(means)[:, None, None]
    return pixels.swapaxes(-1, -2), np.float32(1) / np.float32(count), count


def wiener_filter(noisy_vectors, pilot_vectors, block, sigma, epsilon, matrix):
    noisy = haar_forward(noisy_vectors[:, 1:].reshape(-1, block, block))
    pilot = haar_forward(pilot_vectors[:, 1:].reshape(-1, block, block))
    square = pilot * pilot
    variance = np.float32(sigma) * np.float32(sigma)
    gain = np.divide(square, square + variance, out=np.ones_like(square),
                     where=(square + variance) != 0)
    noisy *= gain
    squares = (gain * gain).transpose(1, 2, 0).ravel()
    total = np.cumsum(np.r_[np.float32(epsilon), squares], dtype=np.float32)[-1]
    pixels = inverse_dct(haar_inverse(noisy), matrix)
    return pixels, np.float32(1) / total, int(np.count_nonzero(gain))
