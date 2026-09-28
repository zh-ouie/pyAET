"""Reimplementation of ``Sample_withcomments.m`` using NumPy/SciPy.

The numerical stages mirror the MATLAB workflow while avoiding GUI input so the
pipeline can run in batch jobs. The BM3D kernel is a separate dependency whose
parity with the supplied legacy binary must be validated independently.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import warnings
import numpy as np
from scipy import ndimage, optimize, signal, sparse
from scipy.io import loadmat

from pyaet.fft_backend import fftn, fftshift, ifftshift


@dataclass
class PreprocessConfig:
    registration_size: int = 550
    crop_size: int = 300
    output_size: int = 210
    background_disk: int = 140
    otsu_scale: float = 0.9
    remove_indices: tuple = ()
    final_indices: tuple | None = None
    commonline_range: tuple = (-5.0, 5.0, 0.1)
    commonline_workers: int = 32
    registration_workers: int = 1
    trim_edge_projections: bool = True
    apply_commonline_rotation: bool = False
    matlab_angle_table: bool = True
    bm3d_backend: str = 'package'


@dataclass
class PreprocessResult:
    projections: np.ndarray
    angles: np.ndarray
    masks: np.ndarray
    support: np.ndarray
    noise: dict
    crop_centers: np.ndarray


_MATLAB_DISK_DECOMPOSITION = {
    # (axis half length, diagonal half length).  The Minkowski sum of these
    # four line elements exactly matches R2025b strel('disk', radius).
    6: (3, 1),
    10: (5, 2),
    12: (5, 3),
    20: (9, 5),
    140: (61, 39),
}


def _shift2(image, dy, dx):
    # Fourier shift matches MATLAB's My_FourierShift more closely than roll.
    f = fftn(image, backend="numpy")
    return fftn(
        ndimage.fourier_shift(f, (dy, dx)), inverse=True, backend="numpy"
    ).real


def _matlab_round(values):
    """MATLAB round: ties are rounded away from zero."""
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _matlab_disk(radius):
    """Return the flat neighborhood used by MATLAB R2025b ``strel``."""
    radius = int(radius)
    if radius in _MATLAB_DISK_DECOMPOSITION:
        axis_half, diagonal_half = _MATLAB_DISK_DECOMPOSITION[radius]
        extent = axis_half + 2 * diagonal_half
        l1_limit = 2 * axis_half + 2 * diagonal_half
        yy, xx = np.ogrid[-extent:extent + 1, -extent:extent + 1]
        return (np.maximum(np.abs(xx), np.abs(yy)) <= extent) & (
            np.abs(xx) + np.abs(yy) <= l1_limit
        )
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _matlab_grey_opening(image, radius):
    """Efficient flat opening for MATLAB's decomposed disk element."""
    image = np.asarray(image, dtype=np.float64)
    radius = int(radius)
    if radius not in _MATLAB_DISK_DECOMPOSITION:
        footprint = _matlab_disk(radius)
        return ndimage.grey_opening(image, footprint=footprint, mode="nearest")
    axis_half, diagonal_half = _MATLAB_DISK_DECOMPOSITION[radius]
    elements = (
        np.ones((1, 2 * axis_half + 1), dtype=bool),
        np.ones((2 * axis_half + 1, 1), dtype=bool),
        np.eye(2 * diagonal_half + 1, dtype=bool),
        np.fliplr(np.eye(2 * diagonal_half + 1, dtype=bool)),
    )
    extent = axis_half + 2 * diagonal_half
    result = np.pad(image, extent, mode="constant", constant_values=np.inf)
    for footprint in elements:
        result = ndimage.grey_erosion(
            result, footprint=footprint, mode="constant", cval=np.inf
        )
    result = result[extent:-extent, extent:-extent]
    # MATLAB repeats the boundary pixels of the eroded image during the
    # dilation half of imopen.  Pad once by the complete element so the
    # decomposed line operations retain that behavior.
    result = np.pad(result, extent, mode="edge")
    for footprint in elements:
        result = ndimage.grey_dilation(
            result, footprint=footprint, mode="constant", cval=-np.inf
        )
    return result[extent:-extent, extent:-extent]


def _phase_shift(reference, image):
    a = reference - reference.mean(); b = image - image.mean()
    c = fftn(
        fftn(a, backend="numpy") * np.conj(fftn(b, backend="numpy")),
        inverse=True,
        backend="numpy",
    )
    y, x = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    if y > c.shape[0] // 2: y -= c.shape[0]
    if x > c.shape[1] // 2: x -= c.shape[1]
    return float(y), float(x)


def _registration_fourier_shift(image, dy, dx):
    """Centered inverse/forward FFT pair, preserving MATLAB single inputs."""
    from scipy import fft
    from ._registration_fft import ordered_fft2
    ny, nx = image.shape
    yy = np.arange(-(ny // 2), (ny - 1) // 2 + 1)[:, None]
    xx = np.arange(-(nx // 2), (nx - 1) // 2 + 1)[None, :]
    # MATLAB ifftn/fftn traverse dimensions in column-major order. Explicit
    # one-dimensional calls preserve that order for real single input too.
    centered = fft.ifftshift(image)
    spectrum = fft.fftshift(ordered_fft2(centered, inverse=True))
    # MATLAB constructs Pfactor in double precision, then the product with
    # the single-precision spectrum is stored as single.  Rounding Pfactor
    # to complex64 before multiplication changes the registered pixels.
    phase = np.exp(2j * np.pi * (dx * xx / nx + dy * yy / ny))
    product = (spectrum.astype(np.complex128) * phase).astype(spectrum.dtype)
    product = fft.ifftshift(product)
    return fft.fftshift(ordered_fft2(product)).real


def _normxcorr2(template, image):
    """MATLAB normxcorr2 for a template no larger than the image."""
    template = np.asarray(template, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    centered = template - template.mean()
    template_energy = np.sum(centered * centered)
    numerator = signal.correlate(image, centered, mode="full", method="fft")
    ones = np.ones(template.shape, dtype=np.float64)
    local_sum = signal.convolve(image, ones, mode="full", method="fft")
    local_sum2 = signal.convolve(image * image, ones, mode="full", method="fft")
    local_energy = np.maximum(local_sum2 - local_sum * local_sum / template.size, 0)
    denominator = np.sqrt(template_energy * local_energy)
    result = np.zeros_like(numerator)
    valid = denominator > np.finfo(float).eps * np.max(denominator)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _matlab_registration_shift(reference, image, resolution=0.05,
                               search_range=(20, 20), window_half_size=(210, 210)):
    """Port My_NormXcorr_subpixel_align_setWindow_atomicRegion.m."""
    if reference.shape != image.shape:
        raise ValueError("registration images must have the same shape")
    if not 0 < resolution < 0.5:
        raise ValueError("registration resolution must be between 0 and 0.5")
    height, width = reference.shape
    crop_half = min(int(np.floor((height + 1) / 2 + 0.5)),
                    int(np.floor((width + 1) / 2 + 0.5))) - 20
    if crop_half <= 0:
        return (*_phase_shift(reference, image), np.nan)
    center_y = int(np.floor((height + 1) / 2 + 0.5)) - 1
    center_x = int(np.floor((width + 1) / 2 + 0.5)) - 1
    crop_y = slice(center_y - crop_half, center_y + crop_half + 1)
    crop_x = slice(center_x - crop_half, center_x + crop_half + 1)
    reference = np.asarray(reference[crop_y, crop_x])
    image = np.asarray(image[crop_y, crop_x])
    if image.dtype != np.float32:
        image = image.astype(np.float64)
    # The original EMD is single. MATLAB keeps the FFT and its phase product
    # single before normxcorr2 converts the shifted image to double.
    from scipy import fft
    from ._registration_fft import ordered_fft2
    # Candidate generation must use the same ordered FFT path as the applied
    # registration shift. A one-ULP candidate change can reverse close peaks
    # and produce a full search-grid jump (HEA-23, view 13, repeated frame 3).
    spectrum = fft.fftshift(ordered_fft2(fft.ifftshift(image)))
    ky = np.arange(-(image.shape[0] // 2), (image.shape[0] - 1) // 2 + 1)[:, None]
    kx = np.arange(-(image.shape[1] // 2), (image.shape[1] - 1) // 2 + 1)[None, :]

    def candidate_shift(dy, dx):
        phase = np.exp(2j * np.pi * (dy * ky / image.shape[0] + dx * kx / image.shape[1]))
        # Match the applied shift: form the phase in double precision, multiply
        # before rounding the product to the spectrum precision.
        product = (spectrum.astype(np.complex128) * phase).astype(spectrum.dtype)
        return fft.fftshift(ordered_fft2(fft.ifftshift(product), inverse=True)).real

    window_center_y = int(np.floor((reference.shape[0] + 1) / 2 + 0.5)) - 1
    window_center_x = int(np.floor((reference.shape[1] + 1) / 2 + 0.5)) - 1
    half_y = min(window_half_size[0], window_center_y)
    half_x = min(window_half_size[1], window_center_x)
    window_start_y = window_center_y - half_y
    window_start_x = window_center_x - half_x
    template = reference[
        window_start_y:window_center_y + half_y + 1,
        window_start_x:window_center_x + half_x + 1,
    ]
    range_y = min(search_range[0], window_center_y - window_start_y)
    range_x = min(search_range[1], window_center_x - window_start_x)
    # A normxcorr2 peak encodes the zero-based template origin as
    # peak-(template_size-1). Restrict it to the MATLAB +/- search window.
    peak_center_y = template.shape[0] - 1 + window_start_y
    peak_center_x = template.shape[1] - 1 + window_start_x
    search_y = slice(peak_center_y - range_y, peak_center_y + range_y + 1)
    search_x = slice(peak_center_x - range_x, peak_center_x + range_x + 1)

    divisions = 4
    iterations = abs(int(np.floor(np.log(resolution) / np.log(divisions))))
    start_y = start_x = -0.5
    end_y = end_x = 0.5
    selected_y = selected_x = 0.0
    for _ in range(iterations):
        boundary_y = np.linspace(start_y, end_y, divisions + 1)
        boundary_x = np.linspace(start_x, end_x, divisions + 1)
        candidates_y = (boundary_y[:-1] + boundary_y[1:]) / 2
        candidates_x = (boundary_x[:-1] + boundary_x[1:]) / 2
        metrics = np.empty((divisions, divisions), dtype=np.float64)
        for iy, fractional_y in enumerate(candidates_y):
            for ix, fractional_x in enumerate(candidates_x):
                # The refinement function uses the opposite Fourier phase from
                # My_FourierShift and subtracts it from the integer shift.
                shifted = candidate_shift(fractional_y, fractional_x)
                correlation = _normxcorr2(template, shifted)
                metrics[iy, ix] = np.max(correlation[search_y, search_x])
        flat = int(np.argmax(metrics.ravel(order="F")))
        iy, ix = np.unravel_index(flat, metrics.shape, order="F")
        selected_y = float(candidates_y[iy])
        selected_x = float(candidates_x[ix])
        start_y, end_y = boundary_y[iy], boundary_y[iy + 1]
        start_x, end_x = boundary_x[ix], boundary_x[ix + 1]

    shifted = candidate_shift(selected_y, selected_x)
    correlation = _normxcorr2(template, shifted)
    restricted = np.full(correlation.shape, -np.inf)
    restricted[search_y, search_x] = correlation[search_y, search_x]
    flat = int(np.argmax(restricted.ravel(order="F")))
    peak_y, peak_x = np.unravel_index(flat, restricted.shape, order="F")
    integer_y = peak_center_y - peak_y
    integer_x = peak_center_x - peak_x
    return (
        float(integer_y - selected_y),
        float(integer_x - selected_x),
        float(correlation[peak_y, peak_x]),
    )


def _cubic_convolution(image, row_coordinates, column_coordinates):
    """Keys bicubic interpolation used by MATLAB interp2(..., 'cubic', 0)."""
    if np.asarray(image).dtype == np.float32:
        return _cubic_convolution_single(image, row_coordinates, column_coordinates)
    image = np.asarray(image, dtype=np.float64)
    rows, columns = np.broadcast_arrays(row_coordinates, column_coordinates)
    outside = ((rows < 0) | (rows > image.shape[0] - 1) |
               (columns < 0) | (columns > image.shape[1] - 1))
    # MATLAB cubic interpolation extrapolates one support sample quadratically
    # at each boundary. Only query points outside the original grid use zero.
    if min(image.shape) < 3:
        raise ValueError('Cubic interpolation requires at least three samples per axis')
    extended = np.pad(image, 1)
    extended[0, 1:-1] = 3 * image[0] - 3 * image[1] + image[2]
    extended[-1, 1:-1] = 3 * image[-1] - 3 * image[-2] + image[-3]
    extended[:, 0] = 3 * extended[:, 1] - 3 * extended[:, 2] + extended[:, 3]
    extended[:, -1] = 3 * extended[:, -2] - 3 * extended[:, -3] + extended[:, -4]
    image = extended
    rows, columns = rows + 1, columns + 1
    floor_rows = np.floor(rows).astype(np.int64)
    floor_columns = np.floor(columns).astype(np.int64)

    def kernel(distance):
        distance = np.abs(distance)
        return np.where(
            distance <= 1,
            1.5 * distance**3 - 2.5 * distance**2 + 1,
            np.where(
                distance < 2,
                -0.5 * distance**3 + 2.5 * distance**2 - 4 * distance + 2,
                0,
            ),
        )

    result = np.zeros(rows.shape, dtype=np.float64)
    for row_offset in (-1, 0, 1, 2):
        source_rows = floor_rows + row_offset
        row_weight = kernel(rows - source_rows)
        for column_offset in (-1, 0, 1, 2):
            source_columns = floor_columns + column_offset
            valid = (
                (source_rows >= 0)
                & (source_rows < image.shape[0])
                & (source_columns >= 0)
                & (source_columns < image.shape[1])
            )
            weight = row_weight * kernel(columns - source_columns)
            result[valid] += image[source_rows[valid], source_columns[valid]] * weight[valid]
    result[outside] = 0
    return result


def _cubic_convolution_single(image, row_coordinates, column_coordinates):
    """Match alignDriftCorr_YY's interp2 on transposed single input.

    The source transposes its image and swaps query axes. Its row-then-column
    accumulation differs numerically from interp2 on an untransposed image.
    """
    image = np.asarray(image, dtype=np.float32)
    if min(image.shape) < 3:
        raise ValueError('Cubic interpolation requires at least three samples per axis')
    rows, columns = np.broadcast_arrays(row_coordinates, column_coordinates)
    outside = ((rows < 0) | (rows > image.shape[0] - 1) |
               (columns < 0) | (columns > image.shape[1] - 1))
    extended = np.pad(image, 1)
    extended[0, 1:-1] = 3 * (image[0] - image[1]) + image[2]
    extended[-1, 1:-1] = 3 * (image[-1] - image[-2]) + image[-3]
    extended[:, 0] = 3 * (extended[:, 1] - extended[:, 2]) + extended[:, 3]
    extended[:, -1] = 3 * (extended[:, -2] - extended[:, -3]) + extended[:, -4]
    ri, ci = np.floor(rows).astype(np.int64), np.floor(columns).astype(np.int64)

    def weights(fraction):
        t = fraction.astype(np.float32)
        return (((-.5*t+1)*t-.5)*t, 1+t*t*(1.5*t-2.5),
                ((-1.5*t+2)*t+.5)*t, t*t*(.5*t-.5))

    rw, cw = weights(rows-ri), weights(columns-ci)
    result = np.zeros(rows.shape, dtype=np.float32)
    for column in range(4):
        intermediate = np.zeros(rows.shape, dtype=np.float32)
        for row in range(4):
            intermediate += extended[
                np.clip(ri+row, 0, extended.shape[0]-1),
                np.clip(ci+column, 0, extended.shape[1]-1)] * rw[row]
        result += intermediate * cw[column]
    result[outside] = 0
    return result


def _correct_linear_scan_drift(stack, shifts):
    """Port alignDriftCorr_YY.m; returns its imageCorr, xyMean, stackCorr."""
    stack = np.asarray(stack)
    shifts = np.asarray(shifts, dtype=np.float64)
    relative = np.diff(shifts, axis=1)
    mean_y = float(np.mean(relative[0]))
    raw_mean_x = float(np.mean(relative[1]))
    mean_x = raw_mean_x * stack.shape[1] / (stack.shape[1] - raw_mean_x)
    indices = np.arange(1, stack.shape[1] + 1, dtype=np.float64) / stack.shape[1]
    indices -= indices.mean()
    row_delta = -mean_y * indices
    column_delta = -mean_x * indices
    rows = np.arange(stack.shape[0], dtype=np.float64)[:, None] + row_delta[None, :]
    columns = np.arange(stack.shape[1], dtype=np.float64)[None, :] + column_delta[:, None]
    corrected = np.stack(
        [_cubic_convolution(stack[:, :, index], rows, columns)
         for index in range(stack.shape[2])],
        axis=2,
    )
    # The MATLAB destination stack is allocated as double even for single input.
    corrected = corrected.astype(np.float64)
    return corrected.sum(axis=2), np.array([mean_y, mean_x]), corrected


def _center_resize_2d(images, output_size):
    """Center crop or zero-pad trailing image axes using MATLAB centers."""
    images = np.asarray(images)
    target = (int(output_size), int(output_size))
    if target[0] <= 0:
        raise ValueError("output_size must be positive")
    output = np.zeros(images.shape[:-2] + target, dtype=images.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(images.shape[-2:], target):
        overlap = min(source_size, target_size)
        source_start = source_size // 2 - overlap // 2
        target_start = target_size // 2 - overlap // 2
        source_slices.append(slice(source_start, source_start + overlap))
        target_slices.append(slice(target_start, target_start + overlap))
    output[(..., *target_slices)] = images[(..., *source_slices)]
    return output


def _register_repeated_frames(frames):
    """Register one view; return (row, column, frame) data and (dy, dx) shifts."""
    frames = np.asarray(frames)
    if frames.dtype != np.float32:
        frames = frames.astype(np.float64)
    frames = np.moveaxis(frames, 0, -1)
    shifted = frames.copy()
    shifts = np.zeros((2, frames.shape[2]), dtype=np.float64)
    for index in range(1, frames.shape[2]):
        reference = shifted[:, :, :index].mean(axis=2)
        dy, dx, _ = _matlab_registration_shift(reference, frames[:, :, index])
        shifts[:, index] = (dy, dx)
        shifted[:, :, index] = _registration_fourier_shift(frames[:, :, index], dy, dx)
    return shifted, shifts


def _register_projection(frames, output_size):
    shifted, shifts = _register_repeated_frames(frames)
    projection, _, corrected = _correct_linear_scan_drift(shifted, shifts)
    corrected = np.moveaxis(corrected, -1, 0)
    return _center_resize_2d(projection, output_size), _center_resize_2d(corrected, output_size)


def register_frames(raw, output_size=550, workers=1):
    """Register frames for each projection. Input is (frame, proj, y, x)."""
    raw = np.asarray(raw)
    if raw.ndim == 3: raw = raw[None, ...]
    nframe, nproj, _, _ = raw.shape
    out = np.empty((output_size, output_size, nproj), np.float64)
    aligned = np.empty((output_size, output_size, nproj, nframe), np.float64)
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        results = map(lambda p: _register_projection(raw[:, p], output_size), range(nproj))
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(lambda p: _register_projection(raw[:, p], output_size), range(nproj))
    try:
        for p, (projection, corrected) in enumerate(results):
            out[:, :, p] = projection
            aligned[:, :, p, :] = np.moveaxis(corrected, 0, -1)
    finally:
        if workers != 1:
            executor.shutdown()
    return out, aligned


def detect_crop_centers(stack, crop_size, smooth_sigma=8.0):
    """Select reproducible particle centers to replace MATLAB GUI clicks."""
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("center detection expects a 3-D projection stack")
    half = int(np.floor(crop_size / 2 + 0.5))
    centers = np.empty((stack.shape[2], 2), dtype=np.float64)
    for index in range(stack.shape[2]):
        image = np.asarray(stack[:, :, index], dtype=np.float64)
        smoothed = ndimage.gaussian_filter(image, smooth_sigma)
        valid = image != 0
        values = smoothed[valid]
        if values.size == 0:
            center = (np.array(image.shape) - 1) / 2
        else:
            histogram, edges = np.histogram(values, bins=256)
            probability = histogram / max(histogram.sum(), 1)
            omega = np.cumsum(probability)
            mu = np.cumsum(probability * np.arange(histogram.size))
            between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
            threshold = edges[int(np.nanargmax(between))]
            labels, count = ndimage.label((smoothed > threshold) & valid)
            if count == 0:
                center = (np.array(image.shape) - 1) / 2
            else:
                label_ids = np.arange(1, count + 1)
                areas = ndimage.sum(np.ones(image.shape), labels, label_ids)
                particle = labels == label_ids[int(np.argmax(areas))]
                weights = np.maximum(smoothed - threshold, 0) * particle
                center = np.asarray(ndimage.center_of_mass(weights))
        # TruncMat_seleCent cannot crop beyond an image boundary.
        centers[index] = np.clip(center, half - 1, np.array(image.shape) - half - 1)
    return centers


def crop_stack(stack, crop_size, centers=None):
    """Crop using TruncMat_seleCent's even/odd and MATLAB-rounding rules.

    ``centers`` uses zero-based floating-point ``(row, column)`` coordinates.
    They are converted to the equivalent one-based MATLAB click positions.
    """
    stack = np.asarray(stack)
    h, w, n = stack.shape[:3]
    if centers is None:
        centers = np.tile([(h - 1) / 2, (w - 1) / 2], (n, 1))
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (n, 2):
        raise ValueError(f"centers must have shape ({n}, 2)")
    result = np.zeros((crop_size, crop_size, n), stack.dtype)
    matlab_half = int(np.floor(crop_size / 2 + 0.5))
    for i, (cy, cx) in enumerate(centers):
        matlab_y = int(np.floor(cy + 1.5))
        matlab_x = int(np.floor(cx + 1.5))
        ys, xs = matlab_y - matlab_half, matlab_x - matlab_half
        y0, x0 = max(0, ys), max(0, xs)
        y1, x1 = min(h, ys + crop_size), min(w, xs + crop_size)
        result[y0 - ys:y1 - ys, x0 - xs:x1 - xs, i] = stack[y0:y1, x0:x1, i]
    return result


def estimate_noise(aligned, dark_floor=1000):
    """Match Sample_withcomments.m and Main_getAlphaSigma_parameters.m."""
    from ._gaussian_numeric import dark_gaussian_filter
    vals, dark = [], []
    for p in range(aligned.shape[2]):
        frames = aligned[:, :, p, :].astype(np.float64)
        dark_image = frames.copy()
        dark_image[dark_image <= dark_floor] = 1e10
        dark_image = dark_gaussian_filter(dark_image)
        dc = float(np.min(dark_image))
        dark.append(dc)
        vals.append(frames - dc)
    x = np.stack(vals, axis=2)
    x[x < -1000] = 0
    alpha = np.empty(x.shape[2], dtype=np.float64)
    sigma = np.empty(x.shape[2], dtype=np.float64)
    for projection in range(x.shape[2]):
        frames = x[:, :, projection, :]
        background_mask = _noise_background_mask(frames[:, :, 0])
        alpha[projection], sigma[projection] = _poisson_gaussian_parameters(
            frames, background_mask
        )
    return x, np.asarray(dark), alpha, sigma


def fit_dark_current(angles, dark):
    """Fit the script's diagnostic a*1e4/cos(tilt)+b*1e4 dark-current curve.

    This bounded linear least-squares diagnostic is saved for review; the
    original script does not feed it back into image subtraction or BM3D.
    """
    angles = np.asarray(angles, dtype=float)
    tilt = angles[:, 1] if angles.ndim == 2 else angles
    dark = np.asarray(dark, dtype=float).ravel()
    cosine = np.cos(np.deg2rad(tilt))
    if tilt.shape != dark.shape or not np.isfinite(dark).all() or np.any(np.abs(cosine) < 1e-12):
        raise ValueError('Dark-current fit needs finite, matching tilts away from 90 degrees')
    design = np.column_stack((1e4 / cosine, np.full(len(tilt), 1e4)))
    fit = optimize.lsq_linear(design, dark, bounds=([0., 0.], [1., 1.5]), tol=1e-14)
    if not fit.success:
        raise RuntimeError('Dark-current diagnostic fit did not converge')
    fitted = design @ fit.x
    return dict(parameters=fit.x, fitted=fitted, resnorm=float(np.sum((fitted-dark)**2)))


def _edge_preserve_flat(image, window_size=3):
    """Port edge_preserve_smoothing_YY(image, window_size, 1)."""
    image = np.asarray(image, dtype=np.float64)
    width = 1 + 2 * window_size
    averaging = np.ones((width, width), dtype=np.float64) / width**2
    variance = (
        signal.convolve2d(image * image, averaging, mode="same")
        - signal.convolve2d(image, averaging, mode="same") ** 2
    )
    cut_height = image.shape[0] - 2 * width + 2
    cut_width = image.shape[1] - 2 * width + 2
    if cut_height <= 0 or cut_width <= 0:
        return image.copy()
    variance_windows = np.lib.stride_tricks.sliding_window_view(variance, (width, width))[
        window_size:window_size + cut_height,
        window_size:window_size + cut_width,
    ]
    # MATLAB stores the first window offset fastest (column-major).
    minimum = np.argmin(variance_windows.swapaxes(-1, -2).reshape(
        cut_height, cut_width, width**2
    ), axis=2)
    offset_row = minimum % width
    offset_column = minimum // width
    patch_means = signal.convolve2d(image, averaging, mode="valid")
    base_row, base_column = np.indices((cut_height, cut_width))
    output = image.copy()
    output[width - 1:image.shape[0] - width + 1,
           width - 1:image.shape[1] - width + 1] = patch_means[
               base_row + offset_row, base_column + offset_column
           ]
    return output


def _matlab_otsu_threshold(image):
    # hist(x, 0.5:255.5) folds out-of-range observations into the end bins.
    edges = np.concatenate(([-np.inf], np.arange(1, 256), [np.inf]))
    counts, _ = np.histogram(np.asarray(image).ravel(), bins=edges)
    probability = counts / counts.sum()
    theta = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu - mu[-1] * theta) ** 2 / (theta * (1 - theta))
    return int(np.nanargmax(between)) + 1


def _remove_small_components(mask, minimum_area):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(np.ones(mask.shape), labels, np.arange(1, count + 1))
    keep = np.flatnonzero(sizes >= minimum_area) + 1
    return np.isin(labels, keep)


def _gaussian_mask_expansion(mask, half_width=3.0, threshold=0.1):
    height, width = mask.shape
    yy, xx = np.meshgrid(np.arange(width), np.arange(height))
    kernel = np.exp(
        -((yy - np.ceil((width - 1) / 2)) ** 2
          + (xx - np.ceil((height - 1) / 2)) ** 2)
        * np.log(2) / half_width**2
    )
    shape = (2 * height - 1, 2 * width - 1)
    start_y = int(np.floor((height + 1) / 2)) - 1
    start_x = int(np.floor((width + 1) / 2)) - 1
    mask_pad = np.zeros(shape, dtype=np.float64)
    kernel_pad = np.zeros(shape, dtype=np.float64)
    mask_pad[start_y:start_y + height, start_x:start_x + width] = mask
    kernel_pad[start_y:start_y + height, start_x:start_x + width] = kernel
    mask_spectrum = fftn(ifftshift(mask_pad), backend="numpy")
    kernel_spectrum = fftn(ifftshift(kernel_pad), backend="numpy")
    convolution = fftshift(
        fftn(mask_spectrum * kernel_spectrum, inverse=True, backend="numpy")
    ).real
    return convolution[start_y:start_y + height, start_x:start_x + width] > threshold


def _center_crop_or_pad(image, output_shape):
    """MATLAB ``My_paddzero``/``My_stripzero`` center convention."""
    output_shape = tuple(int(v) for v in output_shape)
    out = np.zeros(output_shape, dtype=np.asarray(image).dtype)
    source = np.asarray(image)
    source_slices, target_slices = [], []
    for src_size, dst_size in zip(source.shape, output_shape):
        overlap = min(src_size, dst_size)

        def center_start(full_size, kept_size):
            return full_size // 2 - kept_size // 2

        src_start = center_start(src_size, overlap)
        dst_start = center_start(dst_size, overlap)
        source_slices.append(slice(src_start, src_start + overlap))
        target_slices.append(slice(dst_start, dst_start + overlap))
    out[tuple(target_slices)] = source[tuple(source_slices)]
    return out


def _my_circshift_trunc_edge(image, shift_y, shift_x):
    """Zero-filled integer shift used by ``My_circshift_TruncEdge``."""
    image = np.asarray(image)
    out = np.zeros_like(image)
    sy, sx = int(shift_y), int(shift_x)
    src_y0, src_y1 = max(0, -sy), min(image.shape[0], image.shape[0] - sy)
    src_x0, src_x1 = max(0, -sx), min(image.shape[1], image.shape[1] - sx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[src_y0 + sy:src_y1 + sy, src_x0 + sx:src_x1 + sx] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def _matlab_com(image):
    from ._noise_numeric import ordered_sum
    image = np.asarray(image, dtype=np.float64)
    total = float(ordered_sum(image.ravel(order='F')))
    if total == 0:
        return (image.shape[0] + 1) / 2, (image.shape[1] + 1) / 2
    yy, xx = np.indices(image.shape, dtype=np.float64)
    # My_COM sums the weighted images across rows first, then down columns.
    # MATLAB's non-leading dimension uses serial accumulation, not four lanes.
    row_moment = ordered_sum(np.cumsum((yy + 1) * image, axis=1)[:, -1])
    col_moment = ordered_sum(np.cumsum((xx + 1) * image, axis=1)[:, -1])
    return float(row_moment / total), float(col_moment / total)


def _matlab_edge_preserve(image, window_size=2):
    return _edge_preserve_flat(image, window_size)


def _matlab_otsu_255(image):
    image = np.asarray(image, dtype=np.float64)
    if not np.any(np.isfinite(image)) or np.nanmax(image) <= 0:
        return 0.0
    scaled = image / np.nanmax(image) * 255.0
    return float(_matlab_otsu_threshold(scaled) * np.nanmax(image) / 255.0)


def _mask_from_bgsub_stage(image, otsu_scale=0.9, first_scale=0.6):
    """Port the two masks built inside ``BGsub_Main``."""
    image = _wt_rotate(np.asarray(image, dtype=np.float64), 0)
    first = _matlab_edge_preserve(image, 2)
    threshold = _matlab_otsu_255(first) * first_scale
    first_mask = _remove_small_components(first > threshold, 3000)
    first_mask = ndimage.binary_dilation(first_mask, structure=_matlab_disk(6))
    masked = image * first_mask
    com_y, com_x = _matlab_com(masked)
    ref_y, ref_x = _matlab_round((np.array(image.shape) + 1) / 2).astype(int)
    shift_y, shift_x = int(_matlab_round(ref_y - com_y)), int(_matlab_round(ref_x - com_x))
    shifted = np.roll(image, (shift_y, shift_x), axis=(0, 1))
    second = _matlab_edge_preserve(shifted, 2)
    threshold2 = _matlab_otsu_255(second) * otsu_scale
    second_mask = _remove_small_components(second > threshold2, 10000)
    second_mask = _gaussian_mask_expansion(_center_crop_or_pad(second_mask, (second_mask.shape[0] + 128, second_mask.shape[1] + 128)), 5, 0.1)
    second_mask = ndimage.binary_dilation(second_mask, structure=_matlab_disk(10))
    second_mask = ndimage.binary_erosion(second_mask, structure=_matlab_disk(12))
    second_mask = _center_crop_or_pad(second_mask, image.shape).astype(bool)
    original_mask = _my_circshift_trunc_edge(second_mask, -shift_y, -shift_x).astype(bool)
    return shifted, second_mask, original_mask, (shift_y, shift_x)


def _regionfill(image, mask):
    """Solve the discrete Laplace equation used by MATLAB ``regionfill``."""
    from ._regionfill_backend import regionfill_backend, solve_regionfill
    backend = regionfill_backend()
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return image.copy()
    if np.all(mask):
        return np.zeros_like(image)
    coordinates = np.argwhere(mask.T)[:, ::-1] if backend == 'cholmod' else np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[tuple(coordinates.T)] = np.arange(len(coordinates))
    rows, columns, values = [], [], []
    rhs = np.zeros(len(coordinates), dtype=np.float64)
    for equation, (row, column) in enumerate(coordinates):
        degree = 0
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row, neighbor_column = row + delta_row, column + delta_column
            if not (0 <= neighbor_row < image.shape[0] and 0 <= neighbor_column < image.shape[1]):
                continue
            degree += 1
            if mask[neighbor_row, neighbor_column]:
                rows.append(equation)
                columns.append(index[neighbor_row, neighbor_column])
                values.append(-1.0)
            else:
                rhs[equation] += image[neighbor_row, neighbor_column]
        rows.append(equation)
        columns.append(equation)
        values.append(float(degree))
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(coordinates), len(coordinates))
    )
    result = image.copy()
    result[tuple(coordinates.T)] = solve_regionfill(matrix, rhs, backend)
    return result


def _noise_background_mask(first_frame):
    smoothed = _edge_preserve_flat(first_frame, 3)
    maximum = float(np.max(smoothed))
    if maximum <= 0:
        return np.ones(smoothed.shape, dtype=bool)
    scaled = smoothed / maximum * 255
    threshold = _matlab_otsu_threshold(scaled) * maximum / 255 * 0.9
    particle = _remove_small_components(smoothed > threshold, 3000)
    # MATLAB R2025b strel('disk', 20) uses its default four-line
    # decomposition. getnhood() is this 39x39 octagon, not a 41x41 disk.
    yy, xx = np.ogrid[-19:20, -19:20]
    matlab_disk20 = (np.abs(xx) <= 19) & (np.abs(yy) <= 19) & (
        np.abs(xx) + np.abs(yy) <= 28
    )
    particle = ndimage.binary_dilation(particle, structure=matlab_disk20)
    particle = _gaussian_mask_expansion(particle, 3, 0.1)
    return ~particle


def _binned_temporal_samples(data, bin_size=2):
    samples = []
    for row_shift in range(bin_size):
        for column_shift in range(bin_size):
            shifted = np.roll(data, (-row_shift, -column_shift), axis=(0, 1))
            samples.append(shifted[::bin_size, ::bin_size, :])
    return np.concatenate(samples, axis=2)


def _poisson_gaussian_parameters(data, background_mask):
    from ._noise_numeric import ordered_mean, ordered_variance, alpha_objective
    samples = _binned_temporal_samples(np.asarray(data, dtype=np.float64), 2)
    means = ordered_mean(samples, axis=2)
    threshold = np.sort(means.ravel())[::-1][int(np.floor(means.size + 0.5)) - 1]
    selected = samples[means > threshold]
    # selected is pixels x samples after boolean indexing.
    # Evaluate the original scaled-data statistics at each optimizer call.
    # Replacing them by the algebraically equivalent C/p changes the finite
    # difference Jacobian and the resulting MATLAB stopping point.
    from ._trust_region import least_squares_matlab_trust_region
    fit = least_squares_matlab_trust_region(
        lambda p: np.array([alpha_objective(selected, p[0])]),
        np.array([1.0]), np.array([-np.inf]), np.array([np.inf]), tol_fun=1e-6,
    )
    if fit.status <= 0:
        raise RuntimeError('MATLAB-compatible noise alpha fit did not converge')
    alpha = float(fit.x[0])

    binned_mask_sum = np.zeros(background_mask.shape, dtype=np.float64)
    for row_shift in range(2):
        for column_shift in range(2):
            binned_mask_sum += np.roll(
                background_mask, (-row_shift, -column_shift), axis=(0, 1)
            )
    binned_mask = binned_mask_sum[::2, ::2] / 4
    masked = samples * binned_mask[:, :, None]
    # Logical linear indexing in MATLAB traverses column-major coordinates.
    background = masked.ravel(order='F')
    background = background[background != 0]
    current_mean = float(ordered_mean(background))
    sample_variance = float(ordered_variance(background))
    sigma = float(np.sqrt(max(sample_variance - current_mean * alpha, 0)))
    return alpha, sigma


def _anscombe_vectors():
    bundled = Path(__file__).resolve().parent / 'data'
    source = Path(os.environ.get("AET_ANSCOMBE_DIR", bundled))
    general = loadmat(source / "GenAnscombe_vectors.mat")
    anscombe = loadmat(source / "Anscombe_vectors.mat")
    return general, anscombe


def _linear_interp_extrap(x, xp, fp):
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64).ravel()
    fp = np.asarray(fp, dtype=np.float64).ravel()
    result = np.interp(x, xp, fp)
    below, above = x < xp[0], x > xp[-1]
    if np.any(below):
        result[below] = fp[0] + (x[below] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if np.any(above):
        result[above] = fp[-1] + (x[above] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return result


def anscombe_inverse_exact(denoised, vectors=None):
    vectors = _anscombe_vectors()[1] if vectors is None else vectors
    efz = vectors["Efz"].ravel()
    ez = vectors["Ez"].ravel()
    denoised = np.asarray(denoised, dtype=np.float64)
    result = _linear_interp_extrap(denoised, efz, ez)
    result[denoised > efz.max()] = (denoised[denoised > efz.max()] / 2) ** 2 - 1 / 8
    result[denoised < 2 * np.sqrt(3 / 8)] = 0
    return result


def generalized_anscombe_forward(data, sigma, alpha, mean=0.0):
    data = np.asarray(data, dtype=np.float64)
    return 2 / alpha * np.sqrt(np.maximum(
        0, alpha * data + 3 / 8 * alpha**2 + sigma**2 - alpha * mean
    ))


def generalized_anscombe_inverse_exact(denoised, sigma, alpha, mean=0.0, vectors=None):
    general, anscombe = _anscombe_vectors() if vectors is None else vectors
    normalized_sigma = sigma / alpha
    denoised = np.asarray(denoised, dtype=np.float64)
    if normalized_sigma > general["sigmas"].max():
        result = np.maximum(anscombe_inverse_exact(denoised, anscombe) - normalized_sigma**2, 0)
    elif normalized_sigma > 0:
        sigmas = general["sigmas"].ravel()
        ez = general["Ez"].ravel()
        matrix = general["Efzmatrix"]
        upper = int(np.searchsorted(sigmas, normalized_sigma, side="right"))
        upper = min(max(upper, 1), len(sigmas) - 1)
        lower = upper - 1
        weight = (normalized_sigma - sigmas[lower]) / (sigmas[upper] - sigmas[lower])
        efz = matrix[:, lower] * (1 - weight) + matrix[:, upper] * weight
        result = _linear_interp_extrap(denoised, efz, ez)
        asymptotic = anscombe_inverse_exact(denoised, anscombe) - normalized_sigma**2
        result[denoised > efz.max()] = asymptotic[denoised > efz.max()]
        result[denoised < efz.min()] = 0
    elif normalized_sigma == 0:
        result = anscombe_inverse_exact(denoised, anscombe)
    else:
        raise ValueError("sigma must be non-negative")
    return result * alpha + mean


def _legacy_bm3d_profile():
    from bm3d import BM3DProfile
    profile = BM3DProfile()
    profile.transform_2d_ht_name = "dct"
    profile.transform_2d_wiener_name = "dct"
    profile.transform_3rd_dim_name = "haar"
    profile.bs_ht = 8
    profile.step_ht = 2
    profile.max_3d_size_ht = 16
    profile.search_window_ht = 100
    profile.tau_match = 600
    profile.lambda_thr3d = 3.4
    profile.beta = 0.2
    profile.bs_wiener = 16
    profile.step_wiener = 2
    profile.max_3d_size_wiener = 16
    profile.search_window_wiener = 100
    profile.tau_match_wiener = 400
    profile.beta_wiener = 0.5
    profile.num_threads = 1
    return profile


def denoise(data, alpha=None, sigma=None, factor=1.0, backend='package'):
    """Port ``BM3D_Main`` including generalized Anscombe transforms.

    The Python ``bm3d`` package uses the same two-stage algorithm and receives
    the custom MATLAB parameters. Its compiled kernel is a newer implementation,
    so matching parameters alone does not establish numerical parity.
    ``backend='readable'`` selects the independent experimental NumPy/SciPy
    implementation, which supports the legacy normal-noise profile only.
    """
    if backend not in ('package', 'readable'):
        raise ValueError("BM3D backend must be 'package' or 'readable'")
    data = np.maximum(np.asarray(data, dtype=np.float64), 0)
    n = data.shape[2]
    alpha = np.ones(n) if alpha is None else np.broadcast_to(np.asarray(alpha, dtype=float), (n,))
    sigma = np.zeros(n) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    valid = sigma != 0
    if np.any(valid):
        from ._noise_numeric import ordered_mean
        alpha = np.full(n, ordered_mean(alpha[valid]))
        sigma = np.full(n, ordered_mean(sigma[valid]))
    if backend == 'package':
        try:
            from bm3d import bm3d
        except ImportError as exc:
            raise ImportError("Install bm3d to run the package denoising backend") from exc
        profile = _legacy_bm3d_profile()
    else:
        from .bm3d_reference import legacy_candidate
    from .denoising import prepare_bm3d_input, restore_bm3d_output
    vectors = _anscombe_vectors()
    out = np.empty_like(data)
    for i in range(n):
        z = data[:, :, i]
        transformed, vst_scale = prepare_bm3d_input(z, alpha[i], sigma[i], factor)
        sigma_denoise = vst_scale.noise_std
        if backend == 'package':
            filtered = bm3d(transformed, sigma_psd=sigma_denoise, profile=profile)
        else:
            filtered, _, _ = legacy_candidate(transformed, sigma_denoise)
        # The MATLAB wrapper converts the MEX single output to double before
        # undoing normalization and applying the inverse Anscombe transform.
        out[:, :, i] = restore_bm3d_output(filtered, z, alpha[i], sigma[i], vst_scale, vectors)
    return out


def otsu_mask(image, scale=0.9):
    x = ndimage.gaussian_filter(image, 2)
    hist, edges = np.histogram(x[np.isfinite(x)], bins=256)
    prob = hist / max(hist.sum(), 1)
    omega = np.cumsum(prob); mu = np.cumsum(prob * np.arange(256)); total = mu[-1]
    between = (total * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    t = edges[np.nanargmax(between)] * scale
    m = x > t
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=4)
    m = ndimage.binary_fill_holes(m)
    return ndimage.binary_dilation(m, iterations=3)


def pre_background_subtraction(data, para=1.0, strel_radius=140):
    """MATLAB-compatible equivalent of ``Pre_BGsub``.

    Returns ``(Dset, BackgroundStack, bgStack)`` in the same order as MATLAB.
    """
    data = np.asarray(data)
    out = np.zeros_like(data, dtype=np.float64)
    background_stack = np.zeros_like(data, dtype=np.float64)
    bg_stack = np.zeros_like(data, dtype=np.float64)
    for i in range(data.shape[2]):
        z = ndimage.gaussian_filter(data[:, :, i], 5, radius=10, mode="nearest")
        bg = _matlab_grey_opening(z, strel_radius)
        smooth = ndimage.gaussian_filter(bg, 15, radius=30, mode="nearest")
        background_stack[:, :, i] = bg
        bg_stack[:, :, i] = smooth
        out[:, :, i] = data[:, :, i] - para * smooth
    out[out < 0] = 0
    return out, background_stack, bg_stack


def background_and_align(data, disk=140, otsu_scale=0.9):
    n = data.shape[2]; h, w = data.shape[:2]
    out = np.zeros_like(data, dtype=np.float64)
    shift_stack = np.zeros_like(data, dtype=np.float64)
    mask_stack = np.zeros((h, w, n), bool)
    original_masks = np.zeros((h, w, n), bool)
    com_locations = np.zeros((4, n), dtype=np.float64)
    ref_y, ref_x = _matlab_round((np.array([h, w]) + 1) / 2).astype(int)
    for i in range(n):
        shifted, mask, original, first_shift = _mask_from_bgsub_stage(data[:, :, i], otsu_scale)
        com_locations[:2, i] = first_shift
        com_locations[2:, i] = 0
        com_y, com_x = _matlab_com(shifted * mask)
        xshift, yshift = float(ref_y - com_y), float(ref_x - com_x)
        com_locations[2:, i] = (xshift, yshift)
        final = _shift2(shifted * mask, xshift, yshift)
        out[:, :, i] = final * mask
        shift_stack[:, :, i] = _shift2(shifted, xshift, yshift)
        mask_stack[:, :, i] = mask
        original_masks[:, :, i] = original
    return out, mask_stack


def matlab_bgsub_main(data, otsu_scale=0.9):
    """Full ``BGsub_Main`` port with its diagnostic arrays."""
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[2]
    h, w = data.shape[:2]
    geproj = np.zeros_like(data)
    shift_stack = np.zeros_like(data)
    mask_stack = np.zeros((h, w, n), bool)
    original_masks = np.zeros((h, w, n), bool)
    com_locations = np.zeros((4, n), dtype=np.float64)
    ref_y, ref_x = _matlab_round((np.array([h, w]) + 1) / 2).astype(int)
    for i in range(n):
        shifted, mask, original, first_shift = _mask_from_bgsub_stage(data[:, :, i], otsu_scale)
        com_locations[:2, i] = first_shift
        com_y, com_x = _matlab_com(shifted * mask)
        xshift, yshift = float(ref_y - com_y), float(ref_x - com_x)
        com_locations[2:, i] = (xshift, yshift)
        geproj[:, :, i] = _shift2(shifted * mask, xshift, yshift) * mask
        shift_stack[:, :, i] = _shift2(shifted, xshift, yshift)
        mask_stack[:, :, i] = mask
        original_masks[:, :, i] = original
    return geproj, shift_stack, mask_stack, original_masks, com_locations


def matlab_repre_bgsub(data, original_masks):
    """Port ``rePre_BGsub`` using a Laplacian inpainting fill."""
    data = np.asarray(data, dtype=np.float64)
    masks = np.asarray(original_masks, dtype=bool)
    out = np.zeros_like(data)
    fill = np.zeros_like(data)
    for i in range(data.shape[2]):
        curr = data[:, :, i]
        filled = _regionfill(curr, masks[:, :, i])
        out[:, :, i] = np.maximum(curr - filled, 0)
        fill[:, :, i] = filled
    return out, fill


def image_norm(data, value=None):
    """Match ImageNorm's nested column sums and multiply-then-divide order.

    If value is omitted, use the first projection's total intensity.
    """
    from ._noise_numeric import ordered_sum
    data = np.asarray(data, dtype=np.float64)
    sums = ordered_sum(ordered_sum(data, axis=0), axis=0)
    if value is None:
        value = np.ravel(sums)[0]
    return np.divide(data * float(value), sums, out=np.zeros_like(data), where=sums != 0)


def strip_center_stack(data, output_size):
    data = np.asarray(data)
    return np.stack([_center_crop_or_pad(data[:, :, i], (output_size, output_size)) for i in range(data.shape[2])], axis=2)


def _wt_background_value(image):
    values = np.sort(np.asarray(image).real.astype(np.float64, copy=False).ravel())
    position = int(_matlab_round(values.size / 128.0)) - 1
    return float(values[np.clip(position, 0, values.size - 1)])


def _wt_clear_top_ffts(image):
    spectrum = fftshift(fftn(image, backend="numpy"))
    spectrum[0, :] = 0
    spectrum[:, 0] = 0
    return fftn(ifftshift(spectrum), inverse=True, backend="numpy")


def _wt_add_padding(image):
    image = np.asarray(image)
    rows, columns = image.shape
    output = np.full((rows * 2, columns * 2), _wt_background_value(image), dtype=np.float64)
    row_start = rows // 2 - 1
    column_start = columns // 2 - 1
    output[row_start:row_start + rows, column_start:column_start + columns] = image
    return output


def _wt_strip_padding(image):
    rows, columns = image.shape
    out_rows, out_columns = rows // 2, columns // 2
    row_start = out_rows // 2 - 1
    column_start = out_columns // 2 - 1
    return image[row_start:row_start + out_rows, column_start:column_start + out_columns]


def _wt_shear_image(image, dimension, factor):
    image = np.asarray(image)
    if abs(factor) <= 1e-6:
        return image
    rows, columns = image.shape
    if rows % 2 or columns % 2:
        raise ValueError("wtlib shear requires even image dimensions")
    xx, yy = np.meshgrid(
        np.arange(-rows / 2, rows / 2), np.arange(-columns / 2, columns / 2)
    )
    axis = int(dimension) - 1
    spectrum = fftshift(fftn(image, axes=(axis,), backend="numpy"), axes=(axis,))
    spectrum *= np.exp(-2j * np.pi / image.shape[axis] * xx * yy * factor)
    spectrum[0, :] = 0
    spectrum[:, 0] = 0
    result = fftn(
        ifftshift(spectrum, axes=(axis,)),
        axes=(axis,),
        inverse=True,
        backend="numpy",
    )
    return result.real


def _wt_scale_image(image, dimension, scale_factor):
    if abs(scale_factor - 1) < np.finfo(float).eps:
        return image
    transpose = int(dimension) == 1
    work = np.asarray(image).T if transpose else np.asarray(image)
    n = image.shape[int(dimension) - 1]
    if scale_factor > 1:
        internal = 1.0 / scale_factor
        spectrum = fftshift(fftn(work, backend="numpy"))
        spectrum[0, :] = 0
        spectrum[:, 0] = 0
        scaled = int(_matlab_round(n / internal / 2.0)) * 2
        offset = (scaled - n) // 2
        padded = np.zeros((scaled, n), dtype=complex)
        padded[offset:offset + n, :] = spectrum
        padded[0, :] = 0
        padded[:, 0] = 0
        expanded = fftn(ifftshift(padded), inverse=True, backend="numpy")
        result = expanded[offset:offset + n, :].real
    else:
        internal = 1.0 / scale_factor
        scaled = int(_matlab_round(n * internal / 2.0)) * 2
        offset = (scaled - n) // 2
        padded = np.full((scaled, n), _wt_background_value(image), dtype=np.float64)
        padded[offset:offset + n, :] = np.real(work)
        spectrum = fftshift(fftn(padded, backend="numpy"))
        cropped = spectrum[offset:offset + n, :]
        cropped[0, :] = 0
        cropped[:, 0] = 0
        result = fftn(ifftshift(cropped), inverse=True, backend="numpy").real
    return result.T if transpose else result


def _wt_rotate(image, degrees):
    """Port the Fourier scale/shear rotation in ``wtlib.applytransf2``."""
    angle = np.deg2rad(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    padded = _wt_clear_top_ffts(_wt_add_padding(image))
    if abs(sine) < 1e-15:
        return _wt_strip_padding(padded).real
    # LU factors of [[cos(a), -sin(a)], [sin(a), cos(a)]] for |a| < 45 deg.
    padded = _wt_scale_image(padded, 2, cosine)
    padded = _wt_scale_image(padded, 1, 1.0 / cosine)
    padded = _wt_shear_image(padded, 1, -sine * cosine)
    padded = _wt_shear_image(padded, 2, sine / cosine)
    return _wt_strip_padding(padded)


def commonline_align(data, angle_range=(-5, 5, .1), workers=1):
    if len(angle_range) == 3:
        start, stop, step = map(float, angle_range)
        count = int(np.floor((stop - start) / step + 0.5)) + 1
        angles = start + np.arange(count) * step
    else:
        angles = np.asarray(angle_range)
    def score_angle(a):
        rot = np.stack([_wt_rotate(data[:, :, i], a) for i in range(data.shape[2])], 2)
        lines = rot.sum(0)
        return np.abs(lines - lines.mean(1, keepdims=True)).sum()

    # Each candidate angle is independent.  Parallelizing this outer loop
    # preserves the MATLAB score definition and its candidate ordering while
    # avoiding the much slower Python serial loop.
    workers = max(1, int(workers))
    if workers == 1 or len(angles) == 1:
        scores = [score_angle(a) for a in angles]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(angles))) as pool:
            scores = list(pool.map(score_angle, angles))
    best = float(angles[int(np.argmin(scores))])
    out = np.stack([_wt_rotate(data[:, :, i], best) for i in range(data.shape[2])], 2)
    return out, best, np.asarray(scores)


def preprocess_stack(raw, angles=None, centers=None, config=None):
    cfg = config or PreprocessConfig()
    proj, aligned = register_frames(raw, cfg.registration_size, cfg.registration_workers)
    if angles is None: angles = np.arange(proj.shape[2], dtype=float)
    keep = np.ones(proj.shape[2], bool); keep[list(cfg.remove_indices)] = False
    proj, aligned, angles = proj[:, :, keep], aligned[:, :, keep], np.asarray(angles)[keep]
    if centers is None:
        centers = detect_crop_centers(proj, cfg.crop_size)
    else:
        centers = np.asarray(centers)[keep] if len(centers) == len(keep) else np.asarray(centers)
    proj = crop_stack(proj, cfg.crop_size, centers)
    aligned = np.stack([crop_stack(aligned[:, :, :, k], cfg.crop_size, centers) for k in range(aligned.shape[3])], -1)
    noise_data, dark, alpha, sigma = estimate_noise(aligned)
    dark_fit = fit_dark_current(angles, dark)
    den = denoise(noise_data.sum(-1), alpha=alpha, sigma=sigma, backend=cfg.bm3d_backend)
    # Sample_withcomments.m performs a light opening subtraction, a first
    # center-of-mass pass, Laplace (regionfill) subtraction, and a second
    # center-of-mass pass.  The common-line rotation is computed for review
    # but the script intentionally exports the normalized, unrotated stack.
    pre_bg, _, _ = pre_background_subtraction(den, 1, cfg.background_disk)
    _, _, _, original_mask, _ = matlab_bgsub_main(pre_bg, cfg.otsu_scale)
    redo_bg, _ = matlab_repre_bgsub(den, original_mask)
    bg, _, masks, _, _ = matlab_bgsub_main(redo_bg, cfg.otsu_scale)
    bg = np.transpose(bg, (1, 0, 2))
    bg = image_norm(bg)
    rotated, rot_angle, commonline_scores = commonline_align(
        bg, cfg.commonline_range, workers=cfg.commonline_workers)
    if cfg.apply_commonline_rotation:
        bg = rotated
    if cfg.matlab_angle_table and np.asarray(angles).ndim == 1:
        angle_table = np.zeros((len(angles), 3), dtype=np.float64)
        angle_table[:, 1] = angles
        angles = angle_table
    if cfg.final_indices is not None:
        bg = bg[:, :, cfg.final_indices]
        angles = angles[list(cfg.final_indices)]
        masks = masks[:, :, cfg.final_indices]
    elif cfg.trim_edge_projections and bg.shape[2] > 2:
        bg = bg[:, :, 1:-1]
        angles = angles[1:-1]
        masks = masks[:, :, 1:-1]
    bg = strip_center_stack(bg, cfg.output_size)
    d = cfg.output_size; r = d / 2; grid = np.indices((d, d, d), dtype=float) - d / 2
    support = (grid ** 2).sum(0) <= r ** 2
    return PreprocessResult(
        bg,
        angles,
        masks,
        support.astype(np.float32),
        {"dark": dark, "alpha": alpha, "sigma": sigma,
         "dark_fit_parameters": dark_fit['parameters'],
         "dark_fit_values": dark_fit['fitted'],
         "dark_fit_resnorm": np.asarray(dark_fit['resnorm']),
         "commonline_angle": np.asarray(rot_angle),
         "commonline_scores": commonline_scores},
        np.asarray(centers),
    )
