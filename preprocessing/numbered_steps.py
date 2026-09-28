"""Independent, ordered CPU preprocessing functions for one tilt group.

Image arrays use (row, column); repeated frames use (row, column, frame).
Steps 08, 29 and 30 require parameters or images from the complete tilt series.
The numbering expresses execution order, not a change to the legacy algorithm.
"""


from dataclasses import replace


import numpy as np


from scipy import ndimage


from . import pipeline as p


from ._noise_numeric import ordered_mean


def _initial_mask(image):
    image = p._wt_rotate(np.asarray(image, dtype=float), 0)
    smooth = p._matlab_edge_preserve(image, 2)
    mask = p._remove_small_components(smooth > p._matlab_otsu_255(smooth) * .6, 3000)
    return image, ndimage.binary_dilation(mask, structure=p._matlab_disk(6))


def _integer_com(image, mask):
    com = np.asarray(p._matlab_com(image * mask))
    reference = p._matlab_round((np.array(image.shape) + 1) / 2).astype(int)
    shift = p._matlab_round(reference - com).astype(int)
    return np.roll(image, tuple(shift), axis=(0, 1)), shift


def _refined_mask(shifted, shift, otsu_scale=.9):
    smooth = p._matlab_edge_preserve(shifted, 2)
    mask = p._remove_small_components(smooth > p._matlab_otsu_255(smooth) * otsu_scale, 10000)
    mask = p._gaussian_mask_expansion(p._center_crop_or_pad(mask, (mask.shape[0]+128, mask.shape[1]+128)), 5, .1)
    mask = ndimage.binary_dilation(mask, structure=p._matlab_disk(10))
    mask = ndimage.binary_erosion(mask, structure=p._matlab_disk(12))
    mask = p._center_crop_or_pad(mask, shifted.shape).astype(bool)
    original = p._my_circshift_trunc_edge(mask, -shift[0], -shift[1]).astype(bool)
    return mask, original


def _subpixel_com(shifted, mask):
    com = np.asarray(p._matlab_com(shifted * mask))
    reference = p._matlab_round((np.array(shifted.shape) + 1) / 2).astype(int)
    shift = reference - com
    return p._shift2(shifted * mask, *shift) * mask, shift


def first_mask_and_com(image, otsu_scale=.9):
    """Return inspectable mask/COM checkpoints, in their actual execution order."""
    prepared, first = _initial_mask(image)
    shifted, integer = _integer_com(prepared, first)
    refined, original = _refined_mask(shifted, integer, otsu_scale)
    aligned, subpixel = _subpixel_com(shifted, refined)
    return dict(prepared=prepared, first_mask=first, integer_aligned=shifted,
                integer_shift=integer, refined_mask=refined, original_mask=original,
                aligned=aligned, subpixel_shift=subpixel)


def second_mask_and_com(image, otsu_scale=.9):
    return first_mask_and_com(image, otsu_scale)


def step_01_register_frames(frames):
    """Input (frame,row,column); return registered frames and (2,frame) shifts."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] < 2 or not np.isfinite(frames).all():
        raise ValueError('Expected at least two finite repeated frames')
    return p._register_repeated_frames(frames)


def step_02_correct_scan_drift(registered, shifts, output_size=550):
    """Correct scan drift and center-pad/crop the working canvas; return frames."""
    _, mean_drift, frames = p._correct_linear_scan_drift(registered, shifts)
    frames = np.moveaxis(p._center_resize_2d(np.moveaxis(frames, -1, 0), output_size), 0, -1)
    return frames, mean_drift


def step_03_crop_fixed_center(frames, center, size=300):
    """Crop repeated frames at a zero-based (row,column) center."""
    return np.stack([p.crop_stack(frames[:, :, k:k+1], size, np.asarray(center).reshape(1, 2))[:, :, 0]
                     for k in range(frames.shape[2])], axis=2)


def step_04_estimate_dark_current(frames, dark_floor=1000):
    from ._gaussian_numeric import dark_gaussian_filter
    image = np.asarray(frames, dtype=np.float64).copy()
    image[image <= dark_floor] = 1e10
    return float(np.min(dark_gaussian_filter(image)))


def step_05_subtract_dark_current(frames, dark):
    result = np.asarray(frames, dtype=np.float64) - dark
    result[result < -1000] = 0
    return result


def step_06_estimate_noise_parameters(corrected_frames):
    mask = p._noise_background_mask(corrected_frames[:, :, 0])
    return p._poisson_gaussian_parameters(corrected_frames, mask)


def step_07_sum_frames(corrected_frames):
    return np.maximum(np.asarray(corrected_frames).sum(axis=2), 0)


def step_08_average_noise_parameters(alpha, sigma):
    """Pass full-series parameter vectors, including the current view."""
    alpha, sigma = np.broadcast_arrays(np.asarray(alpha, dtype=float), np.asarray(sigma, dtype=float))
    valid = sigma != 0
    if not valid.any():
        raise ValueError('No nonzero-sigma views; pass per-view parameters directly to step 09')
    return float(ordered_mean(alpha[valid])), float(ordered_mean(sigma[valid]))


def step_09_forward_vst(image, alpha, sigma):
    return p.generalized_anscombe_forward(image, sigma, alpha, 0)


def step_10_scale_vst(transformed, factor=1.0):
    from .denoising import VSTScale
    minimum = float(np.min(transformed))
    span = max(float(np.max(transformed)) - minimum, np.finfo(float).eps)
    scale = VSTScale(minimum, span, .96 / span * .7 * factor)
    return (transformed - minimum) / span * .7 + .15, scale


def step_11_bm3d_hard_threshold(scaled, noise_std):
    from .bm3d_reference import HT_PARAMETERS, stage
    if noise_std * 255 > 40:
        raise ValueError('The validated readable profile requires noise_std*255 <= 40')
    parameters = replace(HT_PARAMETERS, adaptive_nonzero=8, dynamic_positions=True,
                         clip_blocks=True, legacy_ht_rules=True, legacy_matching='ht')
    hard, _ = stage(np.asarray(scaled).T, noise_std, parameters)
    return hard.T


def step_12_bm3d_wiener(scaled, hard, noise_std):
    from .bm3d_reference import WIENER_PARAMETERS, stage
    if noise_std * 255 > 40:
        raise ValueError('The validated readable profile requires noise_std*255 <= 40')
    result, _ = stage(scaled, noise_std,
        replace(WIENER_PARAMETERS, clip_blocks=True, legacy_matching='wiener'), pilot=hard)
    return result


def step_13_unscale_vst(filtered, scale):
    return (np.asarray(filtered, dtype=float) - .15) / .7 * scale.value_range + scale.minimum


def step_14_inverse_vst(transformed, alpha, sigma):
    return p.generalized_anscombe_inverse_exact(transformed, sigma, alpha, 0) / alpha


def step_15_restore_intensity(estimate, original):
    denominator = float(np.sum(estimate * estimate))
    factor = float(np.sum(original * estimate) / denominator) if denominator else 1.0
    return estimate * factor, factor


def step_16_estimate_smooth_background(image, radius=140):
    blurred = ndimage.gaussian_filter(image, 5, radius=10, mode='nearest')
    opening = p._matlab_grey_opening(blurred, radius)
    return ndimage.gaussian_filter(opening, 15, radius=30, mode='nearest')


def step_17_subtract_smooth_background(image, background):
    return np.maximum(image - background, 0)


def step_18_initial_mask(image):
    return _initial_mask(image)


def step_19_integer_com(image, mask):
    return _integer_com(image, mask)


def step_20_refined_mask(shifted, shift, otsu_scale=.9):
    return _refined_mask(shifted, shift, otsu_scale)


def step_21_subpixel_com(shifted, mask):
    return _subpixel_com(shifted, mask)


def step_22_estimate_laplacian_background(denoised, original_mask):
    """Use step 15 image and step 20 original-coordinate mask, not aligned image."""
    return p._regionfill(denoised, original_mask)


def step_23_subtract_laplacian_background(denoised, background):
    return np.maximum(denoised - background, 0)


def step_24_initial_mask(image):
    return _initial_mask(image)


def step_25_integer_com(image, mask):
    return _integer_com(image, mask)


def step_26_refined_mask(shifted, shift, otsu_scale=.9):
    return _refined_mask(shifted, shift, otsu_scale)


def step_27_subpixel_com(shifted, mask):
    return _subpixel_com(shifted, mask)


def step_28_transpose_image(image):
    return np.asarray(image).T


def step_29_normalize_intensity(image, reference_total):
    """reference_total is the first transposed view's total intensity."""
    return p.image_norm(image[:, :, None], value=reference_total)[:, :, 0]


def step_30_estimate_commonline(stack, angle_range=(-5, 5, .1), workers=32):
    """Needs the entire normalized series; returns candidate stack, angle, scores."""
    return p.commonline_align(stack, angle_range, workers=workers)


def step_31_select_rotation(normalized, rotated, apply_rotation=False):
    """The original script exports the normalized, unrotated image."""
    return rotated if apply_rotation else normalized


def step_32_select_views(stack, angles, indices):
    indices = np.asarray(indices, dtype=int)
    return stack[:, :, indices], np.asarray(angles)[indices]


def step_33_final_crop(image, size=210):
    return p._center_crop_or_pad(image, (size, size))


def step_34_make_angle_table(angles):
    angles = np.asarray(angles)
    if angles.ndim == 2 and angles.shape[1] == 3:
        return angles.copy()
    if angles.ndim != 1:
        raise ValueError('angles must have shape (views,) or (views,3)')
    table = np.zeros((len(angles), 3), dtype=float)
    table[:, 1] = angles
    return table


def step_35_make_support(size=210):
    grid = np.indices((size, size, size), dtype=float) - size/2
    return ((grid**2).sum(0) <= (size/2)**2).astype(np.float32)


__all__ = sorted(name for name in globals() if name.startswith("step_"))
