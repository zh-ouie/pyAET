"""Experimental input preparation before the paper's Table 1 preprocessing."""
from dataclasses import dataclass
import numpy as np
from .io import load_emd, load_mat_stack
from .pipeline import register_frames, detect_crop_centers, crop_stack


@dataclass
class PreparedProjections:
    projections: np.ndarray
    frames: np.ndarray
    angles: np.ndarray
    centers: np.ndarray


def prepare_acquisition(raw, angles, config, centers=None):
    """Register/drift-correct frames, select views and crop at fixed centers.

    raw: (frame, view, row, column); output frames: (row, column, view, frame).
    Centers are zero-based (row, column), before cropping.
    """
    raw = np.asarray(raw)
    if angles is None:
        raise ValueError('Measured tilt angles are required; view indices are not physical angles')
    if raw.ndim != 4 or min(raw.shape) == 0 or not np.isfinite(raw).all():
        raise ValueError('raw must be finite (frame, view, row, column)')
    if raw.shape[0] < 2:
        raise ValueError('Acquisition drift correction needs at least two repeated frames; use stage APIs for prepared projections')
    angles = np.asarray(angles)
    count = raw.shape[1]
    if angles.shape not in ((count,), (count, 3)) or not np.isfinite(angles).all():
        raise ValueError('angles must be finite with shape (views,) or (views, 3)')
    projections, frames = register_frames(raw, config.registration_size, config.registration_workers)
    keep = np.ones(count, bool)
    keep[list(config.remove_indices)] = False
    if not keep.any():
        raise ValueError('At least one projection must remain')
    projections, frames, angles = projections[:, :, keep], frames[:, :, keep], angles[keep]
    if centers is None:
        centers = detect_crop_centers(projections, config.crop_size)
    else:
        centers = np.asarray(centers)
        if len(centers) == count:
            centers = centers[keep]
        if centers.shape != (keep.sum(), 2) or not np.isfinite(centers).all():
            raise ValueError('centers must have shape (retained views, 2) or (all views, 2)')
    cropped = crop_stack(projections, config.crop_size, centers)
    cropped_frames = np.stack([crop_stack(frames[:, :, :, k], config.crop_size, centers)
                               for k in range(frames.shape[3])], -1)
    return PreparedProjections(cropped, cropped_frames, angles, np.asarray(centers))


__all__ = ['load_emd', 'load_mat_stack', 'register_frames', 'detect_crop_centers',
           'crop_stack', 'prepare_acquisition', 'PreparedProjections']
