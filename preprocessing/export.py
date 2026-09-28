"""Reconstruction handoff, separate from the paper's image-processing stages."""
import numpy as np
from .pipeline import strip_center_stack
from .io import save_npz


def prepare_reconstruction_input(projections, angles, masks, config):
    """Return the legacy export arrays, preserving its diagnostic-mask axes.

    Masks stay in pre-transpose, pre-final-crop coordinates for compatibility.
    They must not be interpreted as pixel-aligned masks of exported projections.
    """
    angles = np.asarray(angles)
    if config.matlab_angle_table and angles.ndim == 1:
        table = np.zeros((len(angles), 3), dtype=np.float64)
        table[:, 1] = angles
        angles = table
    if config.final_indices is not None:
        indices = list(config.final_indices)
        projections, angles, masks = projections[:, :, indices], angles[indices], masks[:, :, indices]
    elif config.trim_edge_projections and projections.shape[2] > 2:
        projections, angles, masks = projections[:, :, 1:-1], angles[1:-1], masks[:, :, 1:-1]
    projections = strip_center_stack(projections, config.output_size)
    d = config.output_size
    grid = np.indices((d, d, d), dtype=float)-d/2
    support = ((grid**2).sum(0) <= (d/2)**2).astype(np.float32)
    return projections, angles, masks, support


def save_result(path, result):
    save_npz(path, proj=result.projections, angle=result.angles, support=result.support,
             masks=result.masks, crop_centers=result.crop_centers, **result.noise)


__all__ = ['prepare_reconstruction_input', 'save_result']
