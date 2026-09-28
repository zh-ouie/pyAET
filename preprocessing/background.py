"""Table 1 Laplacian/threshold background subtraction with legacy COM passes."""
from dataclasses import dataclass
import numpy as np
from .pipeline import pre_background_subtraction, matlab_bgsub_main, matlab_repre_bgsub


@dataclass
class BackgroundResult:
    projections: np.ndarray
    masks: np.ndarray
    original_masks: np.ndarray
    initial_original_masks: np.ndarray
    initial_shifts: np.ndarray
    final_shifts: np.ndarray


def subtract_background(projections, *, disk=140, otsu_scale=0.9):
    """Retain opening -> mask/COM -> regionfill -> mask/COM numerical order.

    Mask estimation and COM are coupled in the original MATLAB implementation;
    separating their execution would change the numerical algorithm.
    """
    preliminary, _, _ = pre_background_subtraction(projections, 1, disk)
    _, _, _, initial_mask, initial_shifts = matlab_bgsub_main(preliminary, otsu_scale)
    corrected, _ = matlab_repre_bgsub(projections, initial_mask)
    final, _, masks, original_masks, shifts = matlab_bgsub_main(corrected, otsu_scale)
    return BackgroundResult(final, masks, original_masks, initial_mask, initial_shifts, shifts)


__all__ = ['BackgroundResult', 'subtract_background', 'pre_background_subtraction',
           'matlab_repre_bgsub']
