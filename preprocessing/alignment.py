"""Table 1 projection COM alignment, common-line analysis and normalization."""
from dataclasses import dataclass
import numpy as np
from .pipeline import matlab_bgsub_main, image_norm, commonline_align


@dataclass
class AlignmentResult:
    projections: np.ndarray
    commonline_angle: float
    commonline_scores: np.ndarray
    rotation_applied: bool


def align_center_of_mass(projections, otsu_scale=0.9):
    """Return projections, shifted images, masks, original masks and COM shifts."""
    return matlab_bgsub_main(projections, otsu_scale)


def normalize_and_align(projections, *, angle_range=(-5, 5, .1), apply_rotation=False,
                        workers=1):
    """Apply the MATLAB axis transpose and first-view total-intensity convention."""
    projections = np.transpose(projections, (1, 0, 2))
    normalized = image_norm(projections)
    rotated, angle, scores = commonline_align(normalized, angle_range, workers=workers)
    return AlignmentResult(rotated if apply_rotation else normalized, angle, scores, apply_rotation)


__all__ = ['AlignmentResult', 'align_center_of_mass', 'normalize_and_align',
           'image_norm', 'commonline_align']
