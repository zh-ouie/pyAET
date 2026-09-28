"""Paper-organized orchestration with optional inspectable stage checkpoints."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import numpy as np
from .pipeline import PreprocessConfig, PreprocessResult
from .acquisition import prepare_acquisition
from .noise import estimate_projection_noise
from .denoising import denoise_projections
from .background import subtract_background
from .alignment import normalize_and_align
from .export import prepare_reconstruction_input
from ._regionfill_backend import regionfill_backend_info


def matlab_compatible_config(**overrides):
    """Select the readable BM3D implementation and retain legacy step settings.

    Dataset sizes, view selection and crop centers still require user input.
    Backend selection alone does not guarantee bitwise MATLAB equivalence.
    """
    options = dict(bm3d_backend='readable')
    options.update(overrides)
    return PreprocessConfig(**options)


def run_preprocessing(raw, angles, centers=None, config=None, *, checkpoint_dir=None):
    """Run image preprocessing.

    An existing checkpoint directory is rejected to protect existing outputs.
    Each NPZ can be loaded with allow_pickle=False and reused with stage APIs.
    """
    cfg = config if config is not None else matlab_compatible_config()
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    manifest = dict(schema_version=1, paper_doi='10.1038/s41586-025-09857-4',
                    scope='Table 1 preprocessing organized around legacy MATLAB implementation',
                    config=asdict(cfg), complete=False, stages=[],
                    registration_fftw_library=os.environ.get('AET_REGISTRATION_FFTW_LIBRARY'),
                    regionfill=regionfill_backend_info(),
                    mask_coordinates='pre-transpose and pre-export-crop',
                    parity_claim='Numerical parity depends on data and backends; arbitrary-input bitwise identity is not guaranteed')
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def checkpoint(name, **arrays):
        if directory is None:
            return
        np.savez_compressed(directory/f'{name}.npz', **arrays)
        manifest['stages'].append(dict(name=name, file=f'{name}.npz',
            arrays={key: dict(shape=list(np.asarray(value).shape), dtype=str(np.asarray(value).dtype))
                    for key, value in arrays.items()}))
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    prepared = prepare_acquisition(raw, angles, cfg, centers)
    checkpoint('01_acquisition', projections=prepared.projections, frames=prepared.frames,
               angles=prepared.angles, centers=prepared.centers)
    noise = estimate_projection_noise(prepared.frames, prepared.angles)
    checkpoint('02_noise', summed=noise.summed, dark=noise.dark, alpha=noise.alpha,
               sigma=noise.sigma, dark_fit_parameters=noise.dark_fit['parameters'],
               dark_fit_values=noise.dark_fit['fitted'])
    denoised = denoise_projections(noise.summed, noise.alpha, noise.sigma, backend=cfg.bm3d_backend)
    checkpoint('03_denoising', projections=denoised)
    background = subtract_background(denoised, disk=cfg.background_disk, otsu_scale=cfg.otsu_scale)
    checkpoint('04_background', projections=background.projections, masks=background.masks,
               original_masks=background.original_masks, initial_original_masks=background.initial_original_masks,
               initial_shifts=background.initial_shifts, final_shifts=background.final_shifts)
    alignment = normalize_and_align(
        background.projections,
        angle_range=cfg.commonline_range,
        apply_rotation=cfg.apply_commonline_rotation,
        workers=cfg.commonline_workers,
    )
    checkpoint('05_alignment', projections=alignment.projections,
               commonline_angle=alignment.commonline_angle, commonline_scores=alignment.commonline_scores,
               rotation_applied=alignment.rotation_applied)
    projections, angles, masks, support = prepare_reconstruction_input(
        alignment.projections, prepared.angles, background.masks, cfg)
    result = PreprocessResult(projections, angles, masks, support,
        dict(dark=noise.dark, alpha=noise.alpha, sigma=noise.sigma,
             dark_fit_parameters=noise.dark_fit['parameters'], dark_fit_values=noise.dark_fit['fitted'],
             dark_fit_resnorm=np.asarray(noise.dark_fit['resnorm']),
             commonline_angle=np.asarray(alignment.commonline_angle),
             commonline_scores=alignment.commonline_scores), prepared.centers)
    checkpoint('06_export', proj=projections, angle=angles, masks=masks, support=support,
               crop_centers=prepared.centers, **result.noise)
    if directory is not None:
        manifest['complete'] = True
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return result
