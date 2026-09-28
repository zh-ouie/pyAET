# Image preprocessing

Six stages prepare repeated experimental frames for reconstruction. The stages
retain the existing numerical order and share `pyaet/fft_backend.py` with PyAET.
Reconstruction and orientation refinement are separate operations.

## Install and run

From the repository root, install into your Python environment:

```sh
python -m pip install -e .
```

Edit the parameters in `run_image_processing.py`, then run the public entry
point from the repository root:

```sh
python run_image_processing.py
```

The complete workflow writes the processed projections and optional stage
checkpoints. The six auxiliary scripts in
`preprocessing/preprocessing_steps/` expose the same workflow stage by stage.
Edit `preprocessing/preprocessing_steps/settings.py` before using them. Paths
are relative to the current working directory:

```sh
python preprocessing/preprocessing_steps/1_frame_registration_drift_correction_crop.py
python preprocessing/preprocessing_steps/2_dark_current_noise_estimation.py
python preprocessing/preprocessing_steps/3_vst_bm3d_denoising.py
python preprocessing/preprocessing_steps/4_background_subtraction_com_alignment.py
python preprocessing/preprocessing_steps/5_normalization_commonline_analysis.py
python preprocessing/preprocessing_steps/6_export_reconstruction_inputs.py
```

Each script loads the preceding checkpoint and writes a new `.npz` file.
Existing outputs are never overwritten. After changing settings, select a new
output directory and restart at stage 1. The checkpoint files produced by these
scripts contain settings metadata; do not substitute checkpoints from other
runners without adapting their schema.

| Stage | Implementation and callable API | Input | Output / processing |
| --- | --- | --- | --- |
| 1. Frame registration, drift correction and crop | `preprocessing/acquisition.py`: `prepare_acquisition` | Raw `(frame, view, row, column)`, measured angles, optional centers | Register repeated frames, correct scan drift, select views and crop; frames `(row, column, view, frame)` |
| 2. Dark current and noise estimation | `preprocessing/noise.py`: `estimate_projection_noise` | Cropped frames and angles | Correct dark current, estimate alpha/sigma, sum frames, fit dark-current trend |
| 3. VST and BM3D denoising | `preprocessing/denoising.py`: `denoise_projections` | Summed projections and noise parameters | Forward VST, scaling, BM3D hard/Wiener stages, inverse VST and intensity restoration |
| 4. Background subtraction and COM alignment | `preprocessing/background.py`: `subtract_background` | Denoised projections | Preliminary background removal, masks/COM, Laplacian background removal, second masks/COM pass |
| 5. Normalization and common-line analysis | `preprocessing/alignment.py`: `normalize_and_align` | Background-corrected projections | Transpose, normalize to first-view intensity, search common-line angle; optionally apply rotation |
| 6. Reconstruction input export | `preprocessing/export.py`: `prepare_reconstruction_input` | Aligned projections, angles and masks | Select views, final crop, angle table and spherical support |

The functions in the table can also be imported independently. For example:

```python
from preprocessing import denoise_projections
filtered = denoise_projections(projections, alpha, sigma, backend="readable")
```

`preprocessing/numbered_steps.py` exposes finer-grained callable operations
inside these six stages.

## Output arrays

`checkpoints/06_export.npz` contains `proj` (row, column, view), `angle`
(view, 3), `support` (3-D), `masks`, `crop_centers`, and noise diagnostics.
Load with `numpy.load(path, allow_pickle=False)`. Masks retain pre-transpose,
pre-final-crop coordinates and cannot be overlaid directly on `proj`.

## Numerical behavior

- Background masks and center-of-mass alignment are coupled in stage 4. Do not
  add another COM pass when reproducing this workflow.
- Common-line search runs by default, but its rotation is **not applied** unless
  `apply_commonline_rotation=True`.
- Denoising averages valid per-view noise parameters, preserving the original
  workflow. The readable BM3D backend supports its normal-noise branch
  only (normalized sigma times 255 at most 40); unsupported inputs raise errors.
- Crop centers are zero-based `(row, column)`. Automatic center detection is
  available; explicit centers make repeated comparisons reproducible.
- Matching numerical results depends on data, floating-point libraries and
  settings. Tests on one dataset do not establish bitwise identity for all data.

The optional `build_parity_fftw.sh` builds a pinned FFTW library for reproducible FFT computation. To enable this backend:

```sh
sh preprocessing/build_parity_fftw.sh /absolute/new/fftw-prefix
export AET_REGISTRATION_FFTW_LIBRARY=/absolute/new/fftw-prefix/lib/libfftw3f.so
```

The build requires a C compiler, make and curl. Linux x86-64 uses SSE2 codelets
by default. The optional CHOLMOD-compatible solver can be installed with
`python -m pip install -e '.[parity]'`; select it using
`AET_REGIONFILL_BACKEND=cholmod`. Leave backend variables unset for portable
fallbacks, which can have different rounding behavior.
