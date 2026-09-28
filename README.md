# pyAET

pyAET is a Python implementation of an atomic electron tomography workflow for
3D reconstruction, atom tracing, species classification, and position
refinement. The main workflow is organized as four directly runnable steps. The
reconstruction, tracing, and position-refinement scripts support CPU and CUDA.
Projection preprocessing and atom classification run on CPU.

## Contents

- [Installation](#installation)
- [CPU FFT dependency](#cpu-fft-dependency)
- [Projection preprocessing](#projection-preprocessing)
- [Reconstruction and atom-analysis workflow](#workflow)
- [Device selection](#device-selection)
- [CPU thread controls](#cpu-thread-controls)

## Features

- Six-stage projection preprocessing with independent callable functions.
- Torch-based RESIRE reconstruction for CPU and CUDA.
- Torch-based polynomial atom tracing with cubic-spline upsampling.
- K-means based atom classification for traced atomic coordinates.
- Position refinement with MATLAB-style trust-region H/B fitting and optimized
  projection kernels.
- C++ spline interpolation extension used by the reconstruction and
  classification routines.
- Analysis utilities for reconstruction slice and projection comparisons.

## Repository Layout

```text
run_image_processing.py             # Projection preprocessing entry point
preprocessing/                     # Preprocessing functions and numbered scripts
pyaet/
  main_reconstruction1_torch.py      # Step1: reconstruction
  main_polynomial_tracing2.py        # Step2: atom tracing
  main_classification3.py            # Step3: atom classification
  main_position_refinement4.py       # Step4: position refinement
  resire_torch/                      # Torch reconstruction backend
  splinterp_cpp/                     # C++ interpolation extension
  src/                               # Shared numerical kernels and helpers
  analysis/                          # Plotting and analysis utilities
  old/                               # Archived legacy scripts and NumPy backend
tests/
docs/
requirements.txt
```

The active workflow is the Torch CPU/CUDA path. The archived files under
`pyaet/old/` are kept for reference and are not the default execution path.

## Installation

Use Python 3.10 or newer. Run the commands from the repository root.

```bash
conda create -n pyaet python=3.10 -y
conda activate pyaet
python -m pip install -e .
```

For CPU preprocessing, also install both FFTW libraries and run the
[installation check](#configure-and-check-the-installation).

Install the PyTorch wheel that matches your machine. For CUDA machines, use the
official PyTorch CUDA wheel index for the installed CUDA driver/runtime.

Build the C++ interpolation extension:

```bash
cd pyaet/splinterp_cpp
python setup.py build_ext --inplace
cd ../..
```

The extension must import successfully before running reconstruction or
classification:

```bash
python - <<'PY'
from pyaet.splinterp_cpp import mex_function1, mex_function2, mex_function3
print("splinterp_cpp ready")
PY
```

The installation commands below target Linux and macOS. On Windows, use a
Linux environment such as WSL. Installing the Python package alone does not
install the FFTW shared libraries.

### CPU FFT dependency

CPU Fourier transforms require both FFTW single- and double-precision shared
libraries. Choose one installation method.

#### Linux: Debian or Ubuntu packages

```sh
sudo apt-get update
sudo apt-get install libfftw3-single3 libfftw3-double3
```

On other Linux distributions, install the equivalent FFTW runtime packages.

#### macOS: Homebrew

```sh
brew install fftw
export AET_FFTW_SINGLE_LIBRARY="$(brew --prefix fftw)/lib/libfftw3f.dylib"
export AET_FFTW_DOUBLE_LIBRARY="$(brew --prefix fftw)/lib/libfftw3.dylib"
```

#### Build from source without administrator access

The supplied script builds FFTW 3.3.8 in both precisions. It requires a C
compiler, make, curl, tar and a SHA-256 utility. Choose a new absolute install
directory; the script rejects existing directories.

```sh
sh preprocessing/build_parity_fftw.sh "$HOME/.local/pyaet-fftw-3.3.8"
export AET_FFTW_SINGLE_LIBRARY="$HOME/.local/pyaet-fftw-3.3.8/lib/libfftw3f.so"
export AET_FFTW_DOUBLE_LIBRARY="$HOME/.local/pyaet-fftw-3.3.8/lib/libfftw3.so"
```

On macOS, replace `.so` with `.dylib`.

#### Configure and check the installation

Run the check below after installation. If you used Homebrew or built FFTW from
source, run the corresponding `export` commands in the same terminal first.
Add those commands to your shell profile if you want to reuse the configuration
in future terminal sessions.

```sh
python - <<'PYTHON'
import numpy as np
from pyaet.fft_backend import centered_fftn, centered_ifftn

for dtype, tolerance in ((np.float32, 1e-5), (np.float64, 1e-12)):
    image = np.arange(35, dtype=dtype).reshape(5, 7)
    restored = centered_ifftn(centered_fftn(image))
    np.testing.assert_allclose(restored.real, image, atol=tolerance, rtol=tolerance)
print("FFTW single- and double-precision transforms ready")
PYTHON
```

If the check fails, confirm that FFTW is installed and that the two library
paths in your `export` commands point to existing files.

The optional CHOLMOD-compatible solver can be installed with
`python -m pip install -e '.[parity]'`; select it using
`AET_REGIONFILL_BACKEND=cholmod`.

### Optional GPU acceleration

To enable the optional Triton acceleration for CUDA reconstruction, install:

```sh
python -m pip install -e ".[gpu-fft]"
```

## Data Layout

The default settings in the four main scripts expect this project layout:

```text
data/
  1_Measured_data/
    Projections.mat
    Angles.mat
  5_Position_refinement/
    input/
      Local_classification_coord_OriOri.mat
      Local_classification_type.mat
outputs/
```

Edit the input paths and parameters at the top of each step script, then run
the script from the repository root.

## Projection preprocessing

`preprocessing/data/` contains the lookup tables required by inverse
variance-stabilizing transforms after denoising. They are installed with the
package and loaded automatically; no separate download or configuration is needed.

Six stages prepare repeated experimental frames for reconstruction. These stages
are separate from the four reconstruction and atom-analysis steps. Complete the
[installation](#installation), including the [FFTW check](#configure-and-check-the-installation),
before running preprocessing.

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

Run the six scripts in order. Stage 1 reads the EMD file; subsequent stages
read the preceding checkpoint and write a new `.npz` file. Choose a new output
directory when restarting with different settings.

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

### Preprocessing output arrays

The main entry point writes `processed.npz` by default. The final stage
checkpoint is `checkpoints/06_export.npz`. Both contain `proj` (row, column, view), `angle`
(view, 3), `support` (3-D), `masks`, `crop_centers`, and noise diagnostics.
Load with `numpy.load(path, allow_pickle=False)`. Masks retain pre-transpose,
pre-final-crop coordinates and cannot be overlaid directly on `proj`.

### Preprocessing settings

- Stage 4 includes background subtraction and center-of-mass alignment.
- Common-line search runs by default, but its rotation is **not applied** unless
  `apply_commonline_rotation=True`.
- Denoising averages valid per-view noise parameters. The readable BM3D backend
  supports its normal-noise branch
  only (normalized sigma times 255 at most 40); unsupported inputs raise errors.
- Crop centers are zero-based `(row, column)`. Automatic center detection is
  available; provide explicit centers to use your own crop locations.

## Workflow

Each step has a user-editable configuration block near the top of the file:

```python
# ========================= User settings =========================
...
# ================================================================
```

Open the file, edit paths and key parameters there, then run it.

### Step1: Reconstruction

Step1 reconstructs a 3D volume from measured projection images and tilt angles
using the Torch RESIRE backend. Input projections should already be
preprocessed; the reconstruction script does not read raw EMD frames.

```bash
python pyaet/main_reconstruction1_torch.py
```

Important settings in `pyaet/main_reconstruction1_torch.py`:

| Setting | Meaning |
| --- | --- |
| `PROJECTIONS_FILE_PATH` | Input projection stack, usually `Projections.mat`. |
| `ANGLES_FILE_PATH` | Input tilt-angle file, usually `Angles.mat`. |
| `OUTPUT_STEM` | Output file stem for the reconstructed volume and timing files. |
| `DEVICE` | `"auto"`, `"cuda"`, or `"cpu"`. |
| `OVERSAMPLING_RATIO` | RESIRE oversampling ratio. Default: `4`. |
| `NUM_ITERATIONS` | Number of RESIRE reconstruction iterations. Default: `200`. |
| `MONITOR_LOOP_LENGTH` | Interval used when monitoring reconstruction progress. Default: `20`. |
| `BACKPROJ_BACKEND` | Backprojection implementation. Default: `"grid_sample"`. |
| `BACKPROJ_CHUNK_SIZE` | Projection chunk size used by the backprojection backend. |
| `PRECOMPUTE_BACKPROJ_GRID` | `None` chooses automatically; large-memory CUDA GPUs precompute grids. |
| `BACKPROJ_ROT_ON_DEMAND` | `None` chooses automatically; useful for CUDA GPUs with limited memory. |
| `INTERP3_CHUNK_SIZE` | 3D interpolation chunk size; `None` chooses automatically. |

Step1 writes:

```text
outputs/step1/MG_reconstruction_volume.npy
outputs/step1/MG_reconstruction_volume_summary.csv
outputs/step1/MG_reconstruction_volume_iter_timing.csv
```

### Step2: Atom Tracing

Step2 traces candidate atom positions from the reconstructed volume. It uses Torch for cubic-spline upsampling, local maxima detection, and
polynomial peak fitting. A support filter is applied after tracing to remove
positions outside the reconstructed object.

```bash
python pyaet/main_polynomial_tracing2.py
```

Important settings in `pyaet/main_polynomial_tracing2.py`:

| Setting | Meaning |
| --- | --- |
| `VOLUME_FILE_PATH` | Input reconstructed volume from Step1. |
| `OUTPUT_STEM` | Output file stem for traced atom coordinates. |
| `DEVICE` | `"auto"`, `"cuda"`, or `"cpu"`. |
| `MAX_NUM_TH` | Maximum number of local maxima considered during tracing. |
| `MIN_DIST_ANGSTROM` | Minimum allowed inter-atomic distance, in angstrom. |
| `TORCH_CHUNK` | Number of peaks processed per Torch chunk. |

Support-filter defaults are kept in the code and can be overridden by
environment variables when needed:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `PYAET_TRACING_THRESHOLD` | `1.0` | Intensity threshold for local maxima. |
| `PYAET_PIXEL_SIZE_ANGSTROM` | `0.347` | Pixel size used to convert distances. |
| `PYAET_TRACING_SEARCH_RAD` | `3` | Local polynomial fitting radius. |
| `PYAET_SUPPORT_TH_DIS_R_AFTERAV` | `0.9125` | Tight-support threshold parameter. |
| `PYAET_SUPPORT_BW_SIZE` | `50000` | Minimum support component size. |
| `PYAET_SUPPORT_DILATE_SIZE` | `15` | Support dilation size. |
| `PYAET_SUPPORT_ERODE1` | `13` | First support erosion size. |
| `PYAET_SUPPORT_ERODE2` | `18` | Second support erosion size. |

Step2 writes:

```text
outputs/step2/traced_model_inPixel.npy
```

### Step3: Atom Classification

Step3 classifies traced atom positions into chemical species using global and
local k-means classification on the reconstructed intensity volume. This step is
CPU-side and uses the same code in both CPU and GPU workflows.

```bash
python pyaet/main_classification3.py
```

Important settings in `pyaet/main_classification3.py`:

| Setting | Meaning |
| --- | --- |
| `VOLUME_FILE_PATH` | Input reconstructed volume from Step1. |
| `TRACED_MODEL_FILE_PATH` | Input traced atom coordinates from Step2. |
| `OUTPUT_STEM` | Output file stem for atom-type labels. |
| `NUM_SPECIES` | Number of atom species to classify. Default: `3`. |
| `LOCAL_RADIUS_ANGSTROM` | Radius used for local classification. Default: `10.0`. |

Step3 writes:

```text
outputs/step3/Local_classification_type.npy
```

### Step4: Position Refinement

Step4 refines atom coordinates against the measured projections. It fits
scattering amplitudes (H) and displacement parameters (B) with a trust-region
least-squares solver, then updates B parameters and atomic coordinates.

```bash
python pyaet/main_position_refinement4.py
```

Important settings in `pyaet/main_position_refinement4.py`:

| Setting | Meaning |
| --- | --- |
| `PROJECTIONS_FILE_PATH` | Input projection stack used for refinement. |
| `ANGLES_FILE_PATH` | Input tilt-angle file. |
| `MODEL_FILE_PATH` | Input coordinates, shape `(3, N)`. |
| `ATOMS_FILE_PATH` | Input atom-type labels. |
| `OUTPUT_STEM` | Output file stem for refined coordinates and comparison statistics. |
| `DEVICE` | `"auto"`, `"cuda"`, or `"cpu"`. |
| `NUM_OUTER_ITERATIONS` | Number of outer refinement iterations. Default: `10`. |

Important constants:

| Constant | Meaning |
| --- | --- |
| `INNER_GRADIENT_ITERATIONS` | Number of B-gradient and XYZ-gradient iterations per outer loop. Default: `10`. |
| `RESOLUTION_ANGSTROM` | Voxel resolution used by the projector. Default: `0.347`. |
| `Z_BY_TYPE` | Atomic numbers used for each classified species. Default: `[28, 45, 78]`. |
| `MATLAB_FTOL` | Least-squares solver tolerance. Default: `1e-12`. |

Step4 runs 10 outer iterations by default. Set `DEVICE` to select CPU or CUDA
execution.

Step4 writes:

```text
outputs/step4/model_refined_res.npy
outputs/step4/model_refined_res.mat
outputs/step4/model_refined_res_compare_stats.npz
outputs/step4/model_refined_res_compare_stats.mat
```

## Device Selection

Step1, Step2, and Step4 select the device from the `DEVICE` variable in each
script:

- `DEVICE = "auto"`: use CUDA when `torch.cuda.is_available()` is true, otherwise
  use CPU.
- `DEVICE = "cuda"`: require CUDA and fail immediately if CUDA is unavailable.
- `DEVICE = "cpu"`: force CPU execution.

Step3 is CPU-oriented because the classification path uses local intensity
statistics, k-means, and the C++ interpolation extension. It consumes the
Torch-generated Step1/Step2 outputs.

PyTorch is required for reconstruction, atom tracing, and position refinement.

For scripted batch runs, the same settings can be overridden from the command
line, for example:

```bash
python -m pyaet.main_reconstruction1_torch --device cuda
```

## CPU Thread Controls

For workstation or Slurm runs, set thread counts explicitly:

```bash
export PYAET_NUM_THREADS=64
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMBA_NUM_THREADS=64
export SPLINTERP_NUM_THREADS=64
```

Adjust these values to the CPU resources available for the run. Use at most
`64` OpenBLAS threads when the installed build has a 64-thread limit.

## Analysis Utilities

Reconstruction comparison plots are available in:

```text
pyaet/analysis/plot_reconstruction_comparison.py
```

The plotting utilities can generate orthogonal slice comparisons, maximum
intensity projections, and absolute-difference maps for two reconstruction
volumes.

## Optional GUI Dependencies

GUI dependencies are not required for the four-step workflow. Install them only
when working on the GUI:

```bash
pip install -r requirements-optional.txt
```
