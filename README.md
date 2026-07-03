# pyAET

pyAET is a Python implementation of an atomic electron tomography workflow for
3D reconstruction, atom tracing, species classification, and position
refinement. The main workflow is organized as four directly runnable steps. The
same step scripts run on CUDA when a compatible GPU is available and on CPU
otherwise.

## Features

- Torch-based RESIRE reconstruction for CPU and CUDA.
- Torch-based polynomial atom tracing with exact cubic-spline upsampling.
- K-means based atom classification for traced atomic coordinates.
- Position refinement with MATLAB-style trust-region H/B fitting and optimized
  projection kernels.
- C++ spline interpolation extension used by the reconstruction and
  classification routines.
- Analysis utilities for reconstruction slice and projection comparisons.

## Repository Layout

```text
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

Use Python 3.10 or newer.

```bash
conda create -n pyaet python=3.10 -y
conda activate pyaet
pip install -r requirements.txt
```

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

For a normal run, edit the user-settings block at the top of each step script
and run the script directly. Optional command-line arguments are still available
for batch or Slurm jobs, but they are not required for ordinary use.

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
using the Torch RESIRE backend. It is the only step that directly consumes the
raw projection tilt series.

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

Step2 traces candidate atom positions from the reconstructed volume. The active
tracing path uses Torch for cubic-spline upsampling, local maxima detection, and
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

Step4 refines atom coordinates against the measured projections. The default
settings follow the MATLAB-aligned refinement path: H/B fitting with a
trust-region LSQ solver, followed by B-gradient and XYZ-gradient updates.

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
| `MATLAB_FTOL` | MATLAB-aligned LSQ tolerance. Default: `1e-12`. |

Step4 uses the MATLAB-aligned refinement settings by default: 10 outer
iterations, full LSQ function evaluations, CUDA paired-stats projector on GPU,
and the release-parity CPU projector when CUDA is unavailable.

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

PyTorch is a required dependency. Missing PyTorch is treated as an environment
error rather than falling back to an older NumPy entry path.

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

Use `64` or lower for OpenBLAS builds that were compiled with a 64-thread
metadata limit.

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
