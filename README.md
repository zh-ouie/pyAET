# pyAET

`pyAET` is a Python implementation of the Step1 atomic electron tomography (AET)
reconstruction workflow. It reconstructs a 3D volume from measured tilt-series
projections and projection angles using the RESIRE-style iterative reconstruction
pipeline.

This release keeps the reconstruction workflow script-driven and close to the
original MATLAB usage: edit the parameter block near the top of the main script,
then run the script directly.

## Step1 Reconstruction

The supported reconstruction path is the NumPy/C++ implementation:

```text
pyaet/main_reconstruction1_numpy.py
pyaet/resire_numpy/
pyaet/splinterp_cpp/
```

The torch line is also included for GPU-oriented development:

```text
pyaet/main_reconstruction1_torch.py
pyaet/resire_torch/
```

The C++ interpolation extension is required. The old Python interpolation fallback
has been removed from the release path so the code uses the same compiled
interpolation kernels used in the validated workstation runs.

## Repository Layout

```text
pyaet/
  main_reconstruction1_numpy.py   # NumPy/C++ Step1 entry point
  main_reconstruction1_torch.py   # torch Step1 entry point
  resire_numpy/                   # NumPy reconstruction implementation
  resire_torch/                   # torch reconstruction implementation
  splinterp_cpp/                  # pybind11 C++ interpolation extension
  analysis/
    plot_reconstruction_comparison.py
tests/                            # Step1 example/smoke scripts
```

## Installation

Recommended environment:

```bash
conda create -n pyaet python=3.10 -y
conda activate pyaet
pip install numpy scipy matplotlib psutil pybind11
```

Optional dependencies:

```bash
pip install pyfftw
pip install torch
```

## Build the C++ Extension

Build the interpolation extension from the repository root:

```bash
cd pyaet/splinterp_cpp
python setup.py build_ext --inplace
cd ../..
```

Check the build:

```bash
python -c "from pyaet.splinterp_cpp import mex_function1, mex_function2, mex_function3; print('splinterp_cpp ok')"
```

The `setup.py` file only controls how the pybind11 extension is compiled. The
public Python interface remains:

```python
from pyaet.splinterp_cpp import mex_function1, mex_function2, mex_function3
```

## Run Reconstruction

Edit the parameter block in:

```text
pyaet/main_reconstruction1_numpy.py
```

Then run:

```bash
python pyaet/main_reconstruction1_numpy.py
```

The entry script wraps the same reconstruction class workflow used before:

```python
resire = RESIRE_Reconstructor()
resire.read_files()
resire.check_prepare_data()
resire.run_gridding()
reconstruct(resire)
```

Those class methods are still present in `pyaet/resire_numpy/RESIRE_Reconstructor.py`.
The main script now calls them through `main_reconstruction(...)` so tests and
other scripts can reuse the same entry point.

## Inputs and Outputs

Step1 expects:

- projection data, shaped as image height x image width x number of projections
- angle data, shaped as number of projections x 3

Small sample inputs are included:

```text
pyaet/input/sample_projections_amorphous.mat
pyaet/input/sample_angles_amorphous.mat
```

Outputs are saved next to the projection file path using the configured
`OUTPUT_FN`.

## Threading

The C++ interpolation extension reads `SPLINTERP_NUM_THREADS`.

Example:

```bash
export SPLINTERP_NUM_THREADS=16
python pyaet/main_reconstruction1_numpy.py
```

## Analysis Utility

Use the reconstruction comparison utility for visual checks:

```text
pyaet/analysis/plot_reconstruction_comparison.py
```

## Validation

Before committing, run:

```bash
python -m py_compile \
  pyaet/main_reconstruction1_numpy.py \
  pyaet/main_reconstruction1_torch.py \
  pyaet/resire_numpy/RESIRE_Reconstructor.py \
  pyaet/resire_numpy/reconstruct.py \
  pyaet/resire_numpy/interp_pj_realspace.py \
  pyaet/splinterp_cpp/__init__.py \
  pyaet/splinterp_cpp/setup.py
```

## License

See [LICENSE](./LICENSE).
