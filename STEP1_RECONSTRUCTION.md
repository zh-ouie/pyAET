# Step1 Reconstruction

This repository currently keeps two parallel Step1 reconstruction lines:

- `pyaet/resire_numpy/`: stable NumPy/CPP reconstruction line
- `pyaet/resire_torch/`: torch line scaffold for future GPU implementation

For now, the supported Step1 path is the NumPy/CPP line.

## Status of the old `pyaet/resire/`

The older `pyaet/resire/` implementation has been removed from this repository.

Use these two explicit Step1 lines instead:

- `pyaet/resire_numpy/`
- `pyaet/resire_torch/`

## Environment

Recommended:

```bash
conda create -n pyaet python=3.10 -y
conda activate pyaet
pip install numpy scipy psutil pybind11
```

Optional FFT acceleration:

```bash
pip install pyfftw
```

## Build the C++ interpolation extension

The NumPy reconstruction line expects `pyaet.splinterp_cpp` to be importable.
The compiled extension is required for the release path; the old Python
interpolation fallback is not used.

From the repository root:

```bash
cd pyaet/splinterp_cpp
python setup.py build_ext --inplace
cd ../..
```

After that, this import should succeed:

```bash
python -c "from pyaet.splinterp_cpp import mex_function2, mex_function3; print('splinterp_cpp ok')"
```

If the extension is not available, build it before running reconstruction.

## Run the NumPy version directly

Edit the parameter block in:

- `pyaet/main_reconstruction1_numpy.py`

The script defaults are set to the formal MATLAB MG reconstruction parameters:

- `oversampling_ratio = 4`
- `num_iterations = 200`
- `monitorR_loopLength = 20`
- `vector3 = [1, 0, 0]`
- `use_parallel = True`
- `dtype = float32`

Then run from the repository root:

```bash
python pyaet/main_reconstruction1_numpy.py
```

Outputs will be written next to the projection file path using `OUTPUT_FN`.

The script now calls a reusable wrapper:

```python
main_reconstruction(
    PROJECTIONS_FILE_PATH,
    ANGLES_FILE_PATH,
    RESIRE_PARAM,
    OUTPUT_FN,
)
```

Internally this wrapper still uses the same `RESIRE_Reconstructor` workflow:

```python
resire.read_files()
resire.check_prepare_data()
resire.run_gridding()
reconstruct(resire)
```

These methods remain available in `pyaet/resire_numpy/RESIRE_Reconstructor.py`.

## Run on your own MG projections

This repository packages only the sample Step1 input files:

- `pyaet/input/sample_projections_amorphous.mat`
- `pyaet/input/sample_angles_amorphous.mat`

The file `pyaet/input/MG_reconstruction_volume.mat` is already a reconstruction volume and is
used by later steps. It is not the raw Step1 projection input.

For MG Step1, replace the file paths and parameters in:

- `pyaet/main_reconstruction1_numpy.py`

and run:

```bash
python pyaet/main_reconstruction1_numpy.py
```

## Threading note

The compiled interpolation extension uses the `SPLINTERP_NUM_THREADS` environment variable.

Example:

```bash
export SPLINTERP_NUM_THREADS=16
```

If unset, the extension will use its compiled default thread count.

## Torch line

Edit the parameter block in:

- `pyaet/main_reconstruction1_torch.py`

The torch entry uses the same formal MATLAB MG parameter defaults as the numpy entry.

and run:

```bash
python pyaet/main_reconstruction1_torch.py
```

This line uses the same gridding/preparation path as the NumPy/CPP version, but performs the
gradient accumulation and descent update with torch when available.

Set this field in `RESIRE_PARAM` to choose the torch device:

```python
"gpu_grad_device": "cuda"
```

If CUDA is unavailable, the script will fall back to CPU.
