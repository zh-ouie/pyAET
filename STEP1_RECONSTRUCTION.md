# Step1 Reconstruction

Step1 reconstructs a 3D volume from measured projections and tilt angles. The
active Step1 implementation is:

```text
pyaet/main_reconstruction1_torch.py
```

It uses the Torch reconstruction path on both CPU and CUDA. CUDA is selected
automatically when available, and CPU is used otherwise.

## Build Requirement

Build the C++ interpolation extension before running Step1:

```bash
cd pyaet/splinterp_cpp
python setup.py build_ext --inplace
cd ../..
```

## Run

```bash
python -m pyaet.main_reconstruction1_torch \
  --device auto \
  --projections data/1_Measured_data/Projections.mat \
  --angles data/1_Measured_data/Angles.mat \
  --output-stem outputs/step1/MG_reconstruction_volume
```

Use `--device cpu` to force CPU execution or `--device cuda` to require CUDA.

## Main Parameters

- `--iterations`: RESIRE reconstruction iterations.
- `--oversampling-ratio`: gridding oversampling ratio.
- `--backproj-backend`: backprojection backend, default `grid_sample`.
- `--save-pickle`: also write the legacy pickle-style output.

## Outputs

Step1 writes the reconstructed volume and timing summaries under the selected
output stem:

```text
<output-stem>.npy
<output-stem>_summary.csv
<output-stem>_iter_timing.csv
<output-stem>_iter_descent_update_only.csv
```

The `.npy` volume is the input for Step2 tracing.
