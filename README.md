# pyAET

`pyAET` is a Python implementation of the AET workflow used in this project.
The repository is organized around the main processing steps:

1. Step1: reconstruction
2. Step2: tracing
3. Step3: classification
4. Step4: position refinement

The release goal is practical parity with the validated MATLAB workflow while
keeping the code runnable as a regular Python project.

## Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Workflow](#workflow)
- [Dependencies](#dependencies)
- [Build](#build)
- [Usage](#usage)
- [GUI Development](#gui-development)
- [Analysis](#analysis)

## Overview

The codebase keeps the verified MATLAB-aligned logic and the fast backends
where they are useful. Each step has its own entry script so it can be run
independently.

## Repository Layout

```text
pyaet/
  main_reconstruction1_numpy.py
  main_reconstruction1_torch.py
  main_polynomial_tracing2.py
  main_classification3.py
  main_position_refinement4.py
  resire_numpy/
  resire_torch/
  splinterp_cpp/
  src/
  analysis/
  output/
tests/
docs/
```

## Workflow

### Step1: Reconstruction

Reconstruct a 3D volume from measured tilt-series projections and angles.

Entry:

```text
pyaet/main_reconstruction1_numpy.py
```

The C++ interpolation extension is required for the release path.

### Step2: Tracing

Trace atomic candidates from the reconstructed volume and apply support-based
filtering.

Entry:

```text
pyaet/main_polynomial_tracing2.py
```

### Step3: Classification

Classify traced atoms into species using the validated local/global
k-means-based pipeline.

Entry:

```text
pyaet/main_classification3.py
```

### Step4: Position Refinement

Refine the atomic coordinates against measured projections. The refinement
pipeline includes:

- H/B fitting
- `gradient_B_2type_difB`
- `gradient_fixHB_XYZ`

Entry:

```text
pyaet/main_position_refinement4.py
```

Step4 keeps two projector implementations:

- `reference` for MATLAB-like numerical behavior
- `fast` for Numba-backed speed

The current release sets the backend directly in code.

## Dependencies

Recommended environment:

```bash
conda create -n pyaet python=3.10 -y
conda activate pyaet
pip install -r requirements.txt
```

## Build

Build the C++ interpolation extension from the repository root:

```bash
cd pyaet/splinterp_cpp
python setup.py build_ext --inplace
cd ../..
```

## Usage

Run each step through its main script.

Example:

```bash
python pyaet/main_reconstruction1_numpy.py
python pyaet/main_polynomial_tracing2.py
python pyaet/main_classification3.py
python pyaet/main_position_refinement4.py
```

## GUI Development

The repository also keeps GUI-related development notes. The GUI is separate
from the main AET workflow and uses PyQt5.

Install GUI dependencies only when needed:

```bash
pip install PyQt5
pip install PyQt5-tools
```

Typical GUI workflow:

1. Run `pyqt5-tools designer` to open Qt Designer.
2. Design the interface and save it as a `.ui` file.
3. Convert the `.ui` file to Python code:

```bash
pyuic5 -x yourfile.ui -o yourfile.py
```

## Analysis

```text
pyaet/analysis/plot_reconstruction_comparison.py
```

## Notes

- Step1, Step2, Step3, and Step4 each have their own main script.
- The release keeps the validated MATLAB-aligned logic.
- The fast backend is available for Step4 when you want speed.
