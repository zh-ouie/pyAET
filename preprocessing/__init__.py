"""Python implementation of the legacy AET projection pre-processing pipeline."""

from .pipeline import PreprocessConfig, PreprocessResult, preprocess_stack
from .io import load_emd, load_mat_stack
from .parity import compare_arrays, compare_mat
from .workflow import run_preprocessing, matlab_compatible_config
from .acquisition import prepare_acquisition
from .noise import estimate_projection_noise
from .denoising import denoise_projections, prepare_bm3d_input, restore_bm3d_output
from .background import subtract_background
from .alignment import normalize_and_align
from .export import save_result, prepare_reconstruction_input

__all__ = ["PreprocessConfig", "PreprocessResult", "preprocess_stack", "run_preprocessing",
           "save_result", "load_emd", "load_mat_stack", "compare_arrays", "compare_mat",
           "matlab_compatible_config", "prepare_acquisition", "estimate_projection_noise",
           "denoise_projections", "prepare_bm3d_input", "restore_bm3d_output",
           "subtract_background", "normalize_and_align", "prepare_reconstruction_input"]
