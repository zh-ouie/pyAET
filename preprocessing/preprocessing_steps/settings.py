"""Edit the parameters below, then run the numbered scripts in order."""
from pathlib import Path
from preprocessing import matlab_compatible_config
INPUT_FILE = Path("input.emd")
CENTERS_FILE = None  # Optional .npy file: zero-based (row, column), one per view.
OUTPUT_DIR = Path("checkpoints")
CONFIG = matlab_compatible_config(
    registration_size=550, crop_size=300, output_size=210,
    remove_indices=(), final_indices=None, trim_edge_projections=True,
    apply_commonline_rotation=False,
)
