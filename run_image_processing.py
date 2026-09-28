"""Edit the parameters below, then run this file."""
from pathlib import Path
import numpy as np
from preprocessing import (
    load_emd, matlab_compatible_config, run_preprocessing, save_result,
)


INPUT_FILE = Path("input.emd")
OUTPUT_FILE = Path("processed.npz")
CENTERS_FILE = None  # Optional Path to a .npy array of zero-based (row, column) centers.
CHECKPOINT_DIR = Path("checkpoints")

# These dimensions and view selections must match the experiment.
CONFIG = matlab_compatible_config(
    registration_size=550,
    crop_size=300,
    output_size=210,
    remove_indices=(),
    final_indices=None,
    trim_edge_projections=True,
    apply_commonline_rotation=False,
)


def main():
    if OUTPUT_FILE.exists() or OUTPUT_FILE.suffix.lower() != ".npz":
        raise ValueError("OUTPUT_FILE must be a new .npz file")
    raw, angles = load_emd(INPUT_FILE)
    centers = np.load(CENTERS_FILE, allow_pickle=False) if CENTERS_FILE else None
    result = run_preprocessing(raw, angles, centers, CONFIG,
                               checkpoint_dir=CHECKPOINT_DIR)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_result(OUTPUT_FILE, result)
    print(f"Saved {result.projections.shape[2]} projections to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
