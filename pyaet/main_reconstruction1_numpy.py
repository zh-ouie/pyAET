import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pyaet.main_reconstruction1 import main_reconstruction


# Edit this block before running the script directly.
# These defaults are aligned with the formal MATLAB MG reconstruction settings.
# The repository does not package the raw MG Projections/Angles files, so set
# these paths to your local MG inputs before running.
PROJECTIONS_FILE_PATH = "/path/to/Projections.mat"
ANGLES_FILE_PATH = "/path/to/Angles.mat"
OUTPUT_FN = "MG_reconstruction_volume"

RESIRE_PARAM = {
    "oversampling_ratio": 4,
    "num_iterations": 200,
    "monitor_R": True,
    "monitorR_loopLength": 20,
    "gridding_method": 1,
    "vector3": [1, 0, 0],
    "use_parallel": True,
    "dtype": "float32",
}


if __name__ == "__main__":
    main_reconstruction(
        PROJECTIONS_FILE_PATH,
        ANGLES_FILE_PATH,
        RESIRE_PARAM,
        OUTPUT_FN,
    )


__all__ = ["main_reconstruction", "PROJECTIONS_FILE_PATH", "ANGLES_FILE_PATH", "OUTPUT_FN", "RESIRE_PARAM"]
