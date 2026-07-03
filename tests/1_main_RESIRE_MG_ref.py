"""Step1 MG-style smoke run.

This script expects local MG projection and angle files. The repository does
not package the full MG raw projection input, so edit the paths below before
using this script for a full MG run.
"""

from pathlib import Path

from pyaet.main_reconstruction1_torch import main_reconstruction


ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_FILE = ROOT / "pyaet" / "input" / "1Projections.mat"
ANGLES_FILE = ROOT / "pyaet" / "input" / "1Angles.mat"

RESIRE_PARAM = {
    "oversampling_ratio": 4,
    "num_iterations": 5,
    "monitor_R": True,
    "monitorR_loopLength": 2,
    "gridding_method": 1,
    "vector3": [1, 0, 0],
    "use_parallel": True,
    "save_temp": False,
    "dtype": "float32",
    "gpu_grad_device": "auto",
}


if __name__ == "__main__":
    if not PROJECTIONS_FILE.exists() or not ANGLES_FILE.exists():
        raise FileNotFoundError(
            "MG raw projection files are not packaged. Place 1Projections.mat "
            "and 1Angles.mat under pyaet/input/ or edit this script."
        )
    main_reconstruction(
        str(PROJECTIONS_FILE),
        str(ANGLES_FILE),
        RESIRE_PARAM,
        "RESIRE_experiment_result_full",
    )
