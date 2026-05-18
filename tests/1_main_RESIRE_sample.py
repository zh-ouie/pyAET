"""Small Step1 reconstruction smoke run using packaged sample inputs."""

from pathlib import Path

from pyaet.main_reconstruction1_numpy import main_reconstruction


ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_FILE = ROOT / "pyaet" / "input" / "sample_projections_amorphous.mat"
ANGLES_FILE = ROOT / "pyaet" / "input" / "sample_angles_amorphous.mat"

RESIRE_PARAM = {
    "oversampling_ratio": 3,
    "num_iterations": 2,
    "monitor_R": True,
    "monitorR_loopLength": 1,
    "gridding_method": 1,
    "vector3": [1, 0, 0],
    "use_parallel": True,
    "save_temp": False,
    "dtype": "float32",
}


if __name__ == "__main__":
    main_reconstruction(
        str(PROJECTIONS_FILE),
        str(ANGLES_FILE),
        RESIRE_PARAM,
        "sample_amorphous_res",
    )
