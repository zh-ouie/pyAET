"""Backward-compatible alias for NumPy RESIRE reconstruction.

The canonical NumPy entry point is ``main_reconstruction1_numpy.py``. This
module remains so existing imports of ``pyaet.main_reconstruction1`` keep
working.
"""

from pyaet.main_reconstruction1_numpy import (
    ANGLES_FILE_PATH,
    OUTPUT_FN,
    PROJECTIONS_FILE_PATH,
    RESIRE_PARAM,
    RESIRE_Reconstructor,
    main_reconstruction,
    reconstruct,
)


if __name__ == "__main__":
    main_reconstruction(
        PROJECTIONS_FILE_PATH,
        ANGLES_FILE_PATH,
        RESIRE_PARAM,
        OUTPUT_FN,
    )


__all__ = [
    "RESIRE_Reconstructor",
    "reconstruct",
    "main_reconstruction",
    "PROJECTIONS_FILE_PATH",
    "ANGLES_FILE_PATH",
    "OUTPUT_FN",
    "RESIRE_PARAM",
]
