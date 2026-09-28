"""Input/output helpers for EMD/HDF5 and NumPy projection stacks."""
from pathlib import Path
import h5py
import numpy as np


def load_emd(path, data_path="/data/raw/data", angle_path="/data/raw/imageSetParameters",
             *, matlab_order=True):
    """Load raw EMD data.

    Returns ``(data, angles_deg)`` with data in ``(frame, projection, y, x)``
    order, matching MATLAB h5read in Sample_withcomments.m. HDF5 exposes
    these axes in reverse order. Set matlab_order=False only for files
    already stored in the returned order, with a projection-by-parameter table.
    """
    with h5py.File(Path(path), "r") as f:
        data = np.asarray(f[data_path])
        p = np.asarray(f[angle_path]) if angle_path in f else None
    if data.ndim != 4:
        raise ValueError("Raw EMD data must have four axes: frames, projections, y, x")
    if matlab_order:
        data = data.transpose(3, 2, 1, 0)
        if p is not None and p.ndim == 2:
            p = p.T
    angles = None
    if p is not None:
        if p.ndim != 2 or p.shape[1] < 5:
            raise ValueError("EMD angle parameter table must contain MATLAB column 5")
        if p.shape[0] != data.shape[1]:
            raise ValueError("EMD angle count does not match the projection count")
        # This acquisition uses gamma (column 5), not the near-90-degree alpha.
        angles = np.asarray(p[:, 4], dtype=np.float64).ravel()
        if not np.isfinite(angles).all():
            raise ValueError("EMD angles must be finite")
    return data, angles


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def load_mat_stack(path, variable=None):
    """Load a public MATLAB projection stack exported from the EMD workflow.

    The public HEA/CoPdPt repositories expose processed projection stacks as
    ``.mat`` rather than the original microscope ``.emd``. The loader selects
    the requested variable, or the first 3-D numeric variable when omitted.
    """
    from scipy.io import loadmat
    from pathlib import Path
    p = Path(path)
    try:
        obj = loadmat(p)
        if variable is not None:
            return np.asarray(obj[variable])
        candidates = [(k, np.asarray(v)) for k, v in obj.items()
                      if not k.startswith("__") and np.asarray(v).ndim >= 3 and np.issubdtype(np.asarray(v).dtype, np.number)]
        if not candidates:
            raise ValueError(f"no numeric 3-D variable found in {p}")
        return candidates[0][1]
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5 containers.
        with h5py.File(p, "r") as f:
            if variable is not None:
                return np.asarray(f[variable])
            names = []
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
                    names.append(name)
            f.visititems(visit)
            if not names:
                raise ValueError(f"no numeric 3-D dataset found in {p}")
            return np.asarray(f[names[0]])
