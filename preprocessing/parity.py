"""Numerical parity checks between MATLAB checkpoint .mat files and Python arrays."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat


def _numeric(x):
    x = np.asarray(x)
    return np.squeeze(x).astype(np.float64, copy=False)


def compare_arrays(reference, candidate, name="array", atol=1e-6, rtol=1e-5):
    a, b = _numeric(reference), _numeric(candidate)
    if a.shape != b.shape:
        return {"name": name, "shape_ref": list(a.shape), "shape_new": list(b.shape), "pass": False}
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    return {
        "name": name, "shape": list(a.shape), "max_abs": float(np.nanmax(d)),
        "mean_abs": float(np.nanmean(d)), "rmse": float(np.sqrt(np.nanmean(d*d))),
        "max_rel": float(np.nanmax(d / denom)),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0,
        "pass": bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)),
    }


def compare_mat(mat_path, candidates, output=None, atol=1e-6, rtol=1e-5):
    """Compare named variables in a MATLAB checkpoint against Python arrays.

    ``candidates`` is ``{mat_variable_name: numpy_array}``.
    """
    mat = loadmat(Path(mat_path), squeeze_me=False)
    report = []
    for name, candidate in candidates.items():
        if name not in mat:
            report.append({"name": name, "pass": False, "error": "missing in MATLAB file"})
        else:
            report.append(compare_arrays(mat[name], candidate, name, atol, rtol))
    result = {"mat": str(mat_path), "all_pass": all(x.get("pass", False) for x in report), "checks": report}
    if output:
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
