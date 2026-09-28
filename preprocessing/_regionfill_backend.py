"""Optional CHOLMOD solver for Laplacian background filling."""
import os
from importlib.metadata import PackageNotFoundError, version
import numpy as np
from scipy.sparse.linalg import spsolve


def regionfill_backend():
    backend = os.environ.get('AET_REGIONFILL_BACKEND', 'scipy')
    if backend not in ('scipy', 'cholmod'):
        raise ValueError('AET_REGIONFILL_BACKEND must be scipy or cholmod')
    return backend


def regionfill_backend_info():
    backend = regionfill_backend()
    package = 'cholespy' if backend == 'cholmod' else 'scipy'
    try:
        package_version = version(package)
    except PackageNotFoundError as error:
        raise ImportError('Install the cholmod extra to use AET_REGIONFILL_BACKEND=cholmod') from error
    return dict(backend=backend, package=package, version=package_version)


def solve_regionfill(matrix, rhs, backend):
    if backend == 'scipy':
        return spsolve(matrix, rhs)
    try:
        from cholespy import CholeskySolverD, MatrixType
    except ImportError as error:
        raise ImportError('Install the cholmod extra to use AET_REGIONFILL_BACKEND=cholmod') from error
    matrix = matrix.tocsc()
    if matrix.shape[0] >= 2**31 or matrix.nnz >= 2**31:
        raise ValueError('CHOLMOD binding requires 32-bit sparse indices')
    solver = CholeskySolverD(matrix.shape[0], matrix.indptr.astype(np.int32),
                            matrix.indices.astype(np.int32), matrix.data, MatrixType.CSC)
    result = np.empty_like(rhs)
    solver.solve(rhs, result)
    return result
