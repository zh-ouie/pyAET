import numpy as np
from typing import Tuple

def mexFunction(nlhs: int, plhs: np.ndarray, nrhs: int, prhs: np.ndarray) -> None:
    """
    Entry point for a MEX function that performs interpolation.

    Args:
        nlhs (int): Number of output arguments.
        plhs (np.ndarray): Array of output arguments.
        nrhs (int): Number of input arguments.
        prhs (np.ndarray): Array of input arguments.
    """
    if nrhs != 2:
        raise ValueError("Incorrect number of arguments. Syntax is Vq = splinterp1(V,Xq)")

    if np.iscomplexobj(prhs[0]):
        Matrix_r = prhs[0].real
        Matrix_i = prhs[0].imag
        x = prhs[1]
        nrows = max(prhs[0].shape)
        ndims_out = prhs[1].ndim
        dims_out = prhs[1].shape
        npoints = np.prod(dims_out)
        plhs[0] = np.empty(dims_out, dtype=np.complex128)
        result_r = plhs[0].real
        result_i = plhs[0].imag

        splinterp.parallel_interp1_cx(splinterp.interp1_F_cx, Matrix_r, Matrix_i, nrows, x, npoints, result_r, result_i, 1)

    else:
        Matrix = prhs[0]
        x = prhs[1]
        nrows = max(prhs[0].shape)
        ndims_out = prhs[1].ndim
        dims_out = prhs[1].shape
        npoints = np.prod(dims_out)
        plhs[0] = np.empty(dims_out)
        result = plhs[0]

        splinterp.parallel_interp1(splinterp.interp1_F, Matrix, nrows, x, npoints, result, 1)
