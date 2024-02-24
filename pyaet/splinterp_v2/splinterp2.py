import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from typing import Tuple
from splinterp_v2 import splinterp


def mexFunction(nlhs: int, plhs: np.ndarray, nrhs: int, prhs: np.ndarray) -> None:
    """
    Entry point for a MEX function that performs 2-D interpolation.

    Args:
        nlhs (int): Number of output arguments.
        plhs (np.ndarray): Array of output arguments.
        nrhs (int): Number of input arguments.
        prhs (np.ndarray): Array of input arguments.
    """
    if nrhs != 3:
        raise ValueError("Incorrect number of arguments. Syntax is Vq = splinterp2(V,Xq,Yq)")

    if np.iscomplexobj(prhs[0]):
        Matrix_r = prhs[0].real
        Matrix_i = prhs[0].imag
        y = prhs[1]
        x = prhs[2]
        nrows, ncols = prhs[0].shape
        if nrows == 1 or ncols == 1:
            raise ValueError("Input data is not 2D.")

        ndims_out = prhs[1].ndim
        dims_out = prhs[1].shape
        npoints = np.prod(dims_out)
        plhs[0] = np.empty(dims_out, dtype=np.complex128)
        result_r = plhs[0].real
        result_i = plhs[0].imag

        splinterp.parallel_interp2_cx(splinterp.interp2_F_cx, Matrix_r, Matrix_i, nrows, ncols, x, y, npoints, result_r,
                                      result_i, 1)
    else:
        Matrix = prhs[0]
        y = prhs[1]
        x = prhs[2]
        nrows, ncols = prhs[0].shape
        if nrows == 1 or ncols == 1:
            raise ValueError("Input data is not 2D.")

        ndims_out = prhs[1].ndim
        dims_out = prhs[1].shape
        npoints = np.prod(dims_out)
        plhs[0] = np.empty(dims_out)
        result = plhs[0]

        splinterp.parallel_interp2(splinterp.interp2_F, Matrix, nrows, ncols, x, y, npoints, result, 1)
