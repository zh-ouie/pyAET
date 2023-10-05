import numpy as np
import splinterp

def mexFunction(nlhs, plhs, nrhs, prhs):
    """
    MATLAB MEX function equivalent in Python.

    Args:
        nlhs (int): Number of output arguments.
        plhs (list of numpy.ndarray): Output arguments.
        nrhs (int): Number of input arguments.
        prhs (list of numpy.ndarray): Input arguments.

    Raises:
        ValueError: If the number of input arguments is not 3 or if input data is not 2D.
    """
    if nrhs != 3:
        raise ValueError("Incorrect number of arguments. Syntax is Vq = splinterp2(V, Xq, Yq)")

    if np.iscomplexobj(prhs[0]):
        Matrix = prhs[0]
        x = prhs[2]
        y = prhs[1]
        result = np.empty_like(x, dtype=np.complex128)
        splinterp.parallel_interp2_cx(splinterp.interp2_F_cx, Matrix.real, Matrix.imag, x, y, result)
        plhs[0] = result
    else:
        Matrix = prhs[0]
        x = prhs[2]
        y = prhs[1]
        result = np.empty_like(x, dtype=np.float64)
        splinterp.parallel_interp2(splinterp.interp2_F, Matrix, x, y, result)
        plhs[0] = result

# Usage example:
# nlhs = 1
# plhs = [None] * nlhs
# nrhs = 3
# prhs = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5]), np.array([0.25, 1.25])]
# mexFunction(nlhs, plhs, nrhs, prhs)
# result = plhs[0]
