"""Public imports for the C++ splinterp extension.

The compiled extension exposes Python-callable wrappers around the MATLAB MEX
interpolation routines used by reconstruction and classification.
"""

try:
    from .splinterp_cpp import mex_function1, mex_function2, mex_function3
except ImportError:
    from pyaet.splinterp.splinterp1 import mex_function1
    from pyaet.splinterp.splinterp2 import mex_function2
    from pyaet.splinterp.splinterp3 import mex_function3

__all__ = ["mex_function1", "mex_function2", "mex_function3"]
