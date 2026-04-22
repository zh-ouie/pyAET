# C++ implementation of splinterp functions
try:
    from .splinterp_cpp import mex_function2, mex_function3
except ImportError:
    from pyaet.splinterp.splinterp2 import mex_function2
    from pyaet.splinterp.splinterp3 import mex_function3

__all__ = ["mex_function2", "mex_function3"]
