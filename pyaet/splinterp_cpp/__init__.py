"""Public imports for the compiled splinterp extension.

The cleaned internal pipeline intentionally depends on the C++ interpolation
extension. If this import fails, build the extension in this directory with:

    python setup.py build_ext --inplace
"""

try:
    from .splinterp_cpp import mex_function1, mex_function2, mex_function3
except ImportError as exc:
    raise ImportError(
        "pyaet.splinterp_cpp is required. Build it with "
        "`cd pyaet/splinterp_cpp && python setup.py build_ext --inplace`."
    ) from exc

__all__ = ["mex_function1", "mex_function2", "mex_function3"]
