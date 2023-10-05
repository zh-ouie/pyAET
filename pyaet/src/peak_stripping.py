import numpy as np
from scipy.signal import savgol_filter

def peak_stripping(Spectrum, Window):
    """
    Perform peak stripping on a spectrum.

    Parameters:
    - Spectrum: 1D numpy array
    - Window: Integer, the window size for Savitzky-Golay filter

    Returns:
    - Baseline: 1D numpy array, the baseline-corrected spectrum
    - stripping: Boolean, indicating if peak stripping was performed
    """

    stripping = 0
    y = savgol_filter(Spectrum, 0, Window)
    n = len(Spectrum)
    Baseline = np.zeros(n)

    for i in range(n):
        if Spectrum[i] > y[i]:
            stripping = 1
            Baseline[i] = y[i]
        else:
            Baseline[i] = Spectrum[i]

    return Baseline, stripping
