import numpy as np

def my_stripzero(PadVol, orisize):
    """
    Remove zero padding from the input array to match the specified size.

    Parameters:
    PadVol (ndarray): The padded N-dimensional array.
    orisize (tuple): The desired size of the N-dimensional array after removing padded zeros.

    Returns:
    ndarray: The N-dimensional array with zero padding removed to match orisize.

    Raises:
    ValueError: If the dimensions of PadVol and the length of orisize do not match,
                or if orisize is larger than PadVol in any dimension.

    Example:
    stripped_array = my_stripzero(padded_array, (32, 32, 32))
    """
    if len(PadVol.shape) != len(orisize):
        raise ValueError("Input volume dimension and padded size length do not match!")
    if any(PadVol.shape < np.array(orisize)):
        raise ValueError("Padded size should be equal to or smaller than the original volume in all dimensions!")

    currevalstr = 'PadVol['
    for i in range(len(PadVol.shape)):
        if PadVol.shape[i] % 2 == 0:
            if orisize[i] % 2 == 0:
                startind = 1 + (PadVol.shape[i] - orisize[i]) // 2
                endind = orisize[i] + (PadVol.shape[i] - orisize[i]) // 2
            else:
                startind = 1 + (PadVol.shape[i] - orisize[i] + 1) // 2
                endind = orisize[i] + (PadVol.shape[i] - orisize[i] + 1) // 2
        else:
            if orisize[i] % 2 == 0:
                startind = 1 + (PadVol.shape[i] - orisize[i] - 1) // 2
                endind = orisize[i] + (PadVol.shape[i] - orisize[i] - 1) // 2
            else:
                startind = 1 + (PadVol.shape[i] - orisize[i]) // 2
                endind = orisize[i] + (PadVol.shape[i] - orisize[i]) // 2
        currevalstr = f"{currevalstr}{startind}:{endind}"
        if i < len(PadVol.shape) - 1:
            currevalstr = f"{currevalstr},"
    currevalstr = f"{currevalstr}]"
    Vol = eval(currevalstr)
    return Vol
