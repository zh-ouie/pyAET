import numpy as np

def croppedOut(largeArray, cropSize):
    """
    Crop out an m x m region from the center of a larger N x N array.

    Args:
        largeArray (ndarray): The larger N x N array.
        cropSize (int or tuple): The size of the m x m region to be cropped.

    Returns:
        ndarray: The cropped region.

    Notes:
        This function crops out an m x m region from the center of a larger N x N array.
    """
    n = largeArray.shape
    nc = np.round((np.array(n) + 1) / 2).astype(int)

    if isinstance(cropSize, int):
        cropSize = (cropSize,) * len(n)

    cropVec = []
    for ii in range(len(n)):
        vec = np.arange(1, cropSize[ii] + 1)
        cropC = np.round((cropSize[ii] + 1) / 2).astype(int)
        cropVec.append(vec - cropC + nc[ii])

    if len(n) == 2:
        ROI = largeArray[cropVec[0], cropVec[1]]
    elif len(n) == 3:
        ROI = largeArray[cropVec[0], cropVec[1], cropVec[2]]

    return ROI
