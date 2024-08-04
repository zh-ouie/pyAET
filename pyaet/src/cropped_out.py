import numpy as np
from pyaet.src.my_round import my_round_list, my_round_num

def cropped_out(large_array, crop_size):
    """
    Crop out an m x m region from the center of a larger N x N array.

    Args:
        large_array (ndarray): The larger N x N array.
        crop_size (int or tuple): The size of the m x m region to be cropped.

    Returns:
        ndarray: The cropped region.

    Notes:
        This function crops out an m x m region from the center of a larger N x N array.
    """
    n = large_array.shape
    # nc = np.round((np.array(n) + 1) / 2).astype(int)
    nc = my_round_list((np.array(n) + 1) / 2)

    if len(crop_size) == 1:
        crop_size = np.repeat(crop_size, len(n), axis=0)  #todo: check

    cropVec = []
    for ii in range(len(n)):
        vec = np.arange(1, crop_size[ii] + 1)
        # cropC = np.round((crop_size[ii] + 1) / 2).astype(int)
        cropC = my_round_num((crop_size[ii] + 1) / 2)
        cropVec.append(vec - cropC + nc[ii])

    if len(n) == 2:
        # ROI = large_array[cropVec[0]-1, cropVec[1]-1]
        ROI = large_array[cropVec[0] - 1, :]
        ROI = ROI[:, cropVec[1] - 1]
    elif len(n) == 3:
        # ROI = large_array[cropVec[0] - 1, cropVec[1] - 1, cropVec[2] - 1]
        #cannot cut at the same time, so instead, cut dimension by dimension.
        ROI = large_array[cropVec[0] - 1, :, :]
        ROI = ROI[:, cropVec[1] - 1, :]
        ROI = ROI[:, :, cropVec[2] - 1]

    return ROI


data = np.load('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/test_cropped_out.npz')

pj_cal=data['pj_cal']
dimx=data['dimx']
dimy=data['dimy']
Num_pj=data['Num_pj']
