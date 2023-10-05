import numpy as np
import scipy.ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion

def obtain_tight_support(RECvol, para_info):
    """
    Obtain tight support.

    Parameters:
    - RECvol (numpy.ndarray): Reconstruction volume.
    - para_info (dict): Parameters for support generation.

    Returns:
    - curr_Supportt (numpy.ndarray): Tight support.
    """

    th_dis_r_afterav = para_info.get('th_dis_r_afterav', 0.90)
    dilate_size = para_info.get('dilate_size', 11)
    erode_size = para_info.get('erode_size', 11)

    print(f'otsu_threshold: {th_dis_r_afterav:.2f}, dilate size: {dilate_size}, erode size: {erode_size}')

    # Smooth the volume
    curr_RECvol = scipy.ndimage.gaussian_filter(RECvol.astype(float), sigma=9)

    # Otsu threshold
    im = curr_RECvol / curr_RECvol.max() * 255
    ot = threshold_otsu(im)
    ot = ot * curr_RECvol.max() / 255 * th_dis_r_afterav

    curr_Support = (curr_RECvol > ot).astype(int)

    # Make the mask slightly larger
    se = np.ones((3, 3, 3))
    curr_Support = binary_dilation(curr_Support, se)

    # Make the mask quite larger
    se = np.ones((dilate_size, dilate_size, dilate_size))
    curr_Support = binary_dilation(curr_Support, se)

    if 'bw_size' in para_info:
        bw_size = para_info['bw_size']
        curr_Support = scipy.ndimage.binary_opening(curr_Support, structure=np.ones((3, 3, 3)), iterations=bw_size)

    # Make the mask quite smaller
    se = np.ones((erode_size, erode_size, erode_size))
    curr_Supportt = binary_erosion(curr_Support, se)

    return curr_Supportt
