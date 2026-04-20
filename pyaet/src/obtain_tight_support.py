import numpy as np
import scipy.ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from pyaet.src.strel3d import strel3d
from scipy.ndimage import grey_dilation, uniform_filter, grey_erosion
from pyaet.src.otsu_thresh_3D import otsu_thresh_3D

def obtain_tight_support(RECvol, para_info):
    """
    Obtain tight support.

    Parameters:
    - RECvol (numpy.ndarray): Reconstruction volume.
    - para_info (dict): Parameters for support generation.

    Returns:
    - curr_support (numpy.ndarray): Tight support.
    """

    th_dis_r_afterav = para_info.get('th_dis_r_afterav', 0.90)
    dilate_size = para_info.get('dilate_size', 11)
    erode_size = para_info.get('erode_size', 11)

    print(f'otsu_threshold: {th_dis_r_afterav:.2f}, dilate size: {dilate_size}, erode size: {erode_size}')

    # Smooth the volume
    curr_RECvol = uniform_filter(RECvol, size=9) #todo: still diffference

    # Otsu threshold
    im = curr_RECvol / curr_RECvol.max() * 255
    ot, _ = otsu_thresh_3D(im)
    ot = ot * curr_RECvol.max() / 255 * th_dis_r_afterav

    curr_support = np.where((curr_RECvol > ot), 1, 0)

    # Make the mask slightly larger
    se = strel3d(3)
    curr_support = grey_dilation(curr_support, footprint=se)

    # Make the mask quite larger
    se = strel3d(dilate_size)
    curr_support = grey_dilation(curr_support, footprint=se)

    if 'bw_size' in para_info:
        #remove small isolated area by bw_size
        bw_size = para_info['bw_size']
        # curr_support = scipy.ndimage.binary_opening(curr_support, structure=np.ones((3, 3, 3)), iterations=bw_size)
        curr_support = remove_small_objects(curr_support.astype(bool), bw_size) #todo: different

    # Make the mask quite smaller
    se = strel3d(erode_size)
    curr_support = grey_erosion(curr_support, footprint=se)

    return curr_support
