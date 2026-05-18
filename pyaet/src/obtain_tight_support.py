import numpy as np
import scipy.ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from pyaet.src.strel3d import strel3d
from scipy.ndimage import grey_dilation, uniform_filter, grey_erosion
from pyaet.src.otsu_thresh_3D import otsu_thresh_3D


def _matlab_even_origin(footprint):
    return tuple(-1 if size % 2 == 0 else 0 for size in footprint.shape)


def _sample_volume(volume, sample_coords):
    if sample_coords is None:
        return None
    sample_coords = np.asarray(sample_coords, dtype=int)
    if sample_coords.size == 0:
        return np.zeros((0,), dtype=volume.dtype)
    sample_coords = np.clip(sample_coords, 0, np.array(volume.shape) - 1)
    return volume[sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]]


def obtain_tight_support(RECvol, para_info, return_debug=False):
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
    sample_coords = para_info.get('debug_sample_coords')

    print(f'otsu_threshold: {th_dis_r_afterav:.2f}, dilate size: {dilate_size}, erode size: {erode_size}')

    # Smooth the volume
    curr_RECvol = uniform_filter(RECvol.astype(np.float32, copy=False), size=9, mode='nearest').astype(np.float32, copy=False)

    # Otsu threshold
    max_val = np.float32(curr_RECvol.max())
    im = (curr_RECvol / max_val * np.float32(255.0)).astype(np.float32, copy=False)
    ot, _ = otsu_thresh_3D(im)
    ot = np.float32(ot) * max_val / np.float32(255.0) * np.float32(th_dis_r_afterav)

    curr_support = np.where((curr_RECvol > ot), 1, 0)
    threshold_support = curr_support.copy()

    # Make the mask slightly larger
    se = strel3d(3)
    curr_support = grey_dilation(curr_support, footprint=se)
    dilated_small = curr_support.copy()

    # Make the mask quite larger
    se = strel3d(dilate_size)
    curr_support = grey_dilation(curr_support, footprint=se)
    dilated_large = curr_support.copy()

    if 'bw_size' in para_info:
        #remove small isolated area by bw_size
        bw_size = para_info['bw_size']
        # curr_support = scipy.ndimage.binary_opening(curr_support, structure=np.ones((3, 3, 3)), iterations=bw_size)
        curr_support = remove_small_objects(curr_support.astype(bool), bw_size, connectivity=RECvol.ndim)
    cleaned_support = curr_support.copy()

    # Make the mask quite smaller
    se = strel3d(erode_size)
    curr_support = grey_erosion(curr_support, footprint=se, origin=_matlab_even_origin(se))

    if not return_debug:
        return curr_support

    debug_info = {
        'threshold': float(ot),
        'smoothed_sample': _sample_volume(curr_RECvol, sample_coords),
        'threshold_sample': _sample_volume(threshold_support, sample_coords),
        'dilate3_sample': _sample_volume(dilated_small, sample_coords),
        'dilateN_sample': _sample_volume(dilated_large, sample_coords),
        'cleanup_sample': _sample_volume(cleaned_support, sample_coords),
        'final_sample': _sample_volume(curr_support, sample_coords),
        'threshold_voxels': int(np.count_nonzero(threshold_support)),
        'dilate3_voxels': int(np.count_nonzero(dilated_small)),
        'dilateN_voxels': int(np.count_nonzero(dilated_large)),
        'cleanup_voxels': int(np.count_nonzero(cleaned_support)),
        'final_voxels': int(np.count_nonzero(curr_support)),
    }
    return curr_support, debug_info
