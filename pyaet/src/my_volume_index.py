import numpy as np

def My_volumn_index(big_size, ori_size):
    """
    Calculate corresponding index for a small size matrix embedded in a large size matrix.

    Args:
        big_size (ndarray): A NumPy array or list representing the dimensions of the large size matrix.
        ori_size (ndarray): A NumPy array or list representing the dimensions of the original (small) size matrix.

    Returns:
        ndarray: A 2D NumPy array where each row contains the start and end indices for each dimension.

    Notes:
        - This function calculates the corresponding index ranges for embedding a small matrix into a larger matrix.
        - It ensures that the center pixel position is preserved in each dimension.
        - The length of `big_size` and `ori_size` must be the same.
        - `ori_size` should be equal to or smaller than `big_size` in all dimensions.
    """
    vol_ind = np.zeros((len(big_size), 2), dtype=int)

    if len(big_size) != len(ori_size):
        print("Input volume dimension and paddedsize length do not match!")
    elif np.any(np.array(big_size) < np.array(ori_size)):
        print("paddedsize should be equal to or smaller than the original volume in all dimensions!")
    else:
        for i in range(len(big_size)):
            if big_size[i] % 2 == 0:
                if ori_size[i] % 2 == 0:
                    startind = 1 + (big_size[i] - ori_size[i]) // 2
                    endind = ori_size[i] + (big_size[i] - ori_size[i]) // 2
                else:
                    startind = 1 + (big_size[i] - ori_size[i] + 1) // 2
                    endind = ori_size[i] + (big_size[i] - ori_size[i] + 1) // 2
            else:
                if ori_size[i] % 2 == 0:
                    startind = 1 + (big_size[i] - ori_size[i] - 1) // 2
                    endind = ori_size[i] + (big_size[i] - ori_size[i] - 1) // 2
                else:
                    startind = 1 + (big_size[i] - ori_size[i]) // 2
                    endind = ori_size[i] + (big_size[i] - ori_size[i]) // 2

            vol_ind[i, 0] = startind
            vol_ind[i, 1] = endind

    return vol_ind
