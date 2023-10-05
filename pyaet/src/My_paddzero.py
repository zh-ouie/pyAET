import numpy as np

def My_paddzero(Vol, paddedsize):
    """
    Pad a N-dimensional array with zeros to match the specified size.

    Args:
        Vol (ndarray): The original N-dimensional array before padding.
        paddedsize (tuple): The desired size of the N-dimensional array after padding.

    Returns:
        ndarray: The padded N-dimensional array.

    Notes:
        My_paddzero pads the input array with zeros while preserving the center
        pixel position at (n+1)/2 (in case of n=odd) or at n/2+1 (in case of n=even)
        in each dimension.

        The length of `paddedsize` must be equal to the length of `Vol.shape`.
        `paddedsize` should be equal to or larger than the size of `Vol` in all dimensions.
    """
    if len(Vol.shape) != len(paddedsize):
        print("Input volume dimension and paddedsize length do not match!")
    elif np.any(np.array(Vol.shape) > np.array(paddedsize)):
        print("paddedsize should be equal to or larger than the original volume in all dimensions!")
    else:
        PadVol = np.zeros(paddedsize)
        currevalstr = 'PadVol['
        for i in range(len(Vol.shape)):
            if Vol.shape[i] % 2 == 0:
                if paddedsize[i] % 2 == 0:
                    startind = 1 + (paddedsize[i] - Vol.shape[i]) // 2
                    endind = Vol.shape[i] + (paddedsize[i] - Vol.shape[i]) // 2
                else:
                    startind = 1 + (paddedsize[i] - Vol.shape[i] - 1) // 2
                    endind = Vol.shape[i] + (paddedsize[i] - Vol.shape[i] - 1) // 2
            else:
                if paddedsize[i] % 2 == 0:
                    startind = 1 + (paddedsize[i] - Vol.shape[i] + 1) // 2
                    endind = Vol.shape[i] + (paddedsize[i] - Vol.shape[i] + 1) // 2
                else:
                    startind = 1 + (paddedsize[i] - Vol.shape[i]) // 2
                    endind = Vol.shape[i] + (paddedsize[i] - Vol.shape[i]) // 2

            currevalstr += f"{startind}:{endind}"
            if i < len(Vol.shape) - 1:
                currevalstr += ','
        currevalstr += '] = Vol.copy()'
        exec(currevalstr)
    
    return PadVol
