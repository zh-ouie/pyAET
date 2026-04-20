import numpy as np

def otsu_thresh_3D(im):
    """
    Otsu Thresholding for 3D images.

    Parameters:
    - im: 3D numpy array (integer 0-255)

    Returns:
    - ot: The threshold determined
    - x: The thresholded image
    """

    dim1, dim2, dim3 = im.shape
    pixels = dim1 * dim2 * dim3

    bins = np.arange(257) + 0.5
    N = np.histogram(im.reshape(pixels), bins)[0]

    Nnorm = N / np.sum(N)  # Normalizing the bin frequencies to make probabilities
    theta = np.cumsum(Nnorm)  # Cumulative probability
    mu = np.cumsum(Nnorm * np.arange(256))

    sigB2 = (mu - mu[255] * theta) ** 2 / (theta * (1 - theta))  # Evaluate sigB2 over the threshold range

    ot = np.nanargmax(sigB2) +1 # Find the maximum value and the index where it is (this is the Otsu threshold)
    p  = np.nanmax(sigB2)

    x = (im > ot)  # Thresholding #Long edit: I think it should be im > p.
    # x = (im > p)  # Thresholding #Long edit: I think it should be im > p.

    return ot, x
