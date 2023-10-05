import numpy as np

def otsu_thresh_3D(im):
    """
    Otsu Thresholding for 3D images.

    Parameters:
    - im (numpy.ndarray): 3D grayscale image (integer 0-255).

    Returns:
    - ot (int): Threshold determined by Otsu method.
    - x (numpy.ndarray): Thresholded image.
    """
    
    dim1, dim2, dim3 = im.shape
    pixels = dim1 * dim2 * dim3

    bins = np.arange(256)
    N = np.histogram(im, bins=np.arange(257))[0]

    Nnorm = N / np.sum(N)  # Normalizing the bin frequencies to make probabilities
    theta = np.cumsum(Nnorm)  # Cumulative probability
    mu = np.cumsum(Nnorm * np.arange(256))

    sigB2 = (mu - mu[-1] * theta) ** 2 / (theta * (1 - theta))  # Evaluate sigB2 over the t range

    ot = np.argmax(sigB2)  # Find the maximum value and the index where it is (this is the Otsu threshold)

    x = (im > ot)  # Thresholding

    return ot, x
