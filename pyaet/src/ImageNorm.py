import numpy as np

def image_norm(image, value):
    """
    Normalize an image or a stack of images by a given value.

    Parameters:
    image (ndarray): The input image or stack of images.
    value (float): The value by which to normalize the image(s).

    Returns:
    ndarray: The normalized image or stack of images.

    Example:
    >>> image = np.array([[1, 2], [3, 4]])
    >>> value = 10
    >>> normalized_image = image_norm(image, value)
    """
    final_image = np.zeros_like(image)
    
    if len(image.shape) == 3:
        for i in range(image.shape[2]):
            sum_image = np.sum(image[:, :, i])
            final_image[:, :, i] = image[:, :, i] * value / sum_image
    elif len(image.shape) == 2:
        sum_image = np.sum(image)
        final_image = image * value / sum_image
    else:
        raise ValueError("Image can only be 2D or 3D!")

    return final_image
