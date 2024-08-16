import numpy as np

def create_box(box_size):
    """
    Create box coordinates and corresponding spherical mask.

    Args:
        box_size (int): The size of the box, should be an odd number.

    Returns:
        tuple: Contains box coordinates (dictionary with x, y, z), box_center, box_radius, and spherical mask.
    """

    box_center = (box_size + 1) / 2
    box_radius = (box_size - 1) / 2

    # Create box coordinate systems
    box_X, box_Y, box_Z = np.meshgrid(
        np.arange(-box_radius, box_radius + 1),
        np.arange(-box_radius, box_radius + 1),
        np.arange(-box_radius, box_radius + 1),
    )

    box_X = np.transpose(box_X, (1, 0, 2))
    box_Y = np.transpose(box_Y, (1, 0, 2))
    box_Z = np.transpose(box_Z, (1, 0, 2))

    # Calculate spherical mask
    sphere = np.sqrt(box_X ** 2 + box_Y ** 2 + box_Z ** 2) <= box_size / 2

    box_coordinates = {'x': box_X, 'y': box_Y, 'z': box_Z}

    return box_coordinates, box_center, box_radius, sphere
