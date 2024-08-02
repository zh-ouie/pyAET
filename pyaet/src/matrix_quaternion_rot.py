import numpy as np

def matrix_quaternion_rot(vector, theta):
    """
    Change Euler angles to a rotation matrix using quaternion.

    Args:
        vector (ndarray): A 3-element array representing the rotation axis.
        theta (float): The rotation angle in degrees.

    Returns:
        ndarray: The 3x3 rotation matrix.

    Notes:
        This function converts Euler angles to a rotation matrix using quaternion representation.
    """
    theta = theta * np.pi / 180.0
    vector = vector / np.linalg.norm(vector)
    w = np.cos(theta / 2)
    x = -np.sin(theta / 2) * vector[0]
    y = -np.sin(theta / 2) * vector[1]
    z = -np.sin(theta / 2) * vector[2]
    
    RotM = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y + w*z), 2 * (x*z - w*y)],
        [2 * (x*y - w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z + w*x)],
        [2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x**2 + y**2)]
    ])
    
    return RotM
