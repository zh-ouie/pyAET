import numpy as np

def calculate_3D_polynomial_Rogers(X, Y, Z, Pos, Order, Coeff):
    """
    Calculate the 3D polynomial using the Rogers' algorithm.

    Parameters:
    - X (numpy.ndarray): X-coordinate values.
    - Y (numpy.ndarray): Y-coordinate values.
    - Z (numpy.ndarray): Z-coordinate values.
    - Pos (numpy.ndarray): Position of the atom as [X, Y, Z].
    - Order (numpy.ndarray): Polynomial orders as an array of shape (N, 3).
    - Coeff (numpy.ndarray): Coefficients corresponding to each order.

    Returns:
    - calcInt (numpy.ndarray): Calculated intensity values.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    calcInt = np.zeros_like(X, dtype=np.float64)

    for i in range(len(Order)):
        calcInt += Coeff[i] * (X - Pos[0])**Order[i, 0] * (Y - Pos[1])**Order[i, 1] * (Z - Pos[2])**Order[i, 2]

    return calcInt
