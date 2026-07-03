import numpy as np
from scipy.optimize import least_squares
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot


def fit_gauss3D_PD(init_guess, xdata, ydata, *args):
    """
    Fit 3D Gaussian and allow fixed parameters as well as bounds.

    Args:
        init_guess (numpy.ndarray): Initial guess for the parameters.
        xdata (dict): Dictionary containing the x, y, z data points.
        ydata (numpy.ndarray): The data to fit the Gaussian to.
        *args: Optional parameters for fixed parameters, lower bounds, and upper bounds.

    Returns:
        tuple: Fitted parameters, residual norm, and residuals.
    """
    fixed = np.full_like(init_guess, False, dtype=bool)
    lower_bound = -np.inf * np.ones_like(init_guess)
    upper_bound = np.inf * np.ones_like(init_guess)

    if len(args) > 0:
        fixed = np.array(args[0])
    if len(args) > 1:
        lower_bound = np.array(args[1])
    if len(args) > 2:
        upper_bound = np.array(args[2])

    init_guess = np.array(init_guess).astype(float)
    ydata = np.array(ydata).astype(float)

    init_guess_all = init_guess.copy()
    init_guess = init_guess_all[fixed == False]
    lower_bound = lower_bound[fixed == False]
    upper_bound = upper_bound[fixed == False]

    def residuals(x, xdata, ydata):
        x_current = x
        x = init_guess_all.copy()
        x[fixed == False] = x_current
        predictions = calc_gauss3D_PD(x, xdata)
        return predictions.flatten(order='F') - ydata.flatten(order='F')

    ydata = np.nan_to_num(ydata, nan=0)
    res = least_squares(
        residuals,
        init_guess,
        args=(xdata, ydata),
        bounds=(lower_bound, upper_bound),
        method='trf',
        xtol=1e-12,
        verbose=0,
    )

    x = res.x
    residual = res.fun
    resnorm = np.sum(residual**2)

    init_guess_all[fixed == False] = x
    x = init_guess_all

    return x, resnorm, residual


def calc_gauss3D_PD(x, xdata):
    """
    Calculate 3D Gaussian f(v) = exp(-v' * A * v).

    Args:
        x (list or ndarray): Parameters for the Gaussian function.
        xdata (dict): Dictionary containing the x, y, z data points.

    Returns:
        ndarray: The calculated 3D Gaussian values.
    """
    L, M, N = xdata['x'].shape

    v = np.vstack((
        xdata['x'].flatten(order='F') - x[2],
        xdata['y'].flatten(order='F') - x[3],
        xdata['z'].flatten(order='F') - x[4],
    ))

    vector1 = np.array([0, 0, 1])
    rotmat1 = matrix_quaternion_rot(vector1, x[8])

    vector2 = np.array([0, 1, 0])
    rotmat2 = matrix_quaternion_rot(vector2, x[9])

    vector3 = np.array([0, 0, 1])
    rotmat3 = matrix_quaternion_rot(vector3, x[10])

    rotMAT = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.diag([1 / x[5], 1 / x[6], 1 / x[7]])
    A = np.dot(np.dot(rotMAT.T, D), rotMAT)
    y = x[1] * np.exp(
        -np.einsum('ij,ij->j', v, np.dot(A, v))
    ).reshape(L, M, N, order='F') + x[0]

    return y
