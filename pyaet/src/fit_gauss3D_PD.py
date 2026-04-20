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

    # Set standard values for optional parameters
    fixed = np.full_like(init_guess, False, dtype=bool)
    lower_bound = -np.inf * np.ones_like(init_guess)
    upper_bound = np.inf * np.ones_like(init_guess)

    # Check optional parameters
    # In python, args starts from 0 parameters. in matlab, it also counts `init_guess, xdata, ydata`, so it starts from 3 parameters.
    if len(args) > 0:
        fixed = np.array(args[0])
    if len(args) > 1:
        lower_bound = np.array(args[1])
    if len(args) > 2:
        upper_bound = np.array(args[2])

    # Convert to double
    init_guess = np.array(init_guess).astype(float)
    ydata = np.array(ydata).astype(float)

    # Store all initial parameters
    init_guess_all = init_guess.copy()

    # Only fit the variable parameters
    init_guess = init_guess_all[fixed == False]
    lower_bound = lower_bound[fixed == False]
    upper_bound = upper_bound[fixed == False]

    # deal with fixed parameters
    def residuals(x, xdata, ydata):
        x_current = x
        x = init_guess_all.copy()
        x[fixed == False] = x_current
        predictions = calc_gauss3D_PD(x, xdata)

        return predictions.flatten() - ydata.flatten()

    opt = {'xtol': 1e-12}

    ydata = np.nan_to_num(ydata, nan=0) #long add

    # Perform the fit
    res = least_squares(residuals, init_guess, args = (xdata, ydata), bounds = (lower_bound, upper_bound), method = 'trf', xtol = opt['xtol'], verbose = 0)

    x = res.x
    residual = res.fun
    resnorm = np.sum(residual**2)

    # Add fixed parameters to the final result again
    init_guess_all[fixed == False] = x
    x = init_guess_all

    return x, resnorm, residual


def calc_gauss3D_PD(x, xdata):
    """
    Calculate 3D Gaussian f(v) = exp(-v' * A * v)

    Args:
        x (list or ndarray): Parameters for the Gaussian function.
        xdata (dict): Dictionary containing the x, y, z data points.

    Returns:
        ndarray: The calculated 3D Gaussian values.
    """

    L, M, N = xdata['x'].shape
    num = L * M * N

    v = np.vstack((xdata['x'].flatten(order='F') - x[2], xdata['y'].flatten(order='F') - x[3], xdata['z'].flatten(order='F') - x[4]))

    vector1 = np.array([0, 0, 1])
    rotmat1 = matrix_quaternion_rot(vector1, x[8])

    vector2 = np.array([0, 1, 0])
    rotmat2 = matrix_quaternion_rot(vector2, x[9])

    vector3 = np.array([0, 0, 1])
    rotmat3 = matrix_quaternion_rot(vector3, x[10])

    rotMAT = np.dot(np.dot(rotmat3, rotmat2), rotmat1)

    D = np.diag([1 / x[5], 1 / x[6], 1 / x[7]])

    A = np.dot(np.dot(rotMAT.T, D), rotMAT)
    y = x[1] * np.exp(-np.einsum('ij,ij->j', v, np.dot(A, v))).reshape(L, M, N) + x[0]

    return y
