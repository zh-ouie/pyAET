import numpy as np
from scipy.optimize import optimize
from pyaet.src.create_box import create_box
from pyaet.src.fit_gauss3D_PD import fit_gauss3D_PD


def initial_class_L1norm(box_arr, mean_box, O_Ratio, half_size, SPHyn):
    """
    Perform L1 norm classification to distinguish atoms from non-atoms.

    Parameters:
    - box_arr (numpy.ndarray): Array of boxes. 4D array of boxes.
    - mean_box (numpy.ndarray): Mean box data.
    - O_Ratio (float): Ratio for oversampling.
    - half_size (int): Half size of the box.
    - SPHyn (bool): Spherical condition.

    Returns:
    - atomtype (numpy.ndarray): Atom types.
    - Rs (numpy.ndarray): R-factors.
    """
    atomtype = -1 * np.ones(box_arr.shape[3])
    Rs = np.zeros((2, box_arr.shape[3]))

    ds = 1 / O_Ratio

    XX, YY, ZZ = np.meshgrid(
        np.arange(-half_size, half_size + ds, ds),
        np.arange(-half_size, half_size + ds, ds),
        np.arange(-half_size, half_size + ds, ds),
        indexing='ij',
    )

    if SPHyn:
        useInd = np.where(
            ((XX ** 2 + YY ** 2 + ZZ ** 2) <= (half_size + 0.5 * ds) ** 2).flatten(order='F')
        )[0]
    else:
        useInd = np.arange(XX.size)

    box_coordinates, _, _, _ = create_box(box_arr.shape[0])

    fit_param_init = [0, np.nanmax(mean_box), 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0]
    fixed = np.full(11, False, dtype=bool)
    lb = [0, 0, -2, -2, -2, 0, 0, 0, -np.pi, 0, -np.pi]
    ub = [np.inf, np.inf, 2, 2, 2, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi]

    if half_size == 0:
        fit_result = 0
    else:
        fit_result, _, _ = fit_gauss3D_PD(fit_param_init, box_coordinates, mean_box, fixed, lb, ub)

    print(f'fitted constant = {fit_result[0]:.2f}')

    for ind in range(box_arr.shape[3]):
        data_box = box_arr[:, :, :, ind]
        zero_box = np.zeros_like(mean_box, dtype=float) + fit_result[0]
        data_box_flat = data_box.flatten(order='F')
        zero_box_flat = zero_box.flatten(order='F')
        mean_box_flat = mean_box.flatten(order='F')

        R1 = np.sum(np.abs(data_box_flat[useInd] - zero_box_flat[useInd]))
        R2 = np.sum(np.abs(data_box_flat[useInd] - mean_box_flat[useInd]))
        Rs[0, ind] = R1
        Rs[1, ind] = R2

        if R1 > R2:
            atomtype[ind] = 1
        else:
            atomtype[ind] = 0

    print(f'number of inserted atoms: {np.sum(atomtype == 1)} atoms')
    print(f'number of skipped atoms: {np.sum(atomtype == 0)} atoms')

    return atomtype, Rs

# Note: You will need to implement `fit_gauss3D_PD` based on your fitting method.
