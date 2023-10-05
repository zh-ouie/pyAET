import numpy as np

def initial_class_L1norm(box_arr, mean_box, O_Ratio, halfSize, SPHyn):
    """
    Initial classification using L1-norm.

    Parameters:
    - box_arr (numpy.ndarray): 4D array of boxes.
    - mean_box (numpy.ndarray): Mean box.
    - O_Ratio (float): Oversampling ratio.
    - halfSize (int): Half the size of the box.
    - SPHyn (int): Spherical region flag.

    Returns:
    - atomtype (numpy.ndarray): Atom types.
    - Rs (numpy.ndarray): R-factors.
    """

    atomtype = -np.ones(box_arr.shape[3])
    Rs = np.zeros((2, box_arr.shape[3]))

    ds = 1 / O_Ratio

    XX, YY, ZZ = np.meshgrid(np.arange(-halfSize, halfSize + ds, ds),
                             np.arange(-halfSize, halfSize + ds, ds),
                             np.arange(-halfSize, halfSize + ds, ds))

    if SPHyn:
        useInd = np.where(XX**2 + YY**2 + ZZ**2 <= (halfSize + 0.5 * ds)**2)
    else:
        useInd = np.arange(XX.size)

    box2coordinates = create_box(box_arr.shape[0])

    fit_param_init = [0, np.max(mean_box), 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0]
    fixed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lb = [0, 0, -2, -2, -2, 0, 0, 0, -np.pi, 0, -np.pi]
    ub = [np.inf, np.inf, 2, 2, 2, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi]

    # Initialize variables to store fit results
    fit_result = np.zeros((box_arr.shape[3], len(fit_param_init)))

    for ind in range(box_arr.shape[3]):
        DataBox = box_arr[:, :, :, ind]

        ZeroBox = np.zeros(mean_box.shape) + fit_result[0]

        # Calculate squared deviation (= non-normalized r-factor)
        R1 = np.sum(np.abs(DataBox[useInd] - ZeroBox[useInd]))
        R2 = np.sum(np.abs(DataBox[useInd] - mean_box[useInd]))
        Rs[0, ind] = R1
        Rs[1, ind] = R2

        if R1 > R2:
            # This is an atom, add to the density matrix
            atomtype[ind] = 1
        else:
            # Not an atom, add background to the density matrix
            atomtype[ind] = 0

    print(f'number of inserted atoms: {np.sum(atomtype == 1)} atoms')
    print(f'number of skipped atoms: {np.sum(atomtype == 0)} atoms')

    return atomtype, Rs

# Define create_box function
def create_box(size):
    # Your create_box function implementation here
    pass
