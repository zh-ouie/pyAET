import numpy as np
from pyaet.src.get_box_intensity import get_box_intensity

def local_class_kmean_sub(rec, curr_model, curr_types, classify_info):
    """
    Local classification using k-means.

    Parameters:
    - rec (numpy.ndarray): Reconstruction data.
    - curr_model (numpy.ndarray): Current model.
    - curr_types (numpy.ndarray): Current atom types.
    - classify_info (dict): Classification information.

    Returns:
    - temp_model (numpy.ndarray): Updated model.
    - temp_atomtype (numpy.ndarray): Updated atom types.
    """

    lnorm = classify_info.get('lnorm', 2)
    StopCri = classify_info.get('StopCri', 5)
    halfSize = classify_info.get('halfSize', 1)
    O_Ratio = classify_info.get('O_Ratio', 1)
    Radius = classify_info.get('Radius', 15)
    SPHyn = classify_info.get('SPHyn', True)


    box_inten = get_box_intensity(rec, curr_model, halfSize, O_Ratio, SPHyn, 'linear')

    #Long: Note that in matlab, we label atom type from 1, but in python, we start from 0.
    num_types = len(np.unique(curr_types))

    endFlag = False
    currDesc = []
    pre_atomtype = curr_types
    #Long: in python, we label atom type from 0. so we initialize the temporary variable using -1 here.
    new_atomtype = np.ones(len(curr_types)) * (-1)

    while not endFlag:
        print('new round:')

        for i in range(curr_model.shape[1]):
            curr_atompos = curr_model[:, i]
            Dist = np.linalg.norm(curr_model.T - curr_atompos, axis=1)
            BallInd = (Dist != 0) & (Dist < Radius)

            R_arr = np.zeros(num_types)
            for j in range(num_types):
                temp_type = (pre_atomtype == j)

                # R_temp_type = np.linalg.norm((box_inten[:, i] - np.mean(box_inten[:, (BallInd & temp_type)[0]], axis=1)), lnorm)

                true_indices = np.where((BallInd & temp_type)[0])[0]
                mean_box_inten = np.mean(box_inten[:, true_indices], axis=1)
                R_temp_type = np.linalg.norm((box_inten[:, i] - mean_box_inten), lnorm)

                #Long add:replace nan with 0, otherwise it will stop here.
                R_temp_type = np.nan_to_num(R_temp_type, nan=0)

                R_arr[j] = R_temp_type

            MinInd = np.argmin(R_arr)
            new_atomtype[i] = MinInd

        for i in range(num_types):
            print_arr = f'num{i}: {np.sum(new_atomtype == i)}; '
            print(print_arr)

        if np.sum(pre_atomtype != new_atomtype) == 0:
            endFlag = True
            currDesc.append(0)
            pre_atomtype = new_atomtype
        else:
            print(f'discrepency: {np.sum(pre_atomtype != new_atomtype)}')
            currDesc.append(np.sum(pre_atomtype != new_atomtype))
            pre_atomtype = new_atomtype
            if len(currDesc) > StopCri:
                cutCri = currDesc[-StopCri:]
                if np.sum(np.where(cutCri == currDesc[-1], 1, 0)) == len(cutCri):
                    endFlag = True

    temp_model = curr_model
    temp_atomtype = new_atomtype

    return temp_model, temp_atomtype
