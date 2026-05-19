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
    half_size = classify_info.get('half_size', 1)
    O_Ratio = classify_info.get('O_Ratio', 1)
    radius = classify_info.get('radius', 15)
    SPHyn = classify_info.get('SPHyn', True)


    box_inten = get_box_intensity(rec, curr_model, half_size, O_Ratio, SPHyn, 'linear')

    num_types = len(np.unique(curr_types))
    label_start = 1 if classify_info.get('matlab_label', False) else 0

    endFlag = False
    currDesc = []
    pre_atomtype = np.asarray(curr_types).copy()
    new_atomtype = np.zeros_like(pre_atomtype)

    while not endFlag:
        new_atomtype = np.zeros_like(pre_atomtype)

        for i in range(curr_model.shape[1]):
            curr_atompos = curr_model[:, i]
            Dist = np.linalg.norm(curr_model.T - curr_atompos, axis=1)
            BallInd = (Dist != 0) & (Dist < radius)

            R_arr = np.zeros(num_types)
            for j in range(num_types):
                label = j + label_start
                temp_type = (pre_atomtype == label)
                true_indices = np.where(BallInd & temp_type)[0]
                mean_box_inten = np.mean(box_inten[:, true_indices], axis=1)
                R_temp_type = np.linalg.norm((box_inten[:, i] - mean_box_inten), lnorm)

                R_arr[j] = R_temp_type

            MinInd = np.argmin(R_arr)
            new_atomtype[i] = MinInd + label_start

        if np.sum(pre_atomtype != new_atomtype) == 0:
            endFlag = True
            currDesc.append(0)
            pre_atomtype = new_atomtype.copy()
        else:
            currDesc.append(np.sum(pre_atomtype != new_atomtype))
            pre_atomtype = new_atomtype.copy()
            if len(currDesc) > StopCri:
                cutCri = currDesc[-StopCri:]
                if np.sum(np.where(cutCri == currDesc[-1], 1, 0)) == len(cutCri):
                    endFlag = True

    temp_model = curr_model
    temp_atomtype = new_atomtype

    return temp_model, temp_atomtype
