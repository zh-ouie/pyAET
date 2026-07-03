import numpy as np
import os
import time
from scipy.spatial import cKDTree
from pyaet.src.get_box_intensity import get_box_intensity


def _build_radius_neighbor_indices(curr_model, radius):
    """Precompute MATLAB-equivalent BallInd lists for all atom positions."""
    coords = np.asarray(curr_model, dtype=np.float64).T
    tree = cKDTree(coords)
    raw_neighbors = tree.query_ball_point(coords, r=float(radius))
    neighbor_indices = []

    for i, candidates in enumerate(raw_neighbors):
        if not candidates:
            neighbor_indices.append(np.empty((0,), dtype=np.intp))
            continue

        cand = np.asarray(candidates, dtype=np.intp)
        delta = coords[cand] - coords[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        keep = (dist != 0) & (dist < radius)
        # np.where(BallInd)[0] in the original code is ascending; keep that
        # order so reductions are as close as possible to the reference path.
        neighbor_indices.append(np.sort(cand[keep]).astype(np.intp, copy=False))

    return neighbor_indices

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
    profile = os.environ.get("PYAET_CLASS_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
    iteration = 0
    seen_states = {pre_atomtype.tobytes(): 0}

    neighbor_indices = _build_radius_neighbor_indices(curr_model, radius)

    while not endFlag:
        iter_t0 = time.perf_counter()
        iteration += 1
        new_atomtype = np.zeros_like(pre_atomtype)

        for i in range(curr_model.shape[1]):
            local_neighbors = neighbor_indices[i]

            R_arr = np.zeros(num_types)
            for j in range(num_types):
                label = j + label_start
                true_indices = local_neighbors[pre_atomtype[local_neighbors] == label]
                if true_indices.size:
                    mean_box_inten = np.mean(box_inten[:, true_indices], axis=1)
                else:
                    mean_box_inten = np.full(box_inten.shape[0], np.nan, dtype=box_inten.dtype)
                R_temp_type = np.linalg.norm((box_inten[:, i] - mean_box_inten), lnorm)

                R_arr[j] = R_temp_type

            MinInd = np.argmin(R_arr)
            new_atomtype[i] = MinInd + label_start

        changed = int(np.sum(pre_atomtype != new_atomtype))
        if profile:
            print(f"local_class iteration {iteration}: changed={changed}, time_s={time.perf_counter() - iter_t0:.3f}", flush=True)

        if changed == 0:
            endFlag = True
            currDesc.append(0)
            pre_atomtype = new_atomtype.copy()
        else:
            currDesc.append(changed)
            state_key = new_atomtype.tobytes()
            previous_iteration = seen_states.get(state_key)
            if previous_iteration is not None:
                if profile:
                    period = iteration - previous_iteration
                    print(
                        f"local_class cycle detected: previous_iteration={previous_iteration}, "
                        f"current_iteration={iteration}, period={period}",
                        flush=True,
                    )
                endFlag = True
            else:
                seen_states[state_key] = iteration
            pre_atomtype = new_atomtype.copy()
            if not endFlag and len(currDesc) > StopCri:
                cutCri = np.asarray(currDesc[-StopCri:], dtype=np.int64)
                if np.sum(cutCri == currDesc[-1]) == cutCri.size:
                    endFlag = True

    temp_model = curr_model
    temp_atomtype = new_atomtype

    return temp_model, temp_atomtype
