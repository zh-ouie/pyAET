import numpy as np
from sklearn.cluster import KMeans
from pyaet.src.get_box_intensity import get_box_intensity
from pyaet.src.initial_class_L1norm import initial_class_L1norm
import matplotlib.pyplot as plt

#


def _cube_side_or_raise(num_points, num_atoms, half_size, O_Ratio, SPHyn):
    dim_b = round(num_points ** (1 / 3))
    if dim_b ** 3 != num_points:
        raise ValueError(
            "Box intensity point count must be a perfect cube before reshape: "
            f"num_points={num_points}, inferred_dim={dim_b}, "
            f"num_atoms={num_atoms}, "
            f"per_atom_points={num_points / max(num_atoms, 1):.6f}, "
            f"half_size={half_size}, O_Ratio={O_Ratio}, SPHyn={SPHyn}"
        )
    return dim_b


def _build_kmeans(values, n_clusters, random_state):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if values.shape[0] < n_clusters:
        raise ValueError(f"Need at least {n_clusters} samples for kmeans, got {values.shape[0]}")

    rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
    data_min = values.min(axis=0)
    data_max = values.max(axis=0)
    init_centers = rng.uniform(low=data_min, high=data_max, size=(n_clusters, values.shape[1]))

    return KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        algorithm='lloyd',
        random_state=random_state,
    )


def _relabel_by_mean(idx, values, n_clusters):
    idx = np.asarray(idx, dtype=int).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    mean_arr = np.empty(n_clusters, dtype=float)
    for i in range(n_clusters):
        cluster_vals = values[idx == i]
        mean_arr[i] = np.mean(cluster_vals) if cluster_vals.size else np.nan
    sort_mean = np.argsort(mean_arr)
    relabeled = idx.copy()
    for i, cluster_id in enumerate(sort_mean):
        relabeled[idx == cluster_id] = i
    return relabeled


def _refine_1d_cluster_boundaries(values, idx, n_clusters):
    values = np.asarray(values, dtype=float).reshape(-1)
    idx = _relabel_by_mean(idx, values, n_clusters)
    if values.size == 0 or n_clusters <= 1:
        return idx

    order = np.argsort(values, kind='mergesort')
    sorted_vals = values[order]
    sorted_idx = idx[order]

    boundaries = []
    for pos in range(1, sorted_idx.size):
        if sorted_idx[pos] != sorted_idx[pos - 1]:
            boundaries.append(pos)
    if len(boundaries) != n_clusters - 1:
        return idx

    prefix = np.zeros(sorted_vals.size + 1, dtype=float)
    prefix_sq = np.zeros(sorted_vals.size + 1, dtype=float)
    prefix[1:] = np.cumsum(sorted_vals)
    prefix_sq[1:] = np.cumsum(sorted_vals * sorted_vals)

    def seg_cost(start, end):
        count = end - start
        if count <= 0:
            return np.inf
        total = prefix[end] - prefix[start]
        total_sq = prefix_sq[end] - prefix_sq[start]
        return total_sq - (total * total) / count

    boundaries = [int(x) for x in boundaries]
    changed = True
    n = sorted_vals.size
    while changed:
        changed = False
        for boundary_idx in range(len(boundaries)):
            left_edge = 0 if boundary_idx == 0 else boundaries[boundary_idx - 1]
            right_edge = n if boundary_idx == len(boundaries) - 1 else boundaries[boundary_idx + 1]
            candidates = np.arange(left_edge + 1, right_edge, dtype=int)
            left_cost = np.array([seg_cost(left_edge, cand) for cand in candidates], dtype=float)
            right_cost = np.array([seg_cost(cand, right_edge) for cand in candidates], dtype=float)
            best_boundary = int(candidates[np.argmin(left_cost + right_cost)])
            if best_boundary != boundaries[boundary_idx]:
                boundaries[boundary_idx] = best_boundary
                changed = True

    refined_sorted = np.empty_like(sorted_idx)
    start = 0
    for cluster_id, end in enumerate(boundaries + [n]):
        refined_sorted[start:end] = cluster_id
        start = end

    refined_idx = np.empty_like(refined_sorted)
    refined_idx[order] = refined_sorted
    return refined_idx

def initial_class_kmean(rec, curr_model, classify_info):
    """
    Perform k-means classification among real-atoms and non-atoms.
    Classify Non-atoms and atoms first then classify among real-atoms.

    Parameters:
    - rec (numpy.ndarray): Reconstruction volume.
    - curr_model (numpy.ndarray): Current atomic positions as [X, Y, Z].
    - classify_info (dict): Classification information.

    Returns:
    - temp_model (numpy.ndarray): Updated atomic positions.
    - temp_atomtype (numpy.ndarray): Atom types.
    """
    lnorm = classify_info.get('lnorm', 2)
    num_species = classify_info.get('num_species', 3)
    half_size = classify_info.get('half_size', 1)
    plot_half_size = classify_info.get('plot_half_size', 4)
    separate_part = classify_info.get('separate_part', 70)
    O_Ratio = classify_info.get('O_Ratio', 1)
    SPHyn = classify_info.get('SPHyn', True)
    PLOT_YN = classify_info.get('PLOT_YN', False)
    random_state = classify_info.get('random_state', 42)

    # Generate points of intensities
    box_inten = get_box_intensity(rec, curr_model, half_size, O_Ratio, SPHyn, 'linear')
    box_inten_plot = get_box_intensity(rec, curr_model, plot_half_size, O_Ratio, SPHyn, 'linear')

    integ_box_inten = np.sum(box_inten, axis=0).reshape(-1, 1)

    # K-means clustering for non-atoms and atoms
    if lnorm == 2:
        kmeans = _build_kmeans(integ_box_inten, num_species + 1, random_state)
        idx = kmeans.fit_predict(integ_box_inten)
    elif lnorm == 1:
        kmeans = _build_kmeans(integ_box_inten, num_species + 1, random_state)
        idx = kmeans.fit_predict(integ_box_inten)
    idx = _refine_1d_cluster_boundaries(integ_box_inten, idx, num_species + 1)

    # Alignment clustered type into correct species order
    mean_arr = np.zeros(num_species + 1)
    for i in range(num_species + 1):
        mean_arr[i] = np.mean(np.sum(box_inten[:, idx == i], axis=0))

    sortMean = np.argsort(mean_arr)
    for i in range(num_species + 1):
        idx[idx == sortMean[i]] = i + 1000
    idx = idx - 1000
    # Plot histogram
    if PLOT_YN:
        plt.figure(203)
        plt.clf()
        plt.figure(figsize=(4, 9))

        hist_inten_plot, cen_integ_total_plot = np.histogram(np.sum(box_inten_plot, axis=0), bins=separate_part)
        y_up = np.round(np.max(hist_inten_plot) / 10) * 12

        for i in range(num_species + 2):
            plt.subplot(num_species + 2, 1, i + 1)
            if i == 0:
                plt.hist(np.sum(box_inten_plot, axis=0), bins=separate_part)
                plt.title(f'boxsize {2 * half_size + 1}')
            elif i == 1:
                intensity_integ_sub = np.sum(box_inten_plot[:, idx == i - 1], axis=0)
                plt.hist(intensity_integ_sub, bins=cen_integ_total_plot)
                plt.title(f'{np.sum(idx == i - 1)} Non-atoms')
            else:
                intensity_integ_sub = np.sum(box_inten_plot[:, idx == i - 1], axis=0)
                plt.hist(intensity_integ_sub, bins=cen_integ_total_plot)
                plt.title(f'{np.sum(idx == i - 1)} Type {i - 2} atoms')
            plt.xlabel('integrated intensity (a.u.)')
            plt.ylabel('# atoms')
            plt.ylim([0, y_up])
            # plt.xlim([0, np.ceil(np.max(np.sum(box_inten_plot, axis=0)) / 5) * 5])

        plt.show()
    else:
        print('Rough classification:')
        print(f'number of Non-atoms: {np.sum(idx == 0)}')
        for i in range(num_species):
            print(f'number of type {i + 1} atoms: {np.sum(idx == i + 1)}')
        print(f'number of total atoms: {np.sum(idx != 0)}')

    # Intermediate results
    idx = idx - 1

    box_inten_type1 = get_box_intensity(rec, curr_model, half_size, O_Ratio, 0, 'linear')
    dim_b = _cube_side_or_raise(box_inten_type1.shape[0], len(idx), half_size, O_Ratio, 0)
    box_inten_type1 = box_inten_type1.reshape((dim_b, dim_b, dim_b, len(idx)), order='F')
    mean_box_type1 = np.mean(box_inten_type1[:, :, :, idx == 0], axis=3)

    atomtype, _ = initial_class_L1norm(box_inten_type1, mean_box_type1, O_Ratio, half_size, SPHyn)

    temp_model = curr_model[:, atomtype == 1]
    box_inten_sub = box_inten[:, atomtype == 1]
    box_inten_plot_sub = box_inten_plot[:, atomtype == 1]

    # K-means clustering among real-atoms
    integ_box_inten_sub = np.sum(box_inten_sub, axis=0).reshape(-1, 1)
    if lnorm == 2:
        kmeans = _build_kmeans(integ_box_inten_sub, num_species, random_state)
        idx = kmeans.fit_predict(integ_box_inten_sub)
    elif lnorm == 1:
        kmeans = _build_kmeans(integ_box_inten_sub, num_species, random_state)
        idx = kmeans.fit_predict(integ_box_inten_sub)
    idx = _refine_1d_cluster_boundaries(integ_box_inten_sub, idx, num_species)

    # Alignment clustered type into correct species order
    mean_arr = np.zeros(num_species, dtype=float)
    for i in range(num_species):
        mean_arr[i] = np.mean(np.sum(box_inten_sub[:, idx == i], axis=0))

    sortMean = np.argsort(mean_arr)
    for i in range(num_species):
        idx[idx == sortMean[i]] = i + 1000
    idx = idx - 1000

    # Plot histogram
    if PLOT_YN:
        plt.figure(204)
        plt.clf()
        plt.figure(figsize=(4, 9))

        hist_inten_plot_sub, cen_integ_total_plot_sub = np.histogram(np.sum(box_inten_plot_sub, axis=0),
                                                                     bins=separate_part)
        y_up = np.round(np.max(hist_inten_plot_sub) / 10) * 12

        for i in range(num_species + 1):
            plt.subplot(num_species + 1, 1, i + 1)
            if i == 0:
                plt.hist(np.sum(box_inten_plot_sub, axis=0), bins=separate_part)
                plt.title(f'boxsize {2 * half_size + 1}')
            else:
                intensity_integ_sub = np.sum(box_inten_plot_sub[:, idx == i - 1], axis=0)
                plt.hist(intensity_integ_sub, bins=cen_integ_total_plot_sub)
                plt.title(f'{np.sum(idx == i - 1)} Type {i - 1} atoms')
            plt.xlabel('integrated intensity (a.u.)')
            plt.ylabel('# atoms')
            plt.ylim([0, y_up])
            # plt.xlim([0, np.ceil(np.max(np.sum(box_inten_plot_sub, axis=0)) / 5) * 5])

        plt.show()
    else:
        print('Initial classification:')
        for i in range(num_species):
            print(f'number of type {i + 1} atoms: {np.sum(idx == i)}')
        print(f'number of total atoms: {len(idx)}')

    temp_atomtype = idx

    return temp_model, temp_atomtype
