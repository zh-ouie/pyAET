import numpy as np
from pyaet.src.get_box_intensity import get_box_intensity
import matplotlib.pyplot as plt


def _kmeans_1d_lloyd_simple(data_1d: np.ndarray, k: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    MATLAB-compatible 1D k-means (L2) "simple_kmeans" branch.

    Returns:
      labels: (n,) in 1..k (MATLAB style)
      centroids: (k,) float64
    """
    x = np.asarray(data_1d, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((k,), dtype=np.float64)

    min_val = float(np.min(x))
    max_val = float(np.max(x))
    centroids = np.linspace(min_val, max_val, k, dtype=np.float64)
    labels = np.ones((x.size,), dtype=np.int64)

    for _ in range(max_iters):
        dists = np.abs(x[:, None] - centroids[None, :])
        # +1 to match MATLAB's 1-based labels.
        labels = np.argmin(dists, axis=1).astype(np.int64) + 1

        old_centroids = centroids.copy()
        for j in range(1, k + 1):
            cluster_points = x[labels == j]
            if cluster_points.size:
                centroids[j - 1] = float(np.mean(cluster_points))
        if np.array_equal(old_centroids, centroids):
            break
    return labels, centroids


def _kmedians_1d_lloyd_simple(data_1d: np.ndarray, k: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    MATLAB-compatible 1D k-means with cityblock distance (k-medians).

    Returns:
      labels: (n,) in 1..k (MATLAB style)
      centroids: (k,) float64
    """
    x = np.asarray(data_1d, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((k,), dtype=np.float64)

    min_val = float(np.min(x))
    max_val = float(np.max(x))
    centroids = np.linspace(min_val, max_val, k, dtype=np.float64)
    labels = np.ones((x.size,), dtype=np.int64)

    for _ in range(max_iters):
        dists = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1).astype(np.int64) + 1

        old_centroids = centroids.copy()
        for j in range(1, k + 1):
            cluster_points = x[labels == j]
            if cluster_points.size:
                centroids[j - 1] = float(np.median(cluster_points))
        if np.array_equal(old_centroids, centroids):
            break
    return labels, centroids


def initial_class_kmean_sub(rec, curr_model, classify_info):
    """
    Perform k-means classification among real-atoms.
    Only classify among real-atoms

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


    # Generate points of intensities
    box_inten = get_box_intensity(rec, curr_model, half_size, O_Ratio, SPHyn, 'linear')
    box_inten_plot = get_box_intensity(rec, curr_model, plot_half_size, O_Ratio, SPHyn, 'linear')

    box_inten = np.nan_to_num(box_inten, nan=0.0)
    box_inten_plot = np.nan_to_num(box_inten_plot, nan=0.0)
    data_1d = np.nan_to_num(np.sum(box_inten, axis=0).astype(np.float64), nan=0.0)

    use_simple = bool(classify_info.get('simple_kmeans', False))
    if lnorm == 2:
        if use_simple:
            idx, _ = _kmeans_1d_lloyd_simple(data_1d, num_species, max_iters=100)
        else:
            idx, _ = _kmeans_1d_lloyd_simple(data_1d, num_species, max_iters=100)
    elif lnorm == 1:
        if use_simple:
            idx, _ = _kmedians_1d_lloyd_simple(data_1d, num_species, max_iters=100)
        else:
            idx, _ = _kmedians_1d_lloyd_simple(data_1d, num_species, max_iters=100)
    else:
        raise ValueError(f"Unsupported lnorm={lnorm}; expected 1 or 2")

    # Alignment clustered type into correct species order (MATLAB logic).
    mean_arr = np.zeros(num_species, dtype=np.float64)
    for i in range(1, num_species + 1):
        temp_vals = data_1d[idx == i]
        mean_arr[i - 1] = float(np.mean(temp_vals)) if temp_vals.size else 0.0

    sortMean = (np.argsort(mean_arr) + 1).astype(np.int64)  # -> labels 1..K
    for i in range(1, num_species + 1):
        idx[idx == sortMean[i - 1]] = i + 1000
    idx = idx - 1000

    if not classify_info.get('matlab_label', False):
        idx = idx - 1

    # Plot histogram
    if PLOT_YN:
        plt.figure(203)
        plt.clf()
        plt.figure(figsize=(4, 9))

        hist_inten_plot, cen_integ_total_plot = np.histogram(np.sum(box_inten_plot, axis=0), bins=separate_part)
        y_up = np.round(np.max(hist_inten_plot) / 10) * 12

        for i in range(num_species + 1):
            plt.subplot(num_species + 1, 1, i + 1)
            if i == 0:
                plt.hist(np.sum(box_inten_plot, axis=0), bins=separate_part)
                plt.title(f'boxsize {2 * half_size + 1}')
            else:
                intensity_integ_sub = np.sum(box_inten_plot[:, idx == i - 1], axis=0)
                plt.hist(intensity_integ_sub, bins=cen_integ_total_plot)
                plt.title(f'{np.sum(idx == i - 1)} Type {i - 1} atoms')
            plt.xlabel('integrated intensity (a.u.)')
            plt.ylabel('# atoms')
            plt.ylim([0, y_up])
            # plt.xlim([0, np.ceil(np.max(np.sum(box_inten_plot, axis=0)) / 5) * 5])

        plt.show()
    else:
        print('Initial classification sub:')
        if classify_info.get('matlab_label', False):
            for i in range(1, num_species + 1):
                print(f'number of type{i} atoms: {np.sum(idx == i)}')
        else:
            for i in range(num_species):
                print(f'number of type{i} atoms: {np.sum(idx == i)}')
        print(f'number of total atoms: {idx.size}')

    # Final results
    temp_model = curr_model
    temp_atomtype = idx

    return temp_model, temp_atomtype
