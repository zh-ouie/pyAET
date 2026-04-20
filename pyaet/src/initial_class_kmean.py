import numpy as np
from sklearn.cluster import KMeans
from pyaet.src.get_box_intensity import get_box_intensity
from pyaet.src.initial_class_L1norm import initial_class_L1norm
import matplotlib.pyplot as plt

#

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

    # Generate points of intensities
    box_inten = get_box_intensity(rec, curr_model, half_size, O_Ratio, SPHyn, 'linear')
    box_inten_plot = get_box_intensity(rec, curr_model, plot_half_size, O_Ratio, SPHyn, 'linear')

    # K-means clustering for non-atoms and atoms
    if lnorm == 2:
        kmeans = KMeans(n_clusters=num_species + 1)
        idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))
    elif lnorm == 1:
        kmeans = KMeans(n_clusters=num_species + 1)
        idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))

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
    dim_b = round(box_inten_type1.shape[0] ** (1 / 3))
    box_inten_type1 = box_inten_type1.reshape((dim_b, dim_b, dim_b, len(idx)))
    mean_box_type1 = np.mean(box_inten_type1[:, :, :, idx == 0], axis=3)

    atomtype, _ = initial_class_L1norm(box_inten_type1, mean_box_type1, O_Ratio, half_size, SPHyn)

    temp_model = curr_model[:, atomtype == 1]
    box_inten_sub = box_inten[:, atomtype == 1]
    box_inten_plot_sub = box_inten_plot[:, atomtype == 1]

    # K-means clustering among real-atoms
    if lnorm == 2:
        kmeans = KMeans(n_clusters=num_species)
        idx = kmeans.fit_predict(np.sum(box_inten_sub, axis=0).reshape(-1, 1))
    elif lnorm == 1:
        kmeans = KMeans(n_clusters=num_species)
        idx = kmeans.fit_predict(np.sum(box_inten_sub, axis=0).reshape(-1, 1))

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
                plt.title(f'{np.sum(idx == i - 1)} Type {i - 1} atoms') #todo: check
            plt.xlabel('integrated intensity (a.u.)')
            plt.ylabel('# atoms')
            plt.ylim([0, y_up])
            # plt.xlim([0, np.ceil(np.max(np.sum(box_inten_plot_sub, axis=0)) / 5) * 5])

        plt.show()
    else:
        print('Initial classification:')
        for i in range(num_species):
            print(f'number of type {i + 1} atoms: {np.sum(idx == i)}') #todo: check
        print(f'number of total atoms: {len(idx)}')

    temp_atomtype = idx

    return temp_model, temp_atomtype

# Note: You would need to define `initial_class_L1norm` as it is called in this function.
