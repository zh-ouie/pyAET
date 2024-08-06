import numpy as np
from sklearn.cluster import KMeans
from pyaet.src.get_box_intensity import get_box_intensity
import matplotlib.pyplot as plt


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
    Num_species = classify_info.get('Num_species', 3)
    half_size = classify_info.get('half_size', 1)
    plothalfSize = classify_info.get('plothalfSize', 4)
    separate_part = classify_info.get('separate_part', 70)
    O_Ratio = classify_info.get('O_Ratio', 1)
    SPHyn = classify_info.get('SPHyn', True)
    PLOT_YN = classify_info.get('PLOT_YN', False)


    # Generate points of intensities
    box_inten = get_box_intensity(rec, curr_model, half_size, O_Ratio, SPHyn, 'linear')
    box_inten_plot = get_box_intensity(rec, curr_model, plothalfSize, O_Ratio, SPHyn, 'linear')

    # K-means clustering
    # if lnorm == 2:
    #     kmeans = KMeans(n_clusters=Num_species, init='k-means++', n_init=10, random_state=0)
    #     idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))
    # elif lnorm == 1:
    #     kmeans = KMeans(n_clusters=Num_species, init='k-means++', n_init=10, random_state=0)
    #     idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))
    #todo: find a method argument to be similar in matlab's Kmeans for lnorm=1 and 2.

    if lnorm == 2:
        kmeans = KMeans(n_clusters=Num_species)
        idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))
    elif lnorm == 1:
        kmeans = KMeans(n_clusters=Num_species)
        idx = kmeans.fit_predict(np.sum(box_inten, axis=0).reshape(-1, 1))

    # Alignment clustered type into correct species order
    mean_arr = np.zeros(Num_species)
    for i in range(Num_species):
        mean_arr[i] = np.mean(np.sum(box_inten[:, idx == i], axis=0))
    
    sortMean = np.argsort(mean_arr)
    for i in range(Num_species):
        idx[idx == sortMean[i]] = i + 1000
    idx = idx - 1000

    # Plot histogram
    if PLOT_YN:
        plt.figure(203)
        plt.clf()
        plt.figure(figsize=(4, 9))

        # Histogram of integrated intensity
        hist_inten_plot, cen_integ_total_plot = np.histogram(np.sum(box_inten_plot, axis=0), bins=separate_part)
        y_up = np.round(np.max(hist_inten_plot) / 10) * 12

        for i in range(Num_species + 1):
            plt.subplot(Num_species + 1, 1, i + 1)
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
        for i in range(Num_species):
            print(f'number of type {i} atoms: {np.sum(idx == i)}')
        print(f'number of total atoms: {idx.size}')

    # Final results
    temp_model = curr_model
    temp_atomtype = idx

    return temp_model, temp_atomtype
