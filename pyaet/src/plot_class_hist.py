import numpy as np
import matplotlib.pyplot as plt

def plot_class_hist(RecVol_padded, temp_model, temp_type, classify_info):
    """
    Plot histograms of integrated intensity values for different atom types in a 3D reconstruction volume.

    Parameters:
    - RecVol_padded (numpy.ndarray): The original or interpolated reconstruction volume.
    - temp_model (numpy.ndarray): The traced atom positions corresponding to the volume (3*N).
    - temp_type (numpy.ndarray): The classified atom types corresponding to the volume (1*N).
    - classify_info (dict): Input parameters for plotting the histogram.

    Returns:
    - peak_info (numpy.ndarray): Histogram information with shape (N+2)*separate_part.
      The rows represent:
        1. x-axis (location) of the histogram.
        2. Counts for all atoms in the model.
        3 to N+2. Counts corresponding to each classified atom type.
    - intensity_plot_arr (numpy.ndarray): Summed intensity array for plotting the histogram.

    Example:
    classify_info = {
        'plothalfSize': 4,
        'SPHyn': 1,
        'separate_part': 100,
        'PLOT_YN': 0
    }
    peak_info, intensity_plot_arr = plot_class_hist(RecVol_padded, temp_model, temp_type, classify_info)
    """
    if 'plothalfSize' in classify_info:
        plothalfSize = classify_info['plothalfSize']
    else:
        plothalfSize = 4

    if 'SPHyn' in classify_info:
        SPHyn = classify_info['SPHyn']
    else:
        SPHyn = 1

    if 'separate_part' in classify_info:
        separate_part = classify_info['separate_part']
    else:
        separate_part = 100

    if 'PLOT_YN' in classify_info:
        PLOT_YN = classify_info['PLOT_YN']
    else:
        PLOT_YN = 0

    Num_types = len(np.unique(temp_type[temp_type > 0]))
    peak_info = np.zeros((Num_types + 2, separate_part))

    xXp, yYp, zZp = np.meshgrid(
        np.arange(-plothalfSize, plothalfSize + 1),
        np.arange(-plothalfSize, plothalfSize + 1),
        np.arange(-plothalfSize, plothalfSize + 1)
    )
    SphereInd_plot = (xXp ** 2 + yYp ** 2 + zZp ** 2) <= (plothalfSize + 0.5) ** 2

    if SPHyn:
        useInd_plot = np.where(SphereInd_plot)
    else:
        useInd_plot = np.arange(len(xXp))

    intensity_plot_arr = np.zeros((len(useInd_plot[0]), temp_model.shape[1]))

    intensity_integ_plot = np.zeros(temp_model.shape[1])

    for j in range(temp_model.shape[1]):
        curr_pos = np.round(temp_model[:, j]).astype(int)
        box_integ = RecVol_padded[
            curr_pos[0] - plothalfSize : curr_pos[0] + plothalfSize + 1,
            curr_pos[1] - plothalfSize : curr_pos[1] + plothalfSize + 1,
            curr_pos[2] - plothalfSize : curr_pos[2] + plothalfSize + 1
        ]
        intensity_integ_plot[j] = np.sum(box_integ[useInd_plot])
        intensity_plot_arr[:, j] = box_integ[useInd_plot]

    hist_inten_plot, cen_integ_total_plot = np.histogram(intensity_integ_plot, bins=separate_part)
    peak_info[0, :] = cen_integ_total_plot
    peak_info[1, :] = hist_inten_plot
    y_up = round(np.max(hist_inten_plot) / 10) * 12

    if PLOT_YN:
        plt.figure()
        plt.clf()
        plt.set_position([1100, 0, 400, 900])
        plt.subplot(Num_types + 1, 1, 1)
        plt.hist(intensity_integ_plot, bins=separate_part)
        plt.xlim([0, np.ceil(np.max(intensity_integ_plot) / 5) * 5])
        plt.ylim([0, y_up])
        plt.xlabel('Integrated Intensity (a.u.)')
        plt.ylabel('# Atoms')
        plt.title(f'Box Size {plothalfSize * 2 + 1}')

    for i in range(Num_types):
        intensity_integ_sub = intensity_integ_plot[temp_type == i + 1]
        plt.subplot(Num_types + 1, 1, i + 2)
        hist_inten_plot_sub, _ = np.histogram(intensity_integ_sub, bins=cen_integ_total_plot)
        peak_info[i + 2, :] = hist_inten_plot_sub
        if PLOT_YN:
            plt.hist(intensity_integ_sub, bins=cen_integ_total_plot)
            plt.xlabel('Integrated Intensity (a.u.)')
            plt.ylabel('# Atoms')
            plt.title(f'{np.sum(temp_type == i + 1)} Type {i + 1} Atoms')
            plt.ylim([0, y_up])
            plt.xlim([0, np.ceil(np.max(intensity_integ_plot) / 5) * 5])

    plt.show()

    return peak_info, intensity_plot_arr
