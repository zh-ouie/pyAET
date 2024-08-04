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
        'SPHyn': True,
        'separate_part': 100,
        'PLOT_YN': False
    }
    peak_info, intensity_plot_arr = plot_class_hist(RecVol_padded, temp_model, temp_type, classify_info)
    """
    plothalfSize=classify_info.get('plothalfSize', 4)
    SPHyn=classify_info.get('SPHyn',True)
    separate_part=classify_info.get('separate_part',100)
    PLOT_YN=classify_info.get('PLOT_YN',False)
    halfSize=classify_info.get('halfSize',3)



    Num_types = len(np.unique(temp_type[temp_type >= 0]))
    peak_info = np.zeros((Num_types + 2, separate_part))

    xXp, yYp, zZp = np.meshgrid(
        np.arange(-plothalfSize, plothalfSize + 1),
        np.arange(-plothalfSize, plothalfSize + 1),
        np.arange(-plothalfSize, plothalfSize + 1)
    )

    xXp = np.transpose(xXp, (1,0,2))
    yYp = np.transpose(yYp, (1,0,2))
    zZp = np.transpose(zZp, (1,0,2))

    SphereInd_plot = np.where(((xXp ** 2 + yYp ** 2 + zZp ** 2) <= (plothalfSize + 0.5) ** 2).flatten())[0]

    if SPHyn:
        useInd_plot = SphereInd_plot
    else:
        useInd_plot = np.arange(len(xXp))

    intensity_plot_arr = np.zeros((len(useInd_plot), temp_model.shape[1]))

    intensity_integ_plot = np.zeros(temp_model.shape[1])

    for j in range(temp_model.shape[1]):
        curr_pos = np.round(temp_model[:, j]).astype(int)-1
        # curr_pos = np.round(curr_pos/10).astype(int)-1 #todo: just for testing small data. delete this line.
        box_integ = RecVol_padded[
            curr_pos[0] - plothalfSize : curr_pos[0] + plothalfSize + 1,
            curr_pos[1] - plothalfSize : curr_pos[1] + plothalfSize + 1,
            curr_pos[2] - plothalfSize : curr_pos[2] + plothalfSize + 1
        ]
        intensity_integ_plot[j] = np.sum(box_integ.flatten(order='F')[useInd_plot])
        intensity_plot_arr[:, j] = box_integ.flatten(order='F')[useInd_plot]

    hist_inten_plot, cen_integ_total_plot = np.histogram(intensity_integ_plot, bins=separate_part)
    cen_integ_total_plot = (cen_integ_total_plot[:-1] + cen_integ_total_plot[1:]) / 2

    # Plot the histogram
    plt.figure()
    plt.hist(intensity_integ_plot, bins=separate_part)
    plt.show()

    peak_info[0, :] = cen_integ_total_plot
    peak_info[1, :] = hist_inten_plot
    y_up = round(np.max(hist_inten_plot) / 10) * 12

    if PLOT_YN:
        fig, ax = plt.subplots(Num_types + 1, 1, figsize=(4, 9))
        ax[0].hist(intensity_integ_plot,  bins=cen_integ_total_plot)
        ax[0].set_xlim([0, np.ceil(np.max(intensity_integ_plot) / 5) * 5])
        ax[0].set_ylim([0, y_up])
        ax[0].set_xlabel('Integrated Intensity (a.u.)')
        ax[0].set_ylabel('# Atoms')
        ax[0].set_title(f'Box Size {halfSize * 2 + 1}')

    for i in range(Num_types):
        intensity_integ_sub = intensity_integ_plot[temp_type == i]
        hist_inten_plot_sub, _ = np.histogram(intensity_integ_sub, bins=cen_integ_total_plot)

        # peak_info[i + 2, :] = hist_inten_plot_sub
        # todo: #the original code cannot work. Because shape is not the same. difference 1.
        # temporary solution: use :-1 instead, simply skipping the last element.
        peak_info[i + 2, :-1] = hist_inten_plot_sub

        if PLOT_YN:
            ax[i+1].hist(intensity_integ_sub, bins=cen_integ_total_plot)
            ax[i+1].set_xlabel('Integrated Intensity (a.u.)')
            ax[i+1].set_ylabel('# Atoms')
            ax[i+1].set_title(f'{np.sum(temp_type == i + 1)} Type {i + 1} Atoms')
            ax[i+1].set_ylim([0, y_up])
            ax[i+1].set_xlim([0, np.ceil(np.max(intensity_integ_plot) / 5) * 5])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return peak_info, intensity_plot_arr
