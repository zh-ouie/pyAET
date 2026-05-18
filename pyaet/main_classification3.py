import numpy as np
import os
from scipy.interpolate import interpn
from scipy.io import loadmat
from pyaet.src.interp3_spline import interp3_spline
from pyaet.src.my_paddzero import my_paddzero
from pyaet.src.initial_class_kmean_sub import initial_class_kmean_sub
from pyaet.src.plot_class_hist import plot_class_hist
from pyaet.src.local_class_kmean_sub import local_class_kmean_sub
from pyaet.src.my_round import my_round_num
from pyaet.src.io_helper import read_mat_file


def main_classification(Dsetvol_file_path, new_model_file_path, num_species, local_radius, output_fn):
    """
    The main classification function.

    Args:
        Dsetvol_file_path (str): File path. Reconstructed volume, in the shape of (300, 300, 300).
        new_model_file_path (str): File path. Atomic positions, in the shape of (3, 18356).
        num_species (int): Number of atom species.
        local_radius (float): The radius of sphere in local classification. default 10.
        output_fn (str): The output .npy filename, Ex: 'localC_res'.

    Returns:
        Atom types after classification. In the shape of (18356,)
    """
    # Load traced atomic positions and reconstruction volume
    # new_model = np.load(new_model_file_path)
    if new_model_file_path.endswith('.mat'):
        new_model = read_mat_file(new_model_file_path)
    else:
        new_model = np.load(new_model_file_path)

    # Dsetvol = np.load(Dsetvol_file_path)
    FinalVol_double = None
    if Dsetvol_file_path.endswith('.mat'):
        data = loadmat(Dsetvol_file_path)
        if 'FinalVol_single' in data:
            FinalVol_double = data['FinalVol_single'].astype(np.float64)
        elif 'FinalVol' in data:
            FinalVol_double = data['FinalVol'].astype(np.float64)
        else:
            data_keys = [key for key in data.keys() if not key.startswith('__')]
            if data_keys:
                Dsetvol = data[data_keys[0]]
            else:
                raise Exception(f'No valid data found in .mat file: {Dsetvol_file_path}')
    else:
        Dsetvol = np.load(Dsetvol_file_path)

    # new_model = np.load('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/traced_model_inPixel.npy')  # ,allow_pickle=True
    # Dsetvol = np.load('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/MG_reconstruction_volume.npy')

    # new_model_full = np.load(new_model_file_path)  # ,allow_pickle=True
    # Dsetvol_full = np.load(Dsetvol_file_path)
    # new_model = new_model_full[:,2000:5000]
    # Dsetvol = Dsetvol_full[199:230,199:230,199:230]

    # new_model = new_model[:,2000:5000]
    # Dsetvol = Dsetvol[199:230,199:230,199:230]

    # print(new_model.shape)
    # print(Dsetvol.shape)

    output_file_path = os.path.join(os.path.dirname(Dsetvol_file_path), output_fn)

    # Upsample the reconstruction matrix by 3*3*3 by linear interpolation
    if FinalVol_double is None:
        xx = np.arange(Dsetvol.shape[0]) - my_round_num((Dsetvol.shape[0]+1)/2) + 1
        yy = np.arange(Dsetvol.shape[1]) - my_round_num((Dsetvol.shape[1]+1)/2) + 1
        zz = np.arange(Dsetvol.shape[2]) - my_round_num((Dsetvol.shape[2]+1)/2) + 1

        xxi = np.arange(3 * xx[0], xx[-1] * 3 + 1) / 3
        yyi = np.arange(3 * yy[0], yy[-1] * 3 + 1) / 3
        zzi = np.arange(3 * zz[0], zz[-1] * 3 + 1) / 3

        xxi = xxi[2:]
        yyi = yyi[2:]
        zzi = zzi[2:]

        use_spline = True
        if use_spline:
            Dsetvol = interp3_spline(Dsetvol, yy, xx, zz, yyi, xxi, zzi)
        else:
            points = (yy, xx, zz)
            Yi, Xi, Zi = np.meshgrid(yyi, xxi, zzi)
            Dsetvol = interpn(points, Dsetvol, (Yi, Xi, Zi), method='cubic', bounds_error=False, fill_value=0)

        FinalVol = my_paddzero(Dsetvol, np.array(Dsetvol.shape) + 20)
        FinalVol_double = FinalVol.astype(np.float64)
    # check data by FinalVol_single[:,:,50]

    # Apply global k-mean classification on the reconstruction
    classify_info = {
        'num_species': num_species,
        'half_size': 3,
        'plot_half_size': 1,
        'O_Ratio': 1,
        'SPHyn': True,
        'PLOT_YN': False,
        'separate_part': 70,
        'simple_kmeans': True,
        'matlab_label': True
    }

    # Calculate the atomic position with upsampled volume
    new_model_L = (new_model + 2) * 3

    # Apply k-mean classification on the reconstruction
    atom_model, global_class_atomtype = initial_class_kmean_sub(
        FinalVol_double, new_model_L, classify_info)

    # Apply function 'plot_class_hist()' to achieve the histogram information 'peak_info_global_classification'
    peak_info_global_classfication, _ = plot_class_hist(
        FinalVol_double, atom_model, global_class_atomtype, classify_info)

    # Apply local k-mean classification on the reconstruction by the results of global k-mean
    temp_class_atomtype = global_class_atomtype

    classify_info['radius'] = local_radius / 0.347 * 3  # Radius is 10A

    # When there are 5 iterations with the same number of atoms flipped (back and forth),
    # the iteration will be stopped
    atom_model, local_class_atomtype = local_class_kmean_sub(
        FinalVol_double, atom_model, temp_class_atomtype, classify_info)

    # Apply function 'plot_class_hist()' to achieve the histogram information 'peak_info_local_classification'
    # Please see the descriptions in subfunction to get more details
    peak_info_local_classification, _ = plot_class_hist(
        FinalVol_double, atom_model, local_class_atomtype, classify_info)

    # Save 'local_atomtype' to a file or process it further
    np.save(output_file_path+".npy", local_class_atomtype)
    print("classification finished.")
    return

'''
#use the following code to get npy version of data:
import scipy
mat = scipy.io.loadmat(r'input\traced_model_inPixel.mat')
data = mat['traced_model_inPixel']
np.save('traced_model_inPixel.npy', data)
mat = scipy.io.loadmat(r'input\MG_reconstruction_volume.mat')
data = mat['final_Rec']
np.save('MG_reconstruction_volume.npy', data)
'''


# new_model_file_path = 'input/traced_model_inPixel.npy'
# Dsetvol_file_path = 'input/MG_reconstruction_volume.npy'
# num_species = 3
# local_radius = 10
# output_fn='localC_res'

# main_classification(Dsetvol_file_path, new_model_file_path, num_species, local_radius, output_fn)
