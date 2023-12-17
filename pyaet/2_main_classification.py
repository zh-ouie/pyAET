import numpy as np
from scipy.interpolate import interpn
from src.My_paddzero import My_paddzero
from src.initial_class_kmean_sub import initial_class_kmean_sub
from src.plot_class_hist import plot_class_hist
from src.local_class_kmean_sub import local_class_kmean_sub

def main_classification(new_model, Dsetvol):
    # Load traced atomic positions and reconstruction volume
    # new_model = np.load('traced_model_inPixel.npy')
    # Dsetvol = np.load('MG_reconstruction_volume.npy')
    # Upsample the reconstruction matrix by 3*3*3 by linear interpolation
    xx =  np.linspace(-(Dsetvol.shape[0]-1)/2,(Dsetvol.shape[0]-1)/2,Dsetvol.shape[0])
    yy =  np.linspace(-(Dsetvol.shape[1]-1)/2,(Dsetvol.shape[1]-1)/2,Dsetvol.shape[1])
    zz =  np.linspace(-(Dsetvol.shape[2]-1)/2,(Dsetvol.shape[2]-1)/2,Dsetvol.shape[2])

    xxi = np.arange(3 * xx[0], xx[-1] * 3 + 1) / 3
    yyi = np.arange(3 * yy[0], yy[-1] * 3 + 1) / 3
    zzi = np.arange(3 * zz[0], zz[-1] * 3 + 1) / 3

    xxi = xxi[2:]  # Skip the first two elements
    yyi = yyi[2:]  # Skip the first two elements
    zzi = zzi[2:]  # Skip the first two elements
    points=(yy, xx, zz)
    Yi, Xi, Zi = np.meshgrid(yyi, xxi, zzi)
    
    Dsetvol = interpn(points, Dsetvol, (Yi, Xi, Zi), method='cubic', bounds_error=False, fill_value=0)
    '''
    #another possible version:
    from scipy.interpolate import RegularGridInterpolator
    reg=RegularGridInterpolator(points, Dsetvol)
    Dsetvol=reg((Yi, Xi, Zi))
    '''
    FinalVol = My_paddzero(Dsetvol, (Dsetvol.shape[0] + 20, Dsetvol.shape[1] + 20, Dsetvol.shape[2] + 20))
    FinalVol_single = FinalVol.astype(np.single)

    # Apply global k-mean classification on the reconstruction
    classify_info = {
        'Num_species': 3,
        'halfSize': 3,
        'plothalfSize': 1,
        'O_Ratio': 1,
        'SPHyn': 1,
        'PLOT_YN': 1,
        'separate_part': 70
    }

    # Calculate the atomic position with upsampled volume
    new_model_L = (new_model + 2) * 3

    # Apply k-mean classification on the reconstruction
    atom_model, global_class_atomtype = initial_class_kmean_sub(
        FinalVol_single, new_model_L, classify_info)

    # Apply function 'plot_class_hist()' to achieve the histogram information 'peak_info_global_classification'
    # Please see the descriptions in subfunction to get more details
    peak_info_global_classification = plot_class_hist(
        FinalVol_single, atom_model, global_class_atomtype, classify_info)

    # Apply local k-mean classification on the reconstruction by the results of global k-mean
    temp_class_atomtype = global_class_atomtype

    classify_info['Radius'] = 10 / 0.347 * 3  # Radius is 10A

    # When there are 5 iterations with the same number of atoms flipped (back and forth),
    # the iteration will be stopped
    atom_model, local_class_atomtype = local_class_kmean_sub(
        FinalVol_single, atom_model, temp_class_atomtype, classify_info)

    # Apply function 'plot_class_hist()' to achieve the histogram information 'peak_info_local_classification'
    # Please see the descriptions in subfunction to get more details
    peak_info_local_classification = plot_class_hist(
        FinalVol_single, atom_model, local_class_atomtype, classify_info)

    # Save 'local_atomtype' to a file or process it further
    np.save('output/localC_res.npy', local_class_atomtype)

'''
#use the following code to get npy version of data:
mat = scipy.io.loadmat('input\traced_model_inPixel.mat')
data = mat['']
np.save('traced_model_inPixel.npy', data)
mat = scipy.io.loadmat('input\MG_reconstruction_volume.mat')
data = mat['final_Rec']
np.save('MG_reconstruction_volume.npy', data)
'''

newmodel = np.load('traced_model_inPixel.npy',allow_pickle=True)
Dsetvol = np.load('MG_reconstruction_volume.npy')
main_classification(newmodel, Dsetvol)