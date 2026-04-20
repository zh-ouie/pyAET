import numpy as np
import os
from scipy.optimize import least_squares
from scipy.io import loadmat, savemat
from pyaet.src.gradient_B_2type_difB import gradient_B_2type_difB
from pyaet.src.gradient_fixHB_XYZ import gradient_fixHB_XYZ
from pyaet.src.cal_Bproj_2type import cal_Bproj_2type
from pyaet.src.my_paddzero import my_paddzero
from scipy.optimize import least_squares
from pyaet.src.io_helper import read_mat_file 

# Define the fatom_vector and fparameters functions here

def main_position_refinement(projections_file_path, angles_file_path, model_file_path, atoms_file_path, num_iterations, output_fn):
    """
    Refine atomic coordinates with an existing model, type, and reconstruction volume by minimizing
    the error between atomic coordinates and measured projections.

    Args:
        projections_file_path (str): File path. Projections, in the shape of (300, 300, 55).
        angles_file_path (str): File path. Angles, in the shape of (55, 3).
        model_file_path (str): File path. Model, in the shape of (3, 18356).
        atoms_file_path (str): File path. Atoms, in the shape of (1, 18356).
        num_iterations (int): Number of iterations.
        output_fn (str): The output .npy filename, Ex: 'initial_traced_model'.
    """

    # Add paths and load data
    # Add the path to load measured projections and angles
    # You can comment it and move the projections and angles into the input folder
    # projections = sio.loadmat('../1_Measured_data/Projections.mat')['Projections']
    # angles = sio.loadmat('../1_Measured_data/Angles.mat')['Angles']
    # model = sio.loadmat('../1_Measured_data/Local_classification_coord_OriOri.mat')['Local_classification_coord_OriOri']
    # atoms = sio.loadmat('../1_Measured_data/Local_classification_type.mat')['Local_classification_type']

    # projections_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4projections.npy'
    # angles_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4angles.npy'
    # model_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4model.npy'
    # atoms_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4atoms.npy'

    # projections = np.load(projections_file_path)
    if projections_file_path.endswith('.mat'):
        projections = read_mat_file(projections_file_path)
    else:
        projections = np.load(projections_file_path)

    # angles = np.load(angles_file_path)
    if angles_file_path.endswith('.mat'):
        angles = read_mat_file(angles_file_path)
    else:
        angles = np.load(angles_file_path)

    # model = np.load(model_file_path)
    if model_file_path.endswith('.mat'):
        model = read_mat_file(model_file_path)
    else:
        model = np.load(model_file_path)

    # atoms = np.load(atoms_file_path)
    if atoms_file_path.endswith('.mat'):
        atoms = read_mat_file(atoms_file_path)
    else:
        atoms = np.load(atoms_file_path)

    output_file_path = os.path.join(os.path.dirname(projections_file_path), output_fn)

    # Process the data
    projections = np.maximum(projections, 0)
    projections = projections[1:, 1:, :]
    projections = my_paddzero(projections, np.array(projections.shape) + np.array([50, 50, 0]))

    N1, N2, num_pj = projections.shape
    half_width = 4
    # the atomic number for different type:
    # use 28 for type 1, 45 for type 2, 78 for type 3
    Z_arr = [28, 45, 78]
    Res = 0.347

    xdata = {
        'Res': Res,
        'Z_arr': Z_arr,
        'half_width': half_width,
        'atoms': atoms,
        'model': model,
        'angles': angles,
    }

    # Initial parameter guess for H and B factor estimation
    para0 = np.array([[1, 1.36, 2.58], [13.3, 13.3, 13.3]], dtype=np.float64)
    lb = np.array([[0.999999, 1, 1], [5, 5, 5]], dtype=np.float64) #long edit
    ub = np.array([[1.000001, 2, 3], [15, 15, 15]], dtype=np.float64) #long edit
    model_refined = model.copy()

    opt = {'ftol': 1e-12}

    for jjjj in range(num_iterations):
        print(f'Iteration num: {jjjj + 1}')
        x0 = para0.copy()
        x0[0, :] /= x0[0, 0]

        xdata['model'] = model_refined.copy()
        xdata['model_ori'] = model_refined.copy()
        xdata['projections'] = projections

        # Define the residuals function for least_squares
        def residuals(params, xdata, projections):
            y_pred, _ = cal_Bproj_2type(params, xdata, projections)
            return y_pred.flatten() - projections.flatten()

        #in python, params x0 has to be in 1D.
        x0 = x0.flatten()
        lb = lb.flatten()
        ub = ub.flatten()

        result = least_squares(residuals, x0, args=(xdata, projections), bounds=(lb, ub), method='trf', ftol=opt['ftol'])
        # result = least_squares(residuals, x0, args=(xdata, projections), bounds=(lb, ub), method='trf', max_nfev=1)

        para_fit = result.x
        para0 = para_fit.reshape(para0.shape) #reshape it back.

        print(f'H1 = {para0[0, 0]:.3f}, H2 = {para0[0, 1]:.3f}, H3 = {para0[0, 2]:.3f}')
        print(f'B11 = {para0[1, 0]:.3f}, B2 = {para0[1, 1]:.3f}, B3 = {para0[1, 2]:.3f}')

        y_pred, _ = cal_Bproj_2type(para0, xdata, projections, fit_flag=False)

        # xdata['projections'] = None
        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para0, errR = gradient_B_2type_difB(para0, xdata, projections)

        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para, errR, _ = gradient_fixHB_XYZ(para0, xdata, projections)
        model_refined = para[2:5, :]

    model_refined_res = model_refined
    # savemat('output/model_refined_res.mat', {'model_refined_res': model_refined_res})
    np.save(output_file_path+".npy", model_refined_res)
    print("position refinement finished.")
    return


# projections_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4projections.npy'
# angles_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4angles.npy'
# model_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4model.npy'
# atoms_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/4atoms.npy'
# num_iterations = 10
# output_fn = 'model_refined_res'

# main_position_refinement(projections_file_path, angles_file_path, model_file_path, atoms_file_path, num_iterations, output_fn)