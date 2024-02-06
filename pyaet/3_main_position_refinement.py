import numpy as np
import scipy.optimize as optimize
import scipy.io as sio
from src.My_paddzero import My_paddzero
from src.Cal_Bproj_2type import Cal_Bproj_2type
from src.gradient_B_2type_difB import gradient_B_2type_difB
from src.gradient_fixHB_XYZ import gradient_fixHB_XYZ
from src.Cal_Bproj_2type2 import  Cal_Bproj_2type2

# Define the fatom_vector and fparameters functions here

def main_position_refinement():
    """
    Refine atomic coordinates with an existing model, type, and reconstruction volume by minimizing
    the error between atomic coordinates and measured projections.
    """
    # Add paths and load data
    # Add the path to load measured projections and angles
    # You can comment it and move the projections and angles into the input folder
    projections = sio.loadmat('../1_Measured_data/Projections.mat')['Projections']
    angles = sio.loadmat('../1_Measured_data/Angles.mat')['Angles']
    model = sio.loadmat('../1_Measured_data/Local_classification_coord_OriOri.mat')['Local_classification_coord_OriOri']
    atoms = sio.loadmat('../1_Measured_data/Local_classification_type.mat')['Local_classification_type']

    # Process the data
    projections = np.maximum(projections, 0)
    projections = projections[1:, 1:, :]
    projections = My_paddzero(projections, projections.shape + (50, 50, 0), dtype=np.float64)

    N1, N2, num_pj = projections.shape[:3]
    halfWidth = 4
    Z_arr = [28, 45, 78]
    Res = 0.347

    xdata = {
        'Res': Res,
        'Z_arr': Z_arr,
        'halfWidth': halfWidth,
        'atoms': atoms,
        'model': model,
        'angles': angles,
    }

    # Initial parameter guess for H and B factor estimation
    para0 = np.array([[1, 1.36, 2.58], [13.3, 13.3, 13.3]], dtype=np.float64)
    lb = np.array([[1, 1, 1], [5, 5, 5]], dtype=np.float64)
    ub = np.array([[1, 2, 3], [15, 15, 15]], dtype=np.float64)
    model_refined = model.copy()

    opt = {'ftol': 1e-12}

    for jjjj in range(10):
        print(f'Iteration num: {jjjj + 1}')
        x0 = para0.copy()
        x0[0, :] /= x0[0, 0]

        xdata['model'] = model_refined.copy()
        xdata['model_ori'] = model_refined.copy()
        xdata['projections'] = projections

        para0, _, _ = optimize.curve_fit(
            Cal_Bproj_2type2, x0, xdata, projections, bounds=(lb, ub), method='trf', options=opt
        )
        print(f'H1 = {para0[0, 0]:.3f}, H2 = {para0[0, 1]:.3f}, H3 = {para0[0, 2]:.3f}')
        print(f'B11 = {para0[1, 0]:.3f}, B2 = {para0[1, 1]:.3f}, B3 = {para0[1, 2]:.3f}')

        y_pred, _ = Cal_Bproj_2type(para0, xdata, projections)
        xdata['projections'] = None

        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para0, errR = gradient_B_2type_difB(para0, xdata, projections)

        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para, errR = gradient_fixHB_XYZ(para0, xdata, projections)
        model_refined = para[2:5, :]

    model_refined_res = model_refined
    sio.savemat('output/model_refined_res.mat', {'model_refined_res': model_refined_res})
