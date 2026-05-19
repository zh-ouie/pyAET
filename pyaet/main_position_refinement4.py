import numpy as np
import os
import argparse
from pathlib import Path
import sys
from scipy.io import loadmat, savemat

# Allow direct execution: python pyAET/pyaet/main_position_refinement4.py
if __package__ is None or __package__ == "":
    _PKG_ROOT = Path(__file__).resolve().parents[1]  # .../pyAET
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

from pyaet.src.gradient_B_2type_difB import gradient_B_2type_difB
from pyaet.src.gradient_fixHB_XYZ import gradient_fixHB_XYZ
from pyaet.src.cal_Bproj_2type_fast import cal_Bproj_2type, cal_Bproj_2type2
from pyaet.src.my_paddzero import my_paddzero
from pyaet.src.io_helper import read_mat_file 
from pyaet.src.matlab_trust_region_lsq import least_squares_matlab_trust_region

def _run_matlab_trust_region_step(x0, xdata, projections, lb_vec, ub_vec, ftol):
    yflat = projections.ravel(order='F')

    def residual(params):
        y_pred, _ = cal_Bproj_2type2(params, xdata)
        return y_pred.ravel(order='F') - yflat

    return least_squares_matlab_trust_region(
        residual,
        x0,
        lb_vec,
        ub_vec,
        tol_fun=ftol,
        max_fun_evals=None,
        max_iter=400,
    )

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

    if os.path.isabs(output_fn):
        output_file_path = output_fn
    else:
        output_file_path = os.path.join(os.path.dirname(projections_file_path), output_fn)
    out_dir = os.path.dirname(output_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Process the data
    projections = np.maximum(projections, 0)
    projections = projections[1:, 1:, :]
    projections = my_paddzero(projections, np.array(projections.shape) + np.array([50, 50, 0]), dtype=np.float64)

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
    hb_history = []
    hb_after_b_history = []
    errR_b_history = []
    errR_xyz_history = []
    lsq_nfev_history = []
    lsq_cost_history = []
    lsq_status_history = []

    for jjjj in range(num_iterations):
        print(f'Iteration num: {jjjj + 1}')
        x0 = para0.copy()
        x0[0, :] /= x0[0, 0]

        xdata['model'] = model_refined.copy()
        xdata['model_ori'] = model_refined.copy()
        xdata['projections'] = projections

        x0 = x0.reshape(-1, order='F')
        lb_vec = lb.reshape(-1, order='F')
        ub_vec = ub.reshape(-1, order='F')

        lsq_result = _run_matlab_trust_region_step(
            x0,
            xdata,
            projections,
            lb_vec,
            ub_vec,
            opt['ftol'],
        )
        para_fit = lsq_result.x
        lsq_nfev_history.append(lsq_result.nfev)
        lsq_cost_history.append(lsq_result.cost)
        lsq_status_history.append(lsq_result.status)

        para0 = para_fit.reshape(para0.shape, order='F')

        print(f'H1 = {para0[0, 0]:.3f}, H2 = {para0[0, 1]:.3f}, H3 = {para0[0, 2]:.3f}')
        print(f'B1 = {para0[1, 0]:.3f}, B2 = {para0[1, 1]:.3f}, B3 = {para0[1, 2]:.3f}')
        hb_history.append(para0.copy())

        y_pred, _ = cal_Bproj_2type(para0, xdata, projections, fit_flag=False)

        # xdata['projections'] = None
        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para0, errR = gradient_B_2type_difB(para0, xdata, projections)
        hb_after_b_history.append(para0.copy())
        errR_b_history.append(np.asarray(errR, dtype=np.float64).copy())

        xdata['step_sz'] = 1
        xdata['iterations'] = 10
        y_pred, para, errR, _ = gradient_fixHB_XYZ(para0, xdata, projections)
        errR_xyz_history.append(np.asarray(errR, dtype=np.float64).copy())
        model_refined = para[2:5, :]

    model_refined_res = model_refined
    np.save(output_file_path + ".npy", model_refined_res)
    hb_hist_arr = np.stack(hb_history, axis=0) if hb_history else np.zeros((0, 2, 3), dtype=np.float64)
    hb_after_b_arr = np.stack(hb_after_b_history, axis=0) if hb_after_b_history else np.zeros((0, 2, 3), dtype=np.float64)
    errR_b_arr = np.array(errR_b_history, dtype=np.float64)
    errR_xyz_arr = np.array(errR_xyz_history, dtype=np.float64)
    np.savez(
        output_file_path + "_compare_stats.npz",
        hb_history=hb_hist_arr,
        hb_after_b_history=hb_after_b_arr,
        errR_b_history=errR_b_arr,
        errR_xyz_history=errR_xyz_arr,
        lsq_nfev_history=np.asarray(lsq_nfev_history, dtype=np.float64),
        lsq_cost_history=np.asarray(lsq_cost_history, dtype=np.float64),
        lsq_status_history=np.asarray(lsq_status_history, dtype=np.float64),
    )
    savemat(
        output_file_path + "_compare_stats.mat",
        {
            'hb_history': hb_hist_arr,
            'hb_after_b_history': hb_after_b_arr,
            'errR_b_history': errR_b_arr,
            'errR_xyz_history': errR_xyz_arr,
            'lsq_nfev_history': np.asarray(lsq_nfev_history, dtype=np.float64),
            'lsq_cost_history': np.asarray(lsq_cost_history, dtype=np.float64),
            'lsq_status_history': np.asarray(lsq_status_history, dtype=np.float64),
        },
    )
    print(f"position refinement finished. Saved: {output_file_path}.npy")
    print(f"compare stats saved: {output_file_path}_compare_stats.npz/.mat")
    return {
        'model_refined_res': model_refined_res,
        'hb_history': hb_hist_arr,
        'hb_after_b_history': hb_after_b_arr,
        'errR_b_history': errR_b_history,
        'errR_xyz_history': errR_xyz_history,
        'output_file_path': output_file_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    workspace_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run position refinement directly from this script.")
    parser.add_argument(
        "--projections",
        default=str(workspace_root / "matlab/1_Measured_data/Projections.mat"),
        help="Path to projections file (.mat or .npy).",
    )
    parser.add_argument(
        "--angles",
        default=str(workspace_root / "matlab/1_Measured_data/Angles.mat"),
        help="Path to angles file (.mat or .npy).",
    )
    parser.add_argument(
        "--model",
        default=str(workspace_root / "matlab/5_Position_refinement/input/Local_classification_coord_OriOri.mat"),
        help="Path to model file (.mat or .npy).",
    )
    parser.add_argument(
        "--atoms",
        default=str(workspace_root / "matlab/5_Position_refinement/input/Local_classification_type.mat"),
        help="Path to atom-type file (.mat or .npy).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of outer refinement iterations. Default: 2",
    )
    parser.add_argument(
        "--output-stem",
        default=str(workspace_root / "pyAET/pyaet/output/position_refinement/model_refined_res_py_iter2"),
        help="Output file stem without extension.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main_position_refinement(
        projections_file_path=args.projections,
        angles_file_path=args.angles,
        model_file_path=args.model,
        atoms_file_path=args.atoms,
        num_iterations=args.iterations,
        output_fn=args.output_stem,
    )

# python /Users/Research/workshop/Debug/202603/generate_interactive_refine_page.py \
#   --mat /Users/Research/workshop/Debug/202603/matlab/5_Position_refinement/output/model_refined_res-2.mat \
#   --py /Users/Research/workshop/Debug/202603/pyAET/pyaet/output/position_refinement/model_refined_res_py_iter2.npy \
#   --out /Users/Research/workshop/Debug/202603/pyAET/pyaet/output/position_refinement/interactive_compare_iter2.html
