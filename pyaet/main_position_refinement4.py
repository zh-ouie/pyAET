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

# Release settings for Step4.
STEP4_HB_OPTIMIZER = "matlab_trust_region"
STEP4_LSQ_MAX_NFEV = None
STEP4_LSQ_MAX_ITER = 400
from pyaet.src.my_paddzero import my_paddzero
from scipy.optimize import least_squares
from pyaet.src.io_helper import read_mat_file 
from pyaet.src.matlab_trust_region_lsq import least_squares_matlab_trust_region

def _residuals(params, xdata, ydata):
    y_pred, _ = cal_Bproj_2type2(params, xdata)
    return y_pred.ravel(order='F') - ydata.ravel(order='F')

def _pack_params_with_fixed_h1(params5):
    """
    Reconstruct 6-parameter vector in Fortran order with H1 fixed to 1.0.
    Full vector order (F): [H1, B1, H2, B2, H3, B3]
    params5 order:         [B1, H2, B2, H3, B3]
    """
    full = np.empty(6, dtype=np.float64)
    full[0] = 1.0
    full[1] = params5[0]
    full[2] = params5[1]
    full[3] = params5[2]
    full[4] = params5[3]
    full[5] = params5[4]
    return full

def _residuals_fixed_h1(params5, xdata, ydata):
    return _residuals(_pack_params_with_fixed_h1(params5), xdata, ydata)

def _run_lsqcurvefit_like_step(x0, xdata, projections, lb_vec, ub_vec, ftol, max_nfev):
    """
    Keep the parameterization close to MATLAB:
    optimize all 6 variables, but hold H1 with a very tight bound band.

    The release MATLAB code only sets TolFun and does not cap MaxIter, so do
    not force a small max_nfev here.
    """
    result = least_squares(
        _residuals,
        x0,
        args=(xdata, projections),
        bounds=(lb_vec, ub_vec),
        method='trf',
        ftol=ftol,
        jac='2-point',
        max_nfev=max_nfev,
    )
    return result

def _run_matlab_trust_region_step(x0, xdata, projections, lb_vec, ub_vec, ftol, max_nfev, max_iter):
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
        max_fun_evals=max_nfev,
        max_iter=max_iter,
    )

def _run_matlab_trust_region_fixed_h1_step(x0, xdata, projections, ftol, max_nfev, max_iter):
    """
    Match the release MATLAB parameter intent with H1 fixed at exactly 1.0.

    MATLAB's lsqcurvefit accepts equal lower/upper bounds for H1 in
    Main_position_refinement.m. Python's full 6-parameter compatibility path
    uses a narrow H1 bound band instead. This 5-parameter path removes H1 from
    the optimization variables while still evaluating the same 6-parameter
    forward model.
    """
    yflat = projections.ravel(order='F')
    x0 = np.asarray(x0, dtype=np.float64)
    x5 = np.array([x0[1], x0[2], x0[3], x0[4], x0[5]], dtype=np.float64)
    lb5 = np.array([5.0, 1.0, 5.0, 1.0, 5.0], dtype=np.float64)
    ub5 = np.array([15.0, 2.0, 15.0, 3.0, 15.0], dtype=np.float64)

    def residual(params5):
        full = _pack_params_with_fixed_h1(params5)
        y_pred, _ = cal_Bproj_2type2(full, xdata)
        return y_pred.ravel(order='F') - yflat

    result5 = least_squares_matlab_trust_region(
        residual,
        x5,
        lb5,
        ub5,
        tol_fun=ftol,
        max_fun_evals=max_nfev,
        max_iter=max_iter,
    )
    result5.x = _pack_params_with_fixed_h1(result5.x)
    return result5

def _gauss_newton_step(params, xdata, ydata, lb, ub, diff_step, lm_lambda):
    params = np.asarray(params, dtype=np.float64)
    r0 = _residuals(params, xdata, ydata)
    n_params = params.size
    J_cols = []
    JTr = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        step = diff_step * max(1.0, abs(params[i]))
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += step
        p_minus[i] -= step
        r_plus = _residuals(p_plus, xdata, ydata)
        r_minus = _residuals(p_minus, xdata, ydata)
        Jcol = (r_plus - r_minus) / (2 * step)
        J_cols.append(Jcol)
        JTr[i] = np.dot(Jcol, r0)

    JTJ = np.empty((n_params, n_params), dtype=np.float64)
    for i in range(n_params):
        for j in range(i, n_params):
            val = np.dot(J_cols[i], J_cols[j])
            JTJ[i, j] = val
            JTJ[j, i] = val

    if lm_lambda is not None and lm_lambda > 0:
        JTJ = JTJ + lm_lambda * np.eye(n_params, dtype=np.float64)

    step_vec = -np.linalg.solve(JTJ, JTr)
    new_params = params + step_vec
    if lb is not None:
        new_params = np.maximum(new_params, lb)
    if ub is not None:
        new_params = np.minimum(new_params, ub)
    return new_params, r0

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
    use_gn_step = False
    gn_diff_step = 1e-6
    gn_lambda = 1e-6
    lsq_max_nfev = STEP4_LSQ_MAX_NFEV
    lsq_max_iter = STEP4_LSQ_MAX_ITER
    hb_optimizer = STEP4_HB_OPTIMIZER
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

        if use_gn_step:
            para_fit, _ = _gauss_newton_step(x0, xdata, projections, lb_vec, ub_vec, gn_diff_step, gn_lambda)
            lsq_nfev_history.append(np.nan)
            lsq_cost_history.append(np.nan)
            lsq_status_history.append(0)
        elif hb_optimizer in {"matlab_trust_region", "matlab-trust-region", "matlab_trf"}:
            lsq_result = _run_matlab_trust_region_step(
                x0,
                xdata,
                projections,
                lb_vec,
                ub_vec,
                opt['ftol'],
                lsq_max_nfev,
                lsq_max_iter,
            )
            para_fit = lsq_result.x
            lsq_nfev_history.append(lsq_result.nfev)
            lsq_cost_history.append(lsq_result.cost)
            lsq_status_history.append(lsq_result.status)
        elif hb_optimizer in {
            "matlab_trust_region_fixed_h1",
            "matlab-trust-region-fixed-h1",
            "matlab_trf_fixed_h1",
            "matlab_trust_region_fixedh1",
            "matlab-trust-region-fixedh1",
            "matlab_trf_fixedh1",
        }:
            lsq_result = _run_matlab_trust_region_fixed_h1_step(
                x0,
                xdata,
                projections,
                opt['ftol'],
                lsq_max_nfev,
                lsq_max_iter,
            )
            para_fit = lsq_result.x
            lsq_nfev_history.append(lsq_result.nfev)
            lsq_cost_history.append(lsq_result.cost)
            lsq_status_history.append(lsq_result.status)
        else:
            lsq_result = _run_lsqcurvefit_like_step(
                x0,
                xdata,
                projections,
                lb_vec,
                ub_vec,
                opt['ftol'],
                lsq_max_nfev,
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
