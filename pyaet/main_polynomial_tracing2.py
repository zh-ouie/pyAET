import json
import numpy as np
import os
import time
from pathlib import Path
from scipy.ndimage import grey_dilation
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from pyaet.src.strel3d import strel3d
from pyaet.src.my_paddzero import my_paddzero
from pyaet.src.obtain_tight_support import obtain_tight_support
from pyaet.src.my_round import my_round_num
from pyaet.src.calculate_3D_polynomial_Rogers import calculate_3D_polynomial_Rogers
from pyaet.src.calc_dX_dY_dZ_Rogers import calc_dX_dY_dZ_Rogers
from pyaet.src.initial_class_kmean import initial_class_kmean
from pyaet.src.io_helper import read_mat_file
from pyaet.src.interp3_spline import interp3_spline

def main_polynomial_tracing(Dsetvol_file_path, max_num_th, min_dist, output_fn):
    """
    The main polynomial tracing function.

    Args:
        Dsetvol_file_path (str): File path. Reconstructed volume, in the shape of (300, 300, 300).
        max_num_th (int): The maximum atom numbers for tracing.
        min_dist (float): The minimum inter-atomic distance. In units of A.
        output_fn (str): The output .npy filename, Ex: 'initial_traced_model'.

    Returns:

    """
    # Add path for user-defined functions
    # addpath('src/')

    # Add the path to load the reconstruction volume (you can comment it and move
    # the reconstruction into an input folder)
    # addpath('../3_Final_reconstruction_volume/') ;

    # Read in files: reconstruction volume
    # Dsetvol = np.load(Dsetvol_file_path)

    # Dsetvol_full = np.load('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/MG_reconstruction_volume.npy')
    # Dsetvol = Dsetvol_full[199:230, 199:230, 199:230]

    if Dsetvol_file_path.endswith('.mat'):
        Dsetvol = read_mat_file(Dsetvol_file_path)
    else:
        Dsetvol = np.load(Dsetvol_file_path)
    print("PID:", os.getpid())

    output_file_path = os.path.join(os.path.dirname(Dsetvol_file_path), output_fn)
    output_dir = Path(output_fn).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_output_path = os.environ.get("PYAET_TRACING_DEBUG_OUTPUT", "").strip()
    if not debug_output_path:
        debug_output_path = str(output_dir / f"tracing_debug_{max_num_th}")
    initial_class_debug_output = os.environ.get("PYAET_INITIAL_CLASS_DEBUG_OUTPUT", "").strip()
    if not initial_class_debug_output:
        initial_class_debug_output = str(output_dir / f"initial_class_debug_{max_num_th}")
    stage_start = time.perf_counter()

    # Constants
    max_iter = 14
    crit_iter = 7
    Th = 1
    Res = 0.347 / 3
    # min_dist = 2 / Res
    min_dist = min_dist / Res
    search_rad = 3

    # Upsample the reconstruction matrix by 3*3*3 using linear interpolation
    xx = np.arange(Dsetvol.shape[0]) - my_round_num((Dsetvol.shape[0]+1)/2) + 1
    yy = np.arange(Dsetvol.shape[1]) - my_round_num((Dsetvol.shape[1]+1)/2) + 1
    zz = np.arange(Dsetvol.shape[2]) - my_round_num((Dsetvol.shape[2]+1)/2) + 1

    xxi = np.arange(3 * xx[0], xx[-1] * 3 + 1) / 3
    yyi = np.arange(3 * yy[0], yy[-1] * 3 + 1) / 3
    zzi = np.arange(3 * zz[0], zz[-1] * 3 + 1) / 3

    xxi = xxi[2:]  # Skip the first two elements
    yyi = yyi[2:]  # Skip the first two elements
    zzi = zzi[2:]  # Skip the first two elements

    Dsetvol = interp3_spline(Dsetvol, yy, xx, zz, yyi, xxi, zzi)
    FinalVol = my_paddzero(Dsetvol, np.array(Dsetvol.shape) + 20)

    # Get polynomial power array
    fit_coeff = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i + j + k <= 4:
                    if max([i, j, k]) == 4:
                        fit_coeff.append([i, j, k, -1])
                    else:
                        fit_coeff.append([i, j, k, 0])
    fit_coeff = np.array(fit_coeff)

    # Get the local maxima from the reconstruction volume
    se = strel3d(3)

    # dilatedBW = cv2.dilate(FinalVol,se)
    # dilatedBW = label(FinalVol == dilatedBW & (FinalVol > Th))[0]
    # max_pos=np.where(FinalVol==dilatedBW && FinalVol>Th)
    # max_vals = FinalVol[dilatedBW > 0]
    # sort_ind = np.argsort(max_vals)[::-1]
    # max_num = min(max_num_th, len(sort_ind))
    # max_pos = np.argwhere(dilatedBW > 0)[sort_ind[:max_num]]

    # from skimage.morphology import dilation, cube
    # dilatedBW = dilation(FinalVol, se) #same result

    dilatedBW = grey_dilation(FinalVol, footprint=se)

    max_pos = np.where(((FinalVol == dilatedBW) & (FinalVol > Th)).flatten(order='F'))[0]
    # max_pos = np.where((((FinalVol - dilatedBW) < 1e-7) & (FinalVol > Th)).flatten())[0]
    max_vals = FinalVol.flatten(order='F')[max_pos]
    sort_ind = np.argsort(max_vals)[::-1]
    max_num = min(max_num_th, len(sort_ind))
    max_pos = max_pos[sort_ind[:max_num]]

    print('numpeak =', len(max_pos))

    max_XYZ = np.zeros((len(max_pos), 3),dtype=int)
    for i in range(len(max_pos)):
        xx_id, yy_id, zz_id = np.unravel_index(max_pos[i], FinalVol.shape, order='F')
        max_XYZ[i, :] = np.array([xx_id, yy_id, zz_id])
    max_XYZ_base = max_XYZ + 1

    # Initialize parameters
    Q = 0.5
    alpha = 1
    crop_half_size = search_rad
    X, Y, Z = np.meshgrid(np.arange(-crop_half_size, crop_half_size + 1),
                         np.arange(-crop_half_size, crop_half_size + 1),
                         np.arange(-crop_half_size, crop_half_size + 1))
    X = np.transpose(X, (1,0,2))
    Y = np.transpose(Y, (1,0,2))
    Z = np.transpose(Z, (1,0,2))

    SphereInd = np.where((X**2 + Y**2 + Z**2 <= (search_rad + 0.5)**2).flatten(order='F'))[0]
    XYZdata = {'X': X.flatten(order='F')[SphereInd], 'Y': Y.flatten(order='F')[SphereInd], 'Z': Z.flatten(order='F')[SphereInd]}

    orders = fit_coeff[:, :3]
    pos_arr = np.zeros_like(max_XYZ, dtype=float)
    tot_pos_arr = np.zeros_like(max_XYZ, dtype=float)

    exit_flag_arr = np.zeros(len(max_XYZ), dtype=int)
    coeff_arr = np.tile(fit_coeff[:, 3], (len(max_XYZ), 1)).T
    coeff_arr = coeff_arr.astype('float')

    # Perform the main tracing loop
    # todo: still some error in the loop. in python, all exitFlagArr is negative. But we want 0.
    peak_loop_start = time.perf_counter()
    least_squares_s = 0.0
    for i in range(len(max_XYZ)):
        end_flag = False
        consec_accum = 0
        iter_num = 0
        while not end_flag:
            iter_num += 1
            if iter_num > max_iter:
                exit_flag_arr[i] = -4
                end_flag = True
            cropXind = max_XYZ[i, 0] + np.arange(-crop_half_size, crop_half_size + 1)
            cropYind = max_XYZ[i, 1] + np.arange(-crop_half_size, crop_half_size + 1)
            cropZind = max_XYZ[i, 2] + np.arange(-crop_half_size, crop_half_size + 1)

            cropVol = FinalVol[cropXind[0]:cropXind[-1]+1, cropYind[0]:cropYind[-1]+1, cropZind[0]:cropZind[-1]+1]
            # cropVol = np.transpose(cropVol, (1,0,2))

            pos = pos_arr[i, :]
            gauss_weight = np.exp(-1 * alpha * ((XYZdata['X'] - pos[0]) ** 2 +
                                                (XYZdata['Y'] - pos[1]) ** 2 +
                                                (XYZdata['Z'] - pos[2]) ** 2) / crop_half_size ** 2)

            # Define the objective function for optimization
            # def fun(p, xdata):
            #     return calculate_3D_polynomial_Rogers(xdata['X'], xdata['Y'], xdata['Z'], pos, orders, p) * gauss_weight
            #
            #
            # # Initial coefficients for optimization
            # p0 = coeff_arr[:, i]
            #
            # # Use scipy's least_squares function for optimization
            # res = least_squares(fun, p0, args=(XYZdata, cropVol.flatten(order='F')[SphereInd]*gauss_weight), method='trf', verbose=0) #todo: different result.

            def residuals(params, XYZdata, cropVol, pos, orders):
                # Calculate model predictions
                predictions = calculate_3D_polynomial_Rogers(XYZdata['X'], XYZdata['Y'], XYZdata['Z'], pos, orders,
                                                             params) * gauss_weight
                # Calculate residuals
                return predictions - (cropVol.flatten(order='F')[SphereInd]*gauss_weight)

            x0 = coeff_arr[:, i]
            lb = -np.inf * np.ones_like(x0)  # Example lower bounds
            ub = np.inf * np.ones_like(x0)  # Example upper bounds
            lsq_start = time.perf_counter()
            res = least_squares(residuals, x0, args=(XYZdata, cropVol, pos, orders), bounds=(lb, ub), method='trf', verbose=0)
            least_squares_s += time.perf_counter() - lsq_start
            p1 = res.x # optimized parameters
            residuals = res.fun
            fminres1 = np.sum(residuals ** 2) # calculate the squared 2-norm of the residuals

            coeff_arr[:, i] = p1

            dX, dY, dZ = calc_dX_dY_dZ_Rogers(orders, p1)
            if dX == -100 and dY == -100 and dZ == -100:
                exit_flag_arr[i] = -1
                end_flag = True
            else:
                maxedShift = np.maximum(np.array([dX, dY, dZ]).reshape(3,), -1 * np.array([Q, Q, Q]))
                minedShift = np.minimum(maxedShift, np.array([Q, Q, Q]))
                pos_arr[i, :] = pos_arr[i, :] + minedShift
                if np.max(np.abs(pos_arr[i, :])) > crop_half_size:
                    exit_flag_arr[i] = -2
                    end_flag = True
                elif np.max(np.abs(minedShift)) < Q:
                    if consec_accum == crit_iter - 1:
                        goodAtomTotPos = tot_pos_arr[:i, :]
                        goodAtomTotPos = goodAtomTotPos[exit_flag_arr[:i] == 0, :]
                        # dist = np.sqrt(np.sum(
                        #     (goodAtomTotPos - np.tile(pos_arr[i, :] + max_XYZ[i, :], (goodAtomTotPos.shape[0], 1))) ** 2,
                        #     axis=1))
                        # Long change from max_XYZ[i, :] to max_XYZ[i, :] +1
                        dist = np.sqrt(np.sum(
                            (goodAtomTotPos - np.tile(pos_arr[i, :] + max_XYZ_base[i, :], (goodAtomTotPos.shape[0], 1))) ** 2,
                            axis=1))
                        if len(dist) == 0:
                            dist = np.array([np.inf])
                        if np.min(dist) < min_dist:
                            exit_flag_arr[i] = -3
                        else:
                            tot_pos_arr[i, :] = pos_arr[i, :] + max_XYZ_base[i, :]
                        end_flag = True
                    else:
                        consec_accum += 1
                else:
                    consec_accum = 0

        if i % 500 == 0 or i == len(max_XYZ) - 1:
            print(f'peak {i}, flag {exit_flag_arr[i]}')

    # Do raw classification and get all candidates for manual tracing
    FinalVol_single = FinalVol.astype(np.single)

    classify_info = {
        'num_species': 3,
        'half_size': 3,
        'plot_half_size': 1,
        'O_Ratio': 1,
        'SPHyn': True,
        'PLOT_YN': False,
        'separate_part': 120,
        'debug_output_path': initial_class_debug_output,
    }

    atom_pos = tot_pos_arr[exit_flag_arr == 0, :].T
    atom_pos_all = atom_pos / 3 - 2
    unique_flags, unique_counts = np.unique(exit_flag_arr, return_counts=True)
    print("Exit flag summary:")
    for flag, count in zip(unique_flags.tolist(), unique_counts.tolist(), strict=True):
        print(f"  flag {flag}: {count}")
    print(f"  candidate_atoms_before_boundary = {atom_pos.shape[1]}")

    b1 = np.where(np.logical_or(atom_pos[0, :] < 15, atom_pos[0, :] > FinalVol_single.shape[0] - 15))[0]
    b2 = np.where(np.logical_or(atom_pos[1, :] < 15, atom_pos[1, :] > FinalVol_single.shape[1] - 15))[0]
    b3 = np.where(np.logical_or(atom_pos[2, :] < 15, atom_pos[2, :] > FinalVol_single.shape[2] - 15))[0]

    bT = np.union1d(np.union1d(b1, b2), b3)
    atom_pos = np.delete(atom_pos, bT, axis=1)
    print(f"  candidate_atoms_after_boundary = {atom_pos.shape[1]}")

    peak_loop_s = time.perf_counter() - peak_loop_start
    class_start = time.perf_counter()
    temp_model, temp_atomtype = initial_class_kmean(
        FinalVol_single, atom_pos, classify_info)
    classification_s = time.perf_counter() - class_start
    print(f"  atoms_after_initial_classification = {temp_model.shape[1]}")

    atom_pos_o = temp_model / 3 - 2

    # Calculate support from reconstruction and get the atoms inside
    support_sample_coords = np.round(atom_pos_o.T).astype(int) - 1
    support_para = {
        'th_dis_r_afterav': 0.90,
        'dilate_size': 15,
        'erode_size': 15,
        'bw_size': 50000,
        'debug_sample_coords': support_sample_coords,
    }

    # Implement the functions obtain_tight_support and my_paddzero similarly
    tight_support1, support_debug1 = obtain_tight_support(Dsetvol, support_para, return_debug=True)
    support_para['erode_size'] = 20
    tight_support2, support_debug2 = obtain_tight_support(Dsetvol, support_para, return_debug=True)
    print(f"  tight_support1_voxels = {int(np.count_nonzero(tight_support1))}")
    print(f"  tight_support2_voxels = {int(np.count_nonzero(tight_support2))}")

    # Exclude atoms near the boundary first
    bdl_1 = 8
    bdl_2 = Dsetvol.shape[0] - 8
    ind_out1 = np.logical_or.reduce((atom_pos_o[0, :] <= bdl_1, atom_pos_o[1, :] <= bdl_1, atom_pos_o[2, :] <= bdl_1))
    ind_out2 = np.logical_or.reduce((atom_pos_o[0, :] >= bdl_2, atom_pos_o[1, :] >= bdl_2, atom_pos_o[2, :] >= bdl_2))
    atom_pos_o = np.delete(atom_pos_o, ind_out1 | ind_out2, axis=1)

    # Add the missing atoms inside tighter support
    temp_pos_arr1 = []
    ind_arr1 = []
    for i in range(atom_pos_o.shape[1]):
        temp_pos = np.round(atom_pos_o[:, i]).astype(int) - 1
        if tight_support1[temp_pos[0], temp_pos[1], temp_pos[2]] == 1:
            temp_pos_arr1.append([atom_pos_o[:, i][0], atom_pos_o[:, i][1], atom_pos_o[:, i][2]])
            ind_arr1.append(i)
    print(f"  atoms_inside_tight_support1 = {len(ind_arr1)}")

    # Exclude traced atoms outside the looser support
    temp_pos_arr1_arr = np.asarray(temp_pos_arr1, dtype=float)
    if temp_pos_arr1_arr.size == 0:
        temp_pos_arr1_arr = temp_pos_arr1_arr.reshape(0, 3)
    ind_arr2 = []
    for i in range(atom_pos_all.shape[1]):
        if temp_pos_arr1_arr.size and np.min(cdist(atom_pos_all[:, i].reshape(1, 3), temp_pos_arr1_arr, metric='euclidean')[0]) < 1e-4:
            ind_arr2.append(i)
        else:
            temp_pos = np.round(atom_pos_all[:, i]).astype(int) - 1
            if tight_support2[temp_pos[0], temp_pos[1], temp_pos[2]] == 1:
                ind_arr2.append(i)
    print(f"  atoms_after_support_filter = {len(ind_arr2)}")

    support_filter_s = time.perf_counter() - class_start - classification_s
    temp_pos_arr2 = atom_pos_all[:, ind_arr2]
    np.save(output_file_path+".npy", temp_pos_arr2)
    total_s = time.perf_counter() - stage_start
    print("Tracing performance summary:")
    print(f"  total_s = {total_s:.3f}")
    print(f"  peak_count = {len(max_XYZ)}")
    print(f"  peak_loop_s = {peak_loop_s:.3f}")
    print(f"  least_squares_s = {least_squares_s:.3f}")
    print(f"  avg_peak_s = {peak_loop_s / max(len(max_XYZ), 1):.6f}")
    print(f"  classification_s = {classification_s:.3f}")
    print(f"  support_filter_s = {support_filter_s:.3f}")
    print("tracing finished.")

    if debug_output_path:
        debug_npz_path = debug_output_path if debug_output_path.endswith(".npz") else debug_output_path + ".npz"
        debug_json_path = debug_npz_path[:-4] + ".json"
        debug_summary = {
            "candidate_atoms_before_boundary": int(atom_pos_all.shape[1]),
            "candidate_atoms_after_boundary": int(atom_pos.shape[1]),
            "atoms_after_initial_classification": int(temp_model.shape[1]),
            "tight_support1_voxels": int(np.count_nonzero(tight_support1)),
            "tight_support2_voxels": int(np.count_nonzero(tight_support2)),
            "tight_support1_threshold": float(support_debug1["threshold"]),
            "tight_support2_threshold": float(support_debug2["threshold"]),
            "atoms_inside_tight_support1": int(len(ind_arr1)),
            "atoms_after_support_filter": int(len(ind_arr2)),
            "exit_flag_summary": {int(k): int(v) for k, v in zip(unique_flags.tolist(), unique_counts.tolist(), strict=True)},
            "peak_count": int(len(max_XYZ)),
        }
        np.savez_compressed(
            debug_npz_path,
            exit_flag_arr=exit_flag_arr,
            max_XYZ=max_XYZ_base,
            pos_arr=pos_arr,
            tot_pos_arr=tot_pos_arr,
            atom_pos=atom_pos,
            atom_pos_all=atom_pos_all,
            temp_model=temp_model,
            temp_atomtype=temp_atomtype,
            atom_pos_o=atom_pos_o,
            temp_pos_arr1=temp_pos_arr1_arr,
            ind_arr1=np.asarray(ind_arr1, dtype=int),
            ind_arr2=np.asarray(ind_arr2, dtype=int),
            temp_pos_arr2=temp_pos_arr2,
            tight_support1=tight_support1.astype(np.uint8),
            tight_support2=tight_support2.astype(np.uint8),
            support_sample_coords=support_sample_coords,
            support1_smoothed_sample=support_debug1["smoothed_sample"],
            support1_threshold_sample=support_debug1["threshold_sample"],
            support1_dilate3_sample=support_debug1["dilate3_sample"],
            support1_dilateN_sample=support_debug1["dilateN_sample"],
            support1_cleanup_sample=support_debug1["cleanup_sample"],
            support1_final_sample=support_debug1["final_sample"],
            support2_smoothed_sample=support_debug2["smoothed_sample"],
            support2_threshold_sample=support_debug2["threshold_sample"],
            support2_dilate3_sample=support_debug2["dilate3_sample"],
            support2_dilateN_sample=support_debug2["dilateN_sample"],
            support2_cleanup_sample=support_debug2["cleanup_sample"],
            support2_final_sample=support_debug2["final_sample"],
        )
        with open(debug_json_path, "w", encoding="utf-8") as f:
            json.dump(debug_summary, f, indent=2, sort_keys=True)
        print(f"Saved tracing debug data to {debug_npz_path}")
        print(f"Saved tracing debug summary to {debug_json_path}")
    return

# Dsetvol_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/MG_reconstruction_volume.npy'
# Dsetvol_file_path='/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/input/2reconstruction_sample.mat'

# max_num_th=300
# output_fn='initial_traced_model'
# min_dist = 2

# # Call the main function
# main_polynomial_tracing(Dsetvol_file_path, max_num_th,min_dist, output_fn)
