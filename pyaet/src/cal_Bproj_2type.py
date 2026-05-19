import numpy as np
from pyaet.src.my_ifft import my_ifft
from pyaet.src.my_fft import my_fft
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot
from pyaet.src.make_fixed_fa_man import make_fixed_fa_man

try:
    import torch
except Exception:
    torch = None

# Release settings: edit these directly in code when switching Step4 variants.
STEP4_EXP_BACKEND = "numpy"  # "numpy", "torch", or "doublecast"
STEP4_ZSUM_MODE = "numpy"    # "numpy" or "sequential"
STEP4_REFERENCE_CACHE = True

def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def _exp_float32(x):
    x32 = np.asarray(x, dtype=np.float32)
    if STEP4_EXP_BACKEND == "torch":
        if torch is None:
            raise ImportError("AET_STEP4_EXP_BACKEND=torch requires torch.")
        return torch.exp(torch.from_numpy(x32)).numpy()
    if STEP4_EXP_BACKEND in {"doublecast", "double-cast", "double"}:
        return np.exp(x32.astype(np.float64)).astype(np.float32)
    return np.exp(x32).astype(np.float32)


def _sum_z_float32(exp_z):
    if STEP4_ZSUM_MODE == "sequential":
        z_sum = np.zeros((exp_z.shape[0], 1, exp_z.shape[2]), dtype=np.float32)
        for iz in range(exp_z.shape[1]):
            z_sum = (z_sum + exp_z[:, iz:iz + 1, :]).astype(np.float32)
        return z_sum
    return np.sum(exp_z, axis=1, keepdims=True, dtype=np.float32)

def cal_Bproj_2type(para, xdata, ydata, fit_flag=True):
    if fit_flag:
        para = np.abs(para)

    Z_arr = xdata['Z_arr']
    Res = xdata['Res']
    half_width = xdata['half_width']
    model = np.asarray(xdata['model'], dtype=np.float64)
    angles = np.asarray(xdata['angles'], dtype=np.float64)
    atom = np.asarray(xdata['atoms']).ravel(order='F')
    num_atom = atom.size
    atom_type_num = len(np.unique(atom))
    # ydata = xdata['projections']

    N1, N2, _ = ydata.shape
    num_pj = angles.shape[0]

    cache_enabled = STEP4_REFERENCE_CACHE
    cache = xdata.setdefault("_cal_bproj_reference_cache", {}) if cache_enabled else {}
    cache_key = (
        id(xdata["model"]),
        id(xdata["angles"]),
        id(xdata["atoms"]),
        N1,
        N2,
        num_pj,
        num_atom,
        atom_type_num,
        float(Res),
        int(half_width),
        tuple(np.asarray(Z_arr, dtype=np.float64).ravel().tolist()),
    )
    if cache_enabled and cache.get("key") == cache_key:
        fixed_fa = cache["fixed_fa"]
        X_rot = cache["X_rot"]
        Y_rot = cache["Y_rot"]
        Z_rot = cache["Z_rot"]
        X_crop = cache["X_crop"]
        Y_crop = cache["Y_crop"]
        Z_crop = cache["Z_crop"]
        atom_type_masks = cache["atom_type_masks"]
        num_atom_type = cache["num_atom_type"]
        patch_offsets = cache["patch_offsets"]
    else:
        fixed_fa = np.asarray(make_fixed_fa_man([N1, N2], Res, Z_arr).reshape(N1, N2))
        model_scaled = model / Res

        # MATLAB only keeps the rotated coordinates in single precision here.
        coord_dtype = np.float32
        X_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
        Y_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)
        Z_rot = np.zeros((num_pj, num_atom), dtype=coord_dtype)

        for i in range(num_pj):
            R1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
            R2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
            R3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
            R = (np.dot(np.dot(R1, R2), R3)).T

            rot_coords = np.dot(R, model_scaled)
            X_rot[i, :] = rot_coords[0, :]
            Y_rot[i, :] = rot_coords[1, :]
            Z_rot[i, :] = rot_coords[2, :]

        # MATLAB uses ndgrid for [X_crop, Y_crop]
        # MATLAB keeps the rotated coordinates in single precision. Mixed
        # double/single arithmetic in the splat stage also returns single, so keep
        # the local voxel arithmetic in float32 and only accumulate into double.
        grid = np.arange(-half_width, half_width + 1, dtype=np.float32)
        X_crop, Y_crop = np.meshgrid(grid, grid, indexing='ij')
        Z_crop = grid
        atom_type_masks = tuple(atom == (j + 1) for j in range(atom_type_num))
        num_atom_type = np.asarray([np.count_nonzero(mask) for mask in atom_type_masks], dtype=int)
        patch_offsets = np.arange(-half_width, half_width + 1, dtype=int)

        if cache_enabled:
            cache.clear()
            cache.update(
                {
                    "key": cache_key,
                    "fixed_fa": fixed_fa,
                    "X_rot": X_rot,
                    "Y_rot": Y_rot,
                    "Z_rot": Z_rot,
                    "X_crop": X_crop,
                    "Y_crop": Y_crop,
                    "Z_crop": Z_crop,
                    "atom_type_masks": atom_type_masks,
                    "num_atom_type": num_atom_type,
                    "patch_offsets": patch_offsets,
                }
            )

    para = np.reshape(np.asarray(para, dtype=np.float64), [2, atom_type_num], order='F')
    h = para[0, :] / para[0, 0]
    b = (np.pi * Res) ** 2 / para[1, :]

    grad = np.zeros((N1, N2, num_pj, 3), dtype=np.float64)

    for i in range(num_pj):
        for j in range(atom_type_num):
            hj = h[j]
            bj = b[j]
            atom_type_j = atom_type_masks[j]
            X_cen = X_rot[i, atom_type_j].reshape(1, 1, num_atom_type[j])
            Y_cen = Y_rot[i, atom_type_j].reshape(1, 1, num_atom_type[j])
            Z_cen = Z_rot[i, atom_type_j].reshape(1, 1, num_atom_type[j])

            X_round = _matlab_round(X_cen).astype(np.float32)
            Y_round = _matlab_round(Y_cen).astype(np.float32)
            Z_round = _matlab_round(Z_cen).astype(np.float32)

            l2_xy = (X_crop[:, :, None] + (X_round - X_cen)) ** 2 + (
                Y_crop[:, :, None] + (Y_round - Y_cen)
            ) ** 2
            # MATLAB: Z_crop is 1x9 and centers are 1x1xN, so l2_z is 1x9xN.
            # sum(...) collapses the first non-singleton dimension (the crop axis),
            # producing one z-integral scalar per atom with shape 1x1xN.
            l2_z = (Z_crop[None, :, None] + (Z_round - Z_cen)) ** 2

            exp_xy = _exp_float32(-(l2_xy * bj).astype(np.float32))
            exp_z = _exp_float32(-(l2_z * bj).astype(np.float32))
            pj_j = (exp_xy * _sum_z_float32(exp_z)).astype(np.float32)

            pj_j_h = (hj * pj_j).astype(np.float32)
            x_round_flat = X_round.reshape(-1, order='F').astype(np.int32)
            y_round_flat = Y_round.reshape(-1, order='F').astype(np.int32)

            for k in range(num_atom_type[j]):
                indx = x_round_flat[k] + patch_offsets + (N1 + 1) // 2 - 1
                indy = y_round_flat[k] + patch_offsets + (N2 + 1) // 2 - 1

                indx = indx.astype(int)
                indy = indy.astype(int)

                # MATLAB evaluates Grad(double) + pj_j_h(single) as single,
                # then writes the rounded single result back into the double
                # Grad array. Reproduce that per-atom rounding here.
                curr = grad[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i, j].astype(np.float32)
                grad[indx[0]:(indx[-1]+1), indy[0]:(indy[-1]+1), i, j] = (
                    curr + pj_j_h[:, :, k]
                ).astype(np.float32)

    projs = np.sum(grad, axis=3) #todo: check

    for i in range(num_pj):
        projs[:, :, i] = np.real( my_ifft( my_fft(projs[:, :, i]) * fixed_fa ) ) #todo: check

    projs_flat = projs.ravel(order='F')
    ydata_flat = ydata.ravel(order='F')
    k = np.sum(projs_flat * ydata_flat) / np.sum(projs_flat ** 2)
    projs = projs * k

    param = np.vstack([k * h, (np.pi * Res) ** 2 / b])

    return projs, param #long edit

def cal_Bproj_2type2(para, xdata, fit_flag=True):
    ydata = xdata['projections']
    return cal_Bproj_2type(para, xdata, ydata, fit_flag=fit_flag)
