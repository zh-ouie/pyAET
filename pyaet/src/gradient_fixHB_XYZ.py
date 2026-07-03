import numpy as np
import torch

from pyaet.src.make_fixed_fa_man import make_fixed_fa_man
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot


def _matlab_round_torch(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.int64)


def _torch_dtype_from_xdata(xdata):
    dtype_name = str(xdata.get("xyz_torch_dtype", "float32")).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported xyz_torch_dtype: {dtype_name!r}")


def _torch_device_from_xdata(xdata):
    device = xdata.get("xyz_device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _apply_fixed_fa_torch(projs, fixed_fa):
    proj_batch = projs.permute(2, 0, 1).contiguous()
    kspace = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(proj_batch, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )
    kspace = kspace * fixed_fa[None, :, :]
    out = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    ).real
    return out.permute(1, 2, 0).contiguous()


def _rotation_matrices_torch(angles, dtype, device):
    rotations = []
    for i in range(angles.shape[0]):
        rm1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        rm2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        rm3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        rotations.append(np.dot(np.dot(rm1, rm2), rm3))
    return torch.as_tensor(np.stack(rotations, axis=0), dtype=dtype, device=device)


def gradient_fixHB_XYZ_torch(para, xdata, ydata):
    print("\nHB gradient algorithm [torch XYZ backend]")
    errR = []
    model_arr = []

    z_arr = xdata["Z_arr"]
    res = xdata["Res"]
    half_width = int(xdata["half_width"])
    iterations = int(xdata["iterations"])
    step_sz = xdata["step_sz"]
    angles = np.asarray(xdata["angles"], dtype=np.float64)
    atom_np = np.asarray(xdata["atoms"]).ravel(order="F").astype(np.int64)
    para_arr = np.asarray(para)
    if para_arr.ndim == 2 and para_arr.shape[0] == 2:
        atom_type_num = para_arr.shape[1]
    else:
        atom_type_num = max(len(np.unique(atom_np)), int(para_arr.size // 2))

    num_atom = atom_np.size
    n1, n2, num_pj = ydata.shape
    patch_n = 2 * half_width + 1

    device = _torch_device_from_xdata(xdata)
    dtype = _torch_dtype_from_xdata(xdata)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA XYZ backend requested, but CUDA is not available.")

    fixed_fa_np = np.asarray(make_fixed_fa_man([n1, n2], res, z_arr).reshape(n1, n2), dtype=np.float32)
    fixed_fa = torch.as_tensor(fixed_fa_np, dtype=dtype, device=device)
    y_t = torch.as_tensor(ydata, dtype=dtype, device=device)

    model = torch.as_tensor(np.asarray(xdata["model"] / res, dtype=np.float32), dtype=dtype, device=device)
    model_ori = torch.as_tensor(np.asarray(xdata["model_ori"] / res, dtype=np.float32), dtype=dtype, device=device)
    x = model[0, :].clone()
    y = model[1, :].clone()
    z = model[2, :].clone()
    x_ori = model_ori[0, :]
    y_ori = model_ori[1, :]
    z_ori = model_ori[2, :]

    para = np.reshape(para, [2, atom_type_num])
    h_np = np.zeros(num_atom, dtype=np.float32)
    b_np = np.zeros(num_atom, dtype=np.float32)
    if para.shape[1] == atom_type_num:
        for k in range(atom_type_num):
            h_np[atom_np == k + 1] = para[0, k]
            b_np[atom_np == k + 1] = para[1, k]
    elif para.shape[1] == num_atom:
        h_np[:] = para[0, :]
        b_np[:] = para[1, :]
    else:
        raise ValueError("para must contain either per-type or per-atom H/B values")

    b_np = (res * np.pi) ** 2 / b_np
    h = torch.as_tensor(h_np, dtype=dtype, device=device)
    b = torch.as_tensor(b_np, dtype=dtype, device=device)

    offsets = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)
    dx_grid, dy_grid = torch.meshgrid(offsets, offsets, indexing="ij")
    dx_flat = dx_grid.reshape(-1)
    dy_flat = dy_grid.reshape(-1)
    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1
    rotations = _rotation_matrices_torch(angles, dtype, device)
    scale = 1 / res

    for iter_idx in range(iterations):
        projs_flat = torch.zeros((num_pj, n1 * n2), dtype=dtype, device=device)
        x_round_all = torch.empty((num_pj, num_atom), dtype=torch.int64, device=device)
        y_round_all = torch.empty((num_pj, num_atom), dtype=torch.int64, device=device)
        grad_x_set = torch.empty((num_pj, num_atom, patch_n * patch_n), dtype=dtype, device=device)
        grad_y_set = torch.empty_like(grad_x_set)
        grad_z_set = torch.empty_like(grad_x_set)

        coords = torch.stack((x, y, z), dim=0)
        for i in range(num_pj):
            rotation = rotations[i]
            model_rot = rotation.T @ coords
            x_cen = model_rot[0, :]
            y_cen = model_rot[1, :]
            z_cen = model_rot[2, :]

            x_round = _matlab_round_torch(x_cen)
            y_round = _matlab_round_torch(y_cen)
            z_round = _matlab_round_torch(z_cen)
            x_round_all[i, :] = x_round
            y_round_all[i, :] = y_round

            x_round_f = x_round.to(dtype)
            y_round_f = y_round.to(dtype)
            z_round_f = z_round.to(dtype)

            dx = dx_flat[:, None] + (x_round_f - x_cen)[None, :]
            dy = dy_flat[:, None] + (y_round_f - y_cen)[None, :]
            dz = offsets[:, None] + (z_round_f - z_cen)[None, :]

            l2_xy = dx * dx + dy * dy
            l2_z = dz * dz
            exp_l2_z_b = torch.exp(-l2_z * b[None, :])
            exp_l2_xy_b = torch.exp(-l2_xy * b[None, :])

            z_sum = exp_l2_z_b.sum(dim=0)
            pj_j = exp_l2_xy_b * z_sum[None, :] * h[None, :]
            pj_j_b = pj_j * b[None, :]

            r2_dx = ((rotation[0, 0] * dx + rotation[0, 1] * dy) * pj_j_b)
            r2_dy = ((rotation[1, 0] * dx + rotation[1, 1] * dy) * pj_j_b)
            r2_dz = ((rotation[2, 0] * dx + rotation[2, 1] * dy) * pj_j_b)

            sum_dz_exp = (dz * exp_l2_z_b).sum(dim=0)[None, :] * exp_l2_xy_b
            sum_dz_hb = sum_dz_exp * (h * b)[None, :]
            xj_j = r2_dx + rotation[0, 2] * sum_dz_hb
            yj_j = r2_dy + rotation[1, 2] * sum_dz_hb
            zj_j = r2_dz + rotation[2, 2] * sum_dz_hb

            xx = x_round[None, :] + center_x + dx_flat.to(torch.int64)[:, None]
            yy = y_round[None, :] + center_y + dy_flat.to(torch.int64)[:, None]
            valid = (xx >= 0) & (xx < n1) & (yy >= 0) & (yy < n2)
            flat_idx = xx * n2 + yy
            projs_flat[i].scatter_add_(0, flat_idx[valid], pj_j[valid])

            grad_x_set[i, :, :] = xj_j.T.contiguous()
            grad_y_set[i, :, :] = yj_j.T.contiguous()
            grad_z_set[i, :, :] = zj_j.T.contiguous()

        projs = projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
        projs = _apply_fixed_fa_torch(projs, fixed_fa)
        residual = projs - y_t
        err_value = float((torch.sum(torch.abs(residual)) / torch.sum(torch.abs(y_t))).detach().cpu())
        errR.append(err_value)
        print(f"{iter_idx + 1}.f = {err_value:.5f}")

        res_batch = residual.permute(2, 0, 1).reshape(num_pj, n1 * n2).contiguous()
        grad_x = torch.zeros(num_atom, dtype=dtype, device=device)
        grad_y = torch.zeros_like(grad_x)
        grad_z = torch.zeros_like(grad_x)

        for i in range(num_pj):
            x_round = x_round_all[i, :]
            y_round = y_round_all[i, :]
            xx = x_round[:, None] + center_x + dx_flat.to(torch.int64)[None, :]
            yy = y_round[:, None] + center_y + dy_flat.to(torch.int64)[None, :]
            valid = (xx >= 0) & (xx < n1) & (yy >= 0) & (yy < n2)
            flat_idx = xx * n2 + yy
            patch_res = torch.zeros((num_atom, patch_n * patch_n), dtype=dtype, device=device)
            patch_res[valid] = res_batch[i, flat_idx[valid]]
            grad_x += torch.sum(patch_res * grad_x_set[i], dim=1)
            grad_y += torch.sum(patch_res * grad_y_set[i], dim=1)
            grad_z += torch.sum(patch_res * grad_z_set[i], dim=1)

        dt = step_sz / torch.mean(h) ** 2 / torch.mean(b) ** 2 / half_width ** 2 / num_pj / (n1 * n2)
        x = x - dt * grad_x
        y = y - dt * grad_y
        z = z - dt * grad_z

        diff_x = x - x_ori
        diff_y = y - y_ori
        diff_z = z - z_ori
        diff_norm = torch.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
        limited = diff_norm > scale
        safe_norm = torch.where(limited, diff_norm, torch.ones_like(diff_norm))
        x = torch.where(limited, x_ori + scale * diff_x / safe_norm, x)
        y = torch.where(limited, y_ori + scale * diff_y / safe_norm, y)
        z = torch.where(limited, z_ori + scale * diff_z / safe_norm, z)
        h = torch.clamp(h, min=0)
        b = torch.clamp(b, min=0)

        model_arr.append(torch.stack((x, y, z), dim=0).detach().cpu().numpy().astype(np.float64) * res)

    model_np = torch.stack((x, y, z), dim=0).detach().cpu().numpy().astype(np.float64) * res
    h_np_out = h.detach().cpu().numpy().astype(np.float64, copy=False)
    b_np_out = b.detach().cpu().numpy().astype(np.float64, copy=False)
    param = np.vstack([h_np_out, (res * np.pi) ** 2 / b_np_out, model_np]).astype(np.float64, copy=False)
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param, errR, model_arr


def gradient_fixHB_XYZ(para, xdata, ydata):
    return gradient_fixHB_XYZ_torch(para, xdata, ydata)
