import numpy as np
import torch

from pyaet.src.make_fixed_fa_man import make_fixed_fa_man
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot


def _matlab_round_torch(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.int64)


def _torch_dtype_from_xdata(xdata):
    dtype_name = str(xdata.get("bgrad_torch_dtype", "float32")).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported bgrad_torch_dtype: {dtype_name!r}")


def _torch_device_from_xdata(xdata):
    device = xdata.get("bgrad_device", None)
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


def _rotation_matrices_np(angles):
    rotations = []
    for i in range(angles.shape[0]):
        rm1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        rm2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        rm3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        rotations.append(np.dot(np.dot(rm1, rm2), rm3).T.astype(np.float32))
    return np.stack(rotations, axis=0)


def gradient_B_2type_difB_torch(para, xdata, ydata):
    print("\nHB gradient algorithm [torch B backend]")
    errR = []

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

    n1, n2, num_pj = ydata.shape
    patch_n = 2 * half_width + 1

    device = _torch_device_from_xdata(xdata)
    dtype = _torch_dtype_from_xdata(xdata)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA B-gradient backend requested, but CUDA is not available.")

    fixed_fa_np = np.asarray(make_fixed_fa_man([n1, n2], res, z_arr).reshape(n1, n2), dtype=np.float32)
    max_fa = float(np.max(np.abs(fixed_fa_np)))
    fixed_fa = torch.as_tensor(fixed_fa_np, dtype=dtype, device=device)
    y_t = torch.as_tensor(ydata, dtype=dtype, device=device)

    model_np = np.asarray(xdata["model"] / res, dtype=np.float32)
    rotations = torch.as_tensor(_rotation_matrices_np(angles), dtype=dtype, device=device)
    model = torch.as_tensor(model_np, dtype=dtype, device=device)
    rot_coords = torch.einsum("pij,jn->pin", rotations, model)
    x_rot = rot_coords[:, 0, :].contiguous()
    y_rot = rot_coords[:, 1, :].contiguous()
    z_rot = rot_coords[:, 2, :].contiguous()

    para = np.reshape(para, [2, atom_type_num])
    h = torch.as_tensor(para[0, :] / para[0, 0], dtype=dtype, device=device)
    b = torch.as_tensor((np.pi * res) ** 2 / para[1, :], dtype=dtype, device=device)
    atom0_np = atom_np - 1
    atom0 = torch.as_tensor(atom0_np, dtype=torch.long, device=device)
    counts_np = np.array([np.count_nonzero(atom0_np == j) for j in range(atom_type_num)], dtype=np.float32)
    safe_counts_np = np.maximum(counts_np, 1.0)
    active = torch.as_tensor(counts_np > 0, dtype=torch.bool, device=device)
    t = torch.as_tensor((step_sz / max_fa ** 2 / num_pj / n1 ** 2) / safe_counts_np, dtype=dtype, device=device)

    offsets = torch.arange(-half_width, half_width + 1, dtype=dtype, device=device)
    dx_grid, dy_grid = torch.meshgrid(offsets, offsets, indexing="ij")
    dx_flat = dx_grid.reshape(-1)
    dy_flat = dy_grid.reshape(-1)
    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1
    n_pix = n1 * n2

    for iter_idx in range(iterations):
        projs_flat = torch.zeros((num_pj, n_pix), dtype=dtype, device=device)
        grad_h_flat = torch.zeros((num_pj, atom_type_num, n_pix), dtype=dtype, device=device)
        grad_b_flat = torch.zeros_like(grad_h_flat)

        h_atom = h[atom0]
        b_atom = b[atom0]
        for i in range(num_pj):
            x_cen = x_rot[i]
            y_cen = y_rot[i]
            z_cen = z_rot[i]
            x_round = _matlab_round_torch(x_cen)
            y_round = _matlab_round_torch(y_cen)
            z_round = _matlab_round_torch(z_cen)

            x_round_f = x_round.to(dtype)
            y_round_f = y_round.to(dtype)
            z_round_f = z_round.to(dtype)
            z_delta = offsets[:, None] + (z_round_f - z_cen)[None, :]
            l2_z = z_delta * z_delta
            exp_z = torch.exp(-l2_z * b_atom[None, :])
            z_sum = exp_z.sum(dim=0)
            z_l2_sum = (l2_z * exp_z).sum(dim=0)

            x_delta = dx_flat[:, None] + (x_round_f - x_cen)[None, :]
            y_delta = dy_flat[:, None] + (y_round_f - y_cen)[None, :]
            l2_xy = x_delta * x_delta + y_delta * y_delta
            exp_xy = torch.exp(-l2_xy * b_atom[None, :])
            pj = exp_xy * z_sum[None, :]
            pj_h = h_atom[None, :] * pj
            bj = pj_h * l2_xy + h_atom[None, :] * exp_xy * z_l2_sum[None, :]

            xx = x_round[None, :] + center_x + dx_flat.to(torch.long)[:, None]
            yy = y_round[None, :] + center_y + dy_flat.to(torch.long)[:, None]
            valid = (xx >= 0) & (xx < n1) & (yy >= 0) & (yy < n2)
            flat_idx = xx * n2 + yy
            projs_flat[i].scatter_add_(0, flat_idx[valid], pj_h[valid])

            type_flat_idx = atom0[None, :] * n_pix + flat_idx
            grad_h_flat[i].view(-1).scatter_add_(0, type_flat_idx[valid], pj[valid])
            grad_b_flat[i].view(-1).scatter_add_(0, type_flat_idx[valid], bj[valid])

        projs = projs_flat.reshape(num_pj, n1, n2).permute(1, 2, 0).contiguous()
        projs = _apply_fixed_fa_torch(projs, fixed_fa)
        k_scale = torch.sum(projs * y_t) / torch.sum(projs * projs)
        projs = projs * k_scale
        grad_b_flat = grad_b_flat * k_scale

        residual = projs - y_t
        err_value = float((torch.sum(torch.abs(residual)) / torch.sum(torch.abs(y_t))).detach().cpu())
        errR.append(err_value)
        print(f"{iter_idx + 1}.f = {err_value:.5f}")

        res_flat = residual.permute(2, 0, 1).reshape(num_pj, n_pix).contiguous()
        grad_h_sum = torch.sum(res_flat[:, None, :] * grad_h_flat, dim=(0, 2))
        grad_b_sum = torch.sum(res_flat[:, None, :] * grad_b_flat, dim=(0, 2))
        b = torch.where(active, b + (t / patch_n ** 6) * grad_b_sum, b)
        h = torch.where(active, h - t * grad_h_sum, h)
        h = torch.clamp(h, min=0)
        h = h / h[0]
        b = torch.clamp(b, min=0)

    h_np = h.detach().cpu().numpy().astype(np.float64, copy=False)
    b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)
    param = np.vstack([float(k_scale.detach().cpu()) * h_np, (np.pi * res) ** 2 / b_np]).astype(
        np.float64,
        copy=False,
    )
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param, errR


def gradient_B_2type_difB(para, xdata, ydata):
    return gradient_B_2type_difB_torch(para, xdata, ydata)
