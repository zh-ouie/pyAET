import math

import numpy as np

try:
    import torch
    from numba import cuda, float32, int32
except Exception as exc:  # pragma: no cover - optional GPU dependency.
    torch = None
    cuda = None
    _CUDA_IMPORT_ERROR = exc
else:
    _CUDA_IMPORT_ERROR = None

from pyaet.src.make_fixed_fa_man import make_fixed_fa_man
from pyaet.src.matrix_quaternion_rot import matrix_quaternion_rot


def _require_cuda():
    if torch is None or cuda is None:
        raise ImportError("CUDA B-gradient backend requires torch and numba.") from _CUDA_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA B-gradient backend requested but torch.cuda is not available.")


def _torch_dtype(xdata):
    dtype_name = str(xdata.get("bgrad_torch_dtype", "float32")).lower()
    if dtype_name not in {"float32", "single"}:
        raise ValueError("CUDA B-gradient fused backend currently supports float32 only.")
    return torch.float32


def _rotation_matrices_np(angles):
    rotations = []
    for i in range(angles.shape[0]):
        rm1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        rm2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        rm3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        rotations.append(np.dot(np.dot(rm1, rm2), rm3).T.astype(np.float32))
    return np.stack(rotations, axis=0)


@cuda.jit(device=True)
def _round_device(x):
    if x >= 0.0:
        return int32(math.floor(x + 0.5))
    return int32(math.ceil(x - 0.5))


@cuda.jit
def _b_forward_kernel(
    X_rot,
    Y_rot,
    Z_rot,
    atom0,
    h,
    b,
    projs_flat,
    grad_h_flat,
    grad_b_flat,
    n1,
    n2,
    num_atom,
    atom_type_num,
    half_width,
):
    idx = cuda.grid(1)
    total = X_rot.shape[0] * num_atom
    if idx >= total:
        return

    i = idx // num_atom
    k = idx - i * num_atom
    atom_type = atom0[k]
    if atom_type < 0 or atom_type >= atom_type_num:
        return

    patch_n = 2 * half_width + 1
    x_cen = X_rot[i, k]
    y_cen = Y_rot[i, k]
    z_cen = Z_rot[i, k]
    x_round = _round_device(x_cen)
    y_round = _round_device(y_cen)
    z_round = _round_device(z_cen)

    hj = h[atom_type]
    bj = b[atom_type]
    z_sum = 0.0
    z_l2_sum = 0.0
    z_delta0 = float32(z_round) - z_cen
    for dz_i in range(32):
        if dz_i >= patch_n:
            break
        dz_int = dz_i - half_width
        z_delta = float32(dz_int) + z_delta0
        l2z = z_delta * z_delta
        ez = math.exp(-l2z * bj)
        z_sum += ez
        z_l2_sum += l2z * ez

    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1
    n_pix = n1 * n2
    x_delta0 = float32(x_round) - x_cen
    y_delta0 = float32(y_round) - y_cen
    type_offset = i * atom_type_num * n_pix + atom_type * n_pix

    for dx_i in range(32):
        if dx_i >= patch_n:
            break
        dx_int = dx_i - half_width
        xx = x_round + center_x + dx_int
        x_delta = float32(dx_int) + x_delta0
        x_l2 = x_delta * x_delta
        for dy_i in range(32):
            if dy_i >= patch_n:
                break
            dy_int = dy_i - half_width
            yy = y_round + center_y + dy_int
            if 0 <= xx < n1 and 0 <= yy < n2:
                y_delta = float32(dy_int) + y_delta0
                l2xy = x_l2 + y_delta * y_delta
                exp_xy = math.exp(-l2xy * bj)
                pj = exp_xy * z_sum
                pj_h = hj * pj
                grad_b = pj_h * l2xy + hj * exp_xy * z_l2_sum
                pix = xx * n2 + yy
                cuda.atomic.add(projs_flat, (i, pix), pj_h)
                cuda.atomic.add(grad_h_flat, type_offset + pix, pj)
                cuda.atomic.add(grad_b_flat, type_offset + pix, grad_b)


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


def gradient_B_2type_difB_cuda(para, xdata, ydata):
    _require_cuda()
    dtype = _torch_dtype(xdata)
    device = torch.device(xdata.get("bgrad_device", "cuda"))

    print("\nHB gradient algorithm [fused CUDA B backend]")
    errR = []

    Res = xdata["Res"]
    half_width = int(xdata["half_width"])
    iterations = int(xdata["iterations"])
    step_sz = xdata["step_sz"]
    angles = np.asarray(xdata["angles"], dtype=np.float64)
    atom_np = np.asarray(xdata["atoms"]).ravel(order="F").astype(np.int64)
    num_atom = atom_np.size
    para_arr = np.asarray(para)
    atom_type_num = para_arr.shape[1] if para_arr.ndim == 2 and para_arr.shape[0] == 2 else max(len(np.unique(atom_np)), int(para_arr.size // 2))
    N1, N2, num_pj = ydata.shape
    n_pix = N1 * N2
    N_s = 2 * half_width + 1

    fixed_fa_np = np.asarray(make_fixed_fa_man([N1, N2], Res, xdata["Z_arr"]).reshape(N1, N2), dtype=np.float32)
    max_fa = float(np.max(np.abs(fixed_fa_np)))
    fixed_fa = torch.as_tensor(fixed_fa_np, dtype=dtype, device=device)
    y_t = torch.as_tensor(ydata, dtype=dtype, device=device)

    model = np.asarray(xdata["model"] / Res, dtype=np.float32)
    rotations = torch.as_tensor(_rotation_matrices_np(angles), dtype=dtype, device=device)
    model_t = torch.as_tensor(model, dtype=dtype, device=device)
    rot_coords = torch.einsum("pij,jn->pin", rotations, model_t)
    X_rot = rot_coords[:, 0, :].contiguous()
    Y_rot = rot_coords[:, 1, :].contiguous()
    Z_rot = rot_coords[:, 2, :].contiguous()

    para = np.reshape(para, [2, atom_type_num])
    h = torch.as_tensor(para[0, :] / para[0, 0], dtype=dtype, device=device)
    b = torch.as_tensor((np.pi * Res) ** 2 / para[1, :], dtype=dtype, device=device)
    atom0_np = atom_np.astype(np.int32, copy=False) - 1
    atom0 = torch.as_tensor(atom0_np, dtype=torch.int32, device=device)
    counts_np = np.array([np.count_nonzero(atom0_np == j) for j in range(atom_type_num)], dtype=np.float32)
    safe_counts_np = np.maximum(counts_np, 1.0)
    active = torch.as_tensor(counts_np > 0, dtype=torch.bool, device=device)
    t = torch.as_tensor((step_sz / max_fa ** 2 / num_pj / N1 ** 2) / safe_counts_np, dtype=dtype, device=device)

    threads = 128
    blocks = (num_pj * num_atom + threads - 1) // threads

    for iter_idx in range(iterations):
        projs_flat = torch.zeros((num_pj, n_pix), dtype=dtype, device=device)
        grad_h_flat = torch.zeros((num_pj * atom_type_num * n_pix,), dtype=dtype, device=device)
        grad_b_flat = torch.zeros_like(grad_h_flat)

        _b_forward_kernel[blocks, threads](
            cuda.as_cuda_array(X_rot),
            cuda.as_cuda_array(Y_rot),
            cuda.as_cuda_array(Z_rot),
            cuda.as_cuda_array(atom0),
            cuda.as_cuda_array(h),
            cuda.as_cuda_array(b),
            cuda.as_cuda_array(projs_flat),
            cuda.as_cuda_array(grad_h_flat),
            cuda.as_cuda_array(grad_b_flat),
            N1,
            N2,
            num_atom,
            atom_type_num,
            half_width,
        )
        cuda.synchronize()

        projs = projs_flat.reshape(num_pj, N1, N2).permute(1, 2, 0).contiguous()
        projs = _apply_fixed_fa_torch(projs, fixed_fa)
        k_scale = torch.sum(projs * y_t) / torch.sum(projs * projs)
        projs = projs * k_scale
        grad_b_flat = grad_b_flat * k_scale

        res = projs - y_t
        err_value = float((torch.sum(torch.abs(res)) / torch.sum(torch.abs(y_t))).detach().cpu())
        errR.append(err_value)
        print(f"{iter_idx + 1}.f = {err_value:.5f}")

        res_flat = res.permute(2, 0, 1).reshape(num_pj, n_pix).contiguous()
        grad_h = grad_h_flat.reshape(num_pj, atom_type_num, n_pix)
        grad_b = grad_b_flat.reshape(num_pj, atom_type_num, n_pix)
        grad_h_sum = torch.sum(res_flat[:, None, :] * grad_h, dim=(0, 2))
        grad_b_sum = torch.sum(res_flat[:, None, :] * grad_b, dim=(0, 2))
        b = torch.where(active, b + (t / N_s ** 6) * grad_b_sum, b)
        h = torch.where(active, h - t * grad_h_sum, h)
        h = torch.clamp(h, min=0)
        h = h / h[0]
        b = torch.clamp(b, min=0)

    h_np = h.detach().cpu().numpy().astype(np.float64, copy=False)
    b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)
    param = np.vstack([float(k_scale.detach().cpu()) * h_np, (np.pi * Res) ** 2 / b_np]).astype(np.float64, copy=False)
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param, errR
