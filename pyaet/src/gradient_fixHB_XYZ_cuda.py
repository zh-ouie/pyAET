import math
import time

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
        raise ImportError("CUDA XYZ backend requires torch and numba.") from _CUDA_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA XYZ backend requested but torch.cuda is not available.")


def _torch_dtype(xdata):
    dtype_name = str(xdata.get("xyz_torch_dtype", "float32")).lower()
    if dtype_name not in {"float32", "single"}:
        raise ValueError("CUDA XYZ fused backend currently supports float32 only.")
    return torch.float32


def _rotation_matrices_np(angles):
    rotations = []
    for i in range(angles.shape[0]):
        rm1 = matrix_quaternion_rot([0, 0, 1], angles[i, 0])
        rm2 = matrix_quaternion_rot([0, 1, 0], angles[i, 1])
        rm3 = matrix_quaternion_rot([1, 0, 0], angles[i, 2])
        rotations.append(np.dot(np.dot(rm1, rm2), rm3).astype(np.float32))
    return np.stack(rotations, axis=0)


def _matlab_round_float32(x):
    if x >= 0:
        return int(np.floor(x + np.float32(0.5)))
    return int(np.ceil(x - np.float32(0.5)))


@cuda.jit(device=True)
def _round_device(x):
    if x >= 0.0:
        return int32(math.floor(x + 0.5))
    return int32(math.ceil(x - 0.5))


@cuda.jit
def _xyz_forward_kernel(
    X,
    Y,
    Z,
    h,
    b,
    rotations,
    projs_flat,
    x_round_all,
    y_round_all,
    grad_x_set,
    grad_y_set,
    grad_z_set,
    n1,
    n2,
    num_atom,
    half_width,
):
    idx = cuda.grid(1)
    patch_n = 2 * half_width + 1
    patch_size = patch_n * patch_n
    total = rotations.shape[0] * num_atom
    if idx >= total:
        return

    i = idx // num_atom
    k = idx - i * num_atom
    r00 = rotations[i, 0, 0]
    r01 = rotations[i, 0, 1]
    r02 = rotations[i, 0, 2]
    r10 = rotations[i, 1, 0]
    r11 = rotations[i, 1, 1]
    r12 = rotations[i, 1, 2]
    r20 = rotations[i, 2, 0]
    r21 = rotations[i, 2, 1]
    r22 = rotations[i, 2, 2]

    x = X[k]
    y = Y[k]
    z = Z[k]
    x_cen = r00 * x + r10 * y + r20 * z
    y_cen = r01 * x + r11 * y + r21 * z
    z_cen = r02 * x + r12 * y + r22 * z

    x_round = _round_device(x_cen)
    y_round = _round_device(y_cen)
    z_round = _round_device(z_cen)
    x_round_all[i, k] = x_round
    y_round_all[i, k] = y_round

    hk = h[k]
    bk = b[k]
    z_sum = 0.0
    z_grad_sum = 0.0
    z_delta0 = float32(z_round) - z_cen
    for dz_i in range(32):
        dz = dz_i - half_width
        if dz_i >= patch_n:
            break
        z_delta = float32(dz) + z_delta0
        l2z = z_delta * z_delta
        ez = math.exp(-l2z * bk)
        z_sum += ez
        z_grad_sum += z_delta * ez

    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1
    x_delta0 = float32(x_round) - x_cen
    y_delta0 = float32(y_round) - y_cen

    for dx_i in range(32):
        if dx_i >= patch_n:
            break
        dx_int = dx_i - half_width
        xx = x_round + center_x + dx_int
        x_delta = float32(dx_int) + x_delta0
        for dy_i in range(32):
            if dy_i >= patch_n:
                break
            dy_int = dy_i - half_width
            yy = y_round + center_y + dy_int
            patch_idx = dx_i * patch_n + dy_i
            y_delta = float32(dy_int) + y_delta0
            l2xy = x_delta * x_delta + y_delta * y_delta
            exp_xy = math.exp(-l2xy * bk)
            pj = hk * exp_xy * z_sum
            pjb = pj * bk
            sum_dz_hb = exp_xy * z_grad_sum * hk * bk

            gx = (r00 * x_delta + r01 * y_delta) * pjb + r02 * sum_dz_hb
            gy = (r10 * x_delta + r11 * y_delta) * pjb + r12 * sum_dz_hb
            gz = (r20 * x_delta + r21 * y_delta) * pjb + r22 * sum_dz_hb
            grad_x_set[i, k, patch_idx] = gx
            grad_y_set[i, k, patch_idx] = gy
            grad_z_set[i, k, patch_idx] = gz

            if 0 <= xx < n1 and 0 <= yy < n2:
                cuda.atomic.add(projs_flat, (i, xx * n2 + yy), pj)


@cuda.jit
def _xyz_residual_kernel(
    res_flat,
    x_round_all,
    y_round_all,
    grad_x_set,
    grad_y_set,
    grad_z_set,
    grad_x,
    grad_y,
    grad_z,
    n1,
    n2,
    num_atom,
    half_width,
):
    idx = cuda.grid(1)
    patch_n = 2 * half_width + 1
    total = x_round_all.shape[0] * num_atom
    if idx >= total:
        return
    i = idx // num_atom
    k = idx - i * num_atom
    x_round = x_round_all[i, k]
    y_round = y_round_all[i, k]
    center_x = (n1 + 1) // 2 - 1
    center_y = (n2 + 1) // 2 - 1

    sx = 0.0
    sy = 0.0
    sz = 0.0
    for dx_i in range(32):
        if dx_i >= patch_n:
            break
        xx = x_round + center_x + dx_i - half_width
        for dy_i in range(32):
            if dy_i >= patch_n:
                break
            yy = y_round + center_y + dy_i - half_width
            if 0 <= xx < n1 and 0 <= yy < n2:
                patch_idx = dx_i * patch_n + dy_i
                rv = res_flat[i, xx * n2 + yy]
                sx += rv * grad_x_set[i, k, patch_idx]
                sy += rv * grad_y_set[i, k, patch_idx]
                sz += rv * grad_z_set[i, k, patch_idx]
    cuda.atomic.add(grad_x, k, sx)
    cuda.atomic.add(grad_y, k, sy)
    cuda.atomic.add(grad_z, k, sz)


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


def gradient_fixHB_XYZ_cuda(para, xdata, ydata):
    _require_cuda()
    dtype = _torch_dtype(xdata)
    device = torch.device(xdata.get("xyz_device", "cuda"))
    profile = False
    profile_t = [time.perf_counter()]

    def profile_stamp(label):
        if not profile:
            return
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        print(f"profile_{label}_s {now - profile_t[0]:.6f}", flush=True)
        profile_t[0] = now

    print("\nHB gradient algorithm [fused CUDA XYZ backend]")
    errR = []
    model_arr = []
    Res = xdata["Res"]
    half_width = int(xdata["half_width"])
    iterations = int(xdata["iterations"])
    step_sz = xdata["step_sz"]
    angles = np.asarray(xdata["angles"], dtype=np.float64)
    atom = np.asarray(xdata["atoms"]).ravel(order="F").astype(np.int64)
    num_atom = atom.size
    para_arr = np.asarray(para)
    atom_type_num = para_arr.shape[1] if para_arr.ndim == 2 and para_arr.shape[0] == 2 else max(len(np.unique(atom)), int(para_arr.size // 2))
    N1, N2, num_pj = ydata.shape
    patch_size = (2 * half_width + 1) ** 2

    para = np.reshape(para, [2, atom_type_num])
    h_np = np.zeros(num_atom, dtype=np.float32)
    b_np = np.zeros(num_atom, dtype=np.float32)
    for k in range(atom_type_num):
        h_np[atom == k + 1] = para[0, k]
        b_np[atom == k + 1] = para[1, k]
    b_np = ((Res * np.pi) ** 2 / b_np).astype(np.float32)

    model = np.asarray(xdata["model"] / Res, dtype=np.float32)
    model_ori = np.asarray(xdata["model_ori"] / Res, dtype=np.float32)
    X = torch.as_tensor(model[0], dtype=dtype, device=device).clone()
    Y = torch.as_tensor(model[1], dtype=dtype, device=device).clone()
    Z = torch.as_tensor(model[2], dtype=dtype, device=device).clone()
    X_ori = torch.as_tensor(model_ori[0], dtype=dtype, device=device)
    Y_ori = torch.as_tensor(model_ori[1], dtype=dtype, device=device)
    Z_ori = torch.as_tensor(model_ori[2], dtype=dtype, device=device)
    h = torch.as_tensor(h_np, dtype=dtype, device=device)
    b = torch.as_tensor(b_np, dtype=dtype, device=device)
    rotations = torch.as_tensor(_rotation_matrices_np(angles), dtype=dtype, device=device)
    fixed_fa = torch.as_tensor(
        np.asarray(make_fixed_fa_man([N1, N2], Res, xdata["Z_arr"]).reshape(N1, N2), dtype=np.float32),
        dtype=dtype,
        device=device,
    )
    y_t = torch.as_tensor(ydata, dtype=dtype, device=device)
    profile_stamp("setup")

    threads = 128
    blocks = (num_pj * num_atom + threads - 1) // threads
    scale = 1 / Res

    for iter_idx in range(iterations):
        projs_flat = torch.zeros((num_pj, N1 * N2), dtype=dtype, device=device)
        x_round_all = torch.empty((num_pj, num_atom), dtype=torch.int32, device=device)
        y_round_all = torch.empty((num_pj, num_atom), dtype=torch.int32, device=device)
        grad_x_set = torch.empty((num_pj, num_atom, patch_size), dtype=dtype, device=device)
        grad_y_set = torch.empty_like(grad_x_set)
        grad_z_set = torch.empty_like(grad_x_set)
        profile_stamp(f"iter{iter_idx + 1}_alloc")

        _xyz_forward_kernel[blocks, threads](
            cuda.as_cuda_array(X),
            cuda.as_cuda_array(Y),
            cuda.as_cuda_array(Z),
            cuda.as_cuda_array(h),
            cuda.as_cuda_array(b),
            cuda.as_cuda_array(rotations),
            cuda.as_cuda_array(projs_flat),
            cuda.as_cuda_array(x_round_all),
            cuda.as_cuda_array(y_round_all),
            cuda.as_cuda_array(grad_x_set),
            cuda.as_cuda_array(grad_y_set),
            cuda.as_cuda_array(grad_z_set),
            N1,
            N2,
            num_atom,
            half_width,
        )
        cuda.synchronize()
        profile_stamp(f"iter{iter_idx + 1}_forward_kernel")

        projs = projs_flat.reshape(num_pj, N1, N2).permute(1, 2, 0).contiguous()
        projs = _apply_fixed_fa_torch(projs, fixed_fa)
        profile_stamp(f"iter{iter_idx + 1}_fft")
        res = projs - y_t
        err = torch.sum(torch.abs(res)) / torch.sum(torch.abs(y_t))
        err_value = float(err.detach().cpu())
        errR.append(err_value)
        print(f"{iter_idx + 1}.f = {err_value:.5f}")
        profile_stamp(f"iter{iter_idx + 1}_error")

        grad_x = torch.zeros(num_atom, dtype=dtype, device=device)
        grad_y = torch.zeros_like(grad_x)
        grad_z = torch.zeros_like(grad_x)
        res_flat = res.permute(2, 0, 1).reshape(num_pj, N1 * N2).contiguous()
        profile_stamp(f"iter{iter_idx + 1}_grad_alloc")
        _xyz_residual_kernel[blocks, threads](
            cuda.as_cuda_array(res_flat),
            cuda.as_cuda_array(x_round_all),
            cuda.as_cuda_array(y_round_all),
            cuda.as_cuda_array(grad_x_set),
            cuda.as_cuda_array(grad_y_set),
            cuda.as_cuda_array(grad_z_set),
            cuda.as_cuda_array(grad_x),
            cuda.as_cuda_array(grad_y),
            cuda.as_cuda_array(grad_z),
            N1,
            N2,
            num_atom,
            half_width,
        )
        cuda.synchronize()
        profile_stamp(f"iter{iter_idx + 1}_residual_kernel")

        dt = step_sz / torch.mean(h) ** 2 / torch.mean(b) ** 2 / half_width ** 2 / num_pj / (N1 * N2)
        X = X - dt * grad_x
        Y = Y - dt * grad_y
        Z = Z - dt * grad_z
        diff_X = X - X_ori
        diff_Y = Y - Y_ori
        diff_Z = Z - Z_ori
        diff_norm = torch.sqrt(diff_X * diff_X + diff_Y * diff_Y + diff_Z * diff_Z)
        mask = diff_norm > scale
        safe_norm = torch.where(mask, diff_norm, torch.ones_like(diff_norm))
        X = torch.where(mask, X_ori + scale * diff_X / safe_norm, X)
        Y = torch.where(mask, Y_ori + scale * diff_Y / safe_norm, Y)
        Z = torch.where(mask, Z_ori + scale * diff_Z / safe_norm, Z)
        profile_stamp(f"iter{iter_idx + 1}_update")
        model_arr.append(torch.stack((X, Y, Z), dim=0).detach().cpu().numpy().astype(np.float64) * Res)
        profile_stamp(f"iter{iter_idx + 1}_model_copy")

    model_np = torch.stack((X, Y, Z), dim=0).detach().cpu().numpy().astype(np.float64) * Res
    h_np_out = h.detach().cpu().numpy().astype(np.float64, copy=False)
    b_np_out = b.detach().cpu().numpy().astype(np.float64, copy=False)
    param = np.vstack([h_np_out, (Res * np.pi) ** 2 / b_np_out, model_np]).astype(np.float64, copy=False)
    return projs.detach().cpu().numpy().astype(np.float64, copy=False), param, errR, model_arr
