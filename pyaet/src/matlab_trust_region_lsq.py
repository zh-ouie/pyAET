from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatlabTRFResult:
    x: np.ndarray
    cost: float
    nfev: int
    njev: int
    status: int
    message: str
    history: list[dict]


def _seceqn(lam: float, eigval: np.ndarray, alpha: np.ndarray, delta: float) -> float:
    denom = eigval + lam
    coeff = np.empty_like(alpha)
    nz = denom != 0
    coeff[nz] = alpha[nz] / denom[nz]
    coeff[~nz] = np.inf
    coeff[np.isnan(coeff)] = 0.0
    norm_s = np.linalg.norm(coeff)
    if norm_s == 0:
        return 1.0 / delta
    return 1.0 / delta - 1.0 / norm_s


def _secular_root(lam0: float, eigval: np.ndarray, alpha: np.ndarray, delta: float) -> float:
    a = float(lam0)
    fa = _seceqn(a, eigval, alpha, delta)
    dx = abs(a) / 2.0 if a != 0 else 0.5
    b = a + 1.0
    fb = _seceqn(b, eigval, alpha, delta)
    it = 0
    while (fa > 0) == (fb > 0) and it <= 50:
        dx *= 2.0
        b = a + dx
        fb = _seceqn(b, eigval, alpha, delta)
        it += 1

    c = a
    fc = fa
    d = b - a
    e = d
    tol = 1e-12
    for _ in range(60):
        if fb == 0:
            break
        if (fb > 0) == (fc > 0):
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        m = 0.5 * (c - b)
        toler = 2.0 * tol * max(abs(b), 1.0)
        if abs(m) <= toler:
            break
        if abs(e) < toler or abs(fa) <= abs(fb):
            d = m
            e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if (2.0 * p < 3.0 * m * q - abs(toler * q)) and (p < abs(0.5 * e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        a = b
        fa = fb
        if abs(d) > toler:
            b += d
        else:
            b += toler if m > 0 else -toler
        fb = _seceqn(b, eigval, alpha, delta)
    return float(b)


def _trust(g: np.ndarray, hess: np.ndarray, delta: float) -> tuple[np.ndarray, float, int]:
    tol2 = 1e-8
    hess = np.asarray(hess, dtype=np.float64)
    eigval, eigvec = np.linalg.eigh(hess)
    mineig_idx = int(np.argmin(eigval))
    mineig = float(eigval[mineig_idx])
    alpha = -eigvec.T @ g
    sig = np.sign(alpha[mineig_idx]) + (alpha[mineig_idx] == 0)

    key = 0
    lam = 0.0
    posdef = 1
    if mineig > 0:
        coeff = alpha / eigval
        step = eigvec @ coeff
        if np.linalg.norm(step) <= 1.2 * delta:
            key = 1
        else:
            laminit = 0.0
    else:
        laminit = -mineig
        posdef = 0

    if key == 0:
        if _seceqn(laminit, eigval, alpha, delta) > 0:
            b = _secular_root(laminit, eigval, alpha, delta)
            if abs(_seceqn(b, eigval, alpha, delta)) <= tol2:
                lam = b
                key = 2
            else:
                lam = -mineig
                key = 3
        else:
            lam = -mineig
            key = 4

        if key > 2:
            close = np.abs(eigval + lam) < 10 * np.finfo(float).eps * np.maximum(np.abs(eigval), 1)
            alpha = alpha.copy()
            alpha[close] = 0.0
        denom = eigval + lam
        coeff = np.zeros_like(alpha)
        nz = denom != 0
        coeff[nz] = alpha[nz] / denom[nz]
        coeff[~nz & (alpha != 0)] = np.inf
        coeff[np.isnan(coeff)] = 0.0
        step = eigvec @ coeff
        nrms = np.linalg.norm(step)
        if key > 2 and nrms < 0.8 * delta:
            beta = np.sqrt(max(delta * delta - nrms * nrms, 0.0))
            step = step + beta * sig * eigvec[:, mineig_idx]
        if key > 2 and nrms > 1.2 * delta:
            lam = _secular_root(laminit, eigval, alpha, delta)
            denom = eigval + lam
            coeff = np.zeros_like(alpha)
            nz = denom != 0
            coeff[nz] = alpha[nz] / denom[nz]
            coeff[~nz & (alpha != 0)] = np.inf
            coeff[np.isnan(coeff)] = 0.0
            step = eigvec @ coeff

    val = float(g @ step + 0.5 * step @ (hess @ step))
    return step, val, posdef


def _definev(g: np.ndarray, x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros_like(x)
    dv = np.zeros_like(x)
    arg1 = (g < 0) & np.isfinite(ub)
    arg2 = (g >= 0) & np.isfinite(lb)
    arg3 = (g < 0) & ~np.isfinite(ub)
    arg4 = (g >= 0) & ~np.isfinite(lb)
    v[arg1] = x[arg1] - ub[arg1]
    v[arg2] = x[arg2] - lb[arg2]
    v[arg3] = -1.0
    v[arg4] = 1.0
    dv[arg1 | arg2] = 1.0
    return v, dv


def _truncate_to_bounds(step: np.ndarray, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, theta: float):
    arg = np.abs(step) > 0
    if not np.any(arg):
        return step, 1.0, None
    dis = np.maximum((ub[arg] - x[arg]) / step[arg], (lb[arg] - x[arg]) / step[arg])
    ipt_local = int(np.argmin(dis))
    mmdis = float(dis[ipt_local])
    full_idx = np.flatnonzero(arg)[ipt_local]
    alpha = min(1.0, theta * mmdis)
    return alpha * step, mmdis, int(full_idx)


def _perturb_trust_region_reflective(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, delta: float | None = None) -> np.ndarray:
    if delta is None:
        delta = 100.0 * np.finfo(float).eps
    x = x.copy()
    if np.min(np.abs(ub - x)) < delta or np.min(np.abs(x - lb)) < delta:
        upper = (ub - x) < delta
        lower = (x - lb) < delta
        x[upper] -= delta
        x[lower] += delta
    return x


def _quad1d(x: np.ndarray, ss: np.ndarray, delta: float) -> tuple[np.ndarray, float]:
    a = float(x @ x)
    b = float(2.0 * (ss @ x))
    c = float(ss @ ss - delta * delta)
    disc = b * b - 4.0 * a * c
    numer = -(b + np.sign(b) * np.sqrt(max(disc, 0.0)))
    r1 = numer / (2.0 * a)
    r2 = c / (a * r1)
    tau = min(1.0, max(r1, r2))
    if tau <= 0:
        raise FloatingPointError("invalid reflected trust-region step")
    return tau * x, tau


def _trdog(
    x: np.ndarray,
    g: np.ndarray,
    ata: np.ndarray,
    d: np.ndarray,
    delta: float,
    dv: np.ndarray,
    theta: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    n = g.size
    grad = d * g
    dg_diag = np.abs(g) * dv
    big_m = (d[:, None] * ata) * d[None, :] + np.diag(dg_diag)
    try:
        v1 = np.linalg.solve(big_m, -grad)
        posdef = 1
    except np.linalg.LinAlgError:
        v1 = -grad
        posdef = 0
    if np.linalg.norm(v1) > 0:
        v1 = v1 / np.linalg.norm(v1)

    z_cols = [v1]
    if n > 1:
        if posdef < 1:
            v2 = d * np.sign(grad)
        else:
            v2 = grad.copy()
        if np.linalg.norm(v2) > 0:
            v2 = v2 / np.linalg.norm(v2)
        v2 = v2 - v1 * (v1 @ v2)
        if np.linalg.norm(v2) > np.sqrt(np.finfo(float).eps):
            z_cols.append(v2 / np.linalg.norm(v2))
    z = np.column_stack(z_cols)

    def reduced(zmat: np.ndarray):
        w = d[:, None] * zmat
        ww = ata @ w
        w2 = d[:, None] * ww
        mm = zmat.T @ w2 + zmat.T @ (dg_diag[:, None] * zmat)
        rhs = zmat.T @ grad
        return rhs, mm

    rhs, mm = reduced(z)
    st, _, _ = _trust(rhs, mm, delta)
    ss = z @ st
    s = d * ss
    ssave = s.copy()
    sssave = ss.copy()
    stsave = st.copy()
    s, mmdis, ipt = _truncate_to_bounds(s, x, lb, ub, theta)
    alpha = 0.0 if np.linalg.norm(ssave) == 0 else np.linalg.norm(s) / np.linalg.norm(ssave)
    st = alpha * st
    ss = alpha * ss
    qpval1 = float(rhs @ st + 0.5 * st @ (mm @ st))

    qpval3 = np.inf
    ns = None
    nss = None
    r = None
    if n > 1 and np.linalg.norm(mmdis * sssave) < 0.9 * delta and ipt is not None:
        r = mmdis * ssave
        nx = x + r
        st0 = mmdis * stsave
        qp0 = float(rhs @ st0 + 0.5 * st0 @ (mm @ st0))
        ng = ata @ r + g
        ngrad = d * ng + dg_diag * (mmdis * sssave)
        reflected = mmdis * sssave
        reflected[ipt] = -reflected[ipt]
        if np.linalg.norm(reflected) > 0:
            zz = (reflected / np.linalg.norm(reflected)).reshape(-1, 1)
            rhs3, mm3 = reduced(zz)
            nss, tau = _quad1d(reflected, mmdis * sssave, delta)
            nst = tau / np.linalg.norm(reflected)
            ns = d * nss
            ns, _, _ = _truncate_to_bounds(ns, nx, lb, ub, theta)
            nst = nst * (0.0 if np.linalg.norm(d * nss) == 0 else np.linalg.norm(ns) / np.linalg.norm(d * nss))
            nss = ns / np.where(d == 0, 1.0, d)
            qpval3 = float(qp0 + rhs3.item() * nst + 0.5 * nst * (mm3.item() * nst))

    gnorm = np.linalg.norm(grad)
    zz = (grad / (gnorm + (gnorm == 0))).reshape(-1, 1)
    rhs2, mm2 = reduced(zz)
    st2, _, _ = _trust(rhs2, mm2, delta)
    ssg = (zz @ st2).ravel()
    sg = d * ssg
    sg, _, _ = _truncate_to_bounds(sg, x, lb, ub, theta)
    alpha_g = 0.0 if np.linalg.norm(d * ssg) == 0 else np.linalg.norm(sg) / np.linalg.norm(d * ssg)
    st2 = alpha_g * st2
    ssg = alpha_g * ssg
    qpval2 = float(rhs2 @ st2 + 0.5 * st2 @ (mm2 @ st2))

    if qpval2 <= min(qpval1, qpval3):
        return sg, ssg, qpval2, posdef
    if qpval1 <= min(qpval2, qpval3):
        return s, ss, qpval1, posdef
    return ns + r, nss + mmdis * sssave, qpval3, posdef


def _finite_difference_jacobian(fun, x: np.ndarray, f0: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    n = x.size
    cols = []
    nfev = 0
    rel_step = np.sqrt(np.finfo(float).eps)
    for i in range(n):
        h = rel_step * max(abs(x[i]), 1.0)
        if x[i] + h > ub[i]:
            h = -h
        xp = x.copy()
        xp[i] += h
        fp = fun(xp)
        cols.append((fp - f0) / h)
        nfev += 1
    return np.column_stack(cols), nfev


def least_squares_matlab_trust_region(
    fun,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    tol_fun: float = 1e-12,
    max_fun_evals: int | None = None,
    max_iter: int = 400,
):
    x = np.asarray(x0, dtype=np.float64).copy()
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if max_fun_evals is None:
        max_fun_evals = 100 * x.size

    f = fun(x)
    nfev = 1
    jac, fd_evals = _finite_difference_jacobian(fun, x, f, lb, ub)
    nfev += fd_evals
    njev = 1

    g = jac.T @ f
    val = float(f @ f)
    delta = 10.0
    ratio = 0.0
    nrmsx = 1.0
    posdef = 1
    oval = np.inf
    history: list[dict] = []

    for iteration in range(max_iter + 1):
        v, dv = _definev(g, x, lb, ub)
        gopt = v * g
        optnrm = float(np.linalg.norm(gopt, ord=np.inf))
        history.append(
            {
                "iteration": iteration,
                "x": x.copy(),
                "cost_half_sumsq": 0.5 * val,
                "resnorm": val,
                "optnrm": optnrm,
                "delta": delta,
                "ratio": ratio,
                "nfev": nfev,
            }
        )

        diff = abs(oval - val)
        oval = val
        if optnrm < tol_fun and posdef == 1:
            return MatlabTRFResult(x, 0.5 * val, nfev, njev, 1, "first-order optimality below TolFun", history)
        if iteration > 1 and nrmsx < np.finfo(float).eps:
            return MatlabTRFResult(x, 0.5 * val, nfev, njev, 2, "step size below TolX", history)
        if iteration > max_iter:
            break
        if nfev > max_fun_evals:
            return MatlabTRFResult(x, 0.5 * val, nfev, njev, 0, "maximum function evaluations exceeded", history)

        d = np.sqrt(np.abs(v))
        theta = max(0.95, 1.0 - optnrm)
        ata = jac.T @ jac
        sx, snod, qp, posdef = _trdog(x, g, ata, d, delta, dv, theta, lb, ub)
        nrmsx = float(np.linalg.norm(snod))
        newx = np.minimum(np.maximum(x + sx, lb), ub)
        newx = _perturb_trust_region_reflective(newx, lb, ub)

        newf = fun(newx)
        nfev += 1
        newjac, fd_evals = _finite_difference_jacobian(fun, newx, newf, lb, ub)
        nfev += fd_evals
        njev += 1
        newval = float(newf @ newf)
        newgrad = newjac.T @ newf

        aug = 0.5 * float(snod @ ((dv * np.abs(g)) * snod))
        ratio = (0.5 * (newval - val) + aug) / qp if qp != 0 else 0.0
        if ratio >= 0.75 and nrmsx >= 0.9 * delta:
            delta = 2.0 * delta
        elif ratio <= 0.25:
            delta = min(nrmsx / 4.0, delta / 4.0)

        if newval < val:
            x = newx
            f = newf
            jac = newjac
            g = newgrad
            val = newval

    return MatlabTRFResult(x, 0.5 * val, nfev, njev, 0, "maximum iterations exceeded", history)
