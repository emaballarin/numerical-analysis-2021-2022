#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Tuple, Optional, Union, Collection
import numbers
from copy import deepcopy
from functools import partial as fpartial
import numpy as np
from numpy.typing import NDArray

# FUNCTIONS IMPLEMENTATION

# 4th/2nd-order FD scheme
def finite_difference(omega, f: Callable, n: int, bc) -> Tuple[NDArray, NDArray]:

    # A preliminary sanity check
    assert n > 2
    assert len(bc) == 2
    assert len(omega) == 2 and omega[0] < omega[-1]

    # Prepare input
    f = np.vectorize(f)
    bc: NDArray = np.asarray(bc)

    # Build b
    b: NDArray = np.empty(n)
    b[1:-1] = f(np.linspace(*omega, n))[1:-1]
    b[0], b[-1] = bc[0], bc[-1]  # BCs

    # Build A
    h: NDArray = np.asarray((omega[1] - omega[0]) / (n - 1))
    coeff_4th: NDArray = np.array([1, -16, 30]) / 12
    coeff_2nd: NDArray = np.array([-1, 2, 1])

    a: NDArray = np.zeros((n, n))

    # Proper 4th order scheme where applicable
    i: int
    for i in range(2, n - 2):
        a[i, i - 2] = a[i, i + 2] = coeff_4th[0]
        a[i, i - 1] = a[i, i + 1] = coeff_4th[1]
        a[i, i] = coeff_4th[2]

    # Fallback to 2nd order scheme where 4th not applicable
    a[1, 0] = a[1, 2] = a[n - 2, n - 3] = a[n - 2, n - 1] = coeff_2nd[0]
    a[1, 1] = a[n - 2, n - 2] = coeff_2nd[1]
    a[0, 0] = a[-1, -1] = coeff_2nd[2]

    a /= h**2

    return a, b


# LU Decomposition
def lu_decomp(a: NDArray) -> Tuple[NDArray, NDArray]:
    adim: int = len(a)

    # Sanity check
    assert adim > 1

    l: NDArray = np.eye(adim)
    u: NDArray = np.copy(a)
    i: int
    for i in range(adim - 1):
        j: int
        for j in range(i + 1, adim):
            l[j, i] = u[j, i] / u[i, i]
            u[j, i:] -= l[j, i] * u[i, i:]
    return l, u


# Cholesky Decomposition
def chol_decomp(a: NDArray) -> Tuple[NDArray, NDArray]:
    adim: int = len(a)

    # Sanity check
    assert adim > 1

    l: NDArray = np.zeros((adim, adim))
    j: int
    for j in range(adim):
        l[j, j] = (a[j, j] - np.sum(l[j, 0:j] ** 2)) ** 0.5
        i: int
        for i in range(j + 1, adim):
            l[i, j] = (a[i, j] - np.dot(l[i, 0:j], l[j, 0:j])) / l[j, j]
    return np.transpose(l), l


# Forward & Backward substitution
def l_solve(l: NDArray, rhs: NDArray) -> NDArray:
    ldim: int = len(l)

    # Sanity checks
    assert ldim > 1
    assert len(rhs) == ldim

    x: NDArray = np.zeros(ldim)

    i: int
    for i in range(ldim):
        x[i] = rhs[i]
        x[i] -= np.dot(x[0:i], l[i, 0:i])
        x[i] = x[i] / l[i, i]
    return x


def u_solve(u: NDArray, rhs: NDArray) -> NDArray:
    udim: int = len(u)

    # Sanity checks
    assert udim > 1
    assert len(rhs) == udim

    x: NDArray = np.zeros(udim)

    i: int
    for i in range(udim - 1, -1, -1):
        x[i] = rhs[i]
        x[i] -= np.dot(x[i + 1 : udim], u[i, i + 1 : udim])
        x[i] = x[i] / u[i, i]
    return x


# LU-powered systems-solver
def lu_solve(a: NDArray, b: NDArray) -> NDArray:

    # Sanity checks
    assert len(a) > 1
    assert len(a) == len(b)

    l: NDArray
    u: NDArray
    l, u = lu_decomp(a)

    return u_solve(u, l_solve(l, b))


def direct_power_method(
    a: NDArray, x_sample: NDArray, niter: int = 1e4, tol: float = 1e-8
) -> Tuple[NDArray, NDArray]:

    # Sanity checks
    assert len(a) > 1
    assert len(x_sample) == len(a)
    assert niter > 0
    assert tol > 0.0

    it: int
    error: float
    it, error = 0, 1.0
    q: NDArray = x_sample / np.linalg.norm(x_sample, 2)

    while it < niter and error > tol:
        x: NDArray = np.dot(a, q)
        l: NDArray = np.dot(np.transpose(q), x)
        error: NDArray = np.linalg.norm(x - l * q, 2)
        q: NDArray = x / np.linalg.norm(x, 2)
        it += 1

    return l, q


def inverse_power_method(
    a: NDArray, x_sample: NDArray, niter: int = 1e4, tol: float = 1e-8
) -> Tuple[NDArray, NDArray]:

    # Sanity checks
    assert len(a) > 1
    assert len(x_sample) == len(a)
    assert niter > 0
    assert tol > 0.0

    adim: int = len(a)
    eye: NDArray = np.eye(adim)
    la: NDArray
    ua: NDArray
    la, ua = lu_decomp(a)
    acopy: NDArray = np.copy(a)

    i: int
    for i in range(adim):
        acopy[:, i] = l_solve(la, u_solve(ua, eye[:, i]))

    l: NDArray
    q: NDArray
    l, q = direct_power_method(acopy, x_sample, niter, tol)
    return 1 / l, q


def inverse_power_method_shf(
    a: NDArray, x: NDArray, mu: NDArray, niter: int = 1e4, tol: float = 1e-8
) -> Tuple[NDArray, NDArray]:

    # Sanity checks
    assert len(a) > 1
    assert len(x) == len(a)
    assert niter > 0
    assert tol > 0.0

    adim: int = len(a)
    eye: NDArray = np.eye(adim)
    q: NDArray = x / np.linalg.norm(x, 2)

    lm: NDArray
    um: NDArray
    m: NDArray = a - eye * mu
    lm, um = lu_decomp(m)

    it: int = 0
    ediff: float = 1.0 + tol

    while it < niter and ediff > tol:
        lsolve: NDArray = l_solve(lm, q)
        usolve: NDArray = u_solve(um, lsolve)
        q: NDArray = usolve / np.linalg.norm(usolve, 2)
        p1: NDArray = np.dot(a, q)
        p2: NDArray = np.dot(np.transpose(q), p1)
        ediff: NDArray = np.linalg.norm(p1 - p2 * q, 2)
        it += 1

    return p2, q


def condition_number(a: NDArray, inv_shf: bool = False) -> np.float64:

    # Sanity check
    assert len(a) > 1

    x_sample: NDArray = np.random.rand(len(a))
    ldir: NDArray
    linv: NDArray
    ldir, _ = direct_power_method(a, x_sample)
    if inv_shf:
        linv, _ = inverse_power_method_shf(a, x_sample, mu=0)
    else:
        linv, _ = inverse_power_method(a, x_sample)
    return ldir / linv


def conjugate_gradient(
    a, b, p, nmax: Optional[int] = None, eps=1e-10, use_lusolve: bool = False
) -> NDArray:

    # Sanity checks
    assert len(a) > 1
    assert len(a) == len(b) == len(p)
    assert eps > 0.0

    if nmax is not None:
        assert nmax > 0
    else:
        nmax = len(a)

    x: NDArray
    p_aux: NDArray
    x, p_aux = np.zeros_like(b)
    r: NDArray = b - np.dot(a, x)
    e_eps: float
    rz_old: NDArray
    e_eps = rz_old = 1.0
    it: int = 0

    if use_lusolve:
        linalg_solve: Callable[[NDArray, NDArray], NDArray] = lu_solve
    else:
        linalg_solve: Callable[[NDArray, NDArray], NDArray] = np.linalg.solve

    while it < nmax and e_eps > eps:
        z: NDArray = linalg_solve(p, r)
        rz: NDArray = np.dot(r, z)
        beta: NDArray = rz / rz_old
        p_aux: NDArray = beta * p_aux + z
        apaux: NDArray = np.dot(a, p_aux)
        alpha: NDArray = rz / np.dot(p_aux, apaux)
        r: NDArray = r - alpha * apaux
        x += alpha * p_aux
        e_eps: NDArray = np.linalg.norm(r, 2)
        it += 1
        rz_old: NDArray = rz

    return x


def fwd_euler_fd_solve(
    alpha: Callable,
    f: Callable,
    nt: int,
    nx: int,
    bc4pde: Collection,
    ttot: numbers.Real,
):

    # Sanity checks
    assert nx > 2
    assert nt > 1
    assert ttot > 0

    # The method will solve the PDE w.r.t. function u(t, x) for t in [0, ttot],
    # given boundary conditions bc4pde = [[a, b], u(0, x), u(t, a), u(t,b)]
    # collection (of length 5) of suitable types.

    # DISCRETIZATION |->

    # Tidy-up bc4pde
    bc4pde_copy: Collection = deepcopy(bc4pde)
    elnr: int
    elem: Union[Callable, numbers.Real]
    for elnr, elem in enumerate(bc4pde):
        if elnr > 0:
            if isinstance(elem, numbers.Real):
                bc4pde_copy[elnr] = lambda _, elem=elem: elem
            bc4pde_copy[elnr] = np.vectorize(bc4pde_copy[elnr])
    xbc: NDArray
    u0x: Callable[[NDArray], NDArray]
    uta: Callable[[NDArray], NDArray]
    utb: Callable[[NDArray], NDArray]
    xbc, u0x, uta, utb = bc4pde_copy

    # Sanity check
    assert len(xbc) == 2

    # Vectorize the remaining functions
    alpha, f = map(np.vectorize, [alpha, f])

    # Actual discretization

    ht = ttot / (nt - 1)
    hx = (xbc[-1] - xbc[0]) / (nx - 1)

    # Coefficients
    coeff_4th: NDArray = np.array([-1, 16, (-30 + 12 * hx**2 / ht)]) / 12
    coeff_2nd: NDArray = np.array([1, (-2 + hx**2 / ht), 1])

    x = np.linspace(xbc[0], xbc[-1], nx)
    t = np.linspace(0, ttot, nt)

    # Prepare A
    a = np.zeros((nx, nx), float)
    i: int
    for i in range(2, nx - 2):
        a[i, i - 2] = a[i, i + 2] = coeff_4th[0]
        a[i, i - 1] = a[i, i + 1] = coeff_4th[1]
        a[i, i] = coeff_4th[2]

    a[1, 0] = a[1, 2] = a[nx - 2, nx - 3] = a[nx - 2, nx - 1] = coeff_2nd[0]
    a[1, 1] = a[nx - 2, nx - 2] = coeff_2nd[1]
    a[0, 0] = a[-1, -1] = coeff_2nd[2]

    a = ht * a
    a /= hx**2

    # SOLUTION |->

    # Compute functions
    fx: NDArray
    alpht: NDArray
    fx, alpht = f(x), alpha(t)

    solution: NDArray = np.zeros((nt, nx), dtype=float)

    # Apply functional BCs
    solution[0, :] = u0x(x)
    solution[:, 0] = uta(t)
    solution[:, nx - 1] = utb(t)

    # Fill the rest of the solution
    tstep: int
    for tstep in range(1, nt):
        solution[tstep, :] = np.dot(a, solution[tstep - 1, :]) + ht * fx * alpht[tstep]

    return solution


def idx_nearx(x: numbers.Real, among: NDArray) -> numbers.Integral:
    return np.argmin(abs(among - x))


def eigval_lu(a: NDArray, niter: int = 425, eps: float = 1e-8) -> NDArray:

    # Sanity checks
    assert len(a) > 1
    assert niter > 0
    assert eps > 0.0

    it = 0
    ediff = 1.0 + eps

    acopy: NDArray = np.copy(a)
    val_prev: NDArray = np.diag(acopy)

    while it < niter and ediff > eps:
        l, u = lu_decomp(acopy)
        acopy = np.matmul(u, l)
        val = np.diag(acopy)
        ediff = np.linalg.norm(val - val_prev, 2)
        it += 1
        val_prev = val

    return val


def eigv_ipm(
    a: NDArray,
    sample: Optional[NDArray] = None,
    niter_val: int = 425,
    niter_ipm: int = int(1e4),
    eps: float = 1e-8,
    refine_vals: bool = True,
) -> Tuple[NDArray, NDArray]:

    # Save relevant value
    aside = len(a)

    # Sanity checks
    assert aside > 1
    assert niter_val > 0
    assert niter_ipm > 0
    assert eps > 0.0

    if sample is not None:
        assert len(sample) == aside
    else:
        sample = np.random.rand(aside)

    vec: NDArray = np.empty_like(a)
    val: NDArray = eigval_lu(a, niter_val, eps)

    if refine_vals:
        val_ref: NDArray = np.zeros_like(val)

    i: int
    for i, _ in enumerate(val):
        nval: NDArray
        nvec: NDArray
        nval, nvec = inverse_power_method_shf(a, sample, val[i] + eps, niter_val, eps)
        vec[i, :] = nvec
        if refine_vals:
            val_ref[i] = nval

    return val, vec


def inv_lu(a: NDArray) -> NDArray:
    assert len(a) > 1

    inva: NDArray = np.empty_like(a)
    alen: int = len(a)
    eye: NDArray = np.eye(alen)

    i: int
    for i in range(alen):
        inva[:, i] = lu_solve(a, eye[:, i])

    return inva


def newton_method(
    f: Callable[[numbers.Real], numbers.Real],
    fprime: Callable[[numbers.Real], numbers.Real],
    x_guess: numbers.Real,
    niter: int = int(1e3),
    eps: float = 1e-10,
    epsprime=1e-12,
) -> Optional[numbers.Real]:

    # Sanity checks
    assert niter > 0
    assert eps > 0.0
    assert epsprime > 0.0

    x: numbers.Real = x_guess
    _: int
    for _ in range(niter):
        if np.abs(f(x)) < eps:
            return x
        if np.abs(fprime(x)) < epsprime:
            return None
        x: numbers.Real = x - f(x) / fprime(x)
    return x


def backward_euler_method(
    y0: numbers.Real,
    f_tx: Callable[[numbers.Real, numbers.Real], numbers.Real],
    fprime_tx: Callable[[numbers.Real, numbers.Real], numbers.Real],
    tbounds,
    nsteps: int,
) -> NDArray:

    # Sanity checks
    assert len(tbounds) == 2
    assert tbounds[0] < tbounds[-1]
    assert nsteps > 1

    # Numpy-ify
    tbounds: NDArray = np.asarray(tbounds)
    f_tx, fprime_tx = map(np.vectorize, [f_tx, fprime_tx])

    time: NDArray = np.linspace(tbounds[0], tbounds[-1], nsteps)
    ht: NDArray = (tbounds[-1] - tbounds[0]) / (nsteps - 1)

    f_stencil: Callable[[numbers.Real, numbers.Real, numbers.Real], numbers.Real] = (
        lambda x_inner, t, x: x_inner - x - ht * f_tx(t, x_inner)
    )
    fprime_stencil: Callable[
        [numbers.Real, numbers.Real, numbers.Real], numbers.Real
    ] = lambda x_inner, t, _: 1 - ht * fprime_tx(t, x_inner)
    y: NDArray = np.zeros(nsteps)

    y[0] = y0
    it: int
    for it in range(1, nsteps):
        f: Callable[[numbers.Real], numbers.Real] = fpartial(
            f_stencil, t=time[it], x=y[it - 1]
        )
        fprime: Callable[[numbers.Real], numbers.Real] = fpartial(
            fprime_stencil, t=time[it], _=y[it - 1]
        )
        y[it] = newton_method(f, fprime, y[it - 1])

    return y
