# Typing support
from typing import Callable, Tuple
from numpy.typing import NDArray

# Module imports
import numpy as np

# FUNCTIONS IMPLEMENTATION

# 4th/2nd-order FD scheme
def finite_difference(omega, f: Callable, n, bc) -> Tuple[NDArray, NDArray]:

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
    coeff_2nd: NDArray = np.array([-1, 2, h**2])

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
    eye = np.eye(adim)
    la: NDArray
    ua: NDArray
    la, ua = lu_decomp(a)
    acopy = np.copy(a)

    for i in range(adim):
        acopy[:, i] = l_solve(la, u_solve(ua, eye[:, i]))

    l, q = direct_power_method(acopy, x_sample, niter, tol)
    return 1 / l, q


def condition_number(a):
    x_sample: NDArray = np.random.rand(len(a))
    ldir, _ = direct_power_method(a, x_sample)
    linv, _ = inverse_power_method(a, x_sample)
    return ldir / linv
