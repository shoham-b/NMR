import numpy as np
from numba import jit


@jit(nopython=True)
def t1_model(t, M0, T1, alpha):
    """
    T1 Inversion Recovery model.
    M(t) = M0 * (1 - 2 * alpha * exp(-t / T1))
    """
    return M0 * (1 - 2 * alpha * np.exp(-t / T1))


@jit(nopython=True)
def t2_decay_model(t, M0, T2, offset):
    """
    T2 / T2* Exponential Decay model.
    M(t) = M0 * exp(-t / T2) + offset
    """
    return M0 * np.exp(-t / T2) + offset


@jit(nopython=True)
def fid_model(t, M0, T2_star, freq, phase, offset):
    """
    Free Induction Decay (FID) model with oscillation.
    M(t) = M0 * exp(-t / T2_star) * cos(2*pi*freq*t + phase) + offset
    """
    return M0 * np.exp(-t / T2_star) * np.cos(2 * np.pi * freq * t + phase) + offset


@jit(nopython=True)
def lorentzian(f, A, f0, gamma):
    """
    Lorentzian function.
    L(f) = A * gamma / (pi * ((f - f0)^2 + gamma^2))
    Area is A.
    HWHM is gamma.
    Max height is A / (pi * gamma).
    """
    return A * gamma / (np.pi * ((f - f0) ** 2 + gamma**2))


@jit(nopython=True)
def multi_lorentzian(f, params, n_peaks):
    """
    Sum of n Lorentzians + offset.
    params: [offset, A1, f0_1, gamma1, A2, f0_2, gamma2, ...]
    """
    y = np.full_like(f, params[0])  # Start with offset
    for i in range(n_peaks):
        idx = 1 + i * 3
        A = params[idx]
        f0 = params[idx + 1]
        gamma = params[idx + 2]
        y += lorentzian(f, A, f0, gamma)
    return y


@jit(nopython=True)
def magnitude_lorentzian(f, A, f0, gamma):
    """
    Magnitude Mode Lorentzian (Square Root of Lorentzian).
    M(f) = A / sqrt( (f-f0)^2 + gamma^2 )
    At peak (f=f0), Height = A / gamma.
    """
    return A / np.sqrt((f - f0) ** 2 + gamma**2)


@jit(nopython=True)
def multi_magnitude_lorentzian(f, params, n_peaks):
    """
    Sum of n Magnitude Lorentzians + offset.
    params: [offset, A1, f0_1, gamma1, A2, f0_2, gamma2, ...]
    """
    y = np.full_like(f, params[0])  # Start with offset
    for i in range(n_peaks):
        idx = 1 + i * 3
        A = params[idx]
        f0 = params[idx + 1]
        gamma = params[idx + 2]
        y += magnitude_lorentzian(f, A, f0, gamma)
    return y
