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
