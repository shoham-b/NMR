import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def t1_model(t, M0, T1, alpha):
    """
    T1 Inversion Recovery model.
    M(t) = M0 * (1 - 2 * alpha * exp(-t / T1))
    """
    return M0 * (1 - 2 * alpha * np.exp(-t / T1))


@jit(nopython=True, cache=True)
def t2_decay_model(t, M0, T2, offset):
    """
    T2 / T2* Exponential Decay model.
    M(t) = M0 * exp(-t / T2) + offset
    """
    return M0 * np.exp(-t / T2) + offset


@jit(nopython=True, cache=True)
def j_modulated_t2(t, M0, T2, J, offset, depth=1.0):
    """
    J-Modulated T2 Decay with variable modulation depth.

    M(t) = | M0 * exp(-t/T2) * ((1-depth) + depth * cos(pi * J * t)) | + offset

    Parameters:
        depth: Modulation depth (0.0 to 1.0)
               - 1.0 = Full modulation (dips to zero)
               - 0.5 = Half modulation (cosine oscillates between 0.5 and 1.5)
               - 0.0 = No modulation (simple exponential decay)

    This accounts for unmodulated components (e.g., singlets) or partial J-coupling.
    """
    envelope = M0 * np.exp(-t / T2)
    modulation = (1.0 - depth) + depth * np.cos(np.pi * J * t)
    return np.abs(envelope * modulation) + offset
