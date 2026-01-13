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
def fid_model(t, M0, T2_star, freq, phase, offset):
    """
    Free Induction Decay (FID) model with oscillation.
    M(t) = M0 * exp(-t / T2_star) * cos(2*pi*freq*t + phase) + offset
    """
    return M0 * np.exp(-t / T2_star) * np.cos(2 * np.pi * freq * t + phase) + offset


@jit(nopython=True, cache=True)
def lorentzian(f, A, f0, gamma):
    """
    Lorentzian function.
    L(f) = A * gamma / (pi * ((f - f0)^2 + gamma^2))
    Area is A.
    HWHM is gamma.
    Max height is A / (pi * gamma).
    """
    return A * gamma / (np.pi * ((f - f0) ** 2 + gamma**2))


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def magnitude_lorentzian(f, A, f0, gamma):
    """
    Magnitude Mode Lorentzian (Square Root of Lorentzian).
    M(f) = A / sqrt( (f-f0)^2 + gamma^2 )
    At peak (f=f0), Height = A / gamma.
    """
    return A / np.sqrt((f - f0) ** 2 + gamma**2)


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def complex_lorentzian(f, A, f0, gamma):
    """
    Complex Lorentzian: A * gamma / (gamma - 1j * (f - f0)) ?
    Standard definition L(f) ~ 1 / (gamma - i(w-w0)).
    Real part (Abs) is Lorentzian.
    Ideally: 1 / ( (f-f0)*1j + gamma ).
    A is scaling.
    Let's align with magnitude_lorentzian: |L(f0)| = A/gamma.
    If L = 1/z, |L| = 1/|z| = 1/gamma.
    So L(f) = A * gamma / (gamma + 1j * (f - f0)) will have peak height A (wait).
    Magnitude: |A*gamma| / sqrt(gamma^2 + (f-f0)^2).
    At f0: |A*gamma|/gamma = A.
    Yes. This matches `magnitude_lorentzian` scaling (A/gamma * gamma?? No wait).
    `magnitude_lorentzian` return A / sqrt( ... ). Peak A/gamma.
    Our complex one: A * gamma / ( ... ) -> Peak A.
    So we need to return magnitude_lorentzian scaling:
    L(f) = A / (gamma + 1j * (f - f0)). -> Peak |A| / gamma.
    """
    return A / (gamma + 1j * (f - f0))


@jit(nopython=True, cache=True)
def multiplet_lorentzian(f, center, J, multiplicity, gamma, A):
    """
    Multiplet Lorentzian Model (Magnitude of Complex Sum).
    """
    y_complex = np.zeros_like(f, dtype=np.complex128)

    if multiplicity == 1:
        y_complex += complex_lorentzian(f, A, center, gamma)
    elif multiplicity == 2:
        y_complex += complex_lorentzian(f, A, center - J / 2, gamma)
        y_complex += complex_lorentzian(f, A, center + J / 2, gamma)
    elif multiplicity == 3:
        y_complex += complex_lorentzian(f, A, center - J, gamma)
        y_complex += complex_lorentzian(f, 2 * A, center, gamma)
        y_complex += complex_lorentzian(f, A, center + J, gamma)
    elif multiplicity == 4:
        y_complex += complex_lorentzian(f, A, center - 1.5 * J, gamma)
        y_complex += complex_lorentzian(f, 3 * A, center - 0.5 * J, gamma)
        y_complex += complex_lorentzian(f, 3 * A, center + 0.5 * J, gamma)
        y_complex += complex_lorentzian(f, A, center + 1.5 * J, gamma)
    else:
        y_complex += complex_lorentzian(f, A, center, gamma)

    return np.abs(y_complex)


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


@jit(nopython=True, cache=True)
def multi_multiplet_lorentzian(f, params, multiplicities, n_multiplets):
    """
    Sum of multiple multiplets.
    params: [offset,  (A, center, J, gamma) for multiplet 1, (A, center, J, gamma) for multiplet 2, ...]
    multiplicities: array of ints, length n_multiplets
    """
    y = np.full_like(f, params[0])  # Offset

    current_idx = 1
    for i in range(n_multiplets):
        A = params[current_idx]
        center = params[current_idx + 1]
        J = params[current_idx + 2]
        gamma = params[current_idx + 3]

        m = multiplicities[i]

        y += multiplet_lorentzian(f, center, J, m, gamma, A)

        current_idx += 4

    return y
