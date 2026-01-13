import numpy as np
import pytest
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.processing import compute_spectrum
from nmr_analysis.core.types import NMRData, ExperimentType


def test_spectral_fit_synthetic():
    """
    Test Spectral Fit using synthetic data with known Lorentzians.
    """
    # Parameters
    # To resolve sharp peaks, we need long T_total (small df)
    # T2* = 0.05s => FWHM ~ 6Hz. We need df < 1-2 Hz.
    # df = 1 / (N*dt).
    # Let's use dt=1e-4 (10kHz bandwidth, +/- 5kHz).
    # N=16384 => T=1.6s => df=0.6 Hz.
    N = 16384
    dt = 1e-4  # 100us sampling
    t = np.arange(N) * dt

    # 2 Frequencies: f1=100Hz, f2=500Hz
    # T2*: 0.05s, 0.02s
    # M0: 1.0, 0.5
    f1, f2 = 100.0, 500.0
    T2s1, T2s2 = 0.05, 0.02
    A1, A2 = 1.0, 0.5

    # Generate FID: M(t) ~ A * exp(-t/T2*) * exp(i*2*pi*f*t)
    # We use Complex FID for proper spectral separation
    # If the system expects real data, we might need to adjust.
    # Usually NMR data is quadrature (complex).
    # Let's assume input is complex. If standard load is complex?
    # Our commands.py uses `np.abs(data.signal)` for T2*.
    # For Spectrum, we need complex signal to distinguish + / - freq?
    # Or start with real signal (cosine) -> symmetric peaks.
    # Let's create complex signal to simulate quadrature detection.

    sig = A1 * np.exp(-t / T2s1) * np.exp(1j * 2 * np.pi * f1 * t) + A2 * np.exp(
        -t / T2s2
    ) * np.exp(1j * 2 * np.pi * f2 * t)

    # Add noise?
    # sig += np.random.normal(0, 0.01, N) + 1j * np.random.normal(0, 0.01, N)

    data = NMRData(
        time=t, signal=sig, metadata={}, experiment_type=ExperimentType.T2_STAR
    )

    # 1. Compute Spectrum
    freqs, spectrum = compute_spectrum(data)

    # 2. Fit Spectrum
    # We fit the Magnitude Spectrum
    mag_spec = np.abs(spectrum)

    # Call the new function (to be implemented)
    result = Fitter.fit_spectrum(freqs, mag_spec, min_prominence=0.1)

    print("\nFit Results:")
    for p in result.params["peaks"]:
        print(p)

    # Validation
    # Identify peaks by finding closest freq
    peaks = result.params["peaks"]
    assert len(peaks) == 2, f"Expected 2 peaks, found {len(peaks)}"

    # Check peak 1 (~100Hz)
    p1 = min(peaks, key=lambda x: abs(x["f0"] - f1))
    assert abs(p1["f0"] - f1) < 5.0  # Freq within 5Hz
    # T2* from width: FWHM ~ 1/(pi*T2*) => width (gamma in lorentzian?)
    # Lorentzian L(f) ~ 1/((f-f0)^2 + gamma^2)
    # HWHM = gamma. FWHM = 2*gamma. T2* = 1 / (pi * FWHM) = 1 / (2*pi*gamma)
    # Wait, convention: L(f) = A / ( ... )
    # Let's check model implementation plan.
    # Plan: S(f) = sum( A * gamma / (pi * ((f-f0)^2 + gamma^2)) )
    # This is normalized Lorentzian area.
    # Max height (at f0) = A / (pi * gamma)
    # HWHM = gamma.
    # Relation to T2*:
    # FID: exp(-t/T2) <-> Lorentzian: 1 / ( (1/T2)^2 + (2*pi*f)^2 )?
    # Fourier Transform of exp(-t/T2) * exp(i*w0*t) is roughly Lorentzian.
    # Line width (FWHM in Hz) = 1 / (pi * T2)
    # FWHM = 2 * gamma.
    # So 2 * gamma = 1 / (pi * T2) => T2 = 1 / (2 * pi * gamma).

    expected_gamma1 = 1.0 / (np.pi * T2s1) / 2.0  # Wait. FWHM = 1/(pi*T2)
    # expected gamma (HWHM) = FWHM/2 = 1/(2*pi*T2)

    derived_t2_1 = p1["t2_star"]
    assert abs(derived_t2_1 - T2s1) < 0.01, (
        f"T2* mismatch for peak 1: got {derived_t2_1}, expected {T2s1}"
    )

    # Check peak 2 (~500Hz)
    p2 = min(peaks, key=lambda x: abs(x["f0"] - f2))
    assert abs(p2["f0"] - f2) < 5.0
    derived_t2_2 = p2["t2_star"]
    assert abs(derived_t2_2 - T2s2) < 0.01, (
        f"T2* mismatch for peak 2: got {derived_t2_2}, expected {T2s2}"
    )


def test_spectral_multiplet_synthetic():
    """
    Test Multiplet Fitting (J-coupling) with synthetic Triplet.
    """
    N = 16384
    dt = 1e-4
    t = np.arange(N) * dt
    freqs = np.fft.fftfreq(N, dt)
    freqs = np.fft.fftshift(freqs)

    # Synthetic Triplet (1:2:1) centered at 200 Hz, J=10 Hz
    # Peaks at 190, 200, 210
    # Amp A=1 (outer), 2A=2 (center)
    center = 200.0
    J_true = 10.0
    T2 = 0.05
    A = 1000.0

    # Generate FID
    # Triplet in Time Domain = A * exp(-t/T2) * cos(pi*J*t)^2 ?? No.
    # 1:2:1 is (1 + e^iwt)^2? No.
    # Triplet is sum of 3 lines.

    sig = (
        A * np.exp(-t / T2) * np.exp(1j * 2 * np.pi * (center - J_true) * t)
        + 2 * A * np.exp(-t / T2) * np.exp(1j * 2 * np.pi * (center) * t)
        + A * np.exp(-t / T2) * np.exp(1j * 2 * np.pi * (center + J_true) * t)
    )

    spectrum = np.fft.fftshift(np.fft.fft(sig))

    # Config
    # We guess center=200, multiplicity=3
    multiplets_config = [
        {"center": 202.0, "multiplicity": 3, "initial_J": 5.0, "initial_gamma": 2.0}
        # Intentionally slightly off
    ]

    result = Fitter.fit_multiplet_spectrum(freqs, spectrum, multiplets_config)

    print("\nMultiplet Fit Results:")
    for m in result.params["multiplets"]:
        print(m)

    m1 = result.params["multiplets"][0]

    assert abs(m1["center"] - center) < 1.0, f"Center mismatch: {m1['center']}"
    assert abs(m1["J"] - J_true) < 1.0, f"J mismatch: {m1['J']}"
    # Gamma should be related to T2
    # Gamma ~ 1/(pi*T2) ??? No, 1/(2*pi*T2)?
    # Previous test derived_t2 from gamma.
    # Here let's just assert J and Center mainly.


if __name__ == "__main__":
    test_spectral_fit_synthetic()
    test_spectral_multiplet_synthetic()
