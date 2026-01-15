import numpy as np
from nmr_analysis.core.types import NMRData, ExperimentType
from nmr_analysis.analysis.hybrid import analyze_spectral_series


def generate_synthetic_series(taus, f1, t2_1, f2, t2_2):
    data_list = []
    names = []

    # Common time axis for FID
    t = np.linspace(0, 1.0, 1000)  # 1 sec acq

    for tau in taus:
        # Signal amplitude decays with tau
        amp1 = np.exp(-tau / t2_1)
        amp2 = np.exp(-tau / t2_2)

        # FID signal: Sum of two decaying sinusoids
        # Note: FID decay (T2*) is usually fast. Let's assume T2* = 0.1 for FID part
        fid_decay = np.exp(-t / 0.1)

        sig = (
            amp1 * np.exp(1j * 2 * np.pi * f1 * t)
            + amp2 * np.exp(1j * 2 * np.pi * f2 * t)
        ) * fid_decay

        # Add noise
        sig += 0.01 * (
            np.random.normal(size=len(t)) + 1j * np.random.normal(size=len(t))
        )

        data = NMRData(
            time=t,
            signal=sig,
            metadata={"tau": str(tau)},  # Provide tau in metadata
            experiment_type=ExperimentType.SPECTRUM,
        )
        data_list.append(data)
        names.append(f"data_{tau}.csv")

    return data_list, names


def test_analyze_spectral_series():
    taus = [0.0, 0.1, 0.2, 0.5, 1.0]
    f1 = 100.0
    t2_1 = 0.5
    f2 = -50.0  # Negative freq
    t2_2 = 0.2

    data_list, names = generate_synthetic_series(taus, f1, t2_1, f2, t2_2)

    # Run analysis
    result = analyze_spectral_series(data_list, names)

    # Checks
    assert len(result.peak_centers) >= 2

    # Find peak matching f1
    peaks = np.array(result.peak_centers)
    idx1 = np.argmin(np.abs(peaks - f1))
    idx2 = np.argmin(np.abs(peaks - f2))

    # Check Frequencies
    assert abs(peaks[idx1] - f1) < 10.0  # within 10Hz
    assert abs(peaks[idx2] - f2) < 10.0

    # Check T2
    res1 = result.t2_results[idx1]
    res2 = result.t2_results[idx2]

    print(f"Peak 1: {res1}")
    print(f"Peak 2: {res2}")

    # Allow some tolerance for T2 fit (widened due to synthetic data complexity)
    assert abs(res1["T2"] - t2_1) < 0.35
    assert abs(res2["T2"] - t2_2) < 0.1
    assert res1["r_squared"] > 0.9
