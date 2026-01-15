import numpy as np
from nmr_analysis.analysis.fitting import Fitter


def test_outlier_removal_logic():
    """Verify that _remove_outliers_semilog correctly identifies outliers."""
    # Create perfect exponential data
    t = np.linspace(0, 10, 20)
    T2_true = 2.0
    M0_true = 100.0
    y = M0_true * np.exp(-t / T2_true)

    # Introduce an outlier (spike)
    outlier_idx = 10
    y[outlier_idx] = M0_true * 2.0  # Huge spike

    # Run outlier detection
    filtered_t, filtered_y, mask = Fitter._remove_outliers_semilog(t, y)

    # Check if outlier was removed
    assert not mask[outlier_idx], "Outlier should be masked out"
    assert sum(mask) == len(t) - 1, "Only one point should be removed"
    assert len(filtered_t) == len(t) - 1
    assert len(filtered_y) == len(y) - 1


def test_fit_t2_with_outlier():
    """Verify fit_t2 is robust to outliers."""
    t = np.linspace(
        0.01, 10, 50
    )  # Start from 0.01 to avoid log(0) issues if any, though t2 model handles t=0
    T2_true = 2.0
    M0_true = 100.0

    # Clean data
    y_clean = M0_true * np.exp(-t / T2_true)

    # Add noise
    rng = np.random.default_rng(42)
    y_noisy = y_clean + rng.normal(0, 0.5, size=len(t))

    # Add outlier
    outlier_idx = 25
    y_noisy[outlier_idx] = 500.0  # Huge outlier

    # Fit
    params, fit_curve, residuals, r2, errors, mask = Fitter.fit_t2(t, y_noisy)

    # Check results
    print(f"Fitted T2: {params['T2']:.4f}, True T2: {T2_true}")
    print(f"Fitted M0: {params['M0']:.4f}, True M0: {M0_true}")

    # Verify mask
    assert mask[outlier_idx], "Outlier should be detected in the mask"
    # assert np.sum(mask) < 10
    assert not mask[0], "First point should valid"

    # The fit should be reasonably close despite the outlier
    assert np.isclose(params["T2"], T2_true, rtol=0.1), (
        f"T2 estimate {params['T2']} is too far from true value {T2_true}"
    )
    assert np.isclose(params["M0"], M0_true, rtol=0.1), (
        f"M0 estimate {params['M0']} is too far from true value {M0_true}"
    )


def test_fit_modulated_t2_outlier_stage1():
    """Verify that outlier removal helps Stage 1 of modulated fit."""
    t = np.linspace(0, 10, 200)
    T2_true = 3.0
    M0_true = 100.0
    J_true = 5.0
    depth_true = 0.5

    # Modulated model
    # M0 * exp(-t/T2) * ((1-d) + d*cos(pi*J*t))
    mod_term = (1 - depth_true) + depth_true * np.cos(np.pi * J_true * t)
    y_clean = M0_true * np.exp(-t / T2_true) * mod_term

    # Add outlier that would mess up simple exponential fit (Stage 1)
    y_noisy = y_clean.copy()
    y_noisy[10] = 1000.0  # Spike

    # Fit
    params, fit_curve, residuals, r2, errors, mask = Fitter.fit_modulated_t2(
        t, y_noisy, guess_J=5.0
    )

    print(f"Modulated Fit Results: {params}")

    # Verify mask
    assert mask[10], "Spike should be detected as outlier"

    assert np.isclose(params["T2"], T2_true, rtol=0.2), (
        "T2 should be recovered reasonably well"
    )
    assert np.isclose(params["J"], J_true, rtol=0.1), "J coupling should be recovered"


if __name__ == "__main__":
    test_outlier_removal_logic()
    test_fit_t2_with_outlier()
    test_fit_modulated_t2_outlier_stage1()
    print("All verification tests passed!")
