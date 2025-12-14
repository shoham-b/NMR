import numpy as np
import pytest
from nmr_analysis.analysis.fitting import Fitter
from nmr_analysis.analysis.models import t1_model, t2_decay_model
from nmr_analysis.core.types import NMRData, ExperimentType


def test_t1_fitting():
    # Generate synthetic T1 data
    delays = np.linspace(0, 10, 20)
    M0 = 100.0
    T1 = 2.5
    alpha = 1.0

    # M(t) = M0 * (1 - 2 * alpha * exp(-t / T1))
    signal = t1_model(delays, M0, T1, alpha)
    # Add noise
    signal += np.random.normal(0, 1.0, size=delays.shape)

    params, fit_curve, residuals, r2 = Fitter.fit_t1(delays, signal)

    assert r2 > 0.95
    assert np.isclose(params["M0"], M0, rtol=0.2)
    assert np.isclose(params["T1"], T1, rtol=0.2)


def test_t2_fitting():
    # Generate synthetic T2 data
    delays = np.linspace(0, 10, 20)
    M0 = 100.0
    T2 = 1.5

    signal = t2_decay_model(delays, M0, T2, 0.0)
    signal += np.random.normal(0, 1.0, size=delays.shape)

    params, fit_curve, residuals, r2 = Fitter.fit_t2(delays, signal)

    assert r2 > 0.95
    assert np.isclose(params["M0"], M0, rtol=0.2)
    assert np.isclose(params["T2"], T2, rtol=0.2)
