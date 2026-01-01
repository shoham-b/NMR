import pytest
import numpy as np
from nmr_analysis.core.types import NMRData, ExperimentType
from nmr_analysis.analysis.processing import preprocess_data


def test_preprocess_data_time_shift():
    # synthetic data: single peak at index 10
    time = np.linspace(0, 10, 100)  # 0 to 10s
    signal = np.zeros(100)
    signal[10] = 6.0
    signal[9] = 1.0
    signal[11] = 1.0

    data = NMRData(time=time, signal=signal)

    # 1 peak -> P1 is start and fit
    processed, t_orig, amp_orig, _ = preprocess_data(data, smoothing=0, min_height=1.0)

    # Check shift: P1 (index 10) is at 0
    assert processed.time[10] == 0.0

    # Check start is negative
    assert processed.time[0] < 0.0

    # Check NO slicing
    assert len(processed.signal) == 100
    assert np.allclose(processed.signal, signal)

    # Tau should be 0 since P1=P3
    assert t_orig == 0.0
    assert amp_orig == signal[10]


def test_preprocess_data_metadata_Preserved():
    time = np.linspace(0, 10, 100)
    signal = np.zeros(100)
    signal[50] = 6.0
    sorted_signal = signal
    data = NMRData(time=time, signal=signal, metadata={"key": "val"})

    processed, _, _, _ = preprocess_data(data, smoothing=0, min_height=1.0)
    assert processed.metadata["key"] == "val"


def test_preprocess_3_peaks_logic():
    # Scenario: 3 dominant peaks Logic.
    # P1 (Highest Amp) at idx 10 (t=1.0) -> Start of slice
    # P2 (2nd Highest) at idx 30 (t=3.0) -> Ignored (Noise)
    # P3 (3rd Highest) at idx 80 (t=8.0) -> Target (Fit)

    # Amplitudes: P1=10, P2=5, P3=8.
    # Amplitude Rank: P1 (10), P3 (8), P2 (5).
    # Sorted Chronologically from Top 3: [10, 30, 80].
    # Logic: Start=10. Fit=80.

    time = np.linspace(0, 10, 100)  # 0.1 step
    signal = np.zeros(100)

    # Peak 1
    signal[10] = 10.0
    signal[9] = 2
    signal[11] = 2
    # Construct signal with 3 peaks:
    # P1 (highest) at index 100, Amp 10
    # P2 (noise) at index 200, Amp 8
    # P3 (fit target) at index 300, Amp 6
    cols = [(100, 10), (200, 8), (300, 6), (900, 2)]
    # Use 1000 points, time 0 to 10.0
    # idx 100 -> t=1.0
    # idx 300 -> t=3.0

    signal = np.zeros(1000)
    time = np.linspace(0, 10, 1000)
    for idx, amp in cols:
        signal[idx] = amp

    data = NMRData(time=time, signal=signal, metadata={})

    processed, tau, amp, peak_info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Check that time was shifted so P1 (index 100) is at 0
    # Original time at 100 is 1.0. So new time should be t - 1.0.
    assert processed.time[100] == 0.0

    # Check that we have negative times for indices < 100
    assert processed.time[0] < 0

    # Check that data length is preserved (no slicing)
    assert len(processed.time) == len(data.time)
    assert len(processed.signal) == len(data.signal)

    # Check amp and tau
    # P3 is at 300 (time 3.0). Tau = 3.0 - 1.0 = 2.0
    assert abs(tau - 2.0) < 1e-1  # Approx check due to generic setup
    # Amp of P3 is 6.0
    assert abs(amp - 6.0) < 1e-6

    # Verify peak_info
    assert peak_info["p1_idx"] == 100
    assert peak_info["p2_idx"] == 200
    assert peak_info["p3_idx"] == 300
    assert len(peak_info["all_peaks"]) >= 3


def test_preprocess_fallback_2_peaks():
    # Scenario: Only 2 peaks found.
    # Fallback Logic: P1=Start. Last=Fit.
    # P1 at 10 (Amp 10). P2 at 50 (Amp 8).

    time = np.linspace(0, 10, 100)
    signal = np.zeros(100)
    signal[10] = 10.0
    signal[50] = 8.0

    data = NMRData(time=time, signal=signal)

    processed, tau, amp, peak_info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Start=10
    assert processed.time[10] == 0.0
    assert len(processed.signal) == 100
    assert processed.time[0] < 0.0

    # Fit=50
    assert amp == 8.0
    expected_tau = time[50] - time[10]
    assert tau == pytest.approx(expected_tau, abs=1e-3)

    assert peak_info["p1_idx"] == 10
    assert peak_info["fit_idx"] == 50


def test_find_peaks_t1_logic():
    # T1 should use P2 for fit, ignoring P3.
    # P1 (100, 10), P2 (200, 8), P3 (300, 6)
    cols = [(100, 10), (200, 8), (300, 6)]
    signal = np.zeros(1000)
    time = np.linspace(0, 10, 1000)
    for idx, amp in cols:
        signal[idx] = amp

    data = NMRData(time=time, signal=signal, experiment_type=ExperimentType.T1)

    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Fit index should be P2 (idx 200)
    # Check what find_peaks returns. P1=100, P2=200, P3=300.
    # Logic: T1 -> use P2.
    assert info["fit_idx"] == 200
    assert info["p1_idx"] == 100
    assert info["p2_idx"] == 200

    # Tau = time[200] - time[100] = 2.0 - 1.0 = 1.0
    assert abs(tau - 1.0) < 3e-3
    assert amp == 8.0


def test_find_peaks_t2_logic_standard():
    # T2 should use P3 for fit, provided it is far enough from P1.
    # P1 (100, 1.0s), P3 (300, 3.0s). Delta = 2.0s > 0.4s.
    cols = [(100, 10), (200, 8), (300, 6)]
    signal = np.zeros(1000)
    time = np.linspace(0, 10, 1000)
    for idx, amp in cols:
        signal[idx] = amp

    data = NMRData(time=time, signal=signal, experiment_type=ExperimentType.T2)

    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Fit index should be P3 (idx 300)
    assert info["fit_idx"] == 300
    assert info["p3_idx"] == 300

    # Tau = time[300] - time[100] = 3.0 - 1.0 = 2.0
    assert abs(tau - 2.0) < 3e-3
    assert amp == 6.0


def test_find_peaks_t2_logic_close_peaks():
    # T2 should use P2 if P3 is too close (< 0.4s) to P1.
    # P1 at 1.0s. P3 at 1.2s (Delta 0.2s < 0.4s).
    # P2 at 1.1s (irrelevant for condition, but used for fallback).
    # Wait, usually P2 is between P1 and P3?
    # Chronological sort: P1 < P2 < P3.
    # So if P3 is close, P2 must be even closer?
    # User logic: "if the third peak is at time smaller than 0.4 after the first peak ... use the second peak"
    # This implies P2 might NOT be chronologically between P1 and P3?
    # OR maybe P2 is later?
    # Our sorting logic: "3. Sort chronologically: P1, P2, P3".
    # So P1 < P2 < P3 is ALWAYS true by definition of our sorting.
    # If P3 - P1 < 0.4, then P2 - P1 must also be < 0.4.
    # So using P2 would also be "close".
    # BUT, the user explicitly said "use the second peak".
    # Maybe the user implies the "Top 3 by amplitude" are NOT sorted chronologically for P1/P2/P3 labels?
    # My implementation explicitly SORTS them chronologically.
    # User's request: "The first peak (P1) is used to trim... The second peak (P2) is to be ignored... The third peak (P3) is used for fitting"
    # User session update: "Sort chronologically: P1 (Start), P2 (Noise), P3 (Fit)."
    # So P1 < P2 < P3.
    # If P3 - P1 < 0.4, then P2 is definitely < 0.4.
    # Using P2 seems counter-intuitive if the goal is to get away from P1.
    # UNLESS P2 is actually FURTHER?
    # Re-reading: "Logic: 1. Find all peaks... 2. Sort by amplitude... 3. Sort chronologically".
    # If P1 is first in time, P2 is second in time, P3 is third in time.
    # Then P2 is strictly closer to P1 than P3 is.
    # Perhaps the user meant "Second dominant peak" vs "Third dominant peak" WITHOUT chronological sort?
    # Let's check previous context.
    # User said: "The first peak (P1) is used to trim... The second peak (P2) is to be ignored... The third peak (P3) is used for fitting"
    # And "Refine T1/T2 Peak Selection Logic: Implement a '3 Dominant Peaks' logic"
    # Usually in NMR T1/T2, peaks are periodic?
    # Or is it an echo train?
    # If it is an echo train (T2), peaks appear at t=0, t=echo, t=2*echo...
    # If P1 is at 0. P2 at echo. P3 at 2*echo.
    # If P3 is < 0.4s, then echo spacing is very short.
    # Maybe P2 is actually *noise* and P3 is the *real* first echo?
    # But user said P2 is usually ignored as noise.
    # Let's strictly follow the user's instruction: "If P3 < 0.4s from P1, use P2".
    # Even if P2 is closer. I will implement as requested.

    # Construct data where P3 is close.
    # P1 at 1.0s. P2 at 1.1s. P3 at 1.3s.
    cols = [
        (100, 10),
        (110, 8),
        (130, 6),
    ]  # Indices. Time 0 to 10 for 1000 points = 0.01s per point.
    # 100 -> 1.0s
    # 110 -> 1.1s
    # 130 -> 1.3s
    # Delta P3-P1 = 0.3s < 0.4s. Use P2 (index 110).

    signal = np.zeros(1000)
    time = np.linspace(0, 10, 1000)
    for idx, amp in cols:
        signal[idx] = amp

    data = NMRData(time=time, signal=signal, experiment_type=ExperimentType.T2)

    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=1.0)

    # Fit index should be P2 (idx 110/11 depending on linspace resolution)
    # 100 -> 1.0. 110 -> 1.1. 130 -> 1.3.
    # 1.3 - 1.0 = 0.3 < 0.4. Condition met. Use P2.
    assert info["fit_idx"] == 110
    assert info["p3_idx"] == 130
    assert abs(tau - 0.1) < 3e-3


def test_find_peaks_t2_p1_max_logic():
    # Enforce P1 = Max Amplitude for T2.
    # Scenario:
    # Noise peak at t=1.0 (idx 100), Amp=5.0
    # Main Pulse (P1) at t=2.0 (idx 200), Amp=10.0 (MAX)
    # Decay Peak (P3) at t=3.0 (idx 300), Amp=8.0

    # Chronological: Noise, P1, P3.
    # Amplitude: P1(10), P3(8), Noise(5).
    # Logic T2:
    # P1 = Max = Index 200.
    # Others = Index 100, Index 300.
    # Sorted Chrono: P2=100, P3=300.

    # Fit target: P3 (300).
    # Check condition: P3 - P1 = 3.0 - 2.0 = 1.0 > 0.4. OK.

    cols = [(100, 5), (200, 10), (300, 8)]
    signal = np.zeros(1000)  # 10s total
    time = np.linspace(0, 10, 1000)
    for idx, amp in cols:
        signal[idx] = amp

    data = NMRData(time=time, signal=signal, experiment_type=ExperimentType.T2)

    processed, tau, amp, info = preprocess_data(data, smoothing=0, min_height=1.0)

    assert info["p1_idx"] == 200
    assert info["p2_idx"] == 100
    assert info["p3_idx"] == 300
    assert info["fit_idx"] == 300

    # Tau = P3 - P1 = 3.0 - 2.0 = 1.0
    assert abs(tau - 1.0) < 3e-3
    assert amp == 8.0
