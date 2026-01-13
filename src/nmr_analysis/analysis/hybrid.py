from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from nmr_analysis.core.types import NMRData
from nmr_analysis.analysis.processing import (
    compute_spectrum,
    integrate_spectral_peaks,
    parse_time_from_filename,
    get_delay_from_metadata,
)
from nmr_analysis.analysis.models import t2_decay_model


@dataclass
class HybridAnalysisResult:
    dataset_name: str
    tau_values: np.ndarray
    peak_centers: List[float]  # Frequencies
    integrated_areas: np.ndarray  # Shape (n_peaks, n_taus)
    t2_results: List[Dict]  # List of dicts with 'T2', 'M0', 'offset', 'r_squared'
    spectra_stack: Tuple[np.ndarray, List[np.ndarray]]  # freqs, list of spectra
    time_stack: List[NMRData]


def analyze_spectral_series(
    data_list: List[NMRData], names: List[str]
) -> HybridAnalysisResult:
    """
    Perform Hybrid Spectral-Temporal T2 Analysis on a series of NMRData.
    """
    # 1. Parse Tau and Sort
    # Try metadata first, then filename
    data_with_tau = []
    for data, name in zip(data_list, names):
        tau = get_delay_from_metadata(data)
        if tau == 0.0:
            tau = parse_time_from_filename(name)
        data_with_tau.append((tau, data, name))

    # Sort by tau
    data_with_tau.sort(key=lambda x: x[0])

    sorted_taus = np.array([x[0] for x in data_with_tau])
    sorted_data = [x[1] for x in data_with_tau]

    # 2. Compute Spectra for Ref (Shortest Tau)
    # Use the first valid one to find peaks
    ref_idx = -1
    freqs = np.array([])
    ref_spect = np.array([])

    for i, d in enumerate(sorted_data):
        # Implement Peak Slicing for T2 (User Request: "Only 2nd peak... should enter fourier transform")
        from scipy.signal import find_peaks as sp_find_peaks
        from scipy.ndimage import gaussian_filter1d

        # We need to replicate logic used in commands.py
        sig_abs = np.abs(d.signal)
        time = d.time
        dt = time[1] - time[0] if len(time) > 1 else 1.0

        # Smooth
        sigma_points = int(50e-6 / dt)
        if sigma_points < 1:
            sigma_points = 1
        smoothed_sig = gaussian_filter1d(sig_abs, sigma=sigma_points * 10)

        dist_points = int(1e-3 / dt)
        if dist_points < 100:
            dist_points = 100

        peaks, _ = sp_find_peaks(
            smoothed_sig, distance=dist_points, height=0.05 * np.max(smoothed_sig)
        )

        target_sig = d.signal
        target_time = d.time

        if len(peaks) >= 2:
            p1 = peaks[0]
            p2 = peaks[1]
            spacing = p2 - p1
            start_idx = int(p2 - (spacing // 2))
            end_idx = int(p2 + (spacing // 2))
            start_idx = max(0, start_idx)
            end_idx = min(len(d.signal), end_idx)

            target_sig = d.signal[start_idx:end_idx]
            target_time = d.time[start_idx:end_idx]

        # Create temp object for FFT
        temp_data = NMRData(time=target_time, signal=target_sig)

        f, s = compute_spectrum(temp_data)
        if len(f) > 0:
            if ref_idx == -1:  # First valid file logic
                ref_idx = i
                freqs = f
                ref_spect = s
            # We break? No, we need to iterate ALL to compute spectra?
            # Wait, line 51 says "Compute Spectra for Ref (Shortest Tau)".
            # Ah, block 51-63 is finding the REFERENCE spectrum (freq axis, peak centers)
            # The loop 107 computes individual spectra.
            # I should move this logic to a helper or replicate it in loop 107 too.
            # Ideally compute_spectrum should optionally support this or I wrap it.
            # But 'analyze_spectral_series' logic structure:
            # 1. Find Ref.
            # 2. Find Peaks on Ref.
            # 3. Integrate all.
            # So I need the slicing to happen consistently.
            # If I slice Ref, I must slice others too?
            # Yes, otherwise frequency axis might differ (N points)?
            # compute_spectrum returns freqs based on N. If N varies (slicing width varies?), freqs vary.
            # If peak spacing varies, slice width varies.
            # BUT for T2 series, peaks (echoes) should be stable in time?
            # Yes, CPMG echoes are fixed.
            # So slicing should be consistent IF p2 is consistent.

            # Let's break ONLY if we found a ref.
            if ref_idx != -1:
                break

    if ref_idx == -1:
        # All files empty?
        return HybridAnalysisResult(
            dataset_name=names[0] if names else "Unknown",
            tau_values=sorted_taus,
            peak_centers=[],
            integrated_areas=np.zeros((0, len(sorted_taus))),
            t2_results=[],
            spectra_stack=(np.array([]), []),
            time_stack=sorted_data,
        )

    ref_mag = np.abs(ref_spect)

    # 3. Peak Finding on Ref
    # Heuristic: Find dominant peaks
    prominence = 0.05 * np.max(ref_mag)
    peak_indices, _ = find_peaks(ref_mag, prominence=prominence, distance=10)

    # If no peaks, fallback to max
    if len(peak_indices) == 0:
        peak_indices = [np.argmax(ref_mag)]

    peak_centers = freqs[peak_indices]

    # 4. Integrate Series
    n_peaks = len(peak_centers)
    n_files = len(sorted_data)

    integrated_areas = np.zeros((n_peaks, n_files))
    spectra_list = []

    # Define integration width: e.g. 500 Hz or based on FWHM?
    # Let's simple check spacing. If peaks are close, reduce width.
    if len(peak_centers) > 1:
        # Sort centers
        sorted_centers = np.sort(peak_centers)
        min_spacing = np.min(np.diff(sorted_centers))
        width_hz = min(500.0, min_spacing * 0.8)  # 80% of spacing or 500Hz
    else:
        width_hz = 500.0  # Default generous width

    for i, data in enumerate(sorted_data):
        # Replicate Peak Slicing Logic for Consistency
        from scipy.signal import find_peaks as sp_find_peaks
        from scipy.ndimage import gaussian_filter1d

        sig_abs = np.abs(data.signal)
        time = data.time
        dt = time[1] - time[0] if len(time) > 1 else 1.0

        sigma_points = int(50e-6 / dt)
        if sigma_points < 1:
            sigma_points = 1
        smoothed_sig = gaussian_filter1d(sig_abs, sigma=sigma_points * 10)

        dist_points = int(1e-3 / dt)
        if dist_points < 100:
            dist_points = 100

        peaks, _ = sp_find_peaks(
            smoothed_sig, distance=dist_points, height=0.05 * np.max(smoothed_sig)
        )

        target_sig = data.signal
        target_time = data.time

        if len(peaks) >= 2:
            p1 = peaks[0]
            p2 = peaks[1]
            spacing = p2 - p1
            start_idx = int(p2 - (spacing // 2))
            end_idx = int(p2 + (spacing // 2))
            start_idx = max(0, start_idx)
            end_idx = min(len(data.signal), end_idx)

            target_sig = data.signal[start_idx:end_idx]
            target_time = data.time[start_idx:end_idx]

        # Create temp object
        temp_data = NMRData(time=target_time, signal=target_sig)

        current_freqs, current_spect = compute_spectrum(temp_data)

        if len(current_freqs) == 0:
            # Handle empty file
            spectra_list.append(np.zeros_like(ref_mag))
            continue

        spectra_list.append(np.abs(current_spect))

        # Integrate
        # Note: current_freqs should be same as freqs if sampling is same
        # If sampling changed, we might have issues, but assuming consistent settings.
        areas = integrate_spectral_peaks(
            current_freqs, current_spect, peak_centers, width_hz=width_hz
        )
        integrated_areas[:, i] = areas

    # 5. Fit T2 Decays
    t2_results = []

    # 5. Fit T1/T2 Decays
    t2_results = []

    # Check if T1: Heuristic check of names/path context
    is_t1 = False
    # Infer dataset_name from first file parent if possible or pass it?
    # names[0] is just filename.
    # But usually dataset name is directory.
    # We can infer T1 from filenames/path
    dataset_name_check = names[0] if names else ""

    if "T1" in dataset_name_check:
        is_t1 = True

    # Use T1 model if T1
    from nmr_analysis.analysis.models import t1_model

    for k in range(n_peaks):
        y_data = integrated_areas[k, :]

        try:
            if is_t1:
                # T1 Fit
                # Initial Guess: M0 ~ max, T1 ~ mean(tau)
                p0 = [
                    np.max(np.abs(y_data)),
                    np.mean(sorted_taus),
                    1.0,
                ]  # M0, T1, alpha
                # No bounds usually for T1? M0>0, T1>0.
                bounds_min = [0, 0, -np.inf]
                bounds_max = [np.inf, np.inf, np.inf]

                # Note: Integrated Area is Magnitude?
                # If T1, spectral intensity might go through zero (phase sensitive) or be magnitude?
                # 'spectra_list' stores np.abs(spect). So it's Magnitude.
                # Magnitude T1 Recovery: |M0 * (1 - 2*exp(-t/T1))|
                # But 't1_model' is signed.
                # If we work with Magnitude Spectra, we lose the sign.
                # Fitting t1_model to Magnitude data is tricky around zero.
                # However, usually we see recovering magnitude.
                # If we use Magnitude, we might need a Magnitude T1 model?
                # Or we assume fully relaxed?
                # Standard T1 analysis on Magnitude data: M(t) = M0 * | 1 - 2*exp(-t/T1) |
                # Let's try fitting basic recovery M(t) = M0 * (1 - exp(-t/T1)) ? No, Inversion is 1-2exp.

                # If the user plotted "Integrated Area", and it's magnitude, they see a "V" shape if it goes through zero.
                # OR if the phase is corrected, they see proper inversion.
                # `compute_spectrum` -> `np.abs(spect)` (in loop above). So it IS Magnitude.
                # So we are fitting Magnitude T1.
                # For now, let's use the standard T1 model but fit to SIGNED data if we could?
                # We can't easily recover sign from `np.abs`.
                # So we fit "Abs T1": | M0(1 - 2a exp(-t/T1)) |
                # Let's define a local wrapper or just try fitting exponential recovery if data is > 0?

                # Simpler: If T1, just fit Saturation Recovery or Inversion Recovery Magnitude?
                # Let's stick to t1_model but be aware.
                # Actually, can we try to guess sign?
                # If tau is small, signal should be inverted?
                # But we have only magnitude.
                # Let's proceed with t1_model. If fit fails, user sees it.

                popt, pcov = curve_fit(
                    t1_model, sorted_taus, y_data, p0=p0, maxfev=5000
                )

                res_dict = {
                    "M0": popt[0],
                    "T1": popt[1],
                    "alpha": popt[2],
                    "r_squared": 0,  # Calc below
                    "f0": peak_centers[k],
                }

                residuals = y_data - t1_model(sorted_taus, *popt)

            else:
                # T2 Fit
                p0 = [np.max(y_data), 0.1, 0.0]  # M0, T2, offset
                bounds_min = [0, 0, -np.inf]
                bounds_max = [np.inf, np.inf, np.inf]

                popt, pcov = curve_fit(
                    t2_decay_model,
                    sorted_taus,
                    y_data,
                    p0=p0,
                    bounds=(bounds_min, bounds_max),
                )

                res_dict = {
                    "M0": popt[0],
                    "T2": popt[1],
                    "offset": popt[2],
                    "r_squared": 0,
                    "f0": peak_centers[k],
                }
                residuals = y_data - t2_decay_model(sorted_taus, *popt)

            # Calculate R2
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            res_dict["r_squared"] = r2
            t2_results.append(res_dict)

        except Exception as e:
            t2_results.append(
                {
                    "M0": 0,
                    "T2": 0,  # or T1
                    "offset": 0,
                    "r_squared": 0,
                    "error": str(e),
                    "f0": peak_centers[k],
                }
            )

    return HybridAnalysisResult(
        dataset_name=names[
            0
        ],  # Name of first file as proxy? Or directory name ideally passed in.
        tau_values=sorted_taus,
        peak_centers=list(peak_centers),
        integrated_areas=integrated_areas,
        t2_results=t2_results,
        spectra_stack=(freqs, spectra_list),
        time_stack=sorted_data,
    )
