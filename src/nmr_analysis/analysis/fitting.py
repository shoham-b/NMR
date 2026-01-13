from typing import Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit

from nmr_analysis.analysis.models import t1_model, t2_decay_model
from nmr_analysis.core.types import NMRData, AnalysisResult, ExperimentType


class Fitter:
    @staticmethod
    def _remove_outliers_semilog(
        delays: np.ndarray, amplitudes: np.ndarray, threshold: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify outliers based on semi-log linearity.
        Returns: (filtered_delays, filtered_amplitudes, mask)
        """
        # Only consider positive amplitudes for log
        valid_mask = amplitudes > 0

        # If too few points, return original
        if np.sum(valid_mask) < 3:
            return delays, amplitudes, np.ones_like(delays, dtype=bool)

        t_valid = delays[valid_mask]
        log_amp_valid = np.log(amplitudes[valid_mask])

        # Linear fit: log(A) = -t/T2 + log(M0)
        # y = mx + c
        slope, intercept = np.polyfit(t_valid, log_amp_valid, 1)

        predicted_log = slope * t_valid + intercept
        residuals = log_amp_valid - predicted_log

        # Use robust statistics (MAD) instead of STD to be resistant to large outliers
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))

        # 1.4826 is the scaling factor for normal distribution consistency
        sigma_est = 1.4826 * mad

        if sigma_est == 0:
            # If perfect fit or no variation, fall back to std or just keep all
            sigma_est = np.std(residuals)
            if sigma_est == 0:
                return delays, amplitudes, np.ones_like(delays, dtype=bool)

        # Identify outliers
        # We define outliers as deviations from the MEDIAN residual, not mean
        deviation = np.abs(residuals - median_res)
        inlier_mask_subset = deviation <= threshold * sigma_est

        # Create full mask
        final_mask = np.zeros_like(delays, dtype=bool)

        # We need to map back the inlier_mask_subset to the full array
        # valid_mask indices:
        valid_indices = np.where(valid_mask)[0]
        final_mask[valid_indices] = inlier_mask_subset

        return delays[final_mask], amplitudes[final_mask], final_mask

    @staticmethod
    @staticmethod
    def _detect_outliers_post_fit(
        delays: np.ndarray,
        amplitudes: np.ndarray,
        fit_func: callable,
        popt: list,
        threshold: float = 4.0,
    ) -> np.ndarray:
        """
        Detect outliers based on residuals from a fitted model.
        Returns: Boolean mask where True indicates an OUTLIER.
        """
        # Calculate fit and residuals
        fit_curve = fit_func(delays, *popt)
        residuals = amplitudes - fit_curve

        # Robust Noise Estimation (MAD)
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        sigma_est = 1.4826 * mad

        if sigma_est == 0:
            sigma_est = np.std(residuals)
            if sigma_est == 0:
                # Perfect fit, no outliers
                return np.zeros_like(delays, dtype=bool)

        # Deviation from MATCHING curve (residuals)
        deviation = np.abs(residuals - median_res)
        outlier_mask = deviation > (threshold * sigma_est)

        return outlier_mask

    @staticmethod
    def fit_t1(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float, dict, np.ndarray]:
        """
        Fit T1 Inversion Recovery data.
        Returns: params, fit_curve, residuals, r_squared, param_errors, outlier_mask
        """
        # Initial guess
        M0_guess = np.max(np.abs(amplitudes))

        # Smarter T1 Guess
        if len(delays) > 0:
            if amplitudes[0] < -0.5 * M0_guess:
                # Significant inversion
                if amplitudes[-1] < 0:
                    # Still inverted at end -> Long T1
                    T1_guess = np.max(delays) * 5.0
                else:
                    # Crosses zero -> T1 is around zero crossing
                    idx_min = np.argmin(np.abs(amplitudes))
                    # Zero crossing is at t = T1 * ln(2*alpha) approx T1*0.69
                    T1_guess = delays[idx_min] / 0.693
            else:
                T1_guess = np.mean(delays)
        else:
            T1_guess = 1.0

        p0 = [M0_guess, T1_guess, 1.0]

        # Stage 1: Initial Fit (All Points)
        try:
            # First pass: Robust Fit (soft_l1) if supported by scipy, else default
            try:
                popt, pcov = curve_fit(
                    t1_model, delays, amplitudes, p0=p0, maxfev=10000, loss="soft_l1"
                )
            except TypeError:
                # Fallback for older scipy without loss param? (Unlikely for modern env)
                popt, pcov = curve_fit(
                    t1_model, delays, amplitudes, p0=p0, maxfev=10000
                )
        except (RuntimeError, ValueError) as e:
            # If standard fit fails, return empty
            print(f"Fit failed: {e}")
            return (
                {},
                np.zeros_like(delays),
                np.zeros_like(delays),
                0.0,
                {},
                np.zeros_like(delays, dtype=bool),
            )

        # Stage 2: Detect Outliers (Post-Fit)
        outlier_mask = Fitter._detect_outliers_post_fit(
            delays, amplitudes, t1_model, popt, threshold=4.0
        )

        # Stage 3: Refit check
        # If we have outliers, refit without them for better accuracy
        if np.any(outlier_mask) and np.sum(~outlier_mask) > 3:
            try:
                # Filter data
                delays_clean = delays[~outlier_mask]
                amps_clean = amplitudes[~outlier_mask]

                popt_clean, pcov_clean = curve_fit(
                    t1_model, delays_clean, amps_clean, p0=popt, maxfev=10000
                )
                popt = popt_clean
                pcov = pcov_clean
            except (RuntimeError, ValueError):
                # If refit fails, keep original (robust) fit
                pass

        # Final Calculation (on ALL points using final params)
        M0, T1, alpha = popt
        fit_curve = t1_model(delays, *popt)
        residuals = amplitudes - fit_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        perr = np.sqrt(np.diag(pcov))
        param_errors = {"M0": perr[0], "T1": perr[1], "alpha": perr[2]}

        return (
            {"M0": M0, "T1": T1, "alpha": alpha},
            fit_curve,
            residuals,
            r2,
            param_errors,
            outlier_mask,
        )

    @staticmethod
    def fit_t2(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float, dict, np.ndarray]:
        """
        Fit T2 Spin Echo decay.
        Returns: params, fit_curve, residuals, r_squared, param_errors, outlier_mask
        """
        M0_guess = np.max(amplitudes) if len(amplitudes) > 0 else 1.0
        T2_guess = np.mean(delays) if len(delays) > 0 else 1.0
        p0 = [M0_guess, T2_guess, 0.0]

        if len(delays) < 3:
            return (
                {},
                np.zeros_like(delays),
                np.zeros_like(delays),
                0.0,
                {},
                np.zeros_like(delays, dtype=bool),
            )

        # Stage 1: Semi-Log Filter (Pre-filter)
        # We keep this as an initial cleanup step (esp for wildly wrong points)
        _, _, semilog_mask_inliers = Fitter._remove_outliers_semilog(
            delays, amplitudes, threshold=3.0
        )
        # semilog_mask_inliers is True for GOOD points

        # Ensure we have enough points
        if np.sum(semilog_mask_inliers) < 3:
            semilog_mask_inliers = np.ones_like(delays, dtype=bool)

        # Fit on Pre-filtered data
        try:
            delays_s1 = delays[semilog_mask_inliers]
            amps_s1 = amplitudes[semilog_mask_inliers]

            # Use soft_l1 for initial fit
            try:
                popt, pcov = curve_fit(
                    t2_decay_model, delays_s1, amps_s1, p0=p0, loss="soft_l1"
                )
            except TypeError:
                popt, pcov = curve_fit(t2_decay_model, delays_s1, amps_s1, p0=p0)

        except (RuntimeError, ValueError):
            # Fallback to fit on everything
            try:
                popt, pcov = curve_fit(t2_decay_model, delays, amplitudes, p0=p0)
            except (RuntimeError, ValueError):
                return (
                    {},
                    np.zeros_like(delays),
                    np.zeros_like(delays),
                    0.0,
                    {},
                    np.zeros_like(delays, dtype=bool),
                )

        # Stage 2: Post-Fit Outlier Detection (Refinement)
        # Detect outliers against the curve from Stage 1
        outlier_mask = Fitter._detect_outliers_post_fit(
            delays, amplitudes, t2_decay_model, popt, threshold=4.0
        )

        # Stage 3: Refit with clean data (if meaningful change)
        if np.any(outlier_mask) and np.sum(~outlier_mask) > 3:
            try:
                delays_clean = delays[~outlier_mask]
                amps_clean = amplitudes[~outlier_mask]

                popt_clean, pcov_clean = curve_fit(
                    t2_decay_model, delays_clean, amps_clean, p0=popt, maxfev=10000
                )
                popt = popt_clean
                pcov = pcov_clean
            except (RuntimeError, ValueError):
                pass

        M0, T2, offset = popt

        # Fit curve on ALL points
        fit_curve = t2_decay_model(delays, *popt)
        residuals = amplitudes - fit_curve

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        perr = np.sqrt(np.diag(pcov))
        param_errors = {"M0": perr[0], "T2": perr[1], "offset": perr[2]}

        return (
            {"M0": M0, "T2": T2, "offset": offset},
            fit_curve,
            residuals,
            r2,
            param_errors,
            outlier_mask,  # True for OUTLIERS
        )

    @staticmethod
    def fit_modulated_t2(
        delays: np.ndarray, amplitudes: np.ndarray, guess_J: float = 7.0
    ) -> Tuple[dict, np.ndarray, np.ndarray, float, dict, np.ndarray]:
        """
        Fit J-Modulated T2 Spin Echo decay using a two-stage approach with modulation depth.

        Stage 1: Fit simple exponential decay to get T2/M0 estimates.
        Stage 2: Use Stage 1 results as initial guesses for full J-modulated fit with depth.

        Model: | M0 * exp(-t/T2) * ((1-depth) + depth * cos(pi*J*t)) | + offset
        Returns: params, fit_curve, residuals, r_squared, param_errors, outlier_mask
        """
        from nmr_analysis.analysis.models import j_modulated_t2

        # ===== Stage 1: Fit Exponential Decay Envelope =====
        # This gives robust T2 and M0 estimates ignoring modulation
        M0_guess = np.max(amplitudes) if len(amplitudes) > 0 else 1.0
        T2_guess = np.mean(delays) if len(delays) > 0 else 0.5
        offset_guess = np.min(amplitudes) if len(amplitudes) > 0 else 0.0

        # Remove outliers for Stage 1 (Initial T2 estimate)
        # We also capture the mask to calculate noise level on "envelope" points.
        delays_stage1, amplitudes_stage1, mask_stage1 = Fitter._remove_outliers_semilog(
            delays, amplitudes
        )

        # This mask will track points used in Stage 2 fit (True = used, False = outlier)
        # Initialize with the mask from Stage 1, as these are already considered "good" for the envelope.
        final_inlier_mask = mask_stage1.copy()

        if len(delays_stage1) < 3:
            delays_stage1, amplitudes_stage1 = delays, amplitudes
            mask_stage1 = np.ones_like(delays, dtype=bool)
            final_inlier_mask = np.ones_like(
                delays, dtype=bool
            )  # Reset if Stage 1 failed to filter

        try:
            popt_stage1, _ = curve_fit(
                t2_decay_model,
                delays_stage1,
                amplitudes_stage1,
                p0=[M0_guess, T2_guess, offset_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=5000,
            )
            M0_stage1, T2_stage1, offset_stage1 = popt_stage1
        except (RuntimeError, ValueError):
            # Fallback to initial guesses if Stage 1 fails
            M0_stage1, T2_stage1, offset_stage1 = M0_guess, T2_guess, offset_guess
            # If stage 1 failed, we can't reliably filter stage 2, so keep all
            popt_stage1 = None
            final_inlier_mask = np.ones_like(
                delays, dtype=bool
            )  # If Stage 1 fit failed, assume all are inliers for now

        # ===== Stage 2: Fit Full J-Modulated Model with Depth =====
        # Prepare data for Stage 2: Remove high outliers (spikes) but KEEP troughs.
        # Use Stage 1 fit as reference envelope.

        # Initialize delays_final and amplitudes_final with the full data,
        # and then apply further filtering if Stage 1 was successful.
        delays_final, amplitudes_final = delays, amplitudes

        if popt_stage1 is not None and np.sum(mask_stage1) > 2:
            # Calculate Envelope
            envelope = t2_decay_model(delays, M0_stage1, T2_stage1, offset_stage1)

            # Calculate noise sigma from the "inliers" identified in Stage 1
            # (These are points close to the envelope)
            # We calculate residuals of 'inliers' vs the fitted envelope
            clean_residuals = amplitudes[mask_stage1] - envelope[mask_stage1]
            robust_sigma = 1.4826 * np.median(
                np.abs(clean_residuals - np.median(clean_residuals))
            )

            if robust_sigma == 0:
                robust_sigma = np.std(clean_residuals)

            if robust_sigma > 0:
                # Filter Positive Outliers (Spikes)
                # We do NOT filter negative residuals (troughs/modulation)
                # Threshold: 4.0 sigma is safe for spikes
                diff_full = amplitudes - envelope
                # Keep points where diff is NOT huge positive
                stage2_mask = diff_full < (4.0 * robust_sigma)

                # Combine Stage 1 and Stage 2 masks: A point must be an inlier in both stages
                combined_mask = final_inlier_mask & stage2_mask

                # Ensure we don't kill too much data
                if np.sum(combined_mask) > len(delays) // 2:
                    delays_final = delays[combined_mask]
                    amplitudes_final = amplitudes[combined_mask]
                    final_inlier_mask = combined_mask
                else:
                    # If too much data is removed by combined_mask, revert to using only Stage 1 mask
                    # or even full data if Stage 1 mask was too aggressive.
                    # For simplicity, if combined is too aggressive, we just use the Stage 1 mask.
                    # If Stage 1 mask was already too aggressive (len(delays_stage1) < 3),
                    # final_inlier_mask would have been reset to all True.
                    delays_final = delays[final_inlier_mask]
                    amplitudes_final = amplitudes[final_inlier_mask]
            # else: robust_sigma is 0, so no further filtering based on envelope, final_inlier_mask remains as is from Stage 1
        # else: popt_stage1 is None or mask_stage1 has too few points, final_inlier_mask remains all True

        # Use Stage 1 results as initial guesses
        # p0: [M0, T2, J, offset, depth]
        p0 = [M0_stage1, T2_stage1, guess_J, offset_stage1, 0.9]

        try:
            # Bounds: M0>0, T2>0, J>0, offset any, 0<=depth<=1
            bounds_min = [0, 0, 0, -np.inf, 0.0]
            bounds_max = [np.inf, np.inf, 20.0, np.inf, 1.0]

            popt, pcov = curve_fit(
                j_modulated_t2,
                delays_final,
                amplitudes_final,
                p0=p0,
                bounds=(bounds_min, bounds_max),
                maxfev=10000,
            )
            M0, T2, J, offset, depth = popt
            fit_curve = j_modulated_t2(delays, M0, T2, J, offset, depth)
            residuals = amplitudes - fit_curve

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            perr = np.sqrt(np.diag(pcov))
            param_errors = {
                "M0": perr[0],
                "T2": perr[1],
                "J": perr[2],
                "offset": perr[3],
                "depth": perr[4],
            }

            return (
                {"M0": M0, "T2": T2, "J": J, "offset": offset, "depth": depth},
                fit_curve,
                residuals,
                r2,
                param_errors,
                ~final_inlier_mask,  # Outlier mask
            )
        except (RuntimeError, ValueError) as e:
            return (
                {},
                np.zeros_like(delays),
                np.zeros_like(delays),
                0.0,
                {"error": str(e)},
                np.zeros_like(delays, dtype=bool),
            )

    @staticmethod
    def fit_t2_star(
        data: NMRData,
        smoothing: float = 1.0,
        start_trim_percent: float = 0.05,
        end_trim_buffer_percent: float = 0.05,
    ) -> AnalysisResult:
        """
        Fit T2* from a single FID trace.

        Args:
            data: NMRData object
            smoothing: Sigma for gaussian smoothing
            start_trim_percent: Fraction of detected DECAY LENGTH to skip after peak.
            end_trim_buffer_percent: Fraction of detected DECAY LENGTH to buffer before noise.
        """
        from scipy.ndimage import gaussian_filter1d

        time = data.time
        signal = data.signal
        raw_magnitude = np.abs(signal)

        # Apply smoothing for peak finding / trimming logic (not necessarily for final fit)
        if smoothing > 0:
            detection_signal = gaussian_filter1d(raw_magnitude, sigma=smoothing)
        else:
            detection_signal = raw_magnitude

        if len(detection_signal) == 0:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Empty Analysis",
                params={},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
            )

        n_samples = len(raw_magnitude)

        # 1. Find Peak Index (Max of Smoothed Magnitude)
        peak_idx = np.argmax(detection_signal)

        # 2. Determine Decay End (Smart Noise Floor Detection)
        # Step A: Estimate Noise Floor from the last 15% of data
        noise_window_start = int(n_samples * 0.85)
        # Assuming peak is somewhat earlier than the end
        if noise_window_start <= peak_idx:
            # Fallback: very short signal or peak is late
            noise_window_start = min(
                peak_idx + (n_samples - peak_idx) // 2, n_samples - 2
            )

        noise_segment = detection_signal[noise_window_start:]
        if len(noise_segment) > 1:
            noise_mean = np.mean(noise_segment)
            noise_std = np.std(noise_segment)
            # Threshold: Mean + 3*Sigma (Standard 99.7% limit)
            noise_threshold = noise_mean + 3 * noise_std
        else:
            noise_threshold = 0.0

        # Step B: Find where signal First drops below/near noise threshold AFTER peak
        decay_stop_idx = n_samples  # Default to end

        # Search slice from Peak til End
        search_slice = detection_signal[peak_idx:]
        below_noise = search_slice < noise_threshold

        if np.any(below_noise):
            # Found a point below noise
            first_noise_idx = np.argmax(below_noise)  # absolute index within slice
            decay_stop_idx = peak_idx + first_noise_idx

        # 3. Calculate Effective Decay Length
        decay_len = decay_stop_idx - peak_idx
        if decay_len <= 0:
            decay_len = n_samples - peak_idx  # Fallback

        # 4. Calculate Trims based on DECAY LENGTH
        start_trim_points = int(decay_len * start_trim_percent)
        end_buffer_points = int(decay_len * end_trim_buffer_percent)

        global_start_fit_idx = peak_idx + start_trim_points
        global_end_fit_idx = decay_stop_idx - end_buffer_points

        # SAFETY CHECKS

        # 1. Ensure Start < End
        if global_start_fit_idx >= global_end_fit_idx:
            # Buffer might be too aggressive or signal too short.
            # Relax the buffer
            global_end_fit_idx = decay_stop_idx

        # 2. Ensure Start < End (Again)
        if global_start_fit_idx >= global_end_fit_idx:
            # If still invalid, try simply fitting from start to end of signal (ignoring noise logic)
            global_end_fit_idx = n_samples

        # 3. Bounds
        global_start_fit_idx = max(0, min(global_start_fit_idx, n_samples - 1))
        global_end_fit_idx = max(0, min(global_end_fit_idx, n_samples))

        # 4. Final Minimum Points Check
        if global_end_fit_idx - global_start_fit_idx < 5:
            # Force at least some points if possible
            if n_samples > peak_idx + 5:
                global_start_fit_idx = peak_idx
                global_end_fit_idx = min(decay_stop_idx, n_samples)
            else:
                # Signal too short
                global_start_fit_idx = 0
                global_end_fit_idx = n_samples

        # Slice for fitting
        t_fit = time[global_start_fit_idx:global_end_fit_idx]
        # Use Smoothed Data for fit as established
        mag_fit = detection_signal[global_start_fit_idx:global_end_fit_idx]

        # Check for insufficient data points (Need at least 3 for M0, T2, offset)
        if len(t_fit) < 3:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="T2* Analysis (Insufficient Data)",
                params={},
                fit_curve=np.full_like(time, np.nan),
                residuals=np.zeros_like(raw_magnitude),
                r_squared=0.0,
                metadata={
                    "source": "smoothed_fit_v2",
                    "error": f"Insufficient data points for fitting: {len(t_fit)} < 3",
                    "start_index": global_start_fit_idx,
                    "end_index": global_end_fit_idx,
                },
            )

        # Initial guess
        M0_guess = np.max(mag_fit) if len(mag_fit) > 0 else 1.0
        T2_guess = (t_fit[-1] - t_fit[0]) / 3.0 if len(t_fit) > 1 else 1e-3
        p0 = [M0_guess, T2_guess, 0.0]

        try:
            popt, pcov = curve_fit(t2_decay_model, t_fit, mag_fit, p0=p0)
            M0, T2_star, offset = popt

            # Calculate full fit curve (padded with NaN)
            full_fit_curve = np.full_like(time, np.nan)

            # Project onto fitted region
            full_fit_curve[global_start_fit_idx:global_end_fit_idx] = t2_decay_model(
                t_fit, *popt
            )

            # Residuals
            full_residuals = np.full_like(time, np.nan)
            full_residuals[global_start_fit_idx:global_end_fit_idx] = (
                mag_fit - t2_decay_model(t_fit, *popt)
            )

            ss_res = np.sum(
                full_residuals[global_start_fit_idx:global_end_fit_idx] ** 2
            )
            ss_tot = np.sum((mag_fit - np.mean(mag_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Calculate parameter errors (std dev)
            perr = np.sqrt(np.diag(pcov))
            # Parameters are: M0, T2_star, offset
            param_errors = {
                "M0": perr[0],
                "T2_star": perr[1],
                "offset": perr[2],
            }

            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="T2* Analysis",
                params={"M0": M0, "T2_star": T2_star, "offset": offset},
                fit_curve=full_fit_curve,
                residuals=full_residuals,
                r_squared=r2,
                param_errors=param_errors,
                metadata={
                    "source": "smoothed_fit_v2",
                    "start_index": global_start_fit_idx,
                    "end_index": global_end_fit_idx,
                    "smoothing": smoothing,
                    "peak_index": peak_idx,
                    "trim_params": {
                        "start_trim_percent": start_trim_percent,
                        "end_trim_buffer_percent": end_trim_buffer_percent,
                        "decay_stop_idx": decay_stop_idx,
                        "noise_threshold": float(noise_threshold),
                        "decay_len": decay_len,
                    },
                },
            )
        except RuntimeError:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="T2* Analysis (Fit Failed)",
                params={},
                fit_curve=np.full_like(time, np.nan),
                residuals=np.zeros_like(raw_magnitude),
                r_squared=0.0,
                metadata={
                    "source": "smoothed_fit",
                    "error": "Fit Failed",
                },
            )

    @staticmethod
    def fit_diffusion(
        tau_values: np.ndarray,
        rates_r2: np.ndarray,
        gradient_strength: float,
        gyromagnetic_ratio: float = 2.675e8,  # Proton gamma (rad/s/T)
        fixed_intercept: Optional[float] = None,
    ) -> AnalysisResult:
        """
        Fit Diffusion Coefficient D from R2 vs tau^2.

        Formula:
        R2_obs = R2_intrinsic + (1/3) * D * gamma^2 * G^2 * tau^2

        Let y = R2_obs
        Let x = tau^2
        slope = (1/3) * D * gamma^2 * G^2
        intercept = R2_intrinsic

        If fixed_intercept (R2_intrinsic) is provided:
            y - R2_intrinsic = slope * x
            Fit linear model through origin.
        """
        # Linear fit: y = mx + c
        x_linear = tau_values**2
        y_linear = rates_r2

        slope = 0.0
        intercept = 0.0

        try:
            if fixed_intercept is not None:
                # Constrained fit: y = mx + c_fixed => y - c_fixed = mx
                y_adj = y_linear - fixed_intercept

                # Fit through origin: slope = sum(x*y) / sum(x^2)
                # Check for zero denominator
                sum_sq_x = np.sum(x_linear**2)
                if sum_sq_x == 0:
                    slope = 0.0
                else:
                    slope = np.sum(x_linear * y_adj) / sum_sq_x

                intercept = fixed_intercept
                predicted_y = slope * x_linear + intercept
            else:
                # Unconstrained Polyfit degree 1: returns [slope, intercept]
                slope, intercept = np.polyfit(x_linear, y_linear, 1)
                predicted_y = slope * x_linear + intercept

            # Calculate D
            gamma = gyromagnetic_ratio
            G = gradient_strength

            # Avoid divide by zero
            if G == 0:
                D = 0.0
            else:
                D = (3 * slope) / ((gamma**2) * (G**2))

            # Calculate R2 (Coefficient of Determination) for the linear fit
            ss_res = np.sum((y_linear - predicted_y) ** 2)
            ss_tot = np.sum((y_linear - np.mean(y_linear)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Estimate error in slope to get error in D
            n = len(x_linear)
            dof = n - 1 if fixed_intercept is not None else n - 2

            if dof > 0:
                # Standard error of slope
                if fixed_intercept is not None:
                    # For regression through origin:
                    # Var(b) = sigma^2 / sum(x^2)
                    sigma2 = ss_res / dof
                    sx2 = np.sum(x_linear**2)
                    std_err_slope = np.sqrt(sigma2 / sx2) if sx2 > 0 else 0.0
                else:
                    sx2 = np.sum((x_linear - np.mean(x_linear)) ** 2)
                    sy_x2 = ss_res / dof  # variance of residuals
                    std_err_slope = np.sqrt(sy_x2 / sx2) if sx2 > 0 else 0.0

                # Propagate error to D
                k = 3 / ((gamma**2) * (G**2)) if G != 0 else 0
                std_err_D = k * std_err_slope
            else:
                std_err_D = 0.0

            fit_curve = predicted_y  # This is R2_fit, not time domain

            return AnalysisResult(
                experiment_type=ExperimentType.DIFFUSION,
                dataset_name="Diffusion Analysis",
                params={
                    "D": D,
                    "R2_intrinsic": intercept,
                    "T2_intrinsic": 1.0 / intercept if intercept > 0 else 0.0,
                    "slope": slope,
                },
                fit_curve=fit_curve,
                residuals=y_linear - predicted_y,
                r_squared=r2,
                param_errors={"D": std_err_D},
                metadata={
                    "gradient_strength": gradient_strength,
                    "gamma": gyromagnetic_ratio,
                    "x_values": x_linear,  # tau^2
                    "y_values": y_linear,  # R2
                    "fixed_intercept": fixed_intercept,
                },
            )

        except Exception as e:
            return AnalysisResult(
                experiment_type=ExperimentType.DIFFUSION,
                dataset_name="Diffusion Analysis (Fit Failed)",
                params={},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
                metadata={"error": str(e)},
            )

    @staticmethod
    def fit_spectrum(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        method: str = "magnitude_lorentzian",
        min_prominence: Optional[float] = None,
    ) -> AnalysisResult:
        """
        Fit the frequency spectrum with multiple Lorentzians.
        1. Find peaks in magnitude spectrum.
        2. Fit sum of Magnitude Lorentzians (if spectrum is magnitude).
        3. Extract T2* from gamma. T2* = 1 / (2 * pi * gamma).
        """
        from scipy.signal import find_peaks
        from nmr_analysis.analysis.models import multi_magnitude_lorentzian

        mag_spec = np.abs(spectrum)

        # 1. PEAK FINDING
        if min_prominence is None:
            min_prominence = 0.05 * np.max(mag_spec)

        peak_indices, properties = find_peaks(
            mag_spec, prominence=min_prominence, distance=5
        )

        if len(peak_indices) == 0:
            peak_indices = [np.argmax(mag_spec)]

        n_peaks = len(peak_indices)

        # 2. INITIAL GUESS
        current_offset = np.median(mag_spec)
        p0 = [current_offset]

        # Bounds: [offset, A1, f0_1, gamma1, ...]
        bounds_min = [-np.inf]
        bounds_max = [np.inf]

        # Sort peaks by amplitude
        sorted_indices = sorted(peak_indices, key=lambda i: mag_spec[i], reverse=True)
        if len(sorted_indices) > 5:
            sorted_indices = sorted_indices[:5]
            n_peaks = 5

        # Sort by frequency
        sorted_indices = sorted(sorted_indices)

        for idx in sorted_indices:
            amp = mag_spec[idx]
            f0 = freqs[idx]
            df = freqs[1] - freqs[0]
            gamma_guess = 5.0 * df  # initial width guess

            # Initial Guess for A
            # Model: MagnitudeLorentzian Height = A / gamma + offset
            # => A = (Height - offset) * gamma
            height = amp - current_offset
            A_guess = max(height * gamma_guess, 0.0)

            p0.extend([A_guess, f0, gamma_guess])

            bounds_min.extend([0, freqs[0], 0])
            bounds_max.extend([np.inf, freqs[-1], np.inf])

        # 3. FITTING
        try:

            def fit_func(f, *params):
                return multi_magnitude_lorentzian(f, np.array(params), n_peaks)

            popt, pcov = curve_fit(
                fit_func, freqs, mag_spec, p0=p0, bounds=(bounds_min, bounds_max)
            )

            # 4. EXTRACT RESULTS
            offset_fit = popt[0]
            peaks_found = []
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)

            fit_curve = fit_func(freqs, *popt)
            residuals = mag_spec - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mag_spec - np.mean(mag_spec)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            for i in range(n_peaks):
                base_idx = 1 + i * 3
                A = popt[base_idx]
                f0 = popt[base_idx + 1]
                gamma = popt[base_idx + 2]

                # T2* conversion
                # Gamma is parameter in sqrt((f-f0)^2 + gamma^2)
                # It corresponds to 1 / (2*pi*T2)
                t2_star = 0.0
                if gamma > 1e-9:
                    t2_star = 1.0 / (2.0 * np.pi * gamma)

                peaks_found.append(
                    {
                        "amplitude": A,
                        "f0": f0,
                        "gamma": gamma,
                        "t2_star": t2_star,
                        "amplitude_error": perr[base_idx],
                        "f0_error": perr[base_idx + 1],
                        "gamma_error": perr[base_idx + 2],
                    }
                )

            # Sort peaks by f0 for consistent reporting
            peaks_found.sort(key=lambda x: x["f0"])

            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Spectral Analysis",
                params={"offset": offset_fit, "peaks": peaks_found, "n_peaks": n_peaks},
                fit_curve=fit_curve,
                residuals=residuals,
                r_squared=r2,
                param_errors={},
                metadata={"freqs": freqs, "spectrum_magnitude": mag_spec},
            )

        except Exception as e:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Spectral Analysis (Fit Failed)",
                params={"peaks": [], "error": str(e)},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
                metadata={},
            )

    @staticmethod
    def fit_multiplet_spectrum(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        multiplets_config: list,
    ) -> AnalysisResult:
        """
        Fit spectrum with known multiplets.
        multiplets_config: List of dicts:
            [
                {'center': 100.0, 'multiplicity': 3, 'initial_J': 7.0, 'initial_gamma': 5.0},
                ...
            ]
        """
        from nmr_analysis.analysis.models import multi_multiplet_lorentzian

        mag_spec = np.abs(spectrum)

        # 1. Build Initial Guess and Bounds
        # Params: [offset,  (A, center, J, gamma)...]
        offset_guess = np.median(mag_spec)
        p0 = [offset_guess]
        bounds_min = [-np.inf]
        bounds_max = [np.inf]

        multiplicities = []
        n_multiplets = len(multiplets_config)

        for m_conf in multiplets_config:
            center_guess = m_conf.get("center", 0.0)
            mult = m_conf.get("multiplicity", 1)
            J_guess = m_conf.get("initial_J", 7.0)
            gamma_guess = m_conf.get("initial_gamma", 5.0)  # ~10Hz width

            # Estimate Amplitude
            # Multiplet peak height depends on multiplicity.
            # Simplified: take max value near center as rough guess for "A" scale
            idx = np.argmin(np.abs(freqs - center_guess))
            amp_guess = max(mag_spec[idx] - offset_guess, 0.0) * gamma_guess

            p0.extend([amp_guess, center_guess, J_guess, gamma_guess])

            # Bounds
            # A > 0
            # Center +/- 100 Hz? Let's say unbounded or slightly constrained? Unbounded is risky.
            # center +/- 50Hz
            # J: 0.1 to 50 Hz
            # gamma: 0.1 to 100 Hz
            bounds_min.extend([0, center_guess - 50, 0.1, 0.1])
            bounds_max.extend([np.inf, center_guess + 50, 50.0, 100.0])

            multiplicities.append(mult)

        multiplicities_arr = np.array(multiplicities, dtype=np.int32)

        # 2. Fit Function Wrapper
        def fit_func(f, *params):
            return multi_multiplet_lorentzian(
                f, np.array(params), multiplicities_arr, n_multiplets
            )

        try:
            popt, pcov = curve_fit(
                fit_func, freqs, mag_spec, p0=p0, bounds=(bounds_min, bounds_max)
            )

            # 3. Extract Results
            offset_fit = popt[0]
            fit_curve = fit_func(freqs, *popt)
            residuals = mag_spec - fit_curve

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mag_spec - np.mean(mag_spec)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            multiplet_results = []
            perr = np.sqrt(np.diag(pcov))

            current_idx = 1
            for i in range(n_multiplets):
                res = {
                    "multiplicity": int(multiplicities_arr[i]),
                    "amplitude": popt[current_idx],
                    "center": popt[current_idx + 1],
                    "J": popt[current_idx + 2],
                    "gamma": popt[current_idx + 3],
                    "amplitude_err": perr[current_idx],
                    "center_err": perr[current_idx + 1],
                    "J_err": perr[current_idx + 2],
                    "gamma_err": perr[current_idx + 3],
                }
                multiplet_results.append(res)
                current_idx += 4

            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Multiplet Analysis",
                params={"offset": offset_fit, "multiplets": multiplet_results},
                fit_curve=fit_curve,
                residuals=residuals,
                r_squared=r2,
                metadata={"freqs": freqs, "spectrum_magnitude": mag_spec},
            )

        except Exception as e:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Multiplet Analysis (Failed)",
                params={"error": str(e)},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
            )
