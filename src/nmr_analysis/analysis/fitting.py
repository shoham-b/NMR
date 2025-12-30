import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple
from nmr_analysis.core.types import NMRData, AnalysisResult, ExperimentType
from nmr_analysis.analysis.models import t1_model, t2_decay_model


class Fitter:
    @staticmethod
    def fit_t1(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float, dict]:
        """
        Fit T1 Inversion Recovery data.
        Returns: params, fit_curve, residuals, r_squared, param_errors
        """
        # Initial guess
        M0_guess = np.max(np.abs(amplitudes))
        T1_guess = np.mean(delays) if len(delays) > 0 else 1.0
        p0 = [M0_guess, T1_guess, 1.0]

        try:
            popt, pcov = curve_fit(t1_model, delays, amplitudes, p0=p0)
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
            )
        except (RuntimeError, ValueError):
            return {}, np.zeros_like(delays), np.zeros_like(delays), 0.0, {}

    @staticmethod
    def fit_t2(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float, dict]:
        """
        Fit T2 Spin Echo decay.
        Returns: params, fit_curve, residuals, r_squared, param_errors
        """
        M0_guess = np.max(amplitudes) if len(amplitudes) > 0 else 1.0
        T2_guess = np.mean(delays) if len(delays) > 0 else 1.0
        p0 = [M0_guess, T2_guess, 0.0]

        try:
            popt, pcov = curve_fit(t2_decay_model, delays, amplitudes, p0=p0)
            M0, T2, offset = popt
            fit_curve = t2_decay_model(delays, *popt)
            residuals = amplitudes - fit_curve

            # Simple R2
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
            )
        except (RuntimeError, ValueError):
            return {}, np.zeros_like(delays), np.zeros_like(delays), 0.0, {}

    @staticmethod
    def fit_t2_star(
        data: NMRData, smoothing: float = 1.0, trim_percent: float = 0.1
    ) -> AnalysisResult:
        """
        Fit T2* from a single FID trace.
        Trims the first and last `trim_percent` of data.
        Applies Gaussian smoothing to the magnitude.
        Starts fitting from the first peak > 5.0 (on smoothed data).
        """
        from scipy.ndimage import gaussian_filter1d

        time = data.time
        signal = data.signal
        raw_magnitude = np.abs(signal)

        # Apply smoothing first
        if smoothing > 0:
            detection_signal = gaussian_filter1d(raw_magnitude, sigma=smoothing)
        else:
            detection_signal = raw_magnitude

        # Find Peak Index (Max of Smoothed Magnitude)
        peak_idx = np.argmax(detection_signal)

        # Trimming Logic:
        # Start: Peak Index + 20% of the "tail" (data after peak)
        # End: Total Length - 10% of Total Length

        n_samples = len(raw_magnitude)
        tail_length = n_samples - peak_idx

        # 20% of tail
        # User updated requirement: "take the data only starting with the highest amplitude"
        # So start_trim_factor should be 0.0
        start_trim_factor = 0.0
        end_trim_factor = 0.1

        global_start_fit_idx = peak_idx + int(tail_length * start_trim_factor)
        global_end_fit_idx = n_samples - int(n_samples * end_trim_factor)

        # Ensure valid range
        if global_start_fit_idx >= global_end_fit_idx:
            # Fallback: use the peak as start, and no end trim
            global_start_fit_idx = peak_idx
            global_end_fit_idx = n_samples

        # Ensure indices are within bounds
        global_start_fit_idx = max(0, global_start_fit_idx)
        global_end_fit_idx = min(n_samples, global_end_fit_idx)

        # Re-check after bounds adjustment
        if global_start_fit_idx >= global_end_fit_idx:
            # If still invalid, try to get at least one point
            if n_samples > 0:
                global_start_fit_idx = peak_idx
                global_end_fit_idx = peak_idx + 1
            else:
                global_start_fit_idx = 0
                global_end_fit_idx = 0

        # Slice for fitting
        # Use smoothed data for fitting? User: "also use smoothing for the data"
        t_fit = time[global_start_fit_idx:global_end_fit_idx]
        mag_fit = detection_signal[global_start_fit_idx:global_end_fit_idx]

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
