import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional
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

        if len(detection_signal) == 0:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="Empty Analysis",
                params={},
                fit_curve=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
            )

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
