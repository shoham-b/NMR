import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple
from nmr_analysis.core.types import NMRData, AnalysisResult, ExperimentType
from nmr_analysis.analysis.models import t1_model, t2_decay_model


class Fitter:
    @staticmethod
    def fit_t1(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float]:
        """
        Fit T1 Inversion Recovery data.
        Returns: params, fit_curve, residuals, r_squared
        """
        # Initial guess
        # M0 is approx max amplitude
        # T1 is approx time to 1/e or zero crossing?
        # Inversion recovery: starts negative? Or we taking absolute magnitude?
        # If magnitude, model is |M0 (1 - 2 exp(-t/T1))|.
        # For simplicity assuming signed data or user passes signed amplitudes.
        # If amplitudes are absolute, we need a different model or robust guess.

        M0_guess = np.max(np.abs(amplitudes))
        T1_guess = np.mean(delays)  # Very rough
        p0 = [M0_guess, T1_guess, 1.0]

        try:
            popt, pcov = curve_fit(t1_model, delays, amplitudes, p0=p0)
            M0, T1, alpha = popt
            fit_curve = t1_model(delays, *popt)
            residuals = amplitudes - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            return {"M0": M0, "T1": T1, "alpha": alpha}, fit_curve, residuals, r2
        except RuntimeError:
            return {}, np.zeros_like(delays), np.zeros_like(delays), 0.0

    @staticmethod
    def fit_t2(
        delays: np.ndarray, amplitudes: np.ndarray
    ) -> Tuple[dict, np.ndarray, np.ndarray, float]:
        """
        Fit T2 Spin Echo decay.
        """
        M0_guess = np.max(amplitudes)
        T2_guess = np.mean(delays)
        p0 = [M0_guess, T2_guess, 0.0]

        try:
            popt, pcov = curve_fit(t2_decay_model, delays, amplitudes, p0=p0)
            M0, T2, offset = popt
            fit_curve = t2_decay_model(delays, *popt)
            residuals = amplitudes - fit_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            return {"M0": M0, "T2": T2, "offset": offset}, fit_curve, residuals, r2
        except RuntimeError:
            return {}, np.zeros_like(delays), np.zeros_like(delays), 0.0

    @staticmethod
    def fit_t2_star(data: NMRData) -> AnalysisResult:
        """
        Fit T2* from a single FID trace.
        Starts fitting from the first peak > 5.0 if available.
        """
        from scipy.signal import find_peaks

        time = data.time
        signal = data.signal
        magnitude = np.abs(signal)

        # Find start index: first peak > 5.0
        # If no peak > 5.0 found, use max as fallback
        peaks, _ = find_peaks(magnitude, height=5.0)

        if len(peaks) > 0:
            start_idx = peaks[0]
        else:
            start_idx = np.argmax(magnitude)

        # Slice data for fitting
        t_fit = time[start_idx:]
        mag_fit = magnitude[start_idx:]

        M0_guess = np.max(mag_fit)
        T2_guess = (t_fit[-1] - t_fit[0]) / 3.0 if len(t_fit) > 1 else 1e-3
        p0 = [M0_guess, T2_guess, 0.0]

        try:
            popt, pcov = curve_fit(t2_decay_model, t_fit, mag_fit, p0=p0)
            M0, T2_star, offset = popt

            # Calculate full fit curve (padded with NaN before start_idx)
            full_fit_curve = np.full_like(time, np.nan)
            full_fit_curve[start_idx:] = t2_decay_model(t_fit, *popt)

            # Residuals only for the fitted portion (padding rest with 0 or NaN?)
            # AnalysisResult expects residuals matching data shape mostly.
            # Let's use 0 for non-fitted part residuals? Or NaN?
            # Existing code used 'magnitude - fit_curve'. If fit_curve has NaNs, residuals will be NaN.
            # That's fine for Plotly.
            residuals = magnitude - full_fit_curve

            ss_res = np.nansum(residuals**2)
            ss_tot = np.nansum((magnitude - np.nanmean(magnitude)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="T2* Analysis",
                params={"M0": M0, "T2_star": T2_star, "offset": offset},
                fit_curve=full_fit_curve,
                residuals=residuals,
                r_squared=r2,
                metadata={"source": "magnitude_fit", "start_index": start_idx},
            )
        except RuntimeError:
            return AnalysisResult(
                experiment_type=ExperimentType.T2_STAR,
                dataset_name="T2* Analysis (Fit Failed)",
                params={},
                fit_curve=np.full_like(time, np.nan),
                residuals=np.zeros_like(magnitude),
                r_squared=0.0,
            )
