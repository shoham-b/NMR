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
        """
        time = data.time
        signal = data.signal

        # We can fit the magnitude to a simple decay
        # Or fit the real data to the oscillating model.
        # Fitting magnitude is more robust for T2*.
        magnitude = np.abs(signal)

        M0_guess = np.max(magnitude)
        T2_guess = (time[-1] - time[0]) / 3.0
        p0 = [M0_guess, T2_guess, 0.0]

        popt, pcov = curve_fit(t2_decay_model, time, magnitude, p0=p0)
        M0, T2_star, offset = popt

        fit_curve = t2_decay_model(time, *popt)
        residuals = magnitude - fit_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((magnitude - np.mean(magnitude)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return AnalysisResult(
            experiment_type=ExperimentType.T2_STAR,
            dataset_name="FID Analysis",
            params={"M0": M0, "T2_star": T2_star, "offset": offset},
            fit_curve=fit_curve,
            residuals=residuals,
            r_squared=r2,
            metadata={"source": "magnitude_fit"},
        )
