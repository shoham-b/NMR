from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

import numpy as np


class ExperimentType(str, Enum):
    T1 = "t1"
    T2 = "t2"
    T2_STAR = "t2_star"


@dataclass
class NMRData:
    """Raw data from an NMR experiment."""

    time: np.ndarray
    signal: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_type: Optional[ExperimentType] = None

    @property
    def sample_rate(self) -> float:
        if len(self.time) > 1:
            return 1.0 / (self.time[1] - self.time[0])
        return 0.0


@dataclass
class AnalysisResult:
    """Result of an NMR analysis."""

    experiment_type: ExperimentType
    dataset_name: str
    params: Dict[str, float]
    fit_curve: np.ndarray
    residuals: np.ndarray
    r_squared: float
    metadata: Dict[str, Any] = field(default_factory=dict)
