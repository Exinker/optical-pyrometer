
from collections.abc import Sequence
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .alias import Array, kelvin, celsius, nano
from .detector import Detector
from .filter import Filter
from .radiation import RadiationDensity
from .utils import celsius2kelvin


class DetectorSignal:

    @staticmethod
    def _calculate(t: kelvin, characteristic: Callable, span: tuple[nano, nano]) -> float:
        '''calculate a signal at the given temperature and detector's characteristic'''
        lb, ub = span

        return quad(
            lambda x: RadiationDensity.calculate(x, t) * characteristic(x),
            a=1e-9*lb,
            b=1e-9*ub,
        )[0]

    @classmethod
    def calculate(cls, temperature: Sequence[celsius], filter: Filter, detector: Detector) -> Array[float]:
        '''Interface to calculate a signal at the given temperatures and detector's kind'''
        config = detector.config

        x = 1e-9 * np.linspace(*filter.span, 1000)
        characteristic = interp1d(
            x, config.sensitivity(x, fill_value=0) * config.transmittance(x, fill_value=0),
            kind='linear', bounds_error=False, fill_value=0,
        )

        y = np.array([
            cls._calculate(celsius2kelvin(t), characteristic, filter.span)
            for t in temperature
        ])

        return y
