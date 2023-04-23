
from collections.abc import Sequence

import numpy as np

from .alias import meter, kelvin, celsius, Array
from .config import SPECTRAL_RANGE, SPECTRAL_STEP
from .detector import Detector
from .filter import Filter
from .radiation import RadiationDensity
from .utils import calculate_response, celsius2kelvin


class DetectorSignal:

    @staticmethod
    def _calculate(x: Array[meter], response: Array[float], t: kelvin) -> float:
        '''calculate a signal at the given temperature and response'''
        dx = 1e-9*SPECTRAL_STEP

        return dx*np.nansum(RadiationDensity.calculate(x, t) * response)

    @classmethod
    def calculate(cls, temperature: Sequence[celsius], filter: Filter, detector: Detector) -> Array[float]:
        '''Interface to calculate a signal at the given temperatures, filter and detector's kind'''
        lb, ub = SPECTRAL_RANGE
        x = 1e-9*np.arange(lb, ub+SPECTRAL_STEP, SPECTRAL_STEP)

        response = calculate_response(x, filter=filter, detector=detector)

        y = np.array([
            cls._calculate(x, response, t=celsius2kelvin(t))
            for t in temperature
        ])

        return y
