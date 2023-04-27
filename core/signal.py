
from collections.abc import Iterable

import numpy as np

from .alias import meter, kelvin, celsius, Array
from .config import SPECTRAL_RANGE, SPECTRAL_STEP
from .detector import Detector
from .filter import Filter
from .radiation import RadiationDensity
from .utils import calculate_response, celsius2kelvin


def calculate_signal(x: Array[meter], t: kelvin, response: Array[float]) -> float:
    '''calculate a detector's signal, in A/m^2'''
    dx = 1e-9*SPECTRAL_STEP

    return dx*np.nansum(RadiationDensity.calculate(x, t) * response)


class DetectorSignal:

    @classmethod
    def calculate(cls, temperature: celsius | Array[celsius], filter: Filter, detector: Detector) -> float | Array[float]:
        '''Interface to calculate a signal at the given temperatures, filter and detector's kind'''
        lb, ub = SPECTRAL_RANGE
        x = 1e-9*np.arange(lb, ub+SPECTRAL_STEP, SPECTRAL_STEP)

        response = calculate_response(x, filter=filter, detector=detector)

        if isinstance(temperature, Iterable):
            return np.array([
                calculate_signal(x, t=celsius2kelvin(t), response=response)
                for t in temperature
            ])
        t = temperature
        return calculate_signal(x, t=celsius2kelvin(t), response=response)
