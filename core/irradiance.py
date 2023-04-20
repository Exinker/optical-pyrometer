
from scipy.integrate import quad

import numpy as np

from .alias import nm, kelvin, celsius, Array
from .radiation import RadiationDensity
from .utils import celsius2kelvin


class Irradiance:

    @staticmethod
    def _calculate(t: kelvin, sensitivity: tuple[nm, nm]) -> float:
        '''calculate irradiance at the given temperature and sensitivity range'''
        lb, ub = sensitivity

        return quad(
            lambda x: RadiationDensity.calculate(x, t),
            a=lb*1e-9,
            b=ub*1e-9,
        )[0]

    @classmethod
    def calculate(cls, x: Array[celsius], sensitivity: tuple[nm, nm]) -> Array[float]:
        '''calculate irradiance at the given sensitivity range'''

        y = np.array([
            cls._calculate(t, sensitivity)
            for t in celsius2kelvin(x)
        ])

        return y
