
import numpy as np

from .alias import celsius, kelvin, Array
from .detector import Detector
from .filter import Filter


def celsius2kelvin(t: celsius) -> kelvin:
    return t + 273.15


def calculate_response(x: Array, filter: Filter | None = None, detector: Detector | None = None, fill_value: float = np.nan) -> Array:
    y = np.ones(x.shape)

    if filter is not None:
        y *= filter(x, fill_value)

    if detector is not None:
        y *= detector.responce(x, fill_value)

    return y
