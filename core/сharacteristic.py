
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from .alias import Array, meter, micro


class Characteristic(ABC):
    '''Interface for any characteristics'''

    @abstractmethod
    def __call__(self, x: Array[meter], fill_value: float = np.nan) -> Array[float]:
        raise NotImplementedError


@dataclass
class ConstantCharacteristic(Characteristic):
    value: float

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:

        if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
            return self.value
        if isinstance(x, Array):
            return np.full(x.shape, self.value)

        raise ValueError


@dataclass
class WindowCharacteristic(Characteristic):
    span: tuple[float, float]

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:
        
        
        if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
            return self.value
        if isinstance(x, Array):
            return np.full(x.shape, self.value)

        raise ValueError


@dataclass
class DatasheetCharacteristic(Characteristic):
    path: str
    norm: float = field(default=1)
    _x: Array[micro] = field(init=False, repr=False)
    _y: Array[float] = field(init=False, repr=False)

    def __post_init__(self):
        dat = np.genfromtxt(self.path, delimiter=',', dtype=np.float32)

        self._x = dat[:, 0]
        self._y = dat[:, 1]

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:
        return interp1d(
            1e-6*self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x) / self.norm

