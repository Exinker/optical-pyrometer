
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import overload

import numpy as np
from scipy import interpolate, signal

from .alias import Array, meter, micro, nano
from .config import SPECTRAL_RANGE, SPECTRAL_STEP


# --------        base        --------
@overload
def gauss(x: float, x0: float, w: float) -> float: ...
@overload
def gauss(x: Array, x0: float, w: float) -> Array: ...
def gauss(x, x0, w):
    '''
    Normal distribution with position x0 and unit intensity.

    Args:
        x0: position;
        w: width.
    '''

    F = np.exp( -(1/2)*((x - x0) / w)**2 ) / ( np.sqrt(2*np.pi) * w )

    return F


@overload
def rectangular(x: float, x0: float, w: float) -> Array: ...
@overload
def rectangular(x: Array, x0: float, w: float) -> Array: ...
def rectangular(x, x0, w):
    '''
    Rectangular distribution with position x0 and unit intensity.

    Args:
        x0: position;
        w: half width.
    '''

    if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
        F = np.zeros(1,)
    else:
        F = np.zeros(x.shape)

    F[(x > x0 - w) & (x < x0 + w)] = (1/w) / 2
    F[(x == x0 - w) | ( x == x0 + w)] = (1/w) / 4
    # F[(x > x0 - w) & (x < x0 + w)] = 1
    # F[(x == x0 - w) | ( x == x0 + w)] = 1/2

    return F



# --------        CurveBase        --------
class CurveBase(ABC):
    '''Interface for any characteristics'''

    @abstractmethod
    def __call__(self, x: Array[meter], fill_value: float = np.nan) -> Array[float]:
        raise NotImplementedError


@dataclass
class ConstantCurve(CurveBase):
    value: float

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:

        if isinstance(x, float):  # TODO: don't remove! It's for integrate.quad functions!
            return self.value
        if isinstance(x, Array):
            return np.full(x.shape, self.value)

        raise ValueError


@dataclass
class WindowCurve(CurveBase):
    span: tuple[float, float]
    edge: int
    _x: Array[nano] = field(init=False, repr=False)
    _y: Array[float] = field(init=False, repr=False)

    def __post_init__(self):
        lb, ub = SPECTRAL_RANGE

        x0 = np.sum(self.span) / 2
        w = (self.span[-1] - self.span[0]) / 2
        if self.edge:
            grid_x = np.arange(-10*w, +10*w, 1)
            f1 = lambda x: rectangular(x, 0, w=w)
            f2 = lambda x: gauss(x, 0, w=self.edge)
            grid_f = 2 * w * signal.convolve(f1(grid_x), f2(grid_x), mode='same') * (grid_x[-1] - grid_x[0])/len(grid_x)

            x = np.arange(lb, ub+SPECTRAL_STEP, SPECTRAL_STEP)
            y = interpolate.interp1d(
                grid_x,
                grid_f,
                kind='linear',
                bounds_error=False,
                fill_value=0,
            )(x - x0)

        else:
            x = np.arange(lb, ub+1, 1)
            y = 2 * w * rectangular(x, x0, w)

        self._x = x
        self._y = y

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            1e+9*self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)


@dataclass
class DatasheetCurve(CurveBase):
    path: str
    norm: float = field(default=1)
    units: float = field(default=1e-6)  # to transform wavelength to meter units
    _x: Array[micro] = field(init=False, repr=False)  # FIXME: change to nano
    _y: Array[float] = field(init=False, repr=False)

    def __post_init__(self):
        dat = np.genfromtxt(self.path, delimiter=',', dtype=np.float32)

        self._x = self.units*dat[:, 0]
        self._y = dat[:, 1]

    def __call__(self, x: float | Array[meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x) / self.norm
