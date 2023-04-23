from typing import overload

import numpy as np

from .alias import Array


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
def lanczos(x: float, x0: float, a: int = 2) -> float: ...
@overload
def lanczos(x: Array, x0: float, a: int = 2) -> Array: ...
def lanczos(x, x0: float, a: int = 2):
    '''Lanczos distribution with position x0
    Params:
        x0: position;
        a: window width.
    '''

    F = np.sinc(x - x0) * np.sinc((x - x0) / a)
    F[(x - x0 < -a) | (x - x0 > a)] = 0

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


@overload
def voigt(x: float, x0: float, w: float, a: float, r: float) -> float: ...
@overload
def voigt(x: Array, x0: float, w: float, a: float, r: float) -> Array: ...
def voigt(x, x0, w, a, r):
    '''
    Voigt distribution with position x0 and unit intensity.

    Args:
        x0: position;
        w: width;
        a: assymetry;
        r: ratio (in range 0-1).

    A simple asymmetric line shape profile for fitting infrared absorption spectra.
    Aaron L. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    '''

    sigma = 2*w / (1 + np.exp(a*(x - x0)) )
    G = np.sqrt(4*np.log(2)/np.pi) / sigma * np.exp(-4*np.log(2)*((x - x0)/sigma)**2)
    L = 2/np.pi/sigma/(1 + 4*((x - x0)/sigma)**2)
    F = r*L + (1 - r)*G

    return F
