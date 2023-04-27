
from decimal import Decimal
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



# from spectrumlab.picture.formatter improt to_label
PREFIX = {
    -12: 'p',
    -9: 'n',
    -6: '\mu',
    -3: 'm',
    0: '',
    3: 'k',
    6: 'M',
    9: 'G',
}


def format_value(value: float, units: str | None = None, n_digits: int = 2, prefix: bool = False) -> str:
    '''format value and units to label
    
    1e-5 -> r'10.00 \cdot 10^{-6}'
    1e-5, 'A' -> r'10.00 \cdot 10^{-6}, A'
    1e-5, 'A' -> r'10.00, \muA'
    
    
    FIXME: 1e-06, 'A' -> $999.99, nA$

    '''

    assert n_digits >= 0, f'n_digits have to be greater 0'

    #
    sign, digits, exponent = Decimal(value).as_tuple()

    e = len(digits) + exponent - 1
    e_rounded = int(np.sign(e) * (3*np.floor(np.abs(e) / 3)))
    n = e - e_rounded + 1

    #
    result = []

    if n > 0:
        result.append(
            ''.join(map(str, digits[:n])) + '.' * (n_digits != 0) + ''.join(map(str, digits[n:n + n_digits]))
        )
    else:
        m = -n
        result.append(
            '0' + '.' + '0'*(m) + ''.join(map(str, digits[m:n_digits - m + 1]))
        )

    # add units with prefix
    if units:
        if prefix:
            result.append(
                r', {{{}}}{{{}}}'.format(
                    PREFIX.get(e_rounded, '???'),
                    units,
                )
            )
        else:
            result.append(
                fr' \cdot 10^{{{e_rounded}}}' * (e_rounded != 0) + f', {units}'
            )


    else:
        result.append(
            fr' \cdot 10^{{{e_rounded}}}' * (e_rounded != 0)
        )

    return r'{}'.format(
        ''.join(result)
    )
