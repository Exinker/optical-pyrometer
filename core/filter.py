
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from .alias import meter, nano, Array
from .config import SPECTRAL_RANGE
from .сharacteristic import WindowCharacteristic


@dataclass
class Filter(ABC):
    '''Interface for any filter'''

    @abstractmethod
    def __call__(self, x: Array[meter], fill_value: float = np.nan) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    # --------        handlers        --------
    @abstractmethod
    def show(self, ax: plt.Axes | None = None) -> None:
        raise NotImplementedError


@dataclass
class WindowFilter(Filter, WindowCharacteristic):
    span: tuple[nano, nano]
    edge: int

    def __repr__(self) -> str:
        cls = self.__class__
        return '{}, nm'.format('-'.join(map(str, self.span)))

    def __call__(self, x: Array[meter], fill_value: float = np.nan) -> Array[float]:
        return interpolate.interp1d(
            1e-9*self._x, self._y,
            kind='linear', bounds_error=False, fill_value=fill_value,
        )(x)

    # --------        handlers        --------
    def show(self, info: Literal['title', 'text'] = 'text', save: bool = False, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)

        ax.plot(
            self._x, self._y,
            color='black',
        )
        text = 'Filter: {}, нм'.format('-'.join(map(str, self.span)))
        match info:
            case 'title':
                ax.set_title(text)
            case 'text': 
                ax.text(
                    0.05, 0.95,
                    text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                )
        ax.set_xlim(SPECTRAL_RANGE)
        ax.set_ylim([0, 1.25])
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Коэффициент пропускания')
        ax.grid(color='grey', linestyle=':')

        if save:
            filepath = os.path.join('.', 'report', 'img', 'filter-response.png')
            plt.savefig(filepath)

        if not fill:
            plt.show()
