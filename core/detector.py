
import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from .alias import meter, Array
from .сharacteristic import Characteristic, ConstantCharacteristic, DatasheetCharacteristic
from .config import SPECTRAL_RANGE, SPECTRAL_STEP


@dataclass(frozen=True)
class DetectorConfig:
    '''Detectors's config'''
    name: str
    sensitivity: Characteristic
    transmittance: Characteristic

    def __repr__(self) -> str:
        cls = self.__class__
        name = self.name

        return f'{cls.__name__}: {name}'


class Detector(Enum):
    '''Enums with detectors's config'''
    unicorn = DetectorConfig(
        name='Unicorn',
        sensitivity=ConstantCharacteristic(value=1),
        transmittance=ConstantCharacteristic(value=1),
    )
    G12180 = DetectorConfig(
        name='G12180 series',
        sensitivity=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12180', 'photo-sensitivity.csv'),
        ),
        transmittance=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12180', 'window-spectral-transmittance.csv'),
            norm=100,
        ),
    )
    G12183 = DetectorConfig(
        name='G12183 series*',  # exclude G12183-219KA-03 detector
        sensitivity=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12183', 'photo-sensitivity.csv'),
        ),
        transmittance=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12183', 'window-spectral-transmittance.csv'),
            norm=100,
        ),
    )
    G12183_219KA_03 = DetectorConfig(
        name='G12183-219KA-03',
        sensitivity=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12183-219KA-03', 'photo-sensitivity.csv'),
        ),
        transmittance=DatasheetCharacteristic(
            path=os.path.join('.', 'core', 'dat', 'G12183-219KA-03', 'window-spectral-transmittance.csv'),
            norm=100,
        ),
    )

    @property
    def config(self) -> DetectorConfig:
        return self.value

    def __repr__(self) -> str:
        cls = self.__class__
        name = self.config.name

        return f'{cls.__name__}: {name}'

    # --------        handlers        --------
    def responce(self, x: Array[meter], fill_value: float = np.nan) -> Array[float]:
        config = self.config

        return config.sensitivity(x, fill_value) * config.transmittance(x, fill_value)

    def show(self, info: Literal['title', 'text'] = 'text', save: bool = False, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)

        lb, ub = SPECTRAL_RANGE
        x = 1e-9*np.arange(lb, ub+SPECTRAL_STEP, SPECTRAL_STEP)
        y = self.responce(x)
        ax.plot(
            1e+9*x, y,
            color='black',
        )
        text = f'Detector: {self.name}'
        match info:
            case 'title':
                ax.set_title(text)
            case 'text': 
                ax.text(
                    0.05, 0.95,
                    text,
                    transform=ax.transAxes,
                )
        ax.set_xlim(SPECTRAL_RANGE)
        ax.set_ylim([0, 1.25])
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Спектральный отклик фотодетектора')
        ax.grid(color='grey', linestyle=':')

        if save:
            filepath = os.path.join('.', 'report', 'detector-response.png')
            plt.savefig(filepath)

        if not fill:
            plt.show()
