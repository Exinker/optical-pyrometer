
import os

from dataclasses import dataclass
from enum import Enum

from .сharacteristic import Characteristic, ConstantCharacteristic, DatasheetCharacteristic


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
