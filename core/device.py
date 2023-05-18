
import os
from collections.abc import Sequence, Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from spectrumlab.alias import meter, nano, kelvin, celsius, Array
from spectrumlab.emulation.characteristic.filter import Filter
from spectrumlab.emulation.detector import PhotoDiode
from spectrumlab.picture.config import COLOR

from .adc import ADC
from .exceptions import NotFittedError
from .utils import celsius2kelvin


# --------        radiation density        --------
@dataclass
class RadiationDensity:
    wavelength_bounds: tuple[nano, nano] = field(default=(0, 4500))
    wavelength_step: nano = field(default=1)

    @staticmethod
    def calculate(x: Array[meter], t: kelvin) -> Array[float]:
        '''calculate spectral density of radiation of black-body
        Args:
            x: wavelength array, in m
            t: temperature, in K
        Return:
            y: density of radiation, in W/m^{3}, or J/(s⋅m^{3})
        '''
        assert t > 0, 'temperature < 0'

        h = 6.625 * 10**-34  # Planck constant, J⋅s
        c = 2.99792458 * 10**8  # speed of light in vacuum, m/s
        k = 1.3806488 * 10**-23  # Boltzmann constant, J/K

        return (2*np.pi * h * c**2) / (x ** 5) * (1 / (np.exp(h * c/(x * k * t)) - 1))

    def show(self, t: celsius, span: tuple[nano, nano] | None = None, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)
        
        lb, ub = self.wavelength_bounds
        step = self.wavelength_step
        x = 1e-9*np.arange(lb, ub+step, step)
        y = RadiationDensity.calculate(x, t=celsius2kelvin(t))
        ax.plot(
            x*1e+9, y,
            color='black',
        )
        ax.text(
            0.05, 0.90,
            fr'$T = {t}, ^{{\circ}}C$',
            transform=ax.transAxes,
        )
        if span is not None:
            x = np.linspace(1e-9*span[0], 1e-9*span[1], 1000)
            y = RadiationDensity.calculate(x, t=celsius2kelvin(t))
            ax.fill_between(
                x*1e+9, y,
                step='mid',
                alpha=0.1, facecolor=COLOR['blue'], edgecolor=COLOR['blue'],
                label='область\nинтегр.',
            )
        ax.set_xlim(self.wavelength_bounds)
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
        ax.grid(color='grey', linestyle=':')

        if not fill:
            plt.show()
        

def show_radiation_density(temperature: Sequence[celsius], wavelength_bounds: tuple[nano, nano], filter: Filter | None = None, detector: PhotoDiode | None = None, save: bool = False, ax: plt.Axes | None = None) -> None:
    fill = ax is not None
    b = 2.8977719 * 1e-3  # Wien's constant, m·K

    #
    if not fill:
        fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)

    lb, ub = wavelength_bounds
    for t in temperature:
        x = 1e-9*np.arange(lb, ub+1, 1)
        y = RadiationDensity.calculate(x, t=celsius2kelvin(t))
        ax.plot(
            x*1e+9, y,
            color='black', linestyle='-', linewidth=1,
        )

        x = b/celsius2kelvin(t)
        y = RadiationDensity.calculate(x, t=celsius2kelvin(t))
        ax.text(
            x*1e+9, y,
            fr'{t:.0f}$^{{\circ}}C$',
            color='red',
            # va='bottom', ha='center',
        )

    if any([item for item in [filter, detector]]):
        for t in temperature:
            x = 1e-9*np.arange(lb, ub+1, 1)
            y = calculate_response(x, filter=filter, detector=detector) * RadiationDensity.calculate(x, t=celsius2kelvin(t))
            ax.plot(
                x*1e+9, y,
                color=COLOR['blue'], linestyle='-', linewidth=1,
            )

    content = []
    if filter is not None:
        content.append(f'Filter: {filter}')
    if detector is not None:
        content.append(f'PhotoDiode: {detector.config.name}',)
    ax.text(
        0.05/2, 0.95,
        '\n'.join(content),
        transform=ax.transAxes,
        ha='left', va='top',
    )

    lb, *_, ub = temperature
    x = np.array([b/celsius2kelvin(t) for t in np.linspace(lb, ub, 20)])
    y = np.array([RadiationDensity.calculate(x, t=celsius2kelvin(t)) for x, t in zip(x, np.linspace(lb, ub, 20))])
    plt.plot(
        x*1e+9, y,
        color='black', linestyle=':', linewidth=1,
    )

    ax.set_xlim(wavelength_bounds)
    # ax.set_ylim([0, 1.2e+11])
    ax.set_xlabel('$\lambda, нм$')
    ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
    ax.grid(color='grey', linestyle=':')

    if save:
        filepath = os.path.join('.', 'report', 'img', 'radiation-density.png')
        plt.savefig(filepath)

    if not fill:
        plt.show()


# --------        signal        --------
def calculate_response(x: Array, filter: Filter | None = None, detector: PhotoDiode | None = None, fill_value: float = np.nan) -> Array:

    y = np.ones(x.shape)
    if filter is not None:
        y *= filter(x, fill_value)
    if detector is not None:
        y *= detector.responce(x, fill_value)

    return y


def calculate_current(x: Array[meter], t: kelvin, response: Array[float], alpha: float) -> float:
    '''calculate a detector's current, in A'''
    dx = (x[-1] - x[0]) / (len(x) - 1)
    value = np.nansum(response * RadiationDensity.calculate(x, t)) * dx

    return alpha * value


@dataclass
class Signal:
    current: Array[float]
    temperature: Array[celsius]


@dataclass
class Device:
    filter: Filter
    detector: PhotoDiode
    adc: ADC
    alpha: float  # response coeff    

    wavelength_bounds: tuple[nano, nano]
    wavelength_step: nano

    _current: Array[float] = field(default=None, init=False)
    _temperature: Array[celsius] = field(default=None, init=False)

    def calculate_input_current(self, temperature: celsius | Array[celsius]) -> float | Array[float]:
        '''Calculate a input current at the given temperature'''
        lb, ub = self.wavelength_bounds
        x = 1e-9*np.arange(lb, ub + self.wavelength_step, self.wavelength_step)
        response = calculate_response(x, filter=self.filter, detector=self.detector)

        if isinstance(temperature, Iterable):
            return np.array([
                calculate_current(x, t=celsius2kelvin(t), response=response, alpha=self.alpha)
                for t in temperature
            ])

        return calculate_current(x, t=celsius2kelvin(temperature), response=response, alpha=self.alpha)

    def fit(self, temperature: celsius | Array[celsius]) -> 'Device':
        self._current = self.calculate_input_current(temperature)
        self._temperature = temperature

        return self

    def predict(self, current: Array, kind: Literal['unicorn', 'adc']) -> Array:
        assert self._current is not None, NotFittedError.__doc__
        assert self._temperature is not None, NotFittedError.__doc__

        if kind == 'unicorn':
            return interpolate.interp1d(
                self._current, self._temperature,
                kind='linear', bounds_error=False, fill_value=self._temperature[-1],  # FIXME: 
            )(current)

        if kind == 'adc':
            adc = self.adc

            # quantized values
            current_quantized = adc.quantize(current)
            return interpolate.interp1d(
                np.log2(current) if adc.log else current, self._temperature,
                kind='linear', bounds_error=False, fill_value=self._temperature[-1],
            )(current_quantized)


if __name__ == '__main__':
    RadiationDensity().show(t=1250, span=(900, 1700))
