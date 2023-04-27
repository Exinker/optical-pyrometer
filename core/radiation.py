
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .alias import meter, nano, kelvin, celsius, Array
from .config import COLORS, SPECTRAL_RANGE, SPECTRAL_STEP
from .filter import Filter
from .detector import Detector
from .utils import calculate_response, celsius2kelvin


@dataclass
class RadiationDensity:

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

        y = (2*np.pi * h * c**2) / (x ** 5) * (1 / (np.exp(h * c/(x * k * t)) - 1))

        return y

    @classmethod
    def show(cls, t: celsius, span: tuple[nano, nano] | None = None, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)
        
        lb, ub = SPECTRAL_RANGE
        x = 1e-9*np.arange(lb, ub+SPECTRAL_STEP, SPECTRAL_STEP)
        y = cls.calculate(x, t=celsius2kelvin(t))
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
            y = cls.calculate(x, t=celsius2kelvin(t))
            ax.fill_between(
                x*1e+9, y,
                step='mid',
                alpha=0.1, facecolor=COLORS['blue'], edgecolor=COLORS['blue'],
                label='область\nинтегр.',
            )
        ax.set_xlim(SPECTRAL_RANGE)
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
        ax.grid(color='grey', linestyle=':')

        if not fill:
            plt.show()
        

def show_radiation_density(temperature: Sequence[celsius], filter: Filter | None = None, detector: Detector | None = None, save: bool = False, ax: plt.Axes | None = None) -> None:
    fill = ax is not None
    b = 2.8977719 * 1e-3  # Wien's constant, m·K

    #
    if not fill:
        fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)

    lb, ub = SPECTRAL_RANGE
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
                color=COLORS['blue'], linestyle='-', linewidth=1,
            )

    content = []
    if filter is not None:
        content.append(f'Filter: {filter}')
    if detector is not None:
        content.append(f'Detector: {detector.config.name}',)
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

    ax.set_xlim(SPECTRAL_RANGE)
    # ax.set_ylim([0, 1.2e+11])
    ax.set_xlabel('$\lambda, нм$')
    ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
    ax.grid(color='grey', linestyle=':')

    if save:
        filepath = os.path.join('.', 'report', 'img', 'radiation-density.png')
        plt.savefig(filepath)

    if not fill:
        plt.show()


if __name__ == '__main__':
    RadiationDensity.show(t=1250, span=(900, 1700))
