
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
        filepath = os.path.join('.', 'report', 'radiation-density.png')
        plt.savefig(filepath)

    if not fill:
        plt.show()


def calculate_irradiance(t: kelvin, filter: Filter) -> float:
    lb, ub = SPECTRAL_RANGE

    dx = 1e-9
    x = np.arange(1e-9*lb, 1e-9*ub, dx)
    y = dx*np.nansum(RadiationDensity.calculate(x, t) * filter(x))

    return y


def show_irradiance(temperature_range: tuple[float, float], filter: Filter, save: bool = False, ax: plt.Axes | None = None) -> None:
    lb, ub = temperature_range

    temperature = np.linspace(lb, ub, 200)
    irradiance = np.array([
        calculate_irradiance(celsius2kelvin(t), filter=filter)
        for t in temperature
    ])

    #
    fill = ax is not None

    if not fill:
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)
    
    x = temperature
    y = irradiance
    ax_left.plot(
        x, y,
        color='black',
    )
    ax_left.text(
        0.05, 0.95,
        '\n'.join([
            f'Filter: {filter}',
        ]),
        ha='left', va='top',
        transform=ax_left.transAxes,
    )
    ax_left.set_xlabel(r'$T, ^{\circ}C$')
    ax_left.set_ylabel(r'Излучение, $\rm Вт/м^{2}$')
    ax_left.grid(color='grey', linestyle=':')

    x = temperature
    dx = x[1] - x[0]
    dy = np.gradient(irradiance, dx)
    ax_right.plot(
        x, dy,
        color='black',
    )
    # ax_right.set_yscale('log')
    ax_right.set_xlabel(r'$T, ^{\circ}C$')
    ax_right.set_ylabel(r'$\rm d/dT(Излучение), Вт/Tм^{2}$')
    ax_right.grid(color='grey', linestyle=':')
    table = {
        t: calculate_irradiance(celsius2kelvin(t), filter=filter)
        for t in [lb, lb+10, ub-10, ub]
    }
    ax_right.text(
        0.65, 0.05,
        '\n'.join([
            fr'I$\rm _{{{t:<5}}}={{{table[t]:.2E}}}, Вт/м^{2}$'
            for t in [lb, lb+10, ub-10, ub]
        ]),
        ha='left', va='bottom',
        transform=ax_right.transAxes,
    )

    if save:
        filepath = os.path.join('.', 'report', 'irradiance.png')
        plt.savefig(filepath)

    if not fill:
        plt.show()


if __name__ == '__main__':
    RadiationDensity.show(t=1250, span=(900, 1700))
