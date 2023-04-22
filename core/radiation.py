
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .alias import meter, nano, kelvin, celsius, Array
from .config import COLORS
from .detector import Detector
from .utils import celsius2kelvin


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
    def show(cls, t: celsius, span: tuple[nano, nano] | None = None, xlim: tuple[float, float] = (0, 4500), ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)
        
        x = np.linspace(1e-9*xlim[0], 1e-9*xlim[1], 1000)
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
        ax.set_xlim(xlim)
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
        ax.grid(color='grey', linestyle=':')

        if not fill:
            plt.show()
        

def show_radiation_density(temperature: Sequence[celsius], detector: Detector | None = None, xlim: tuple[float, float] = (0, 4500), ax: plt.Axes | None = None) -> None:
    fill = ax is not None
    b = 2.8977719 * 1e-3  # Wien's constant, m·K

    #
    if not fill:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    for t in temperature:
        x = np.linspace(1e-9*xlim[0], 1e-9*xlim[1], 1000)
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
        )

    if detector is not None:
        config = detector.config

        for t in temperature:

            x = 1e-9 * np.linspace(*xlim, 1000)
            characteristic = interp1d(
                x, config.sensitivity(x) * config.transmittance(x),
                kind='linear', bounds_error=False, fill_value=np.nan,
            )
            y = characteristic(x) * RadiationDensity.calculate(x, t=celsius2kelvin(t))
            ax.plot(
                x*1e+9, y,
                color=COLORS['blue'], linestyle='-', linewidth=1,
            )

        ax.text(
            0.05, 0.95,
            f'Detector: {config.name}',
            transform=ax.transAxes,
            ha='left', va='top',
        )


    lb, *_, ub = temperature
    x = np.array([b/celsius2kelvin(t) for t in np.linspace(lb, ub, 20)])
    y = np.array([RadiationDensity.calculate(x, t=celsius2kelvin(t)) for x, t in zip(x, np.linspace(lb, ub, 20))])
    plt.plot(
        x*1e+9, y,
        color='black', linestyle=':', linewidth=1.5,
    )

    ax.set_xlim(xlim)
    ax.set_xlabel('$\lambda, нм$')
    ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
    ax.grid(color='grey', linestyle=':')

    if not fill:
        plt.show()


def calculate_irradiance(t: kelvin, span: tuple[nano, nano]) -> float:
    lb, ub = span

    return quad(
        lambda x: RadiationDensity.calculate(x, t),
        a=1e-9*lb,
        b=1e-9*ub,
    )[0]


def show_irradiance(temperature_range: tuple[float, float], span: tuple[nano, nano], ax: plt.Axes | None = None) -> None:
    lb, ub = temperature_range

    temperature = np.linspace(lb, ub, 200)
    irradiance = np.array([
        calculate_irradiance(celsius2kelvin(t), span)
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
            f'Filter: {list(span)}, нм',
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
        t: calculate_irradiance(celsius2kelvin(t), span)
        for t in [lb, lb+10, ub-10, ub]
    }
    ax_right.text(
        0.65, 0.05,
        '\n'.join([
            fr'I$\rm _{{{t:<5}}}={{{table[t]:.2f}}}, Вт/м^{2}$'
            for t in [lb, lb+10, ub-10, ub]
        ]),
        ha='left', va='bottom',
        transform=ax_right.transAxes,
    )
    if not fill:
        plt.show()


if __name__ == '__main__':
    RadiationDensity.show(t=1250, span=(900, 1700))
