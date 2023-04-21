
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .alias import meter, nm, kelvin, celsius, Array
from.config import COLORS
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
    def show(cls, t: celsius, span: tuple[nm, nm] | None = None, xlim: tuple[float, float] = (0, 4500), ax: plt.Axes | None = None) -> None:
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
        

def show_radiation_density(ts: tuple[celsius], span: tuple[nm, nm] | None = None, xlim: tuple[float, float] = (0, 4500), ax: plt.Axes | None = None) -> None:
    fill = ax is not None
    b = 2.8977719 * 1e-3  # Wien's constant, m·K

    #
    if not fill:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    for t in ts:
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

    if span is not None:
        ax.axvspan(
            *span,
            color=COLORS['blue'],
            alpha=.1,
        )
        for x in span:
            ax.axvline(
                x,
                color=COLORS['blue'], linestyle=':', linewidth=1,
                alpha=.1,
            )

    lb, *_, ub = ts
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


if __name__ == '__main__':
    RadiationDensity.show(t=1250, span=(900, 1700))
