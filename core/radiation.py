
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from .alias import meter, nm, kelvin, celsius, Array
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
    def show(cls, t: celsius, span: tuple[nm, nm] | None = None, ax: plt.Axes | None = None) -> None:
        fill = ax is not None

        if not fill:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), tight_layout=True)

        lb, ub = (10, 3000)
        x = np.linspace(lb*1e-9, ub*1e-9, 1000)
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
            lb, ub = span
            x = np.linspace(lb*1e-9, ub*1e-9, 1000)
            y = cls.calculate(x, t=celsius2kelvin(t))
            ax.fill_between(
                x*1e+9, y,
                step='mid',
                alpha=0.25, facecolor='lightskyblue', edgecolor='lightskyblue',
                label='область\nинтегр.',
            )
        ax.set_xlabel('$\lambda, нм$')
        ax.set_ylabel(r'Плотность излучения, $\rm Вт/м^{3}$')
        ax.grid(color='grey', linestyle=':')

        if not fill:
            plt.show()
        

if __name__ == '__main__':
    RadiationDensity.show(t=1250, span=(900, 1700))
