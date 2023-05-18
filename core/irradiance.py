
import os

import numpy as np
import matplotlib.pyplot as plt

from spectrumlab.alias import kelvin, celsius
from spectrumlab.emulation.characteristic.filter import Filter
from spectrumlab.emulation.detector import PhotoDiode
from spectrumlab.picture.format import format_value

from .config import SPECTRAL_RANGE
from .device import RadiationDensity, calculate_response
from .utils import celsius2kelvin


def calculate_irradiance(t: kelvin, filter: Filter, detector: PhotoDiode | None = None) -> float:
    '''calculate irafiance, W/m^2 (or A/m^2 if detector is not None)'''
    lb, ub = SPECTRAL_RANGE

    dx = 1e-9
    x = np.arange(1e-9*lb, 1e-9*ub, dx)
    y = dx*np.nansum(RadiationDensity.calculate(x, t) * calculate_response(x, filter=filter, detector=detector))

    return y


def show_irradiance(temperature_range: tuple[celsius, celsius], filter: Filter, detector: PhotoDiode | None = None, save: bool = False, ax: plt.Axes | None = None) -> None:
    lb, ub = temperature_range

    temperature = np.linspace(lb, ub, 200)
    irradiance = np.array([
        calculate_irradiance(celsius2kelvin(t), filter=filter, detector=detector)
        for t in temperature
    ])

    #
    fill = ax is not None

    if not fill:
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)

    kind = 'Излучение' if detector is None else 'Ток'
    units = 'Вт/м^{2}' if detector is None else 'А/м^{2}'

    x = temperature
    y = irradiance
    ax_left.plot(
        x, y,
        color='black',
    )
    content = []
    if filter is not None:
        content.append(f'Filter: {filter}')
    if detector is not None:
        content.append(f'PhotoDiode: {detector.config.name}',)
    ax_left.text(
        0.05/2, 0.95,
        '\n'.join(content),
        transform=ax_left.transAxes,
        ha='left', va='top',
    )
    ax_left.set_xlabel(r'$T, ^{\circ}C$')
    ax_left.set_ylabel(fr'{kind}, $\rm {units}$')
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
    ax_right.set_ylabel(fr'$\rm d/dT({kind}), {units}T$')
    ax_right.grid(color='grey', linestyle=':')
    table = {
        t: calculate_irradiance(celsius2kelvin(t), filter=filter, detector=detector)
        for t in [lb, lb+10, ub-10, ub]
    }
    ax_right.text(
        0.70, 0.05,
        '\n'.join([
            r'$\rm {label}={value}$'.format(
                label=fr'I_{{{t:<5}}}',
                value=format_value(table[t], units, n_digits=2, prefix=True),
            )
            for t in [lb, lb+10, ub-10, ub]
        ]),
        ha='left', va='bottom',
        transform=ax_right.transAxes,
    )

    if save:
        filepath = os.path.join('.', 'report', 'img', 'irradiance.png')
        plt.savefig(filepath)

    if not fill:
        plt.show()
