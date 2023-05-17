
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from spectrumlab.alias import celsius
from spectrumlab.emulation.characteristic.filter import Filter
from spectrumlab.emulation.detector import PhotoDiode
from spectrumlab.picture.format import format_value

from .adc import ADC
from .config import SPECTRAL_RANGE, SPECTRAL_STEP
from .signal import Signal


@dataclass
class ExperimentConfig:
    emissivity: float = field(default=.9)  # emissivity coefficient
    square: float = field(default=1)  # square of spot
    ratio: float = field(default=1)  # detected part of radiation

    @property
    def coeff(self) -> float:
        return self.emissivity * self.square * self.ratio

    def __str__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}(emissivity: {self.emissivity:.2f}, k={self.square * self.ratio:.2E}, m^2)'


def run_experiment(temperature_range: tuple[celsius, celsius], filter: Filter, detector: PhotoDiode, adc: ADC, config: ExperimentConfig, relative: bool = False, save: bool = False) -> None:
    lb, ub = temperature_range

    # signal
    signal = Signal(
        wavelength_bounds=SPECTRAL_RANGE,
        wavelength_step=SPECTRAL_STEP,
    )

    # values
    temperature = np.linspace(lb, ub, 200)
    values = config.coeff * signal.calculate(temperature, filter=filter, detector=detector)

    # quantized values
    values_quantized = adc.quantize(values)
    temperature_quantized = interpolate.interp1d(
        np.log2(values) if adc.log else values, temperature,
        kind='linear',
        # kind='linear', bounds_error=False, fill_value=ub,
    )(values_quantized)

    # # approxed values
    # values_quantized = np.log10(values_quantized) if not adc.log else values_quantized
    # p = np.polyfit(values_quantized, temperature, deg=degree)
    # temperature_approxed = np.polyval(p, values_quantized)

    # show
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)

    x, y = values, temperature
    ax_left.plot(
        x, y,
        color='black', linestyle='-',
        label='сигнал',
    )
    x, y = values, temperature_quantized
    ax_left.plot(
        x, y,
        # color='red', linestyle='--',
        label='дискрет. сигнал',
    )
    ax_left.text(
        0.05, 0.95,
        '\n'.join([
            f'Filter: {filter}',
            f'PhotoDiode: {detector.config.name}',
            f'ADC(res={adc.resolution}, log={repr(adc.log)})',
            r'$\rm {label}={value}$'.format(
                label='s',
                value=format_value(config.square*config.ratio, r'm^{2}', n_digits=2, prefix=False),
            ),  # fr'$s={float2latex(m^{2}, 2)}, $',
            fr'$\rm \epsilon={config.emissivity:.2f}$',
        ]),
        ha='left', va='top',
        transform=ax_left.transAxes,
    )
    table = {
        t: config.coeff * signal.calculate(t, filter=filter, detector=detector)
        for t in np.linspace(lb, lb+10, 2)
    }
    ax_left.text(
        0.70, 0.05,
        '\n'.join([
            r'$\rm {label}={value}$'.format(
                label=fr'I_{{{t:<5}}}',
                value=format_value(table[t], 'A', n_digits=2, prefix=True),
            )
            for t in table.keys()
        ]),
        ha='left', va='bottom',
        transform=ax_left.transAxes,
    )
    # ax_left.set_xscale('log')
    ax_left.set_xlabel(r'Ток, А')
    ax_left.set_ylabel(r'$\rm T, ^{\circ}C$')
    ax_left.grid(color='grey', linestyle=':')
    ax_left.legend(loc='upper right')

    x, y = temperature, temperature - temperature_quantized
    ax_right.plot(
        x, 100*y/temperature if relative else y,
        # color='red', linestyle='--',
        label='ошибка дискрет.',
    )
    # x, y = values, temperature - temperature_approxed
    # ax_right.plot(
    #     x, 100*y/temperature if relative else y,
    #     # color='red', linestyle='-',
    #     label='ошибка аппрокс.',
    # )
    # ax_right.set_xscale('log')
    ax_right.set_xlabel(r'$T, ^{\circ}C$')
    ax_right.set_ylabel(r'$\rm \delta T, \%$' if relative else '$\delta T, ^{\circ}C$')
    ax_right.grid(color='grey', linestyle=':')
    ax_right.legend(loc='upper right')

    if save:
        filepath = os.path.join('.', 'report', 'img', 'signal-temperature.png')
        plt.savefig(filepath)

    plt.show()
