
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .adc import ADC
from .irradiance import Irradiance


def run_experiment(temperature_range: tuple[float, float], sensitivity_range: tuple[float, float], adc: ADC, degree: int = 5, relative: bool = False, show: bool = True):
    lb, ub = temperature_range

    # signal
    temperature = np.linspace(lb, ub, 2000)
    irradiance = Irradiance.calculate(temperature, sensitivity=sensitivity_range)

    # quantized signal
    irradiance_quantized = adc.quantize(irradiance)
    temperature_quantized = interp1d(
        np.log2(irradiance) if adc.log else irradiance, temperature,
        kind='linear',
        # kind='linear', bounds_error=False, fill_value=ub,
    )(irradiance_quantized)

    # # approxed signal
    # irradiance_quantized = np.log10(irradiance_quantized) if not adc.log else irradiance_quantized
    # p = np.polyfit(irradiance_quantized, temperature, deg=degree)
    # temperature_approxed = np.polyval(p, irradiance_quantized)

    # show
    if show:
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)

        x, y = irradiance, temperature
        ax_left.plot(
            x, y,
            color='black', linestyle='-',
            label='сигнал',
        )
        x, y = irradiance, temperature_quantized
        ax_left.plot(
            x, y,
            # color='red', linestyle='--',
            label='дискрет. сигнал',
        )
        ax_left.text(
            0.05, 0.95,
            '\n'.join([
                f'ADC(res={adc.resolution}, log={repr(adc.log)})',
            ]),
            ha='left', va='top',
            transform=ax_left.transAxes,
        )
        table = {
            t: Irradiance.calculate(np.array([t]), sensitivity=sensitivity_range).item()
            for t in [lb, lb+10, ub-10, ub]
        }
        ax_left.text(
            0.65, 0.05,
            '\n'.join([
                fr'I$\rm _{{{t:<5}}}={{{table[t]:.2f}}}, Вт/м^{2}$'
                for t in [lb, lb+10, ub-10, ub]
            ]),
            ha='left', va='bottom',
            transform=ax_left.transAxes,
        )
        # x, y = irradiance, temperature_approxed
        # ax_left.plot(
        #     x, y,
        #     # color='red', linestyle='-',
        #     label='аппрокс. сигнал',
        # )
        # ax_left.set_xscale('log')
        ax_left.set_xlabel(r'Излучение, $\rm Вт/м^{2}$')
        ax_left.set_ylabel(r'$\rm T, ^{\circ}C$')
        ax_left.grid(color='grey', linestyle=':')
        ax_left.legend(loc='upper right')

        x, y = temperature, temperature - temperature_quantized
        ax_right.plot(
            x, 100*y/temperature if relative else y,
            # color='red', linestyle='--',
            label='ошибка дискрет.',
        )
        # x, y = irradiance, temperature - temperature_approxed
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

        plt.show()

    #
    if adc.log:
        lv, mv, uv = [np.log2(Irradiance.calculate(np.array([t]), sensitivity=sensitivity_range)).item() for t in [lb, lb+10, ub]]
    else:
        lv, mv, uv = [Irradiance.calculate(np.array([t]), sensitivity=sensitivity_range).item() for t in [lb, lb+10, ub]]

    result = ((uv - lv) / (2**adc.resolution - 1)) < (mv - lv)

    return result
