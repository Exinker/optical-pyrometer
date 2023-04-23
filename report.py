
import os

import matplotlib.pyplot as plt
from fpdf import FPDF

from core.adc import ADC
from core.alias import celsius
from core.detector import Detector
from core.filter import WindowFilter as Filter
from core.experiment import run_experiment
from core.radiation import RadiationDensity, show_radiation_density, show_irradiance

import warnings
warnings.filterwarnings('ignore')

WIDTH = 210  # in mm
HEIGHT = 297  # in mm



def run_report(temperature_range: tuple[celsius, celsius], filter: Filter, detector: Detector) -> None:

    pdf = FPDF(
        orientation='portrait',
        unit='mm',
        format='A4',
    )

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(
        w=WIDTH, h=5,
        txt='Temperature: {}, C'.format('-'.join(map(str, temperature_range))),
        ln=1,
    )
    pdf.cell(
        w=WIDTH, h=5,
        txt=f'Filter: {filter}',
        ln=1,
    )
    pdf.cell(
        w=WIDTH, h=5,
        txt=f'Detector: {detector.config.name}',
        ln=0,
    )

    pdf.set_font('helvetica', '', 8)
    html = f'''
    <p></p>
    <center>
        <img src="./report/img/spectral-response.png" alt="spectral-response"/ width=480>
        <img src="./report/img/irradiance.png" alt="irradiance"/ width=480>
        <img src="./report/img/radiation-density.png" alt="radiation-density"/ width=360>
        <img src="./report/img/signal-temperature.png" alt="signal-temperature"/ width=480>
    </center>
    '''
    pdf.write_html(html)


    filepath = os.path.join('.', 'report', 'report-{filter}-{detector}.pdf'.format(filter=f'{filter}'.strip(', nm'), detector=detector.name))
    pdf.output(filepath)


if __name__ == '__main__':
    
    temperature_range = (400, 1250)  # in celsius
    filter = Filter(
        span=(900, 2500),
        edge=250,
    )
    detector = Detector.G12183

    run_report(
        temperature_range=temperature_range,
        filter=filter,
        detector=detector,
    )