import numpy as np
import astropy.units as u
from astropy.table import QTable

class Spectrum:
    def __init__(self, wvl=[], flux=[]):

        self.wvl = wvl
        self.flux = flux

    def write(self, fname):
        
        spec_table = QTable([self.wvl, self.flux], 
                            names=('Wavelength', 'Flux'))
        spec_table.write(fname, format='fits')

    def read(self, fname):
        
        spec_table = QTable.read(fname, format='fits')
        self.wvl = spec_table['Wavelength']
        self.flux = spec_table['Flux']
