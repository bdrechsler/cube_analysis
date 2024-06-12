import numpy as np
import astropy.units as u
from astropy.table import QTable

class Spectrum:

    r"""
    A class to handle 1D spectral objects

    Base Attributes:
        :attr:`wvl` (numpy.ndarray):
            Array of wavelength values in microns
        :attr:`flux` (numpy.ndarray):
            Array of flux values of spectrum in Jy
    
    """
    def __init__(self, wvl=[], flux=[]):

        self.wvl = wvl
        self.flux = flux

    def write(self, fname):
        r"""
        Write spectrum as a fits file

        Args:
            :attr:`fname` (str):
                name of file to write spectrum to
        
        """
        
        spec_table = QTable([self.wvl, self.flux], 
                            names=('Wavelength', 'Flux'))
        spec_table.write(fname, format='fits')

    def read(self, fname):
        r"""
        Read in spectrum from a fits file

        Args:
            :attr:`fname` (str):
                name of file to read in spectrum from
        
        """
        spec_table = QTable.read(fname, format='fits')
        self.wvl = spec_table['Wavelength']
        self.flux = spec_table['Flux']
