import copy
import numpy as np
import astropy.units as u
from astropy.table import QTable
from scipy.signal import medfilt

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

    def spectral_region(self, wvl1, wvl2, invert=False):

        if invert:
            inds = np.where((self.wvl < wvl1) | (self.wvl > wvl2))
        else:
            inds = np.where((self.wvl >= wvl1) & (self.wvl <= wvl2))

        return Spectrum(wvl=self.wvl[inds], flux=self.flux[inds])
    
    def spectral_region_from_center(self, wvl, width, invert=False):

        wvl1 = wvl - (width/2)
        wvl2 = wvl + (width/2)
        if invert:
            inds = np.where((self.wvl < wvl1) | (self.wvl > wvl2))
        else:
            inds = np.where((self.wvl >= wvl1) & (self.wvl <= wvl2))

        return Spectrum(wvl=self.wvl[inds], flux=self.flux[inds])
    
    def fit_continuum(self, ignore_regions, med_kernel=3, fit_order=3, from_center=True):

        just_continuum  = copy.deepcopy(self)
        for region in ignore_regions:
            if from_center:
                just_continuum = just_continuum.spectral_region_from_center(region[0], region[1], invert=True)
            else:
                just_continuum = just_continuum.spectral_region(region[0], region[1], invert=True)
        
        smooth_flux = medfilt(just_continuum.flux.value, kernel_size=med_kernel)
        fit_params = np.polyfit(just_continuum.wvl.value, smooth_flux, fit_order)
        continuum_flux = np.poly1d(fit_params) * u.Jy

        return Spectrum(self.wvl, continuum_flux)



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

    def __add__(self, spec2):
        if self.wvl != spec2.wvl:
            return "Error: Spectra have different wvl arrays"
        else:
            return Spectrum(self.wvl, self.flux + spec2.flux)
