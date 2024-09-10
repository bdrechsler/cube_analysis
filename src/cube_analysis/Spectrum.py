import copy
import numpy as np
import astropy.units as u
from astropy.constants import c
import pickle
from scipy.signal import medfilt
from astropy.modeling import models, fitting

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
        continuum_flux = np.poly1d(fit_params)

        return Spectrum(self.wvl, continuum_flux(self.wvl.value) * u.Jy)
    
    def attach_line(self, line):
        self.line = line
        self.vel = ((c * (self.wvl - line.wvl)) / line.wvl).to(u.km/u.s)
    
    def get_line_flux(self, n_gauss=1, means=None):
        if hasattr(self, "line"):
            fitter = fitting.LevMarLSQFitter()
            if n_gauss==1:
                g_init = models.Gaussian1D(amplitude=np.nanmax(self.flux).value, mean=self.line.wvl.value,
                                  stddev = self.line.lw.value/2.)
                g = fitter(g_init, self.wvl.value, self.flux.value)
                A = g.amplitude * u.Jy
                sigma = ((g.stddev * u.um) * (c / self.line.wvl**2)).to(u.Hz)
                line_flux = A * sigma * np.sqrt(2*np.pi)
            
            elif n_gauss==2:
                g_init_1 = models.Gaussian1D(amplitude=np.nanmax(self.flux).value, mean=means[0],
                                            stddev=self.line.lw.value/6.)
                g_init_2 = models.Gaussian1D(amplitude=np.nanmax(self.flux).value, mean=means[1],
                                            stddev=self.line.lw.value/6.)
                
                g = fitter(g_init_1 + g_init_2, self.wvl.value, self.flux.value)
                A1, A2 = g.amplitude_0 * u.Jy, g.amplitude_1 * u.Jy
                sigma1 = ((g.stddev_0 * u.um) * (c / self.line.wvl**2)).to(u.Hz)
                sigma2 = ((g.stddev_1 * u.um) * (c / self.line.wvl**2)).to(u.Hz)
                line_flux = np.sqrt(2*np.pi) * (A1*sigma1 + A2*sigma2)
            
            self.line_model = g
            self.line_flux = line_flux.to(u.erg/u.s/u.cm**2)

        else:
            print("No line attatched to spectrum")


    @classmethod
    def load(cls, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
        
    def write(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def __add__(self, spec2):
        if np.array_equal(self.wvl, spec2.wvl):
            return Spectrum(self.wvl, self.flux + spec2.flux)
        else:
            return Spectrum(np.concatenate(self.wvl, spec2.wvl), np.concatenate(self.flux, spec2.flux))
        
    def __sub__(self, spec2):
        if np.array_equal(self.wvl, spec2.wvl):
            return Spectrum(self.wvl, self.flux - spec2.flux)
        else:
            new_wvls = np.setdiff1d(self.wvl, spec2.wvl)
            new_fluxes = self.flux[self.wvl==new_wvls]
            return Spectrum(new_wvls, new_fluxes)
