import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from .Spectrum import Spectrum

class Cube:
    r"""
    A class to store the contents of a spectral data cube

    Base Attributes:
        :attr:`flux` (numpy.ndarray):
            Array of the flux values of the cube in units of Jy
        :attr:`wvl` (nummpy.ndarray):
            Wavelength of each channel in units of microns
        :attr:`header` (fits header):
            header of the original fits file
        :attr:`wcs` (astropy wcs):
            wcs object of the spectral cube
        :attr:`spectra` (list[spectrum]):
            list of 1D extracted spectra
        :attr:`collapsed_img` (numpy.ndarray):
            Representative image of the cube obtained by collapsing along the spectral axis
        :attr:`collapsed_spec` (numpy.ndarray):
            Representative spectrum of the cube obtained by collapsing along the spatial axes
    
    """
    def __init__(self):
        
        self.flux = []
        self.wvl = []
        self.header = []
        self.wcs = []
        self.spectra = []
        self.collapsed_img = []
        self.collapsed_spec = []

    def read(self, path):

        r"""
        Read in spectral cube data from a fits file to populate base attributes
        
        Args:
            :attr:`path` (str):
                The path to the fits file
        
        """

        # read in data from fits file
        data, header = fits.getdata(path, extname='SCI', header=True)
        # data shape is (spectral, y, x)

        # get wvl array from header info
        start_wvl = header['CRVAL3']
        wvl_step = header['CDELT3']
        self.nchan = header['NAXIS3']
        wvl = (np.arange(self.nchan) * wvl_step) + start_wvl

        # make sure units are in Jy
        flux_units = header['BUNIT']
        if flux_units == 'MJy/sr':
            # pixel area in sr
            pix_area = header['PIXAR_SR']
            data *= pix_area * 1e6
        

        self.flux = data * u.Jy
        self.wvl = wvl * u.um
        self.header = header
        self.wcs = WCS(header)

        self.collapsed_img = np.nansum(self.flux, axis=0)
        self.collapsed_spec = np.nansum(self.flux, axis=(1,2))


    def extract_spectrum(self, sky_aps):

        r"""
        Extracted 1D spectrum from a list of apertures

        Args:
            :attr:`sky_aps` (list[sky_apertures])
                List of sky apertures to use to extract one dimensional spectra
        
        Returns:
            :attr:`spec_list` (list[spectrum])
                List of spectrum objects corresponding to the input sky apertures
        
        """

        spec_list = []
        for sky_ap in sky_aps:
            # convert sky ap to pixel coordinates
            pix_ap = sky_ap.to_pixel(self.wcs.celestial)
            # create a mask
            mask = pix_ap.to_mask(method='exact')

            # initialize spectrum array
            spectrum_flux = np.zeros(self.nchan)

            # extract the 1D spectrum
            for i in range(self.nchan):
                # get data of current channel
                chan = self.flux[i, ...].value
                # extract the data in the aperture
                ap_data = mask.get_values(chan)
                # sum to get value for spectrum
                spectrum_flux[i] = np.nansum(ap_data)

            # append to list of extracted spectra
            spectrum = Spectrum(self.wvl, spectrum_flux)
            spec_list.append(spectrum)
        
        self.spectra = spec_list