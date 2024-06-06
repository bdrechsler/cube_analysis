import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from .Spectrum import Spectrum

class Cube:
    def __init__(self, path=None):
        if path:
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

            # initially empty attributes
            self.spectra = []

    def extract_spectrum(self, sky_aps):

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