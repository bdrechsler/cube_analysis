import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from photutils.aperture import EllipticalAperture, CircularAperture
from photutils.centroids import centroid_com
from scipy.signal import fftconvolve
from scipy.ndimage import shift

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
        self.pixel_aps = []
        self.sky_aps = []

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

        self.collapsed_img = np.nanmedian(self.flux, axis=0)
        self.collapsed_spec = np.nansum(self.flux, axis=(1,2))

    def extract_spectrum(self, params_list):

        r"""
        Extracted 1D spectrum from a list of apertures

        Args:
            :attr:`params_list` (list[dict])
                Parameters to define apertures
        
        """

        pixel_aps = []
        sky_aps = []

        for params in params_list:
            if type(params) == dict:
                if params["shape"] == "circle":
                    pixel_ap = CircularAperture(params["pos"], params["r"])
                if params["shape"] == "ellipse":
                    pixel_ap = EllipticalAperture(params['pos'], params['a'], params['b'], params['theta'])
                
                self.pixel_aps.append(pixel_ap)
                self.sky_aps.append(pixel_ap.to_sky(self.wcs.celestial))
                
            else:
                self.sky_aps.append(params)
                self.pixel_aps.append(params.to_pixel(self.wcs.celestial))

        spec_list = []
        for pix_ap in self.pixel_aps:

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

    def align(self, cube2, corr_box_dims=(4, 7)):
        
        # take the cross correlation
        corr = fftconvolve(np.nan_to_num(cube2.collapsed_img), np.nan_to_num(self.collapsed_img[::-1, ::-1]),
                           mode='same')
        
        # find index of peak of correlation
        peak_ind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        x_peak = peak_ind[1]
        y_peak = peak_ind[2]

        # create a box around peak of correlation
        size_x = corr_box_dims[0]
        size_y = corr_box_dims[1]
        corr_box = corr[y_peak - size_y: y_peak + size_y + 1,
                        x_peak - size_x: x_peak + size_x + 1]
        
        # take centroid of box to get sub-pixel position of peak
        box_cent = centroid_com(corr_box)

        # get location of peak in entire array
        x_peak = box_cent[0] + x_peak - size_x
        y_peak = box_cent[1] + y_peak - size_y
        
        # get coordinates of the center of the array
        x0 = (self.flux.shape[2] - 1) / 2.
        y0 = (self.flux.shape[1] - 1) / 2.

        # get offset from from center to get shift
        x_offset = x_peak - x0
        y_offset = y_peak - y0

        shifted_flux = np.copy(self.flux)

        # shift each channel
        for i in range(len(self.wvl)):
            shifted_flux[i] = shift(np.nan_to_num(self.flux[i]), [y_offset, x_offset],
                                    cval=np.nan)
        
        shifted_cube = Cube()
        shifted_cube.flux = shifted_flux
        shifted_cube.wvl = self.wvl
        shifted_cube.header = self.header
        shifted_cube.wcs = self.wcs
        shifted_cube.collapsed_img = np.nanmedian(shifted_flux, axis=0)
        shifted_cube.collapsed_spec = np.nansum(shifted_flux, axis=(1, 2))
        
        return shifted_cube
        


        
