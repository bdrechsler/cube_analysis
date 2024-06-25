import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import photutils.aperture as ap
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
        :attr:`pixel_aps` (list[photutils apertures]):
            List of pixel apertures used to extract spectra
        :attr: `sky_aps` (list[photutils sky apertures]):
            List of sky apertures used to extract spectra
    
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

        self.collapsed_img = np.nanmedian(self.flux.value, axis=0)
        self.collapsed_spec = np.nansum(self.flux.value, axis=(1,2))

    def extract_spectrum(self, aperture_list):

        r"""
        Extract 1D spectrum from a list of apertures

        Args:
            :attr:`params_list` (list[photutils apertures (sky or pixel)])
                Parameters to define apertures
        
        """

        for aperture in aperture_list:
            if isinstance(aperture, ap.EllipticalAperture) or isinstance(aperture, ap.CircularAperture):
                self.pixel_aps.append(aperture)
                self.sky_aps.append(aperture.to_sky(self.wcs.celestial))
            elif isinstance(aperture, ap.SkyEllipticalAperture) or isinstance(aperture, ap.SkyCircularAperture):
                self.pixel_aps.append(aperture.to_pixel(self.wcs))
                self.sky_aps.append(aperture)
        
        spec_list = []
        for pix_ap in self.pixel_aps:

            # create a mask
            mask = pix_ap.to_mask(method='exact')

            # initialize spectrum array
            spectrum_flux = np.zeros(len(self.wvl))

            # extract the 1D spectrum
            for i in range(len(spectrum_flux)):
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

    def __add__(self, cube2, corr_box_dims=(4, 7)):
        r"""
        Align then combine two spectral cubes:

        Args:
            :attr:`cube2` (Cube)
                Second cube to combine with the first
        Returns:
            :attr:`combined_cube` (Cube)
                averaged spectral cubes after alignment
        """

        # take the cross correlation
        corr = fftconvolve(np.nan_to_num(self.collapsed_img), np.nan_to_num(cube2.collapsed_img[::-1, ::-1]),
                           mode='same')
        
        # find index of peak of correlation
        peak_ind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        x_peak = peak_ind[1]
        y_peak = peak_ind[0]

        # create a box around peak of correlation
        size_x = corr_box_dims[1]
        size_y = corr_box_dims[0]
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

        combined_flux = np.copy(self.flux)

        # shift each channel
        for i in range(len(self.wvl)):
            shifted_img = shift(np.nan_to_num(cube2.flux[i].value), [y_offset, x_offset],
                                    cval=np.nan) * u.Jy
            # cut off borders off
            shifted_img[:4, :] = np.nan
            shifted_img[-4:, :] = np.nan
            shifted_img[:, :4] = np.nan
            shifted_img[:, -4:] = np.nan

            nan_inds_1 = np.where(np.isnan(self.flux[i]))
            nan_inds_2 = np.where(np.isnan(shifted_img))

            img_combined = np.nan_to_num(shifted_img) + np.nan_to_num(self.flux[i]) / 2.
            img_combined[nan_inds_2] = self.flux[i][nan_inds_2]
            img_combined[nan_inds_1] = np.nan
            combined_flux[i] = img_combined

            
        combined_cube = Cube()
        combined_cube.flux = combined_flux
        combined_cube.wvl = self.wvl
        combined_cube.header = self.header
        combined_cube.wcs = self.wcs
        combined_cube.collapsed_img = np.nanmedian(combined_flux.value, axis=0)
        combined_cube.collapsed_spec = np.nansum(combined_flux.value, axis=(1, 2))
        
        return combined_cube


        
