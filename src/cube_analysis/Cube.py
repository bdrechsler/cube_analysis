import copy
import pickle
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import photutils.aperture as ap
from photutils.centroids import centroid_com
from scipy.signal import fftconvolve
from scipy.ndimage import shift

from .Spectrum import Spectrum
from .Maps import Maps

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
        :attr:`collapsed_spec` (spectrum):
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
        self.spectra = {}
        self.collapsed_img = []
        self.collapsed_spec = []
        self.pixel_aps = {}
        self.sky_aps = {}

    @classmethod
    def read_from_fits(cls, path):

        r"""
        Read in spectral cube data from a fits file to populate base attributes
        
        Args:
            :attr:`path` (str):
                The path to the fits file
        
        """
        
        cube = cls()

        # read in data from fits file
        data, header = fits.getdata(path, extname='SCI', header=True)
        # data shape is (spectral, y, x)

        # get wvl array from header info
        start_wvl = header['CRVAL3']
        wvl_step = header['CDELT3']
        nchan = header['NAXIS3']
        wvl = (np.arange(nchan) * wvl_step) + start_wvl

        # make sure units are in Jy
        flux_units = header['BUNIT']
        if flux_units == 'MJy/sr':
            # pixel area in sr
            pix_area = header['PIXAR_SR']
            data *= pix_area * 1e6
        

        # set cube attributes
        cube.flux = data * u.Jy
        cube.wvl = wvl * u.um
        cube.header = header
        cube.wcs = WCS(header)

        cube.collapsed_img = np.nanmedian(cube.flux.value, axis=0)
        cube.collapsed_spec = Spectrum(cube.wvl, np.nansum(cube.flux.value, axis=(1,2))*u.Jy)

        return cube
    
    @classmethod
    def load(cls, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    
    def write(self, fname):

        with open(fname, "wb") as f:
            pickle.dump(self, f)


    def extract_spectrum(self, aperture_dict):

        r"""
        Extract 1D spectrum from a dictionary of photutil apertures

        Args:
            :attr:`aperture_dict` (dict))
                Dictionary to define apertures used to extract spectra. In the format: {"name": aperture}
        
        """

        # iterate through all the apertures in the dictionary
        for name, aperture in aperture_dict.items():
            # check if provided aperture is a pixel or sky aperture and convert it accordingly
            if isinstance(aperture, ap.EllipticalAperture) or isinstance(aperture, ap.CircularAperture):
                self.pixel_aps[name] = aperture
                self.sky_aps[name] = aperture.to_sky(self.wcs.celestial)
            elif isinstance(aperture, ap.SkyEllipticalAperture) or isinstance(aperture, ap.SkyCircularAperture):
                self.pixel_aps[name] = aperture.to_pixel(self.wcs.celestial)
                self.sky_aps[name] = aperture
        
        # iterate through all of the pixel apertures
        for name, pix_ap in self.pixel_aps.items():
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

            # add to the dictionary of 1D spectra
            spectrum = Spectrum(self.wvl, spectrum_flux * u.Jy)
            self.spectra[name] = spectrum
        
        self.spectra["total"] = self.spectra["left"] + self.spectra["right"]


    def spectral_region(self, wvl1, wvl2, invert=False):
        r"""
        Extract a spectral sub region from the spectral cube

        Args:
            :attr:`wvl1` (astropy.units.quantity)
                lower wavelength of spectral region
            :attr:`wvl2` (astropy.units.aquantity)
                upper wavelength of spectral region
            :attr:`invert` (bool)
                if True, will consider the spectral region outside of given range
        
        """

        # find the indicies either inside (or outside) the provided range
        if invert:
            inds = np.where((self.wvl < wvl1) | (self.wvl > wvl2))
        else:
            inds = np.where((self.wvl >= wvl1) & (self.wvl <= wvl2))

        region_cube = copy.deepcopy(self)
        region_cube.flux = self.flux[inds]
        region_cube.wvl = self.wvl[inds]

        return region_cube
    
    def spectral_region_from_center(self, wvl, width, invert=False):
        
        wvl1 = wvl - (width/2)
        wvl2 = wvl + (width/2)

        if invert:
            inds = np.where((self.wvl < wvl1) | (self.wvl > wvl2))
        else:
            inds = np.where((self.wvl >= wvl1) & (self.wvl <= wvl2))

        region_cube = copy.deepcopy(self)
        region_cube.flux = self.flux[inds]
        region_cube.wvl = self.wvl[inds]

        return region_cube

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

        combined_cube = copy.deepcopy(self)
        combined_cube.flux = combined_flux
        combined_cube.collapsed_img = np.nanmedian(combined_flux.value, axis=0)
        combined_cube.collapsed_spec = np.nansum(combined_flux.value, axis=(1, 2))

        return combined_cube
    

    
    def create_maps(self, line, winsize=0.08*u.um, method="peak"):
        
        win_cube = self.spectral_region_from_center(line.wvl, winsize)

        ny, nx = self.collapsed_img.shape

        line_map = np.zeros((ny, nx))
        cont_map = np.zeros((ny, nx))

        for i in range(ny):
            for j in range(nx):
                spectrum = Spectrum(wvl=win_cube.wvl, flux=np.nan_to_num(win_cube.flux[:, i, j]))
                continuum = spectrum.fit_continuum(ignore_regions=[(line.wvl, line.lw)])
                cont_sub = spectrum - continuum
                
                cont_sub_just_line = cont_sub.spectral_region_from_center(line.wvl, line.lw)

                if np.isnan(spectrum.flux).all():
                    line_map[i, j] = np.nan
                    cont_map[i, j] = np.nan

                else:
                    if method == "peak":
                        line_map[i, j] = np.nanmax(cont_sub_just_line.flux.value)
                    elif method=="sum":
                        line_map[i, j] = np.nansum(cont_sub_just_line.flux.value)

                    cont_map[i, j] = np.nanmean(continuum.flux.value)
        
        return Maps(line, line_map, cont_map)
                




