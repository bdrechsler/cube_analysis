import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

def map_grid(maps, map_type, center=(69.9744522, 26.0526275), width = 0.00119):
    
    plt.close()
    fig = plt.figure(4, 3, dpi=300, figsize=(12,14))
    grid = fig.add_gridspec(4, 3, hspace=0., wspace=0.2)

    # get indicies of plots in the grid
    row, col = np.indices((4, 3))
    row_inds = row.flatten()
    col_inds = col.flatten()

    for i in range(maps):
        map = maps[i]
        line_map = map.line_map * 1e6
        cont_map = map.cont_map * 1e6
        ratio_map = map.ratio_map * 1e6
        line_name = map.line.plot_name
        if map_type == "line":
            plot_map = line_map
        elif map_type == "ratio":
            plot_map = ratio_map
        
        wcs = map.wcs.celestial
        header = wcs.to_header()

        # get map shape
        ny, nx = plot_map.shape
        # estimate sigma for contours
        sigma = np.nanstd(cont_map)
        # contour levels
        levels = [sigma, 3*sigma, 6*sigma]

        # get au per pixel for scale bar
        deg_per_pixel = header['CDELT1']
        arcsec_per_pix = (deg_per_pixel * u.deg).to(u.arcsec)
        au_per_pix = arcsec_per_pix * 140. # multiply by distance in pc
        pix_in_100au = 100 / au_per_pix

        # use percentile to get image stretch
        plot_map_flat = plot_map.flatten()
        plot_map_flat = plot_map_flat[~np.isnan(plot_map_flat)] # ignore nans
        vmin = np.percentile(plot_map_flat, 0.5)
        vmax = np.perentile(plot_map_flat, 99.5)

        # get center and width of plot in pixel coords
        center_sky = SkyCoord(center[0] * u.deg, center[1] * u.deg, fname='icrs')
        center_pix = wcs.world_to_pixel(center_sky)
        width_pix = width / deg_per_pixel

        # get bounds of field of view in pixel coords
        left = center_pix[0] - (0.5* width_pix)
        right = center_pix[0] + (0.5* width_pix)
        down = center_pix[1] - (0.5* width_pix)
        up = center_pix[1] + (0.5* width_pix)

        # add subplot to figure
        ax = fig.add_subplot(grid[row_inds[i], col_inds[i]])
        # plot the map
        im = ax.imshow(np.nan_to_num(plot_map), origin="lower", cmap="magma", 
                       vmin=vmin, vmax=vmax)
        # add continuum contours
        ax.contour(cont_map, levels=levels, colors="white", alpha=0.7,
                   linewidths=1.2)
        # set the FOV
        ax.set_xlim(left, right)
        ax.set_ylim(down, up)

        # add the colorbar
        plt.colorbar(im, use_gridspec=True, fraction=0.046, pad=0.04)

        # transform from axes to data units
        inv = ax.transLimits.inverted()
        # starting position of scalebar
        scalex1, scaley1 = inv.transform((0.6, 0.1))
        # ending position of scalebar
        scalex2 = scalex1 + pix_in_100au

        # plot the scale bar
        scale_x = np.linspace(scalex1, scalex2, 10)
        scale_y = [scaley1] * len(scale_x)
        ax.plot(scale_x, scale_y, color="white")

        # add text to scale bar
        ax.text(0.6, 0.13, "100 AU", transform=plt.gca().transAxes,
                color="white", fontsize=12)
        
        # add line name to plot
        ax.text(0.07, 0.92, line_name, ha="left", va="top",
                transform = plt.gca().transAxes, color="white", 
                fontsize=15)
        
        # get ra and dec
        ra = ax.coords[0]
        dec = ax.coords[1]

        # set where tick labels should be shown
        if row_inds[i] == 3:
            ra.set_axislabel("RA", minpad=0.5, fontsize=10)
        else:
            ra.set_ticklabel_visible(False)
        if col_inds[i] == 0:
            dec.set_axislabel("DEC", minpad=-0.5, fontsize=10)
        else:
            dec.set_ticklabel_visible(False)
        
        # set ticklabel fontsize and number of ticks
        ra.set_ticklabel(fontsize=8)
        dec.set_ticklabel(fontsize=8)
        ra.set_ticks(number=4)
        dec.set_ticks(number=4)

    plt.show()