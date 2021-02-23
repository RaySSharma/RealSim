#!/bin/env python

"""
The statistical observational realism suite presented in Bottrell et al 2019b. If you use this suite in your research or for any other purpose, I would appreciate a citation. If you have questions or suggestions on how to improve or broaden the suite, please contact me at cbottrel "at" uvic "dot" ca.

Version 0.3

Update History:
(v0_1) - February-2016 Correction to the way that the poisson noise is handled. Changed incorrect float padding to a sampling of a true Poisson distribution with correct implementation of the Gain quantity. SkyServer 'sky' and 'skysig' quantities are added to the header keywords when using real SDSS images.
(v0_2) - January-2019 Spec2SDSS_gri now incorporates the redshift (factor of (1+z)**5) to the wavelength specific intensities when generating the bandpass AB surface brightnesses. The redshift factor is now removed from intensity scaling step in this version.
(v0_3) - January-2019 Computes ra,dec from the source mask when determining image position. This avoids image registration offsets in row and column numbers for each band. Prepared for public release.
"""

import numpy as np
from astropy.io import fits


def ObsRealism(
    inputName,
    candels_args={
        "candels_field": "GN",  # candels field
        "candels_ra": 236.1900,  # ra for image centroid
        "candels_dec": -0.9200,  # dec for image centroid
    },
):

    """
    Add realism to idealized unscaled image.
    """

    # img header and data
    hdu = fits.open(inputName, "append")
    img_data = hdu["MockImage"].data
    header = hdu["MockImage"].header

    # add image to real sdss sky
    """
    Extract field from galaxy survey database using
    effectively weighted by the number of galaxies in
    each field. For this to work, the desired field
    mask should already have been generated and the
    insertion location selected.
    """
    ra = candels_args["candels_ra"]
    dec = candels_args["candels_dec"]
    colc = candels_args["candels_colc"]
    rowc = candels_args["candels_rowc"]
    real_im = candels_args["candels_im"]
    field = candels_args['candels_field']

    # convert to integers
    colc, rowc = int(np.around(colc)), int(np.around(rowc))

    # add real sky pixel by pixel to image in nanomaggies
    corr_ny, corr_nx = real_im.shape
    ny, nx = img_data.shape
    for xx in range(nx):
        for yy in range(ny):
            corr_x = int(colc - nx / 2 + xx)
            corr_y = int(rowc - ny / 2 + yy)
            if (
                corr_x >= 0
                and corr_x <= corr_nx - 1
                and corr_y >= 0
                and corr_y <= corr_ny - 1
            ):
                img_data[yy, xx] += real_im[corr_y, corr_x]

    hdu_out = fits.ImageHDU(img_data, header=header)
    hdu_out.header["EXTNAME"] = "RealSim"
    hdu_out.header["CANDELS_RA"] = float(ra)
    hdu_out.header["CANDELS_DEC"] = float(dec)

    hdu.append(hdu_out)
    hdu.flush()

    hdu_out = fits.ImageHDU(real_im, header=header)
    hdu_out.header["EXTNAME"] = "Real"
    hdu_out.header["CANDELS_RA"] = float(ra)
    hdu_out.header["CANDELS_DEC"] = float(dec)
    hdu_out.header['CANDELS_FIELD'] = field

    hdu.append(hdu_out)
    hdu.flush()
    hdu.close()


"""
Script executions start here. This version grabs a corrected image
based on a basis set of galaxies from a database, runs source 
extractor to produce a mask, and selects the location in which to
place the image in the SDSS sky. The final science cutout includes
PSF blurring (real SDSS), SDSS sky from the corrected image, and 
Poisson noise added. The final image is in nanomaggies.
"""

# get field,column,row from database data
def rrcf_radec(field_info, input_dir, cutout_size, field_name, pixsize):
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D

    # choose from a random CANDELS field
    if field_name is None:
        field_name = np.random.choice(list(field_info.keys()))

    catalog = fits.getdata(input_dir + "Catalogs/" + field_info[field_name][0], 1)
    field, field_header = fits.getdata(
        input_dir + "Fields/" + field_info[field_name][1], header=True
    )

    # filter only non-contaminated sources
    catalog = catalog[(catalog['FLAGS'] == 0) & (catalog['WFC3_F160W_FLUX'] > 0)]

    segmap_found = False
    # Loop through the catalog until a field is found with a viable segmap.
    while not segmap_found:
        # randomly select from basis set
        index = np.random.randint(low=0, high=len(catalog) - 1)

        position = catalog["X_image"][index], catalog["Y_image"][index]
        size = (cutout_size, cutout_size)
        im = Cutout2D(field, position, size).data

        # convert to counts nanojanskies, where PHOTFNU is inverse sensitivity in units Jy*sec/electron
        im *= field_header["PHOTFNU"] * 1e9
        try:
            segmap = detect_sources(im, pixsize=pixsize)
            segmap_found = True
        except TypeError as err:
            segmap_found = False
            continue
        ## Run photutils

    # get info from mask header
    mask_nx, mask_ny = segmap.shape
    # define an initial pixel location by row,col
    colc = np.random.randint(low=int(0.1 * mask_nx), high=int(0.9 * mask_nx))
    rowc = np.random.randint(low=int(0.1 * mask_ny), high=int(0.9 * mask_ny))
    # iterate until the pixel location does not overlap with existing source
    while segmap[rowc, colc] != 0:
        colc = np.random.randint(low=int(0.1 * mask_nx), high=int(0.9 * mask_nx))
        rowc = np.random.randint(low=int(0.1 * mask_ny), high=int(0.9 * mask_ny))
    # get wcs mapping
    w = WCS(field_header)
    # determine ra,dec to prevent image registration offsets in each band
    ra, dec = w.wcs_pix2world(colc, rowc, 1, ra_dec_order=True)
    return field_name, ra, dec, colc, rowc, im


def make_candels_args(field_info, input_dir="./", cutout_size=1024, field_name=None, pixsize=0.06):
    field_name, ra, dec, colc, rowc, im = rrcf_radec(field_info, input_dir, cutout_size, field_name, pixsize)
    candels_args = {
        "candels_field": field_name,  # candels field
        "candels_ra": ra,  # ra for image centroid
        "candels_dec": dec,  # dec for image centroid
        "candels_colc": colc,  # pixel x-coordinate for image centroid
        "candels_rowc": rowc,  # pixel y-coordinate for image centroid
        "candels_im": im,  # candels cutout
    }
    return candels_args


def detect_sources(image, pixsize):
    import photutils
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel

    # Run basic source detection

    # build kernel for pre-filtering.  How big?
    # don't assume redshift knowledge here
    typical_kpc_per_arcsec = 8.0

    kernel_kpc_fwhm = 5.0
    kernel_arcsec_fwhm = kernel_kpc_fwhm / typical_kpc_per_arcsec
    kernel_pixel_fwhm = kernel_arcsec_fwhm / pixsize

    sigma = kernel_pixel_fwhm * gaussian_fwhm_to_sigma
    nsize = int(5 * kernel_pixel_fwhm)
    kernel = Gaussian2DKernel(sigma, x_size=nsize, y_size=nsize)

    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(image, (50, 50), bkg_estimator=bkg_estimator)
    thresh = bkg.background + (5.0 * bkg.background_rms)
    
    segmap_obj = photutils.detect_sources(
        image, thresh, npixels=10, filter_kernel=kernel
    )

    segmap_obj = photutils.deblend_sources(
        image, segmap_obj, npixels=10, filter_kernel=kernel, nlevels=32, contrast=0.01
    )

    segmap = segmap_obj.data

    return segmap
