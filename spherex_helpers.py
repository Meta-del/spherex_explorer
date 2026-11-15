import numpy as np
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


BAD_FLAG_BITS = (0, 1, 2, 6, 7, 9, 10, 11, 15, 17, 19, 20, 22, 25, 26, 27, 28, 29)
PHOTOMETRY_BAD_FLAG_BITS = (2, 6, 9, 10, 11, 15, 17, 26, 27, 28, 29)
BACKGROUND_BAD_FLAG_BITS = PHOTOMETRY_BAD_FLAG_BITS + (0, 1, 19, 21)


@st.cache_data(show_spinner=False)
def load_cutout(url, ra, dec):
    with fits.open(url, memmap=False) as hdul:
        image_hdu = hdul["IMAGE"]
        image = image_hdu.data.astype(float)
        flags = hdul["FLAGS"].data.astype(np.int32)
        header = image_hdu.header

        sky_position = SkyCoord(ra=ra, dec=dec, unit="deg")
        spatial_wcs = WCS(header)
        x, y = spatial_wcs.world_to_pixel(sky_position)

        wavelength_wcs = WCS(
            header=header,
            fobj=hdul,
            key="W",
        )
        wavelength_wcs.sip = None
        wavelength, bandwidth = wavelength_wcs.pixel_to_world(x, y)

        metadata = {
            "detector": header.get("DETECTOR", "Unknown"),
            "units": header.get("BUNIT", "Unknown"),
            "image_date": header.get("DATE", "Unknown"),
            "psf_fwhm": header.get("PSF_FWHM", "Unknown"),
            "wavelength_um": float(wavelength.to_value(u.um)),
            "bandwidth_um": float(bandwidth.to_value(u.um)),
        }

    return image, flags, metadata, header.copy()


def flagged_pixels(flags, bits=BAD_FLAG_BITS):
    bad = np.zeros(flags.shape, dtype=bool)

    for bit in bits:
        bad |= (flags & (1 << bit)) != 0

    return bad


def observation_label(results, index):
    row = results[index]
    return f"{row['date_utc']} | {row['obs_id']}"
