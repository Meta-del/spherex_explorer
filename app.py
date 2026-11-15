import csv
import base64
import io
import socket
import time
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.error import HTTPError, URLError
from html import escape
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp

from spherex_helpers import (
    BACKGROUND_BAD_FLAG_BITS,
    PHOTOMETRY_BAD_FLAG_BITS,
    flagged_pixels,
    load_cutout,
    observation_label,
)
from spherex_ui import apply_styles, render_page_header


st.set_page_config(
    page_title="SPHEREx Sky Explorer",
    page_icon="🔭",
    layout="wide",
)


TARGETS = {
    "Custom coordinates": (304.693508808, 42.4436872991),
    "Barnard's Star (approx. 2026)": (269.4463, 4.7682),
    "Epsilon Indi B": (331.044146, -56.782844),
    "WISE 0855": (133.795129, -7.245147),
    "RW Aurigae": (76.956525, 30.401438),
    "R Draconis": (248.167611, 66.754979),
}

IRSA_ARCHIVE_GUIDE = (
    "https://caltech-ipac.github.io/spherex-archive-documentation/"
)
SPHEREX_MISSION = "https://spherex.caltech.edu/"
SPHEREX_CUBE_GUIDE = (
    "https://caltech-ipac.github.io/irsa-tutorials/spherex-hipscube/"
)


def render_loading_experience(
    stage,
    title,
    detail,
    *,
    progress_fraction=None,
    progress_label="",
    ra=None,
    dec=None,
    dates=(),
    wavelengths=(),
    preview_data=None,
    focus_blur=12,
    error=False,
):
    stage_order = {
        "coordinate": 0,
        "retrieve": 1,
        "retrieved": 1,
        "process": 2,
        "spectrum": 3,
        "sequence": 4,
        "ready": 4,
    }
    active_index = stage_order[stage]
    stage_labels = [
        "Coordinate scan",
        "Observations",
        "Image alignment",
        "Wavelengths",
        "Time sequence",
    ]
    stage_items = []
    for index, label in enumerate(stage_labels):
        state = "upcoming"
        if index < active_index or (stage in {"retrieved", "ready"} and index <= active_index):
            state = "done"
        elif index == active_index and stage != "retrieved":
            state = "active"
        stage_items.append(
            f'<span class="cosmic-step {state}"><i>{index + 1}</i>{escape(label)}</span>'
        )

    progress_html = ""
    if progress_fraction is not None:
        fraction = max(0.0, min(1.0, float(progress_fraction)))
        progress_html = (
            '<div class="cosmic-progress-wrap">'
            f'<div class="cosmic-progress"><span style="width:{fraction * 100:.3f}%"></span></div>'
            f'<div class="cosmic-progress-label">{escape(progress_label)}</div>'
            '</div>'
        )

    coordinates_html = ""
    if ra is not None and dec is not None:
        coordinates_html = (
            '<div class="cosmic-coordinates">'
            f'<span>RA {float(ra):.6f}°</span><b>·</b>'
            f'<span>Dec {float(dec):+.6f}°</span></div>'
        )

    date_values = [str(value).replace("T", " ")[:19] for value in dates]
    date_marks = []
    if len(date_values) <= 60:
        selected_dates = list(enumerate(date_values))
        marker_caption = f"{len(date_values)} returned observation date(s)"
    else:
        selected_indices = np.linspace(0, len(date_values) - 1, 60, dtype=int)
        selected_dates = [(int(index), date_values[int(index)]) for index in selected_indices]
        marker_caption = f"60 date markers sampled from {len(date_values)} returned observations"
    for index, date in selected_dates:
        position = 0 if len(date_values) <= 1 else index / (len(date_values) - 1) * 100
        date_marks.append(
            f'<span class="cosmic-date-mark" style="left:{position:.3f}%" title="{escape(date)}"></span>'
        )
    timeline_html = ""
    if date_values:
        timeline_html = (
            '<div class="cosmic-timeline-block">'
            '<div class="cosmic-mini-label">OBSERVATION TIMELINE</div>'
            f'<div class="cosmic-timeline" aria-label="{escape(marker_caption)}">'
            '<span class="cosmic-timeline-line"></span>'
            + "".join(date_marks)
            + '</div>'
            f'<div class="cosmic-timeline-ends"><span>{escape(date_values[0])}</span>'
            f'<span>{escape(date_values[-1])}</span></div>'
            f'<div class="cosmic-timeline-caption">{escape(marker_caption)}</div>'
            '</div>'
        )

    channel_marks = "".join(
        f'<i class="{"no-channel" if index in (51, 52) else ""}" title="Nominal channel {index}"></i>'
        for index in range(1, 103)
    )
    wavelength_values = [float(value) for value in wavelengths if np.isfinite(value)]
    wavelength_caption = "Wavelength samples appear as real FITS headers are read."
    if wavelength_values:
        wavelength_caption = (
            f"{len(wavelength_values)} real FITS wavelength sample(s) · "
            f"{min(wavelength_values):.3f}–{max(wavelength_values):.3f} μm"
        )
    wavelength_dots = ""
    if wavelength_values:
        for wavelength in wavelength_values:
            position = max(0.0, min(100.0, (wavelength - 0.75) / 4.25 * 100))
            wavelength_dots += (
                f'<span class="cosmic-wave-dot" style="left:{position:.2f}%" '
                f'title="{wavelength:.4f} microns"></span>'
            )

    preview_html = (
        f'<img class="cosmic-real-preview" src="data:image/png;base64,{preview_data}" '
        f'style="--focus-blur:{max(0, float(focus_blur)):.1f}px" alt="Actual SPHEREx image preview">'
        if preview_data
        else '<div class="cosmic-scan-target"><span></span><i></i><b></b></div>'
    )
    coordinate_label = (
        f"{float(ra):.6f}, {float(dec):+.6f}"
        if ra is not None and dec is not None
        else "Waiting for sky coordinates"
    )
    if error:
        error_class = " cosmic-error"
    else:
        error_class = ""
    receipt_class = " cosmic-results-receipt" if stage == "retrieved" else ""

    return dedent(f'''
    <section class="cosmic-loader{error_class}{receipt_class}" aria-live="polite">
        <div class="cosmic-loader-topline"><span class="cosmic-live-dot"></span>SPHEREx · LIVE DATA WORKFLOW</div>
        <div class="cosmic-loader-grid">
            <div class="cosmic-visual" aria-label="Astronomical image processing visualization">
                <div class="cosmic-visual-stars"></div>
                {preview_html}
                <div class="cosmic-reticle"><i></i><b></b><span></span></div>
                <div class="cosmic-visual-coordinate">{escape(coordinate_label)}</div>
            </div>
            <div class="cosmic-loader-copy">
                <div class="cosmic-mini-label">{escape("OBSERVATIONS RECEIVED" if stage == "retrieved" else stage_labels[active_index].upper())}</div>
                <h3>{escape(title)}</h3>
                <p>{escape(detail)}</p>
                {coordinates_html}
                {progress_html}
                <div class="cosmic-steps">{"".join(stage_items)}</div>
                {timeline_html}
                <div class="cosmic-spectrum-block">
                    <div class="cosmic-spectrum-heading"><span class="cosmic-mini-label">102-CHANNEL INSTRUMENT REFERENCE</span><span>51–52 blank</span></div>
                    <div class="cosmic-spectrum">{channel_marks}{wavelength_dots}</div>
                    <div class="cosmic-spectrum-caption">{escape(wavelength_caption)} · Colors are an instrument guide, not a progress meter.</div>
                </div>
            </div>
        </div>
    </section>
    ''')


def make_loading_preview(image):
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]
    if not finite.size:
        return None
    low, high = np.nanpercentile(finite, [5, 99])
    if high <= low:
        high = low + 1
    figure, axis = plt.subplots(figsize=(2.4, 1.8), dpi=80)
    figure.patch.set_facecolor("#05080d")
    axis.set_facecolor("#05080d")
    axis.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        cmap="magma",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    axis.axis("off")
    figure.tight_layout(pad=0)
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=80,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def query_sia_with_retries(*, attempts=3, **query_options):
    coordinate, radius = query_options["pos"]
    collections = query_options["collection"]
    if isinstance(collections, str):
        collections = [collections]
    required_columns = [
        "obs_id",
        "t_min",
        "t_max",
        "energy_bandpassname",
        "access_url",
        "obs_collection",
        "s_ra",
        "s_dec",
        "em_min",
        "em_max",
        "s_fov",
        "dataproduct_type",
    ]
    parameters = [
        ("COLLECTION", collection) for collection in collections
    ]
    parameters.extend(
        [
            (
                "POS",
                "circle "
                f"{coordinate.ra.deg:.8f} "
                f"{coordinate.dec.deg:.8f} "
                f"{radius.to_value(u.deg):.10f}",
            ),
            ("DPTYPE", str(query_options.get("data_type", "image"))),
            ("RESPONSEFORMAT", "CSV"),
        ]
    )

    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                "https://irsa.ipac.caltech.edu/SIA",
                params=parameters,
                timeout=(8, 60),
            )
            response.raise_for_status()
            reader = csv.reader(io.StringIO(response.text))
            header = next(reader, [])
            if not header or any(name not in header for name in required_columns):
                raise ValueError(
                    "IRSA SIA returned an unexpected CSV table."
                )
            positions = [header.index(name) for name in required_columns]
            rows = []
            for row in reader:
                if len(row) > max(positions):
                    rows.append([row[position] for position in positions])
            table = Table(rows=rows, names=required_columns)
            for name in (
                "t_min",
                "t_max",
                "s_ra",
                "s_dec",
                "em_min",
                "em_max",
                "s_fov",
            ):
                table[name] = np.asarray(table[name], dtype=float)
            return table
        except (requests.RequestException, StopIteration, ValueError) as error:
            last_error = error
            if attempt == attempts - 1:
                raise
            time.sleep(1.0 * (2 ** attempt))
    raise last_error


@st.cache_data(ttl=21600, show_spinner=False)
def query_reference_catalog(catalog_id, ra, dec, radius_arcmin):
    if catalog_id == "gaia_dr3_source":
        columns = "source_id,ra,dec,ref_epoch,pmra,pmdec,phot_g_mean_mag"
    elif catalog_id == "allwise_p3as_psd":
        columns = "designation,ra,dec,w1mpro,w2mpro"
    else:
        raise ValueError("Unsupported reference catalog.")
    response = requests.get(
        "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query",
        params={
            "catalog": catalog_id,
            "spatial": "cone",
            "radius": f"{float(radius_arcmin) * 60:.6f}",
            "radunits": "arcsec",
            "objstr": f"{float(ra):.8f} {float(dec):.8f}",
            "outfmt": "1",
            "outrows": "1500",
            "selcols": columns,
        },
        timeout=(8, 35),
    )
    response.raise_for_status()
    return ascii.read(response.text, format="ipac")


def catalog_positions_at_epoch(table, catalog_id, observation_time):
    ra_values = np.asarray(table["ra"], dtype=float)
    dec_values = np.asarray(table["dec"], dtype=float)
    positions = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
    if catalog_id != "gaia_dr3_source" or observation_time is None:
        return positions

    required = ("pmra", "pmdec", "ref_epoch")
    if not all(name in table.colnames for name in required):
        return positions

    pmra = np.ma.asarray(table["pmra"], dtype=float).filled(np.nan)
    pmdec = np.ma.asarray(table["pmdec"], dtype=float).filled(np.nan)
    epochs = np.ma.asarray(table["ref_epoch"], dtype=float).filled(np.nan)
    valid = (
        np.isfinite(ra_values)
        & np.isfinite(dec_values)
        & np.isfinite(pmra)
        & np.isfinite(pmdec)
        & np.isfinite(epochs)
    )
    if np.any(valid):
        moving_positions = SkyCoord(
            ra=ra_values[valid] * u.deg,
            dec=dec_values[valid] * u.deg,
            pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
            pm_dec=pmdec[valid] * u.mas / u.yr,
            obstime=Time(epochs[valid], format="jyear"),
            frame="icrs",
        ).apply_space_motion(new_obstime=observation_time)
        ra_values = ra_values.copy()
        dec_values = dec_values.copy()
        ra_values[valid] = moving_positions.ra.deg
        dec_values[valid] = moving_positions.dec.deg
        positions = SkyCoord(ra=ra_values * u.deg, dec=dec_values * u.deg)
    return positions


def parse_shared_search_params():
    defaults = {
        "ra": TARGETS["Custom coordinates"][0],
        "dec": TARGETS["Custom coordinates"][1],
        "fov_arcmin": 6,
    }
    try:
        ra = float(st.query_params.get("ra", defaults["ra"]))
        dec = float(st.query_params.get("dec", defaults["dec"]))
        fov = int(float(st.query_params.get("fov_arcmin", defaults["fov_arcmin"])))
    except (TypeError, ValueError):
        return defaults, ""
    if not (0 <= ra <= 360 and -90 <= dec <= 90 and 2 <= fov <= 20):
        return defaults, ""
    collections = str(st.query_params.get("collections", ""))
    return {"ra": ra, "dec": dec, "fov_arcmin": fov}, collections


def load_cutout_with_retries(url, ra, dec, attempts=3):
    retryable_statuses = {429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(attempts):
        try:
            return load_cutout(url, ra, dec)
        except HTTPError as error:
            last_error = error
            if error.code not in retryable_statuses or attempt == attempts - 1:
                raise
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as error:
            last_error = error
            if attempt == attempts - 1:
                raise

        time.sleep(0.8 * (2**attempt))

    raise last_error


@st.cache_data(ttl=3600, show_spinner=False)
def available_image_collections():
    table = Irsa.list_collections(servicetype="SIA", filter="spherex")
    candidates = ["collection", "collection_name", "collection_id"]
    column = next(
        (name for name in candidates if name in table.colnames),
        None,
    )

    if column is None:
        for name in table.colnames:
            values = [str(value).strip() for value in table[name]]
            if values and all(value.startswith("spherex_") for value in values):
                column = name
                break

    if column is None:
        raise ValueError("IRSA did not return a recognizable collection list.")

    names = sorted(
        {
            str(value).strip()
            for value in table[column]
            if str(value).strip().startswith("spherex_")
            and not str(value).strip().endswith("_cube")
        }
    )
    if not names:
        raise ValueError("IRSA returned no SPHEREx image collections.")
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return names, checked_at


def clear_available_collections_cache():
    available_image_collections.clear()


def collection_profiles(names):
    profiles = {}

    def release_number(name, marker):
        suffix = name.split(marker, 1)[1]
        digits = "".join(
            character for character in suffix if character.isdigit())
        return int(digits or 0)

    wide_names = [name for name in names if not name.endswith("_deep")]
    deep_names = [name for name in names if name.endswith("_deep")]
    latest_dr = max(
        (name for name in wide_names if "_dr" in name),
        key=lambda name: release_number(name, "_dr"),
        default=None,
    )
    latest_qr = max(
        (name for name in wide_names if "_qr" in name),
        key=lambda name: release_number(name, "_qr"),
        default=None,
    )
    recommended = [name for name in (latest_dr, latest_qr) if name]

    if recommended:
        release_labels = [
            name.removeprefix("spherex_").upper() for name in recommended
        ]
        label = "Latest standard releases: " + " + ".join(release_labels)
        profiles[label] = recommended

    all_collections = sorted(set(wide_names + deep_names))
    if all_collections and all_collections != recommended:
        profiles["All discovered image collections"] = all_collections

    for name in names:
        short_name = name.removeprefix("spherex_").upper()
        profiles[f"{short_name} only"] = [name]

    return profiles


def reset_animation():
    st.session_state["playing"] = False
    st.session_state["prepared_signature"] = None
    st.session_state["prepared_frames"] = []
    st.session_state["prepared_metadata"] = []
    st.session_state["prepared_stretch"] = None
    st.session_state["prepared_reference_pixel"] = None
    st.session_state["prepared_wcs_header"] = None
    st.session_state["prepared_mean_stack"] = None
    st.session_state["preload_failures"] = []
    st.session_state["preload_failure_signature"] = None


def apply_target():
    target_name = st.session_state["target"]

    if target_name != "Custom coordinates":
        st.session_state["ra_input"], st.session_state["dec_input"] = (
            TARGETS[target_name]
        )

    st.session_state["results"] = None
    st.session_state["search_context"] = None
    st.session_state["frame_index"] = 0
    st.session_state["start_index"] = 0
    st.session_state["workspace"] = "Search & Results"
    reset_animation()


def choose_band():
    st.session_state["frame_index"] = 0
    st.session_state["start_index"] = 0
    reset_animation()


def choose_start_frame():
    st.session_state["frame_index"] = st.session_state["start_index"]
    st.session_state["playing"] = False


def navigate_workspace(destination):
    current = st.session_state.get("workspace", "Search & Results")
    if current != destination:
        st.session_state["previous_workspace"] = current
        st.session_state["workspace"] = destination
        st.session_state["last_workspace"] = destination


def remember_workspace_change():
    current = st.session_state.get("workspace", "Search & Results")
    previous = st.session_state.get("last_workspace", "Search & Results")
    if current != previous:
        st.session_state["previous_workspace"] = previous
    st.session_state["last_workspace"] = current


def return_to_previous_workspace():
    destination = st.session_state.get(
        "previous_workspace", "Search & Results"
    )
    navigate_workspace(destination)


def make_cutout_url(image_url, ra, dec, size_deg):
    separator = "&" if "?" in image_url else "?"
    return f"{image_url}{separator}center={ra},{dec}&size={size_deg}"


@st.cache_data(show_spinner=False)
def load_variance_cutout(url):
    with fits.open(url, memmap=False) as hdul:
        return np.asarray(hdul["VARIANCE"].data, dtype=float)


@st.cache_data(show_spinner=False, max_entries=2)
def load_cube_channel(url, channel_number):
    with fits.open(url, memmap=True, cache=True) as hdul:
        image_hdu = hdul["IMAGE"]
        hits_hdu = hdul["NHITS"]
        flags_hdu = hdul["FLAGS"]
        image = np.asarray(
            image_hdu.data[channel_number - 1], dtype=np.float32
        ).copy()
        hits = np.asarray(
            hits_hdu.data[channel_number - 1], dtype=np.int32
        ).copy()
        flags = np.asarray(
            flags_hdu.data[channel_number - 1], dtype=np.int32
        ).copy()
        wavelength = float(
            hdul["SPECTRAL_CHANNELS"].data["WAVELENGTH"][channel_number - 1]
        )
        image_wcs = WCS(image_hdu.header, fobj=hdul).celestial
        wcs_header = image_wcs.to_header().tostring(sep="\n")

    return image, hits, flags, wavelength, wcs_header


@st.cache_data(show_spinner=False, max_entries=1)
def extract_cube_quick_spectrum(url, ra, dec, aperture_radius, annulus_inner, annulus_outer):
    wavelengths = []
    fluxes_mjy = []
    flagged_percent = []

    with fits.open(url, memmap=True, cache=True) as hdul:
        cube = hdul["IMAGE"].data
        flags_cube = hdul["FLAGS"].data
        wavelengths = np.asarray(
            hdul["SPECTRAL_CHANNELS"].data["WAVELENGTH"], dtype=float
        )
        wcs = WCS(hdul["IMAGE"].header, fobj=hdul).celestial
        image_units = u.Unit(hdul["IMAGE"].header.get("BUNIT", "MJy / sr"))

        x, y = wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg"))
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(
                "The searched coordinate is outside the cube WCS.")

        first = wcs.pixel_to_world(x, y)
        right = wcs.pixel_to_world(x + 1, y)
        up = wcs.pixel_to_world(x, y + 1)
        east_x, north_x = first.spherical_offsets_to(right)
        east_y, north_y = first.spherical_offsets_to(up)
        pixel_area_sr = abs(
            east_x.to_value(u.rad) * north_y.to_value(u.rad)
            - north_x.to_value(u.rad) * east_y.to_value(u.rad)
        )

        outer = int(np.ceil(annulus_outer))
        y_size, x_size = cube.shape[1:]
        x0 = max(0, int(np.floor(x - outer)))
        x1 = min(x_size, int(np.ceil(x + outer)) + 1)
        y0 = max(0, int(np.floor(y - outer)))
        y1 = min(y_size, int(np.ceil(y + outer)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_squared = (xx - x) ** 2 + (yy - y) ** 2
        aperture = distance_squared <= aperture_radius**2
        annulus = (
            (distance_squared >= annulus_inner**2)
            & (distance_squared <= annulus_outer**2)
        )

        for channel_index in range(cube.shape[0]):
            if channel_index in (50, 51):
                fluxes_mjy.append(np.nan)
                flagged_percent.append(100.0)
                continue

            image = np.asarray(cube[channel_index, y0:y1, x0:x1], dtype=float)
            flags = np.asarray(
                flags_cube[channel_index, y0:y1, x0:x1], dtype=np.int32
            )
            valid = np.isfinite(image) & ~flagged_pixels(flags)
            aperture_valid = aperture & valid
            annulus_valid = annulus & valid
            aperture_fraction = (
                np.count_nonzero(aperture_valid)
                / max(1, np.count_nonzero(aperture))
            )

            flagged_percent.append(100 * (1 - aperture_fraction))
            if (
                aperture_fraction < 0.8
                or np.count_nonzero(annulus_valid) < 10
            ):
                fluxes_mjy.append(np.nan)
                continue

            background = float(np.nanmedian(image[annulus_valid]))
            net_sum = float(np.sum(image[aperture_valid] - background))
            flux_mjy = (
                net_sum * image_units * pixel_area_sr * u.sr
            ).to_value(u.mJy)
            fluxes_mjy.append(float(flux_mjy))

    return wavelengths, fluxes_mjy, flagged_percent


def aperture_measurement(image, flags, variance, header, ra, dec, radius):
    image = np.asarray(image, dtype=float)
    flags = np.asarray(flags)
    variance = np.asarray(variance, dtype=float)
    wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg"))

    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("The coordinate could not be mapped into this image.")
    if variance.shape != image.shape or flags.shape != image.shape:
        raise ValueError("IMAGE, FLAGS, and VARIANCE dimensions do not match.")

    inner_radius = radius + 2
    outer_radius = radius + 5
    y_size, x_size = image.shape
    x0 = max(0, int(np.floor(x - outer_radius)))
    x1 = min(x_size, int(np.ceil(x + outer_radius)) + 1)
    y0 = max(0, int(np.floor(y - outer_radius)))
    y1 = min(y_size, int(np.ceil(y + outer_radius)) + 1)

    if x0 >= x1 or y0 >= y1:
        raise ValueError("The coordinate falls outside the image cutout.")

    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_squared = (xx - x) ** 2 + (yy - y) ** 2
    aperture = distance_squared <= radius**2
    annulus = (
        (distance_squared >= inner_radius**2)
        & (distance_squared <= outer_radius**2)
    )

    image_crop = image[y0:y1, x0:x1]
    flags_crop = flags[y0:y1, x0:x1]
    variance_crop = variance[y0:y1, x0:x1]
    finite_image = np.isfinite(image_crop)
    valid_variance = np.isfinite(variance_crop) & (variance_crop > 0)
    aperture_valid = (
        aperture
        & finite_image
        & valid_variance
        & ~flagged_pixels(flags_crop, PHOTOMETRY_BAD_FLAG_BITS)
    )
    annulus_valid = (
        annulus
        & finite_image
        & valid_variance
        & ~flagged_pixels(flags_crop, BACKGROUND_BAD_FLAG_BITS)
    )
    aperture_coverage = (
        np.count_nonzero(aperture_valid) / max(1, np.count_nonzero(aperture))
    )

    if aperture_coverage < 0.8:
        aperture_count = max(1, int(np.count_nonzero(aperture)))
        flagged_count = int(
            np.count_nonzero(
                aperture
                & flagged_pixels(flags_crop, PHOTOMETRY_BAD_FLAG_BITS)
            )
        )
        invalid_image_count = int(
            np.count_nonzero(aperture & ~np.isfinite(image_crop))
        )
        invalid_variance_count = int(
            np.count_nonzero(
                aperture
                & np.isfinite(image_crop)
                & (~np.isfinite(variance_crop) | (variance_crop <= 0))
            )
        )
        raw_flags = np.asarray(flags_crop[aperture], dtype=np.uint32)
        set_bits = [
            f"{bit}:{int(np.count_nonzero(raw_flags & (1 << bit)))}"
            for bit in range(32)
            if np.any(raw_flags & (1 << bit))
        ]
        bit_summary = ", ".join(set_bits) if set_bits else "none"
        raise ValueError(
            f"Only {aperture_coverage * 100:.0f}% of the aperture is usable "
            f"({np.count_nonzero(aperture_valid)}/{aperture_count} pixels). "
            f"Rejected by quality flags: {flagged_count}; non-finite image: "
            f"{invalid_image_count}; invalid variance: {invalid_variance_count}. "
            f"Flag bits present (bit:pixel count): {bit_summary}."
        )
    if np.count_nonzero(annulus_valid) < 10:
        raise ValueError(
            "Too few usable pixels remain in the background annulus.")

    background = float(np.nanmedian(image_crop[annulus_valid]))
    aperture_values = image_crop[aperture_valid]
    net_pixel_sum = float(np.sum(aperture_values - background))
    n_aperture = int(np.count_nonzero(aperture_valid))
    n_background = int(np.count_nonzero(annulus_valid))
    background_variance = float(np.nanmedian(variance_crop[annulus_valid]))
    total_variance = float(np.sum(variance_crop[aperture_valid]))
    total_variance += n_aperture**2 * background_variance / n_background
    net_pixel_error = float(np.sqrt(max(0, total_variance)))

    pixel_area_sr = (
        proj_plane_pixel_area(wcs) * u.deg**2
    ).to_value(u.sr)
    image_units = str(header.get("BUNIT", "Unknown"))
    normalized_units = image_units.lower().replace(" ", "")

    if "mjy/sr" in normalized_units or "mjy*sr" in normalized_units:
        flux = net_pixel_sum * pixel_area_sr * 1e12
        uncertainty = net_pixel_error * pixel_area_sr * 1e12
        flux_units = "μJy"
    else:
        flux = net_pixel_sum
        uncertainty = net_pixel_error
        flux_units = f"{image_units} × pixels"

    return {
        "flux": flux,
        "uncertainty": uncertainty,
        "flux_units": flux_units,
        "aperture_coverage_percent": 100 * aperture_coverage,
        "background": background,
        "pixel_area_sr": pixel_area_sr,
        "x": float(x),
        "y": float(y),
    }


def detect_bright_peaks(image, sigma_threshold, max_sources=100, reference_pixel=None):
    image = np.asarray(image, dtype=float)
    valid_values = image[np.isfinite(image)]
    if valid_values.size < 9:
        return []
    background = float(np.nanmedian(valid_values))
    noise = float(1.4826 * np.nanmedian(np.abs(valid_values - background)))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.nanstd(valid_values))
    if not np.isfinite(noise) or noise <= 0:
        return []

    working = np.where(np.isfinite(image), image, -np.inf)
    is_peak = np.isfinite(image) & (image > background + sigma_threshold * noise)
    height, width = image.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = np.full_like(working, -np.inf)
            sy0, sy1 = max(0, -dy), min(height, height - dy)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            ty0, ty1, tx0, tx1 = sy0 + dy, sy1 + dy, sx0 + dx, sx1 + dx
            neighbor[ty0:ty1, tx0:tx1] = working[sy0:sy1, sx0:sx1]
            is_peak &= working >= neighbor

    peak_pixels = np.argwhere(is_peak)
    peak_pixels = sorted(peak_pixels, key=lambda pos: image[pos[0], pos[1]], reverse=True)
    selected, candidates = [], []
    for y, x in peak_pixels:
        if x < 2 or y < 2 or x >= width - 2 or y >= height - 2:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 2.5**2 for py, px in selected):
            continue
        selected.append((int(y), int(x)))
        patch = image[y - 2:y + 3, x - 2:x + 3]
        finite = np.isfinite(patch)
        local_background = float(np.nanmedian(patch[finite]))
        weights = np.where(finite, np.maximum(patch - local_background, 0), 0)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            yy, xx = np.mgrid[y - 2:y + 3, x - 2:x + 3]
            cx, cy = float(np.sum(xx * weights) / weight_sum), float(np.sum(yy * weights) / weight_sum)
        else:
            cx, cy = float(x), float(y)
        candidates.append({
            "x_pixel": cx,
            "y_pixel": cy,
            "peak_value": float(image[y, x]),
            "approx_snr": float((image[y, x] - background) / noise),
        })

    if reference_pixel is not None:
        ref_x, ref_y = reference_pixel
        for candidate in candidates:
            candidate["search_offset_px"] = float(np.hypot(candidate["x_pixel"] - ref_x, candidate["y_pixel"] - ref_y))
        candidates.sort(key=lambda candidate: (candidate["search_offset_px"], -candidate["approx_snr"]))
    else:
        candidates.sort(key=lambda candidate: candidate["approx_snr"], reverse=True)
    return candidates[:max_sources]


def link_candidate_track(detections_by_frame, mjds, anchor_index, anchor_candidate, max_step_px):
    track = {anchor_index: anchor_candidate}
    for direction in (-1, 1):
        indices = range(anchor_index - 1, -1, -1) if direction < 0 else range(anchor_index + 1, len(mjds))
        history = [anchor_index]
        for index in indices:
            prior_index = history[-1]
            prior = track[prior_index]
            predicted_x, predicted_y = float(prior["x_pixel"]), float(prior["y_pixel"])
            if len(history) >= 2:
                earlier_index = history[-2]
                earlier = track[earlier_index]
                elapsed = float(mjds[prior_index] - mjds[earlier_index])
                target_elapsed = float(mjds[index] - mjds[prior_index])
                if elapsed != 0:
                    predicted_x += ((prior["x_pixel"] - earlier["x_pixel"]) / elapsed) * target_elapsed
                    predicted_y += ((prior["y_pixel"] - earlier["y_pixel"]) / elapsed) * target_elapsed
            ranked = sorted(
                [
                    (float(np.hypot(c["x_pixel"] - predicted_x, c["y_pixel"] - predicted_y)), c)
                    for c in detections_by_frame[index]
                ],
                key=lambda item: item[0],
            )
            if not ranked:
                continue
            distance, nearest = ranked[0]
            ambiguous = len(ranked) > 1 and ranked[1][0] <= max_step_px and ranked[1][0] - distance < 0.5
            if distance <= max_step_px and not ambiguous:
                track[index] = nearest
                history.append(index)
    return dict(sorted(track.items()))


def calendar_date(value):
    return str(value).strip().replace("T", " ", 1).split()[0]


def make_observation_table(table):
    display_columns = [
        name
        for name in [
            "obs_id",
            "date_utc",
            "energy_bandpassname",
            "obs_collection",
            "access_url",
        ]
        if name in table.colnames
    ]

    dataframe = table[display_columns].to_pandas()
    column_config = {}

    if "access_url" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"access_url": "FITS product"}
        )
        column_config["FITS product"] = st.column_config.LinkColumn(
            "Original IRSA FITS product",
            display_text="Download FITS",
            help=(
                "Opens the original SPHEREx product from IRSA. "
                "Full FITS files can be large."
            ),
        )

    return dataframe, column_config


def render_cube_explorer(search_context):
    with st.expander("Explore the 102-channel DR1 spectral cube"):
        st.write(
            "The DR1 all-sky cube contains 102 wavelength channels. "
            "IRSA serves complete sky tiles rather than a small coordinate "
            "cutout here, so this app finds the real tile but does not "
            "download its large FITS file automatically."
        )
        st.caption(
            "Channels 51 and 52 are blank in the published cube. A cube tile "
            "is useful for spectral exploration, but it is not a source "
            "spectrum or a detection by itself."
        )

        if st.button(
            "Find the cube tile covering these coordinates",
            key="find_cube_tile",
        ):
            cube_coord = SkyCoord(
                ra=float(search_context["ra"]),
                dec=float(search_context["dec"]),
                unit="deg",
            )
            try:
                with st.spinner("Cassiopeia is locating the DR1 cube tile in IRSA…"):
                    st.session_state["cube_results"] = Irsa.query_sia(
                        pos=(cube_coord, 1 * u.arcsec),
                        collection="spherex_dr1_cube",
                        data_type="cube",
                    )
                st.session_state["cube_results_context"] = (
                    float(search_context["ra"]),
                    float(search_context["dec"]),
                )
            except Exception as error:
                st.session_state["cube_results"] = None
                st.error(f"The cube search failed: {error}")

        cube_results = st.session_state.get("cube_results")
        cube_context = st.session_state.get("cube_results_context")
        current_context = (
            float(search_context["ra"]),
            float(search_context["dec"]),
        )

        if cube_results is not None and cube_context == current_context:
            if len(cube_results) == 0:
                st.info("IRSA returned no DR1 cube tile for this position.")
            else:
                cube_columns = [
                    name
                    for name in ("obs_id", "obs_collection", "access_url")
                    if name in cube_results.colnames
                ]
                cube_table = cube_results[cube_columns].to_pandas()
                cube_config = {}
                if "access_url" in cube_table.columns:
                    cube_table = cube_table.rename(
                        columns={"access_url": "Cube FITS product"}
                    )
                    cube_config["Cube FITS product"] = (
                        st.column_config.LinkColumn(
                            "Original IRSA cube FITS product",
                            display_text="Open cube FITS",
                            help="Opens the original DR1 cube tile. The file may be large.",
                        )
                    )
                st.write(f"Cube tiles found: {len(cube_results)}")
                st.dataframe(
                    cube_table,
                    column_config=cube_config,
                    hide_index=True,
                    use_container_width=True,
                )
                st.link_button(
                    "Read about viewing SPHEREx cube channels",
                    SPHEREX_CUBE_GUIDE,
                )

                cube_urls = [
                    str(value)
                    for value in cube_results["access_url"]
                ] if "access_url" in cube_results.colnames else []

                if cube_urls:
                    cube_labels = [
                        str(cube_results["obs_id"][index])
                        if "obs_id" in cube_results.colnames
                        else f"DR1 cube tile {index + 1}"
                        for index in range(len(cube_urls))
                    ]
                    cube_choice = st.selectbox(
                        "Cube tile to inspect",
                        range(len(cube_urls)),
                        format_func=lambda index: cube_labels[index],
                        key="cube_tile_choice",
                    )
                    selected_cube_url = cube_urls[cube_choice]
                    st.warning(
                        "Loading a tile downloads the full 102-channel FITS "
                        "cube (about 300 MB). This is separate from the "
                        "time-lapse and happens only after you choose to load it."
                    )

                    if st.button(
                        "Load this tile and browse its channels",
                        key="load_cube_tile",
                    ):
                        st.session_state["loaded_cube_url"] = selected_cube_url
                        st.session_state["cube_spectrum"] = None
                        st.session_state["cube_spectrum_key"] = None

                    if st.session_state.get("loaded_cube_url") == selected_cube_url:
                        channel_number = st.slider(
                            "Spectral channel",
                            min_value=1,
                            max_value=102,
                            value=1,
                            help="Channels 51 and 52 are blank in the published cube.",
                        )

                        try:
                            with st.spinner(
                                "Cassiopeia is opening this wavelength channel…"
                            ):
                                channel_image, channel_hits, channel_flags, wavelength, channel_header = (
                                    load_cube_channel(
                                        selected_cube_url,
                                        channel_number,
                                    )
                                )
                            channel_wcs = WCS(
                                fits.Header.fromstring(
                                    channel_header,
                                    sep="\n",
                                )
                            ).celestial
                            target_x, target_y = channel_wcs.world_to_pixel(
                                SkyCoord(
                                    ra=float(search_context["ra"]),
                                    dec=float(search_context["dec"]),
                                    unit="deg",
                                )
                            )

                            if channel_number in (51, 52):
                                st.warning(
                                    f"Channel {channel_number} is one of the "
                                    "two documented blank channels."
                                )
                            finite_channel = channel_image[
                                np.isfinite(channel_image)
                            ]
                            if finite_channel.size == 0:
                                st.info(
                                    "This channel has no finite image pixels.")
                            else:
                                low, high = np.nanpercentile(
                                    finite_channel,
                                    [2, 99],
                                )
                                if high <= low:
                                    high = low + 1
                                quality_col, hits_col = st.columns(2)
                                quality_col.metric(
                                    "Pixels with nonzero flags",
                                    f"{100 * np.count_nonzero(channel_flags) / channel_flags.size:.1f}%",
                                )
                                hits_col.metric(
                                    "Median contributing hits",
                                    f"{np.nanmedian(channel_hits):.1f}",
                                )
                                figure, axis = plt.subplots(figsize=(7, 6))
                                plot = axis.imshow(
                                    np.ma.masked_invalid(channel_image),
                                    origin="lower",
                                    cmap="inferno",
                                    vmin=low,
                                    vmax=high,
                                    interpolation="nearest",
                                )
                                if np.isfinite(target_x) and np.isfinite(target_y):
                                    axis.scatter(
                                        target_x,
                                        target_y,
                                        marker="+",
                                        s=120,
                                        color="#45ddff",
                                    )
                                axis.set_title(
                                    f"Channel {channel_number} · {wavelength:.4f} μm"
                                )
                                axis.set_xlabel("Cube tile pixel")
                                axis.set_ylabel("Cube tile pixel")
                                figure.colorbar(
                                    plot,
                                    ax=axis,
                                    label="Surface brightness (MJy/sr)",
                                )
                                figure.tight_layout()
                                st.pyplot(
                                    figure,
                                    use_container_width=True,
                                    clear_figure=True,
                                )
                                plt.close(figure)
                        except Exception as error:
                            st.error(
                                f"Could not load this cube channel: {error}")

                        st.divider()
                        st.markdown(
                            "#### Quick-look spectrum at the searched coordinate")
                        st.caption(
                            "This uses a circular aperture and local background "
                            "annulus across the cube channels. The result is for "
                            "visual exploration, not precision source photometry."
                        )
                        radius_col, annulus_col = st.columns(2)
                        with radius_col:
                            spectrum_aperture = st.slider(
                                "Spectrum aperture radius (cube pixels)",
                                min_value=1,
                                max_value=8,
                                value=2,
                                key="cube_spectrum_aperture",
                            )
                        with annulus_col:
                            spectrum_annulus = st.select_slider(
                                "Background annulus outer radius (pixels)",
                                options=[6, 8, 10, 12, 16],
                                value=8,
                                key="cube_spectrum_annulus",
                            )
                        spectrum_inner = spectrum_aperture + 2
                        if spectrum_annulus <= spectrum_inner:
                            st.warning(
                                "Choose a larger background radius so the "
                                "annulus stays outside the source aperture."
                            )
                        elif st.button(
                            "Extract quick-look spectrum",
                            key="extract_cube_spectrum",
                        ):
                            spectrum_key = (
                                selected_cube_url,
                                float(search_context["ra"]),
                                float(search_context["dec"]),
                                spectrum_aperture,
                                spectrum_inner,
                                spectrum_annulus,
                            )
                            try:
                                with st.spinner(
                                    "Cassiopeia is reading the cube’s real wavelength channels…"
                                ):
                                    st.session_state["cube_spectrum"] = (
                                        extract_cube_quick_spectrum(
                                            selected_cube_url,
                                            float(search_context["ra"]),
                                            float(search_context["dec"]),
                                            spectrum_aperture,
                                            spectrum_inner,
                                            spectrum_annulus,
                                        )
                                    )
                                st.session_state["cube_spectrum_key"] = spectrum_key
                            except Exception as error:
                                st.session_state["cube_spectrum"] = None
                                st.error(
                                    f"Spectrum extraction failed: {error}")

                        current_spectrum_key = (
                            selected_cube_url,
                            float(search_context["ra"]),
                            float(search_context["dec"]),
                            spectrum_aperture,
                            spectrum_inner,
                            spectrum_annulus,
                        )
                        quick_spectrum = st.session_state.get("cube_spectrum")
                        if (
                            quick_spectrum is not None
                            and st.session_state.get("cube_spectrum_key")
                            == current_spectrum_key
                        ):
                            spectrum_wavelengths, spectrum_fluxes, spectrum_flags = quick_spectrum
                            spectrum_figure, spectrum_axis = plt.subplots(
                                figsize=(9, 4.5)
                            )
                            spectrum_axis.plot(
                                spectrum_wavelengths,
                                spectrum_fluxes,
                                marker="o",
                                markersize=3,
                                linewidth=1,
                                color="#50c8ff",
                            )
                            spectrum_axis.set_xlabel("Wavelength (μm)")
                            spectrum_axis.set_ylabel(
                                "Background-subtracted flux (mJy)")
                            spectrum_axis.set_title(
                                "SPHEREx DR1 quick-look spectrum")
                            spectrum_axis.grid(alpha=0.25)
                            spectrum_figure.tight_layout()
                            st.pyplot(
                                spectrum_figure,
                                use_container_width=True,
                                clear_figure=True,
                            )
                            plt.close(spectrum_figure)
                            spectrum_rows = [
                                {
                                    "channel": index + 1,
                                    "wavelength_um": float(spectrum_wavelengths[index]),
                                    "flux_mJy": float(spectrum_fluxes[index]),
                                    "flagged_aperture_percent": float(spectrum_flags[index]),
                                }
                                for index in range(len(spectrum_wavelengths))
                            ]
                            st.dataframe(
                                spectrum_rows,
                                hide_index=True,
                                use_container_width=True,
                            )
                            spectrum_csv = io.StringIO()
                            spectrum_writer = csv.DictWriter(
                                spectrum_csv,
                                fieldnames=list(spectrum_rows[0]),
                            )
                            spectrum_writer.writeheader()
                            spectrum_writer.writerows(spectrum_rows)
                            st.download_button(
                                "Download quick-look spectrum as CSV",
                                data=spectrum_csv.getvalue(),
                                file_name="spherex_quicklook_spectrum.csv",
                                mime="text/csv",
                                key="download_cube_spectrum",
                            )
                            st.warning(
                                "The cube is a visualization mosaic. This "
                                "quick-look extraction has no uncertainty "
                                "estimate or PSF correction; use IRSA’s "
                                "spectrophotometry tool for science-grade "
                                "measurements."
                            )


shared_search_params, shared_collections = parse_shared_search_params()
shared_signature = (
    shared_search_params["ra"],
    shared_search_params["dec"],
    shared_search_params["fov_arcmin"],
    shared_collections,
)
shared_params_changed = (
    st.session_state.get("loaded_shared_signature") != shared_signature
)
if shared_params_changed:
    st.session_state["ra_input"] = shared_search_params["ra"]
    st.session_state["dec_input"] = shared_search_params["dec"]
    st.session_state["fov_arcmin"] = shared_search_params["fov_arcmin"]
    st.session_state["target"] = "Custom coordinates"
    st.session_state["loaded_shared_signature"] = shared_signature

state_defaults = {
    "intro_complete": False,
    "target": "Custom coordinates",
    "ra_input": shared_search_params["ra"],
    "dec_input": shared_search_params["dec"],
    "collection_profile": None,
    "cube_results": None,
    "cube_results_context": None,
    "loaded_cube_url": None,
    "cube_spectrum": None,
    "cube_spectrum_key": None,
    "fov_arcmin": shared_search_params["fov_arcmin"],
    "results": None,
    "search_context": None,
    "frame_index": 0,
    "start_index": 0,
    "playing": False,
    "prepared_signature": None,
    "prepared_frames": [],
    "prepared_metadata": [],
    "prepared_stretch": None,
    "prepared_reference_pixel": None,
    "prepared_wcs_header": None,
    "preload_failures": [],
    "preload_failure_signature": None,
    "workspace": "Search & Results",
    "previous_workspace": "Search & Results",
    "last_workspace": "Search & Results",
    "playback_interval": 0.75,
    "lightcurve_rows": None,
    "lightcurve_signature": None,
    "track_candidate_index": 0,
    "catalog_overlay_tables": {},
    "catalog_overlay_signature": None,
    "catalog_overlay_failures": [],
}

for key, value in state_defaults.items():
    st.session_state.setdefault(key, value)


apply_styles()

if not st.session_state["intro_complete"]:
    st.markdown(
        """
        <section class="cass-intro" aria-label="Welcome from Cassiopeia">
            <div class="cass-constellation" aria-hidden="true">
                <svg viewBox="0 0 220 150">
                    <path class="cass-star-path" d="M22 90L64 43L104 98L151 35L198 82"/>
                    <path class="cass-star-trail" d="M22 90L64 43L104 98L151 35L198 82"/>
                    <path class="cass-star-glow" d="M22 90L64 43L104 98L151 35L198 82"/>
                    <circle class="cass-constellation-star star-a" cx="22" cy="90" r="7"/>
                    <circle class="cass-constellation-star star-b" cx="64" cy="43" r="10"/>
                    <circle class="cass-constellation-star star-c" cx="104" cy="98" r="8"/>
                    <circle class="cass-constellation-star star-d" cx="151" cy="35" r="12"/>
                    <circle class="cass-constellation-star star-e" cx="198" cy="82" r="8"/>
                    <path class="cass-tiny-star" d="M31 29V41M25 35H37M181 113V125M175 119H187"/>
                    <circle class="cass-orbit-dot" cx="108" cy="25" r="2.5"/>
                </svg>
            </div>
            <div class="cass-intro-copy">
                <div class="cass-type-lines" aria-label="Cassiopeia welcomes you to explore the sky.">
                    <div class="cass-type-line type-one">HEY, SKY EXPLORER.</div>
                    <div class="cass-type-line type-two">I’M CASSIOPEIA.</div>
                    <div class="cass-type-line type-three">YOUR SKY IS WAITING...</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    intro_col1, intro_col2, intro_col3 = st.columns([1, 1.3, 1])
    with intro_col2:
        if st.button(
            "ENTER THE SKY  →",
            type="primary",
            key="continue_from_welcome",
        ):
            st.session_state["intro_complete"] = True
            st.session_state["workspace"] = "Search & Results"
            st.rerun()
    st.stop()

render_page_header(
    "explore",
    "Explore the infrared sky over time",
    "Search real SPHEREx observations, prepare a time-lapse, "
    "and inspect how the sky changes.",
)

workspace_col, workspace_hint = st.columns([1.25, 3.75])
with workspace_col:
    st.selectbox(
        "Explore tools",
        ["Search & Results", "Animation", "Light Curve", "Track Sources"],
        key="workspace",
        on_change=remember_workspace_change,
    )
with workspace_hint:
    st.markdown(
        '<div class="workspace-hint"><span>EXPLORE</span> '
        'Choose a task here, or use the quick actions beside your observations.</div>',
        unsafe_allow_html=True,
    )


if st.session_state["workspace"] == "Search & Results":
    st.subheader("Search the SPHEREx archive")
    st.caption(
        "Choose a known target or enter coordinates. "
        "The search uses NASA/IPAC IRSA archive data."
    )

    with st.expander("SPHEREx data sources and help"):
        st.write(
            "Images come from the NASA/IPAC Infrared Science Archive "
            "(IRSA). Each observation in the results list links to its "
            "original FITS data product."
        )

        source_col1, source_col2 = st.columns(2)

        with source_col1:
            st.link_button(
                "SPHEREx archive guide",
                IRSA_ARCHIVE_GUIDE,
                use_container_width=True,
            )

        with source_col2:
            st.link_button(
                "SPHEREx mission website",
                SPHEREX_MISSION,
                use_container_width=True,
            )

        st.link_button(
            "IRSA 102-channel cube guide",
            SPHEREX_CUBE_GUIDE,
            use_container_width=True,
        )

    st.selectbox(
        "Challenge target",
        list(TARGETS.keys()),
        key="target",
        on_change=apply_target,
    )

    coordinate_col1, coordinate_col2 = st.columns(2)

    with coordinate_col1:
        ra = st.number_input(
            "Right Ascension (degrees)",
            min_value=0.0,
            max_value=360.0,
            key="ra_input",
        )

    with coordinate_col2:
        dec = st.number_input(
            "Declination (degrees)",
            min_value=-90.0,
            max_value=90.0,
            key="dec_input",
        )

    with st.expander("Data release and image size", expanded=False):
        try:
            image_collections, release_list_checked_at = available_image_collections()
            st.session_state["known_image_collections"] = image_collections
            st.session_state["release_list_checked_at"] = release_list_checked_at
        except Exception as error:
            image_collections = st.session_state.get(
                "known_image_collections",
                ["spherex_dr1", "spherex_dr1_deep", "spherex_qr3", "spherex_qr3_deep"],
            )
            release_list_checked_at = st.session_state.get(
                "release_list_checked_at", "unavailable in this session"
            )
            st.warning(
                "Could not reach IRSA for the release list. Showing the last "
                "collection list available in this session, if any. "
                f"Details: {error}"
            )

        st.caption(f"IRSA collection list checked: {release_list_checked_at}")

        profiles = collection_profiles(image_collections)
        profile_names = list(profiles)
        if not profile_names:
            st.error("No SPHEREx image collections are available right now.")
            st.stop()

        if shared_params_changed and shared_collections:
            requested_collections = set(shared_collections.split(","))
            matched_profile = next(
                (
                    name
                    for name, values in profiles.items()
                    if set(values) == requested_collections
                ),
                None,
            )
            if matched_profile:
                st.session_state["collection_profile"] = matched_profile

        if st.session_state["collection_profile"] not in profiles:
            st.session_state["collection_profile"] = profile_names[0]

        search_option_col1, search_option_col2 = st.columns(2)

        with search_option_col1:
            collection_profile = st.selectbox(
                "Data release",
                profile_names,
                key="collection_profile",
                help=(
                    "The list refreshes from IRSA. DR1 contains reprocessed "
                    "Survey 1 and 2 data; QR3 contains Survey 3 and 4 data."
                ),
            )
            selected_collections = profiles[collection_profile]
            st.caption("Searching: " + ", ".join(selected_collections))
            st.button(
                "↻ Check for new releases",
                key="refresh_available_releases",
                help=(
                    "Ask IRSA for its current collection list now. It also "
                    "refreshes automatically each hour."
                ),
                on_click=clear_available_collections_cache,
            )

        with search_option_col2:
            fov_arcmin = st.slider(
                "Image field of view (arcminutes)",
                min_value=2,
                max_value=20,
                step=1,
                key="fov_arcmin",
                help=(
                    "Controls the size of displayed image cutouts. "
                    "The archive search checks the chosen coordinates."
                ),
            )


    search_col, _ = st.columns([1, 3])

    if search_col.button(
        "Search observations",
        type="primary",
        use_container_width=True,
    ):
        search_status = st.empty()
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            search_status.html(
                render_loading_experience(
                    "coordinate",
                    "Scanning your sky coordinates",
                    "Contacting IRSA for matching SPHEREx observations. The archive does not provide a partial search percentage.",
                    ra=ra,
                    dec=dec,
                )
            )
            new_results = query_sia_with_retries(
                pos=(coord, 1 * u.arcsec),
                collection=selected_collections,
                data_type="image",
            )
            search_status.empty()

            new_results.sort("t_min")

            if len(new_results) > 0:
                new_results["date_utc"] = Time(
                    new_results["t_min"].astype(float),
                    format="mjd",
                ).to_value("iso")

                bands = sorted(
                    {
                        str(value)
                        for value in new_results["energy_bandpassname"]
                    }
                )
                st.session_state["selected_band"] = bands[0]

            st.session_state["results"] = new_results
            st.session_state["cube_results"] = None
            st.session_state["cube_results_context"] = None
            st.session_state["loaded_cube_url"] = None
            st.session_state["cube_spectrum"] = None
            st.session_state["cube_spectrum_key"] = None
            st.session_state["search_context"] = {
                "ra": ra,
                "dec": dec,
                "collection": collection_profile,
                "collections": list(selected_collections),
                "fov_arcmin": fov_arcmin,
                "target": st.session_state["target"],
            }
            st.session_state["frame_index"] = 0
            st.session_state["start_index"] = 0

            reset_animation()

        except Exception as error:
            search_status.empty()
            st.session_state["results"] = None
            st.session_state["search_context"] = None
            reset_animation()
            st.error(
                "IRSA could not be reached after retrying. This is usually "
                "a temporary archive or internet connection problem."
            )
            with st.expander("Show technical details"):
                st.code(str(error))

    results = st.session_state["results"]
    search_context = st.session_state["search_context"]

    if results is not None:
        st.divider()
        st.subheader("Observation results")
        st.metric("Observations found", len(results))
        if st.button(
            "Create shareable search link",
            key="create_shareable_search_link",
            help=(
                "Adds this search position, field of view, and collection "
                "to the browser address. The link pre-fills the search; "
                "the recipient starts the IRSA search themselves."
            ),
        ):
            st.query_params["ra"] = str(search_context["ra"])
            st.query_params["dec"] = str(search_context["dec"])
            st.query_params["fov_arcmin"] = str(
                search_context["fov_arcmin"]
            )
            st.query_params["collections"] = ",".join(
                search_context["collections"]
            )
            st.success(
                "Search link ready. Copy the browser address to share it. "
                "The link opens with these search settings filled in."
            )
        retrieved_dates = (
            [str(value) for value in results["date_utc"]]
            if "date_utc" in results.colnames
            else []
        )
        st.html(
            render_loading_experience(
                "retrieve",
                f"IRSA returned {len(results)} observation record(s)",
                (
                    "These are real archive records for your selected coordinates and collection. "
                    "Dates below come from the returned observations."
                    if len(results)
                    else "No matching observation dates were returned for this search."
                ),
                ra=search_context["ra"],
                dec=search_context["dec"],
                dates=retrieved_dates,
            )
        )

        if len(results) == 0:
            st.info(
                "No observations were found for these coordinates and "
                "collection. Try another target or collection."
            )
        else:
            if (
                float(ra) != float(search_context["ra"])
                or float(dec) != float(search_context["dec"])
                or collection_profile != search_context["collection"]
                or fov_arcmin != search_context["fov_arcmin"]
            ):
                st.info(
                    "These results are from your previous search. "
                    "Search again to apply the changed coordinates, "
                    "collection, or field of view."
                )

            bands = sorted(
                {
                    str(value)
                    for value in results["energy_bandpassname"]
                }
            )

            selected_band = st.selectbox(
                "Filter results by SPHEREx band",
                bands,
                key="selected_band",
                on_change=choose_band,
            )

            band_results = results[
                results["energy_bandpassname"] == selected_band
            ]
            band_results.sort("t_min")

            st.caption(
                f"Showing {len(band_results)} observations in "
                f"**{selected_band}**. Open the list to review dates, "
                "observation IDs, and original data links."
            )

            observation_table, column_config = make_observation_table(
                band_results
            )

            with st.expander(
                f"View all {len(band_results)} observations and source links"
            ):
                st.dataframe(
                    observation_table,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True,
                )

            st.success(
                "Your real SPHEREx images are ready. Start by playing them "
                "in date order."
            )
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                st.button(
                    "▶  Animate these images",
                    type="primary",
                    use_container_width=True,
                    key="go_animate_from_results",
                    on_click=navigate_workspace,
                    args=("Animation",),
                )
            with action_col2:
                st.page_link(
                    "pages/2_Compare_Images.py",
                    label="Compare two dates",
                    icon="🌓",
                    use_container_width=True,
                )
            with action_col3:
                st.button(
                    "Brightness over time",
                    use_container_width=True,
                    key="go_lightcurve_from_results",
                    on_click=navigate_workspace,
                    args=("Light Curve",),
                )

        with st.expander("Explore the 102-channel sky cube", expanded=False):
            render_cube_explorer(search_context)


elif st.session_state["workspace"] == "Animation":
    back_col, compare_col, lightcurve_col = st.columns([1.2, 1, 1])
    with back_col:
        st.button(
            "← Back to observations",
            use_container_width=True,
            key="animation_back_to_results",
            on_click=navigate_workspace,
            args=("Search & Results",),
        )
    with compare_col:
        st.page_link(
            "pages/2_Compare_Images.py",
            label="Compare two dates",
            icon="🌓",
            use_container_width=True,
        )
    with lightcurve_col:
        st.button(
            "Brightness over time",
            use_container_width=True,
            key="animation_to_lightcurve",
            on_click=navigate_workspace,
            args=("Light Curve",),
        )

    st.subheader("SPHEREx time-lapse")
    st.caption(
        "The app aligns the frames to one sky grid and applies the same "
        "brightness stretch to every frame before playback."
    )

    results = st.session_state["results"]
    search_context = st.session_state["search_context"]

    if results is None or len(results) == 0 or search_context is None:
        st.info(
            "Search for observations first. Then return here to prepare "
            "and play the available images."
        )
        st.stop()

    active_ra = float(search_context["ra"])
    active_dec = float(search_context["dec"])
    active_fov_arcmin = int(search_context["fov_arcmin"])
    cutout_size_deg = active_fov_arcmin / 60.0

    st.info(
        f"Target: **{search_context['target']}** · "
        f"RA {active_ra:.6f}° · Dec {active_dec:.6f}° · "
        f"{search_context['collection']} · "
        f"{active_fov_arcmin} arcminute field of view"
    )

    if search_context["target"] in {
        "Barnard's Star (approx. 2026)",
        "WISE 0855",
    }:
        st.warning(
            "This is a high-proper-motion target. The red cross marks "
            "your searched coordinate; it does not track the target's "
            "predicted position at each date. Compare the source with "
            "nearby stationary stars."
        )

    bands = sorted(
        {
            str(value)
            for value in results["energy_bandpassname"]
        }
    )

    selected_band = st.selectbox(
        "SPHEREx band",
        bands,
        key="selected_band",
        on_change=choose_band,
    )

    band_results = results[
        results["energy_bandpassname"] == selected_band
    ]
    band_results.sort("t_min")

    if len(band_results) == 0:
        st.info("There are no observations for this band.")
        st.stop()

    first_mjd = float(band_results["t_min"][0])
    last_mjd = float(band_results["t_min"][-1])
    date_span_days = last_mjd - first_mjd

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Frames in time-lapse", len(band_results))

    with summary_col2:
        st.metric("Date range", f"{date_span_days:.1f} days")

    observation_table, column_config = make_observation_table(band_results)

    with st.expander(
        f"Observation dates and original data links · "
        f"{len(band_results)} frames"
    ):
        st.dataframe(
            observation_table,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )

    st.session_state["start_index"] = min(
        st.session_state["start_index"],
        len(band_results) - 1,
    )

    selected_index = st.selectbox(
        "Start playback at",
        range(len(band_results)),
        format_func=lambda index: observation_label(
            band_results, index
        ),
        key="start_index",
        on_change=choose_start_frame,
    )

    cutout_urls = [
        make_cutout_url(
            str(band_results["access_url"][index]),
            active_ra,
            active_dec,
            cutout_size_deg,
        )
        for index in range(len(band_results))
    ]

    preparation_signature = (
        "common-sky-grid-fixed-stretch-stack-track-v3",
        tuple(cutout_urls),
    )

    if (
        st.session_state["preload_failure_signature"]
        != preparation_signature
    ):
        st.session_state["preload_failures"] = []
        st.session_state["preload_failure_signature"] = (
            preparation_signature
        )

    is_prepared = (
        st.session_state["prepared_signature"]
        == preparation_signature
        and len(st.session_state["prepared_frames"])
        == len(band_results)
    )

    controls_col1, controls_col2 = st.columns([2, 1])

    with controls_col1:
        st.markdown("#### Prepare the time-lapse")
        st.caption(
            f"This loads and aligns all {len(cutout_urls)} real cutouts "
            f"in **{selected_band}**. The aligned frames are then cached "
            "for playback."
        )

    with controls_col2:
        st.selectbox(
            "Playback pace",
            options=[0.5, 0.75, 1.0, 1.5, 2.0],
            format_func=lambda seconds: {
                0.5: "Fast · 0.5 sec",
                0.75: "Medium · 0.75 sec",
                1.0: "Relaxed · 1 sec",
                1.5: "Slow · 1.5 sec",
                2.0: "Cinematic · 2 sec",
            }[seconds],
            key="playback_interval",
        )

    if st.button("Prepare animation", type="primary"):
        st.session_state["playing"] = False
        st.session_state["prepared_signature"] = None
        st.session_state["prepared_frames"] = []
        st.session_state["prepared_metadata"] = []
        st.session_state["prepared_stretch"] = None
        st.session_state["prepared_reference_pixel"] = None
        st.session_state["prepared_wcs_header"] = None
        st.session_state["prepared_mean_stack"] = None
        st.session_state["preload_failures"] = []

        loading_status = st.empty()
        observation_dates = [
            str(value) for value in band_results["date_utc"]
        ]
        loading_status.html(
            render_loading_experience(
                "retrieve",
                "Your animation is getting ready",
                "Get your eyes ready. Cassiopeia is retrieving the real image cutouts for these observation dates.",
                progress_fraction=0,
                progress_label=f"Archive responses finished: 0 of {len(cutout_urls)}",
                ra=active_ra,
                dec=active_dec,
                dates=observation_dates,
            )
        )
        failures = []
        aligned_frames = [None] * len(cutout_urls)
        frame_metadata = [None] * len(cutout_urls)
        reference_wcs = None
        reference_header = None
        reference_shape = None
        reference_pixel = None

        downloaded = [None] * len(cutout_urls)
        future_indices = {}
        completed_count = 0
        preview_data = None
        download_started = time.monotonic()
        waiting_messages = [
            "Still gathering the images. A big sky can take a little longer…",
            "Each frame is a real SPHEREx observation. I’m bringing them together…",
            "Eyes ready? I’m lining up the sky so you can watch it change…",
            "Thanks for waiting with me. The archive is still sending your frames…",
        ]
        last_wait_message = -1

        with ThreadPoolExecutor(max_workers=2) as executor:
            for index, cutout_url in enumerate(cutout_urls):
                future = executor.submit(
                    load_cutout_with_retries,
                    cutout_url,
                    active_ra,
                    active_dec,
                )
                future_indices[future] = index

            pending_futures = set(future_indices)
            while pending_futures:
                completed_futures, _ = wait(
                    pending_futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                elapsed = time.monotonic() - download_started
                if not completed_futures:
                    if elapsed >= 4:
                        message_index = max(0, int(elapsed // 4) - 1) % len(waiting_messages)
                        if message_index != last_wait_message:
                            loaded_wavelengths = [
                                item[2]["wavelength_um"]
                                for item in downloaded
                                if item is not None
                            ]
                            loading_status.html(
                                render_loading_experience(
                                    "retrieve",
                                    "Your animation is getting ready",
                                    waiting_messages[message_index],
                                    progress_fraction=completed_count / len(cutout_urls),
                                    progress_label=(
                                        f"Archive responses finished: "
                                        f"{completed_count} of {len(cutout_urls)}"
                                    ),
                                    ra=active_ra,
                                    dec=active_dec,
                                    dates=observation_dates,
                                    wavelengths=loaded_wavelengths,
                                    preview_data=preview_data,
                                    focus_blur=(
                                        12 * (1 - .35 * completed_count / len(cutout_urls))
                                    ),
                                )
                            )
                            last_wait_message = message_index
                    continue

                for future in completed_futures:
                    pending_futures.remove(future)
                    index = future_indices[future]
                    try:
                        downloaded[index] = future.result()
                        if preview_data is None:
                            try:
                                preview_data = make_loading_preview(
                                    downloaded[index][0]
                                )
                            except Exception:
                                preview_data = None
                    except Exception as error:
                        failures.append(
                            (
                                index,
                                str(error).splitlines()[0][:180],
                            )
                        )
                    completed_count += 1

                if completed_count == len(cutout_urls):
                    status = (
                        "All archive requests have finished. I’m aligning the frames now…"
                        if not failures
                        else "All archive requests have finished. I’m checking the responses…"
                    )
                else:
                    status = (
                        f"Cassiopeia: {completed_count} of {len(cutout_urls)} "
                        "archive responses have finished. I’m fetching the rest…"
                    )
                loaded_wavelengths = [
                    item[2]["wavelength_um"]
                    for item in downloaded
                    if item is not None
                ]
                loading_status.html(
                    render_loading_experience(
                        "retrieve",
                        "Your animation is getting ready",
                        status,
                        progress_fraction=completed_count / len(cutout_urls),
                        progress_label=(
                            f"Archive responses finished: {completed_count} "
                            f"of {len(cutout_urls)}"
                        ),
                        ra=active_ra,
                        dec=active_dec,
                        dates=observation_dates,
                        wavelengths=loaded_wavelengths,
                        preview_data=preview_data,
                        focus_blur=(
                            12 * (1 - .35 * completed_count / len(cutout_urls))
                        ),
                    )
                )

        if not failures:
            for index, loaded_cutout in enumerate(downloaded):
                try:
                    image, flags, metadata, header = loaded_cutout
                    image = np.asarray(image, dtype=float)
                    bad_pixels = flagged_pixels(flags)
                    usable_image = image.copy()
                    usable_image[bad_pixels] = np.nan
                    image_wcs = WCS(header).celestial

                    if reference_wcs is None:
                        reference_wcs = image_wcs
                        reference_header = header.copy()
                        reference_shape = usable_image.shape
                        reference_pixel = reference_wcs.world_to_pixel(
                            SkyCoord(
                                ra=active_ra,
                                dec=active_dec,
                                unit="deg",
                            )
                        )
                        aligned_image = usable_image
                    else:
                        aligned_image, footprint = reproject_interp(
                            (usable_image, image_wcs),
                            reference_wcs,
                            shape_out=reference_shape,
                        )
                        aligned_image = np.asarray(aligned_image, dtype=float)
                        aligned_image[footprint <= 0] = np.nan

                    aligned_frames[index] = aligned_image
                    frame_metadata[index] = {
                        "detector": str(metadata["detector"]),
                        "units": str(metadata["units"]),
                        "image_date": str(metadata["image_date"]),
                        "wavelength_um": float(metadata["wavelength_um"]),
                        "psf_fwhm": str(metadata["psf_fwhm"]),
                        "flagged_percent": (
                            100 * np.count_nonzero(bad_pixels) / bad_pixels.size
                        ),
                    }

                    actual_wavelengths = [
                        item["wavelength_um"]
                        for item in frame_metadata
                        if item is not None
                    ]
                    loading_status.html(
                        render_loading_experience(
                            "process",
                            "Bringing the real images into focus",
                            (
                                f"Cassiopeia is aligning frame {index + 1} of "
                                f"{len(cutout_urls)} to the shared sky view."
                            ),
                            progress_fraction=(index + 1) / len(cutout_urls),
                            progress_label=f"Frames aligned: {index + 1} of {len(cutout_urls)}",
                            ra=active_ra,
                            dec=active_dec,
                            dates=observation_dates,
                            wavelengths=actual_wavelengths,
                            preview_data=preview_data,
                            focus_blur=7.8 * (1 - (index + 1) / len(cutout_urls)),
                        )
                    )
                except Exception as error:
                    failures.append(
                        (index, str(error).splitlines()[0][:180])
                    )
                    break

        if failures:
            loading_status.html(
                render_loading_experience(
                    "process",
                    "Cassiopeia hit a snag while preparing a frame",
                    f"{len(failures)} image request(s) or processing step(s) did not finish. Review the failed observation and retry.",
                    ra=active_ra,
                    dec=active_dec,
                    dates=observation_dates,
                    preview_data=preview_data,
                    error=True,
                )
            )

        st.session_state["preload_failures"] = failures
        st.session_state["preload_failure_signature"] = (
            preparation_signature
        )

        if not failures and all(
            frame is not None for frame in aligned_frames
        ):
            actual_wavelengths = [
                item["wavelength_um"]
                for item in frame_metadata
                if item is not None
            ]
            loading_status.html(
                render_loading_experience(
                    "spectrum",
                    "Reading the real wavelength samples",
                    (
                        "Each value below came from an aligned image’s FITS wavelength header. "
                        "The colored strip is the instrument reference, not a claim that the full cube was loaded."
                    ),
                    progress_fraction=1,
                    progress_label=(
                        f"FITS wavelength headers read: {len(actual_wavelengths)} "
                        f"of {len(cutout_urls)} images · channel guide spans 0.75–5.00 μm"
                    ),
                    ra=active_ra,
                    dec=active_dec,
                    dates=observation_dates,
                    wavelengths=actual_wavelengths,
                    preview_data=preview_data,
                    focus_blur=0,
                )
            )

            sampled_values = []

            for frame in aligned_frames:
                finite_values = frame[np.isfinite(frame)]
                if finite_values.size:
                    stride = max(1, finite_values.size // 10000)
                    sampled_values.append(finite_values[::stride])

            if sampled_values:
                all_samples = np.concatenate(sampled_values)
                low, high = np.nanpercentile(all_samples, [5, 99])

                if high <= low:
                    high = low + 1

                st.session_state["prepared_frames"] = aligned_frames
                st.session_state["prepared_metadata"] = frame_metadata
                st.session_state["prepared_stretch"] = (
                    float(low), float(high))
                st.session_state["prepared_reference_pixel"] = (
                    float(reference_pixel[0]),
                    float(reference_pixel[1]),
                )
                st.session_state["prepared_wcs_header"] = (
                    reference_header.tostring(sep="\n")
                )
                stack_sum = np.zeros_like(aligned_frames[0], dtype=float)
                stack_count = np.zeros_like(
                    aligned_frames[0], dtype=np.uint16
                )
                for stack_index, frame in enumerate(aligned_frames):
                    valid = np.isfinite(frame)
                    stack_sum[valid] += frame[valid]
                    stack_count[valid] += 1
                    loading_status.html(
                        render_loading_experience(
                            "sequence",
                            "Cassiopeia is arranging the time sequence",
                            "The observation dates are ordered from archive timestamps while I build a reference stack for the viewer.",
                            progress_fraction=(stack_index + 1) / len(aligned_frames),
                            progress_label=(
                                f"Chronological frames added to reference stack: "
                                f"{stack_index + 1} of {len(aligned_frames)}"
                            ),
                            ra=active_ra,
                            dec=active_dec,
                            dates=observation_dates,
                            wavelengths=actual_wavelengths,
                            preview_data=preview_data,
                            focus_blur=0,
                        )
                    )
                mean_stack = np.full_like(stack_sum, np.nan)
                np.divide(
                    stack_sum,
                    stack_count,
                    out=mean_stack,
                    where=stack_count > 0,
                )
                st.session_state["prepared_mean_stack"] = mean_stack
                st.session_state["prepared_signature"] = (
                    preparation_signature
                )
                loading_status.html(
                    render_loading_experience(
                        "sequence",
                        "Your time sequence is ready",
                        "Every prepared frame is sorted by its returned observation time. The viewer is opening now.",
                        progress_fraction=1,
                        progress_label=(
                            f"Frames aligned and sequence ready: "
                            f"{len(aligned_frames)} of {len(aligned_frames)}"
                        ),
                        ra=active_ra,
                        dec=active_dec,
                        dates=observation_dates,
                        wavelengths=[
                            item["wavelength_um"]
                            for item in frame_metadata
                            if item is not None
                        ],
                        preview_data=preview_data,
                        focus_blur=0,
                    )
                )
            else:
                st.session_state["preload_failures"] = [
                    (0, "No usable unmasked image pixels were found.")
                ]

        if st.session_state["preload_failures"]:
            st.error("The time-lapse could not be fully prepared. Check the failed frames and try again.")
            with st.expander("Show frame preparation details"):
                for failed_index, reason in st.session_state["preload_failures"][:8]:
                    st.caption(f"{observation_label(band_results, failed_index)} — {reason}")
            loading_status.empty()
            st.stop()

        loading_status.empty()

    is_prepared = (
        st.session_state["prepared_signature"]
        == preparation_signature
        and len(st.session_state["prepared_frames"])
        == len(band_results)
    )
    failures = st.session_state["preload_failures"]

    if is_prepared:
        wavelengths = [
            metadata["wavelength_um"]
            for metadata in st.session_state["prepared_metadata"]
        ]

        st.success(
            f"All {len(cutout_urls)} frames are aligned and ready. "
            "The display brightness is fixed across the sequence."
        )
        st.caption(
            f"Frame wavelengths range from "
            f"{min(wavelengths):.3f} to {max(wavelengths):.3f} μm. "
            "The selected SPHEREx band is fixed, but wavelength can vary "
            "between observations."
        )

    elif failures:
        st.warning(
            f"{len(failures)} frame(s) could not be prepared. "
            "Playback is enabled after every frame loads successfully. "
            "Try preparing again."
        )

        for failed_index, reason in failures[:5]:
            st.caption(
                f"{observation_label(band_results, failed_index)} — "
                f"{reason}"
            )

    else:
        st.info(
            "Preparing aligns every image to the same sky view and sets "
            "one brightness scale for the whole time-lapse."
        )

    if not is_prepared:
        st.info(
            "Choose a starting date above, then prepare the time-lapse. "
            "The image sequence appears here when preparation is complete."
        )
        st.stop()

    catalog_choices = {
        "Gaia DR3 star catalog": "gaia_dr3_source",
        "WISE AllWISE infrared catalog": "allwise_p3as_psd",
    }
    with st.expander("Optional Gaia / WISE reference overlays"):
        st.caption(
            "These are catalog positions plotted over the SPHEREx image. "
            "They help you inspect known sources; they do not confirm that "
            "a catalog object is the changing source. Gaia positions with "
            "proper motions are propagated to each observation date. Up to "
            "500 closest catalog markers are drawn to keep crowded fields readable."
        )
        selected_catalog_labels = st.multiselect(
            "Catalog layers",
            list(catalog_choices),
            key="selected_catalog_overlays",
            help=(
                "Queries a cone covering the selected square field of view. "
                "Results are cached for six hours."
            ),
        )
        if st.button(
            "Load catalog references",
            disabled=not selected_catalog_labels,
            key="load_catalog_references",
        ):
            overlay_signature = (
                active_ra,
                active_dec,
                float(fov_arcmin),
                tuple(
                    catalog_choices[label]
                    for label in selected_catalog_labels
                ),
            )
            catalog_tables = {}
            catalog_failures = []
            radius_arcmin = float(fov_arcmin) / np.sqrt(2)
            with st.spinner(
                "Querying IRSA catalogs for sources in this field…"
            ):
                for label in selected_catalog_labels:
                    catalog_id = catalog_choices[label]
                    try:
                        catalog_tables[catalog_id] = query_reference_catalog(
                            catalog_id,
                            active_ra,
                            active_dec,
                            radius_arcmin,
                        )
                    except Exception as error:
                        catalog_failures.append(
                            f"{label}: {str(error).splitlines()[0][:240]}"
                        )
            st.session_state["catalog_overlay_tables"] = catalog_tables
            st.session_state["catalog_overlay_signature"] = overlay_signature
            st.session_state["catalog_overlay_failures"] = catalog_failures
            if catalog_failures:
                st.warning(
                    "One or more catalogs could not be loaded. You can still "
                    "use the SPHEREx images and retry the catalog query."
                )
            else:
                st.success(
                    "Catalog query complete. Available reference positions "
                    "will appear on the animation frames."
                )

    current_overlay_signature = (
        active_ra,
        active_dec,
        float(fov_arcmin),
        tuple(
            catalog_choices[label]
            for label in selected_catalog_labels
        ),
    )
    active_overlay_tables = (
        st.session_state.get("catalog_overlay_tables", {})
        if st.session_state.get("catalog_overlay_signature")
        == current_overlay_signature
        else {}
    )
    if st.session_state.get("catalog_overlay_failures"):
        for catalog_failure in st.session_state["catalog_overlay_failures"]:
            st.caption(catalog_failure)
    overlay_wcs = WCS(
        fits.Header.fromstring(
            st.session_state["prepared_wcs_header"], sep="\n"
        )
    ).celestial

    with st.container(key="animation-playback-controls"):
        play_col, pause_col, replay_col, previous_col, next_col = st.columns(5)

        if play_col.button(
            "Play",
            disabled=st.session_state["playing"],
            use_container_width=True,
        ):
            st.session_state["playing"] = True

        if pause_col.button(
            "Pause",
            disabled=not st.session_state["playing"],
            use_container_width=True,
        ):
            st.session_state["playing"] = False

        if replay_col.button(
            "Replay",
            use_container_width=True,
        ):
            st.session_state["frame_index"] = 0
            st.session_state["playing"] = True

        if previous_col.button(
            "Prev",
            disabled=st.session_state["playing"],
            use_container_width=True,
        ):
            st.session_state["frame_index"] = (
                st.session_state["frame_index"] - 1
            ) % len(band_results)

        if next_col.button(
            "Next",
            disabled=st.session_state["playing"],
            use_container_width=True,
        ):
            st.session_state["frame_index"] = (
                st.session_state["frame_index"] + 1
            ) % len(band_results)

    reference_x, reference_y = (
        st.session_state["prepared_reference_pixel"]
    )
    stretch_low, stretch_high = st.session_state["prepared_stretch"]

    with st.expander("Image display options", expanded=False):
        image_appearance = st.radio(
            "Image rendering",
            ["Smoothed preview", "Native pixels"],
            horizontal=True,
            help=(
                "Smoothing changes only how pixels are drawn. It does not "
                "add detail or alter the underlying FITS data."
            ),
        )
        show_candidate_peaks = st.checkbox(
            "Mark bright source candidates",
            value=False,
            help="These are visual peak candidates, not confirmed object detections.",
        )
        detection_threshold = 5.0
        if show_candidate_peaks:
            detection_threshold = st.slider(
                "Candidate brightness threshold",
                min_value=3.0,
                max_value=8.0,
                value=5.0,
                step=0.5,
                help="Approximate number of background-noise levels above the background.",
            )
            st.caption(
                "Candidates are for human inspection, not confirmed sources."
            )

    interpolation = "bilinear" if image_appearance == "Smoothed preview" else "nearest"

    @st.fragment(
        run_every=(
            st.session_state["playback_interval"]
            if st.session_state["playing"]
            else None
        )
    )
    def show_frame():
        index = st.session_state["frame_index"] % len(band_results)
        frame = st.session_state["prepared_frames"][index]
        metadata = st.session_state["prepared_metadata"][index]
        candidates = (
            detect_bright_peaks(frame, detection_threshold)
            if show_candidate_peaks
            else []
        )

        st.markdown(
            f"### Frame {index + 1} of {len(band_results)}"
        )
        st.write(observation_label(band_results, index))

        image_col1, image_col2, image_col3 = st.columns([0.8, 2.4, 0.8])

        with image_col2:
            figure, axis = plt.subplots(figsize=(4.4, 4.4))
            color_map = plt.get_cmap("gray").copy()
            color_map.set_bad("#101522")

            axis.imshow(
                np.ma.masked_invalid(frame),
                origin="lower",
                cmap=color_map,
                vmin=stretch_low,
                vmax=stretch_high,
                interpolation=interpolation,
            )
            axis.scatter(
                reference_x,
                reference_y,
                marker="+",
                s=110,
                linewidths=2,
                color="red",
            )
            if active_overlay_tables:
                try:
                    observation_time = Time(
                        str(metadata["image_date"]), scale="utc"
                    )
                except Exception:
                    observation_time = None
                for catalog_id, catalog_table in active_overlay_tables.items():
                    if len(catalog_table) == 0:
                        continue
                    sky_positions = catalog_positions_at_epoch(
                        catalog_table,
                        catalog_id,
                        observation_time,
                    )
                    overlay_x, overlay_y = overlay_wcs.world_to_pixel(
                        sky_positions
                    )
                    valid_overlay = (
                        np.isfinite(overlay_x)
                        & np.isfinite(overlay_y)
                        & (overlay_x >= 0)
                        & (overlay_y >= 0)
                        & (overlay_x < frame.shape[1])
                        & (overlay_y < frame.shape[0])
                    )
                    valid_indices = np.flatnonzero(valid_overlay)
                    if len(valid_indices) > 500:
                        separations = sky_positions[valid_indices].separation(
                            SkyCoord(
                                ra=active_ra,
                                dec=active_dec,
                                unit="deg",
                            )
                        ).arcsec
                        valid_indices = valid_indices[
                            np.argsort(separations)[:500]
                        ]
                    if catalog_id == "gaia_dr3_source":
                        axis.scatter(
                            np.asarray(overlay_x)[valid_indices],
                            np.asarray(overlay_y)[valid_indices],
                            marker="o",
                            s=30,
                            facecolors="none",
                            edgecolors="#45dcff",
                            linewidths=0.8,
                            alpha=0.9,
                            label="Gaia DR3 positions",
                        )
                    else:
                        axis.scatter(
                            np.asarray(overlay_x)[valid_indices],
                            np.asarray(overlay_y)[valid_indices],
                            marker="s",
                            s=34,
                            facecolors="none",
                            edgecolors="#ffb454",
                            linewidths=0.9,
                            alpha=0.9,
                            label="WISE AllWISE positions",
                        )
            if candidates:
                axis.scatter(
                    [candidate["x_pixel"] for candidate in candidates],
                    [candidate["y_pixel"] for candidate in candidates],
                    marker="o",
                    s=42,
                    facecolors="none",
                    edgecolors="#ffd166",
                    linewidths=1.1,
                )
            axis.set_title(
                f"{metadata['wavelength_um']:.3f} μm"
            )
            if active_overlay_tables:
                axis.legend(
                    loc="upper right",
                    fontsize=7,
                    facecolor="#080d12",
                    edgecolor="#5c6b78",
                    labelcolor="#eff5ff",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            figure.tight_layout(pad=1)

            with st.container(key=f"animation-frame-image-{index}"):
                st.pyplot(
                    figure,
                    use_container_width=False,
                    clear_figure=True,
                )
            plt.close(figure)

        date_col, wavelength_col, detector_col = st.columns(3)

        with date_col:
            st.metric("Observation date", metadata["image_date"])

        with wavelength_col:
            st.metric(
                "Wavelength",
                f"{metadata['wavelength_um']:.3f} μm",
            )

        with detector_col:
            st.metric("Detector", metadata["detector"])

        if show_candidate_peaks:
            st.caption(
                f"Bright peak candidates in this frame: {len(candidates)}. "
                "Yellow circles mark their pixel positions."
            )
            if candidates:
                st.dataframe(
                    candidates[:12],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "peak_value": st.column_config.NumberColumn(
                            "Peak image value",
                            format="%.5g",
                        ),
                        "approx_snr": st.column_config.NumberColumn(
                            "Approx. significance",
                            format="%.2f",
                        ),
                    },
                )

        with st.expander("Frame details and source data"):
            st.caption(
                "The red cross marks the searched coordinate. "
                "It does not track a moving object."
            )
            st.write(f"**Image units:** {metadata['units']}")
            st.write(f"**PSF FWHM:** {metadata['psf_fwhm']}")
            st.write(
                f"**Pixels flagged in this original cutout:** "
                f"{metadata['flagged_percent']:.1f}%"
            )
            st.link_button(
                "Open original FITS product from IRSA",
                str(band_results["access_url"][index]),
                use_container_width=True,
            )

        if st.session_state["playing"]:
            st.session_state["frame_index"] = (
                index + 1
            ) % len(band_results)

    show_frame()

    if active_overlay_tables:
        with st.expander("Catalog reference results"):
            for catalog_id, catalog_table in active_overlay_tables.items():
                catalog_title = (
                    "Gaia DR3" if catalog_id == "gaia_dr3_source"
                    else "WISE AllWISE"
                )
                st.markdown(f"**{catalog_title}: {len(catalog_table)} sources returned**")
                if len(catalog_table):
                    if len(catalog_table) == 1500:
                        st.caption(
                            "The catalog query reached its 1,500-row safety "
                            "limit; additional catalog sources may exist."
                        )
                    st.dataframe(
                        catalog_table[:100].to_pandas(),
                        hide_index=True,
                        use_container_width=True,
                    )
                    if len(catalog_table) > 100:
                        st.caption(
                            "Showing the first 100 table rows. Up to 1,500 "
                            "closest catalog positions are drawn on each frame."
                        )

    with st.expander("View an exploratory mean stack of all prepared dates"):
        st.write(
            "This combines the currently prepared observations in the "
            "selected detector band using an equal-weight mean. Quality "
            "flagged pixels and uncovered areas are excluded."
        )
        st.warning(
            "This is a visual preview, not a calibrated science coadd: "
            "it is not variance-weighted, and the images are not matched "
            "for point-spread function or background. Small wavelength "
            "differences can remain. A moving source may blur in the stack."
        )
        mean_stack = st.session_state.get("prepared_mean_stack")
        if mean_stack is None:
            st.info("Prepare the time-lapse again to create this mean stack.")
        else:
            stack_col1, stack_col2, stack_col3 = st.columns([1, 2, 1])
            with stack_col2:
                figure, axis = plt.subplots(figsize=(5, 5))
                color_map = plt.get_cmap("gray").copy()
                color_map.set_bad("#101522")
                axis.imshow(
                    np.ma.masked_invalid(mean_stack),
                    origin="lower",
                    cmap=color_map,
                    vmin=stretch_low,
                    vmax=stretch_high,
                )
                axis.scatter(
                    reference_x,
                    reference_y,
                    marker="+",
                    s=110,
                    linewidths=2,
                    color="red",
                )
                axis.set_title(
                    f"Equal-weight mean · {len(band_results)} dates"
                )
                axis.set_xticks([])
                axis.set_yticks([])
                figure.tight_layout(pad=1)
                st.pyplot(
                    figure,
                    use_container_width=False,
                    clear_figure=True,
                )
                plt.close(figure)
        st.caption(
            "For the selected pairwise A − B subtraction and blink view, "
            "open the Compare page."
        )


elif st.session_state["workspace"] == "Light Curve":
    st.button(
        "← Back to previous view",
        key="lightcurve_back",
        on_click=return_to_previous_workspace,
    )
    st.subheader("Measure brightness over time")
    st.write(
        "This makes an exploratory light curve from real SPHEREx image "
        "products. It measures a circular aperture at the searched sky "
        "coordinate, estimates local background, and uses the FITS "
        "VARIANCE extension for measurement uncertainty."
    )

    results = st.session_state["results"]
    search_context = st.session_state["search_context"]

    if results is None or len(results) == 0 or search_context is None:
        st.info("Search for SPHEREx observations before measuring a light curve.")
        st.stop()

    bands = sorted(
        {str(value) for value in results["energy_bandpassname"]}
    )
    if st.session_state.get("lightcurve_band") not in bands:
        st.session_state["lightcurve_band"] = bands[0]

    band = st.selectbox(
        "SPHEREx band",
        bands,
        key="lightcurve_band",
    )
    band_results = results[
        results["energy_bandpassname"] == band
    ]
    band_results.sort("t_min")

    aperture_radius = st.slider(
        "Aperture radius (image pixels)",
        min_value=1,
        max_value=6,
        value=2,
        help=(
            "One SPHEREx spectral-image pixel is about 6.15 arcseconds. "
            "The background is estimated from an annulus outside this aperture."
        ),
    )

    active_ra = float(search_context["ra"])
    active_dec = float(search_context["dec"])
    cutout_size_deg = float(search_context["fov_arcmin"]) / 60.0
    urls = [
        make_cutout_url(
            str(band_results["access_url"][index]),
            active_ra,
            active_dec,
            cutout_size_deg,
        )
        for index in range(len(band_results))
    ]
    measurement_signature = (
        "wavelength-aware-lightcurve-v2",
        active_ra,
        active_dec,
        band,
        aperture_radius,
        tuple(urls),
    )

    st.caption(
        f"{len(band_results)} observations · "
        f"{search_context['collection']} · "
        f"RA {active_ra:.6f}°, Dec {active_dec:.6f}°"
    )

    if search_context["target"] in {
        "Barnard's Star (approx. 2026)",
        "WISE 0855",
    }:
        st.warning(
            "The aperture stays at the searched coordinate for every date. "
            "It does not follow a high-proper-motion object, so this light "
            "curve may stop measuring the target as it moves."
        )

    st.warning(
        "This is exploratory aperture photometry, not a confirmed target "
        "measurement. Nearby sources can blend at SPHEREx resolution. "
        "The aperture is not corrected for the fraction of the PSF outside "
        "it, and background or PSF changes can affect the result. Image and "
        "background pixels use separate masks based on the current IRSA "
        "photometry flag guidance; flagged pixels are not all treated alike."
    )

    if st.button("Measure brightness for these dates", type="primary"):
        measured_rows = []
        measurement_failures = []
        progress = st.progress(0, text="Starting photometry…")

        for index, cutout_url in enumerate(urls):
            try:
                image, flags, metadata, header = load_cutout(
                    cutout_url,
                    active_ra,
                    active_dec,
                )
                variance = load_variance_cutout(cutout_url)
                measurement = aperture_measurement(
                    image,
                    flags,
                    variance,
                    header,
                    active_ra,
                    active_dec,
                    aperture_radius,
                )
                measured_rows.append(
                    {
                        "date": str(band_results["date_utc"][index]),
                        "mjd": float(band_results["t_min"][index]),
                        "obs_id": str(band_results["obs_id"][index]),
                        "flux": measurement["flux"],
                        "uncertainty": measurement["uncertainty"],
                        "units": measurement["flux_units"],
                        "wavelength_um": metadata["wavelength_um"],
                        "bandwidth_um": metadata["bandwidth_um"],
                        "detector": metadata["detector"],
                        "aperture_coverage_percent": (
                            measurement["aperture_coverage_percent"]
                        ),
                    }
                )
            except Exception as error:
                measurement_failures.append(
                    (
                        observation_label(band_results, index),
                        str(error).splitlines()[0][:500],
                    )
                )

            progress.progress(
                (index + 1) / len(urls),
                text=f"Measuring observation {index + 1} of {len(urls)}…",
            )

        progress.empty()
        st.session_state["lightcurve_rows"] = measured_rows
        st.session_state["lightcurve_failures"] = measurement_failures
        st.session_state["lightcurve_signature"] = measurement_signature

    if st.session_state.get("lightcurve_signature") == measurement_signature:
        measured_rows = st.session_state.get("lightcurve_rows") or []
        measurement_failures = st.session_state.get(
            "lightcurve_failures") or []

        if measured_rows:
            flux_units = measured_rows[0]["units"]
            same_units = all(
                row["units"] == flux_units for row in measured_rows)

            if same_units:
                wavelengths = np.asarray(
                    [row["wavelength_um"] for row in measured_rows],
                    dtype=float,
                )
                bandwidths = np.asarray(
                    [row["bandwidth_um"] for row in measured_rows],
                    dtype=float,
                )
                median_bandwidth = float(np.nanmedian(bandwidths))
                minimum_wavelength = float(np.nanmin(wavelengths))
                maximum_wavelength = float(np.nanmax(wavelengths))

                st.subheader("Match the wavelength before comparing dates")
                st.caption(
                    "A detector band covers multiple wavelengths. The plot "
                    "below includes only measurements near the selected "
                    "wavelength, rather than mixing the full detector range."
                )

                if maximum_wavelength > minimum_wavelength:
                    target_wavelength = st.number_input(
                        "Wavelength center (μm)",
                        min_value=minimum_wavelength,
                        max_value=maximum_wavelength,
                        value=float(np.nanmedian(wavelengths)),
                        step=0.005,
                        format="%.4f",
                    )
                else:
                    target_wavelength = minimum_wavelength
                    st.metric(
                        "Wavelength center (μm)",
                        f"{target_wavelength:.4f}",
                    )

                tolerance_options = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
                default_tolerance = min(
                    tolerance_options,
                    key=lambda value: abs(value - median_bandwidth / 2),
                )
                wavelength_tolerance = st.select_slider(
                    "Allowed wavelength difference (μm)",
                    options=tolerance_options,
                    value=default_tolerance,
                    help=(
                        "A narrower range is a fairer time comparison. "
                        "The default is approximately half the median "
                        "channel bandwidth."
                    ),
                )
                matched_rows = [
                    row
                    for row in measured_rows
                    if abs(row["wavelength_um"] - target_wavelength)
                    <= wavelength_tolerance
                ]
                matched_dates = {
                    row["date"][:10]
                    for row in matched_rows
                }
                summary_col1, summary_col2 = st.columns(2)
                summary_col1.metric(
                    "Wavelength-matched measurements",
                    len(matched_rows),
                )
                summary_col2.metric(
                    "Distinct dates matched",
                    len(matched_dates),
                )

                if len(matched_dates) >= 3:
                    dates = Time(
                        [row["mjd"] for row in matched_rows],
                        format="mjd",
                    ).to_datetime()
                    flux_values = np.asarray(
                        [row["flux"] for row in matched_rows], dtype=float
                    )
                    errors = np.asarray(
                        [row["uncertainty"] for row in matched_rows],
                        dtype=float,
                    )

                    figure, axis = plt.subplots(figsize=(9, 4.5))
                    axis.errorbar(
                        dates,
                        flux_values,
                        yerr=errors,
                        fmt="o",
                        capsize=3,
                        color="#3bb8ff",
                        ecolor="#a9dfff",
                    )
                    axis.set_title(
                        f"{search_context['target']} · {band} · "
                        f"{target_wavelength:.3f} μm"
                    )
                    axis.set_xlabel("Observation date (UTC)")
                    axis.set_ylabel(f"Aperture flux ({flux_units})")
                    axis.grid(alpha=0.25)
                    figure.autofmt_xdate()
                    figure.tight_layout()
                    st.pyplot(
                        figure,
                        use_container_width=True,
                        clear_figure=True,
                    )
                    plt.close(figure)

                    st.dataframe(
                        matched_rows,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "flux": st.column_config.NumberColumn(
                                "Aperture flux",
                                format="%.5g",
                            ),
                            "uncertainty": st.column_config.NumberColumn(
                                "Uncertainty",
                                format="%.3g",
                            ),
                            "wavelength_um": st.column_config.NumberColumn(
                                "Wavelength (μm)",
                                format="%.4f",
                            ),
                            "bandwidth_um": st.column_config.NumberColumn(
                                "Channel width (μm)",
                                format="%.4f",
                            ),
                            "aperture_coverage_percent": st.column_config.NumberColumn(
                                "Usable aperture (%)",
                                format="%.1f",
                            ),
                        },
                    )

                    matched_csv = io.StringIO()
                    matched_writer = csv.DictWriter(
                        matched_csv,
                        fieldnames=list(matched_rows[0]),
                    )
                    matched_writer.writeheader()
                    matched_writer.writerows(matched_rows)
                    st.download_button(
                        "Download wavelength-matched measurements",
                        data=matched_csv.getvalue(),
                        file_name="spherex_wavelength_matched_lightcurve.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        "There are fewer than three distinct dates at this "
                        "wavelength. The app will not present these points "
                        "as a temporal light curve. Try a nearby wavelength, "
                        "but keep the range narrow."
                    )

                with st.expander("Review all measured samples"):
                    st.dataframe(
                        measured_rows,
                        hide_index=True,
                        use_container_width=True,
                    )
            else:
                st.error(
                    "The returned measurements have inconsistent units, "
                    "so a single light curve cannot be plotted."
                )

        if measurement_failures:
            st.warning(
                f"Could not measure {len(measurement_failures)} observation(s)."
            )
            for failed_label, reason in measurement_failures[:8]:
                st.caption(f"{failed_label} — {reason}")

        if not measured_rows and not measurement_failures:
            st.info("No measurements were produced. Try another band or target.")

    st.caption(
        "The plotted points use the same searched coordinate and selected "
        "detector band. The listed wavelength can vary from date to date, "
        "so interpret small changes with that spectral variation in mind."
    )


else:
    st.button(
        "← Back to previous view",
        key="track_back",
        on_click=return_to_previous_workspace,
    )
    st.subheader("Track a source candidate across dates")
    st.write(
        "Choose a bright peak in one observation. The app links nearby "
        "peaks in aligned images and checks whether the measured drift is "
        "large enough compared with the scatter to interpret. The result is "
        "a candidate for inspection, not an automatic discovery."
    )

    results = st.session_state["results"]
    search_context = st.session_state["search_context"]

    if results is None or len(results) == 0 or search_context is None:
        st.info("Search and prepare observations before tracking a source.")
        st.stop()

    bands = sorted(
        {str(value) for value in results["energy_bandpassname"]}
    )
    if st.session_state.get("track_band") not in bands:
        st.session_state["track_band"] = bands[0]

    track_band = st.selectbox(
        "SPHEREx band to track",
        bands,
        key="track_band",
    )
    band_results = results[
        results["energy_bandpassname"] == track_band
    ]
    band_results.sort("t_min")

    active_ra = float(search_context["ra"])
    active_dec = float(search_context["dec"])
    cutout_size_deg = float(search_context["fov_arcmin"]) / 60.0
    track_urls = [
        make_cutout_url(
            str(band_results["access_url"][index]),
            active_ra,
            active_dec,
            cutout_size_deg,
        )
        for index in range(len(band_results))
    ]
    track_signature = (
        "common-sky-grid-fixed-stretch-stack-track-v3",
        tuple(track_urls),
    )

    if (
        st.session_state["prepared_signature"] != track_signature
        or len(st.session_state["prepared_frames"]) != len(band_results)
    ):
        st.info(
            "First open **Animation** and prepare this band. Tracking uses "
            "those aligned frames so apparent telescope pointing shifts are "
            "reduced before candidate matching."
        )
        st.stop()

    threshold = st.slider(
        "Peak detection threshold",
        min_value=3.0,
        max_value=8.0,
        value=5.0,
        step=0.5,
        help="A candidate must be brighter than this approximate number of background-noise levels.",
    )
    max_step = st.slider(
        "Maximum match distance between observations (pixels)",
        min_value=1.0,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="A candidate must stay within this distance of the predicted position. Time gaps do not automatically widen the search.",
    )

    reference_pixel = st.session_state.get("prepared_reference_pixel")
    with st.spinner("Cassiopeia is checking the prepared frames for bright candidates…"):
        detections_by_frame = [
            detect_bright_peaks(frame, threshold, max_sources=80, reference_pixel=reference_pixel)
            for frame in st.session_state["prepared_frames"]
        ]

    mjds = np.asarray(band_results["t_min"], dtype=float)
    anchor_index = st.selectbox(
        "Choose an observation containing the source",
        range(len(band_results)),
        format_func=lambda index: observation_label(band_results, index),
    )
    anchor_candidates = detections_by_frame[anchor_index]

    if not anchor_candidates:
        st.info(
            "No bright peaks passed the threshold in this frame. Try a lower "
            "threshold or another observation."
        )
        st.stop()

    if st.session_state["track_candidate_index"] >= len(anchor_candidates):
        st.session_state["track_candidate_index"] = 0

    candidate_index = st.selectbox(
        "Choose the peak to follow",
        range(len(anchor_candidates)),
        format_func=lambda index: (
            f"Candidate {index + 1} · "
            f"pixel ({anchor_candidates[index]['x_pixel']:.1f}, "
            f"{anchor_candidates[index]['y_pixel']:.1f}) · "
            f"{anchor_candidates[index]['approx_snr']:.1f} approximate SNR · "
            f"{anchor_candidates[index].get('search_offset_px', 0):.1f} px from searched coordinate"
        ),
        key="track_candidate_index",
    )
    anchor_candidate = anchor_candidates[candidate_index]
    st.caption(
        "Candidates are ordered by distance from the searched coordinate. "
        "That helps choose a nearby source but does not identify it as your target. "
        "Centroids are approximate image-based positions, not precision astrometry."
    )

    track = link_candidate_track(
        detections_by_frame,
        mjds,
        anchor_index,
        anchor_candidate,
        max_step,
    )

    if len(track) < 2:
        st.warning(
            "The selected peak matched in fewer than two observations. "
            "Try a different source, lower threshold, or larger search distance."
        )
        st.stop()

    header_text = st.session_state.get("prepared_wcs_header")
    if not header_text:
        st.error("The reference WCS is missing. Prepare the animation again.")
        st.stop()

    reference_header = fits.Header.fromstring(header_text, sep="\n")
    reference_wcs = WCS(reference_header).celestial
    matched_indices = list(track)
    x_pixels = np.asarray(
        [track[index]["x_pixel"] for index in matched_indices], dtype=float
    )
    y_pixels = np.asarray(
        [track[index]["y_pixel"] for index in matched_indices], dtype=float
    )
    matched_mjds = mjds[matched_indices]
    time_span_days = float(np.ptp(matched_mjds))
    distinct_dates = len(
        {
            calendar_date(band_results["date_utc"][index])
            for index in matched_indices
        }
    )
    sky_positions = reference_wcs.pixel_to_world(x_pixels, y_pixels)
    anchor_position = sky_positions[matched_indices.index(anchor_index)]
    east_offsets, north_offsets = anchor_position.spherical_offsets_to(
        sky_positions
    )
    east_arcsec = east_offsets.to_value(u.arcsec)
    north_arcsec = north_offsets.to_value(u.arcsec)

    track_rows = []
    for row_index, observation_index in enumerate(matched_indices):
        track_rows.append(
            {
                "date_utc": str(band_results["date_utc"][observation_index]),
                "obs_id": str(band_results["obs_id"][observation_index]),
                "ra_deg": float(sky_positions.ra.deg[row_index]),
                "dec_deg": float(sky_positions.dec.deg[row_index]),
                "x_pixel": float(x_pixels[row_index]),
                "y_pixel": float(y_pixels[row_index]),
                "peak_value": track[observation_index]["peak_value"],
                "approx_snr": track[observation_index]["approx_snr"],
            }
        )

    days_from_anchor = matched_mjds - float(mjds[anchor_index])
    east_rate = north_rate = rate_uncertainty = None
    fit_rms = None
    epochs = {}
    for position_index, observation_index in enumerate(matched_indices):
        date_key = calendar_date(band_results["date_utc"][observation_index])
        epochs.setdefault(date_key, {"mjd": [], "east": [], "north": []})
        epochs[date_key]["mjd"].append(float(matched_mjds[position_index]))
        epochs[date_key]["east"].append(float(east_arcsec[position_index]))
        epochs[date_key]["north"].append(float(north_arcsec[position_index]))

    epoch_mjds = np.asarray(
        [np.median(epochs[date]["mjd"]) for date in sorted(epochs)],
        dtype=float,
    )
    epoch_east = np.asarray(
        [np.median(epochs[date]["east"]) for date in sorted(epochs)],
        dtype=float,
    )
    epoch_north = np.asarray(
        [np.median(epochs[date]["north"]) for date in sorted(epochs)],
        dtype=float,
    )
    epoch_years_from_anchor = (
        epoch_mjds - float(mjds[anchor_index])
    ) / 365.25

    if len(epoch_mjds) >= 3 and np.ptp(epoch_mjds) > 0:
        east_rate, east_intercept = np.polyfit(
            epoch_years_from_anchor,
            epoch_east,
            1,
        )
        north_rate, north_intercept = np.polyfit(
            epoch_years_from_anchor,
            epoch_north,
            1,
        )
        east_fit = east_rate * epoch_years_from_anchor + east_intercept
        north_fit = north_rate * epoch_years_from_anchor + north_intercept
        east_residual = epoch_east - east_fit
        north_residual = epoch_north - north_fit
        fit_rms = float(
            np.sqrt(np.mean(east_residual**2 + north_residual**2))
        )
        time_variance = float(
            np.sum(
                (epoch_years_from_anchor
                 - np.mean(epoch_years_from_anchor)) ** 2
            )
        )
        if time_variance > 0 and len(epoch_mjds) > 2:
            east_error = np.sqrt(
                np.sum(east_residual**2)
                / (len(epoch_mjds) - 2)
                / time_variance
            )
            north_error = np.sqrt(
                np.sum(north_residual**2)
                / (len(epoch_mjds) - 2)
                / time_variance
            )
            rate_uncertainty = float(
                np.hypot(east_error, north_error)
            )

        total_rate = float(np.hypot(east_rate, north_rate))
        fitted_displacement = total_rate * time_span_days / 365.25
        scatter_to_motion = (
            fit_rms / fitted_displacement
            if fitted_displacement > 0
            else float("inf")
        )
        motion_resolved = distinct_dates >= 3 and fitted_displacement > 2 * fit_rms
        rate_col1, rate_col2, rate_col3 = st.columns(3)
        rate_col1.metric(
            "Motion fit",
            f"{total_rate:.3f} arcsec/year" if motion_resolved else "Unresolved",
        )
        rate_col2.metric(
            "East / north components",
            f"{east_rate:.3f} / {north_rate:.3f} arcsec/year" if motion_resolved else "—",
        )
        rate_col3.metric(
            "Linear-fit scatter",
            f"{fit_rms:.2f} arcsec",
        )
        if motion_resolved and rate_uncertainty is not None:
            st.caption(
                f"Approximate fit uncertainty: ±{rate_uncertainty:.3f} "
                "arcsec/year. This reflects fit scatter only; it does not "
                "include a full astrometric error model."
            )
        st.write(
            f"**Time baseline:** {time_span_days:.2f} days across "
            f"{distinct_dates} calendar date(s). **Fitted travel across "
            f"that baseline:** {fitted_displacement:.2f} arcsec."
        )
        if distinct_dates < 3:
            st.warning(
                "Reliability check: the track covers fewer than three "
                "calendar dates. Repeated exposures on one night do not "
                "replace a longer time baseline. Treat the fitted rate as "
                "provisional."
            )
        elif fitted_displacement <= 2 * fit_rms:
            st.error(
                "Reliability check: the fitted travel is not clearly larger "
                "than the positional scatter. The motion is unresolved in "
                "this sequence; do not interpret the arcsec/year number as "
                "a measurement. This is a conservative screening rule, not "
                "a formal statistical significance test."
            )
        elif scatter_to_motion >= 0.5:
            st.warning(
                "Reliability check: positional scatter is a substantial "
                "fraction of the fitted travel. This is a weak candidate "
                "track; inspect the images and nearby sources before "
                "interpreting the rate."
            )
        else:
            st.info(
                "Reliability check: the fitted travel exceeds the scatter "
                "screen. This supports further inspection, but it does not "
                "confirm the source identity or establish a catalog-grade "
                "proper motion."
            )
    else:
        st.info(
            "At least three distinct calendar dates are needed for a motion "
            "fit. The current track is provisional."
        )

    st.caption(
        "The motion fit uses the median position from each calendar date, "
        "so several near-consecutive exposures on one date do not count as "
        "independent time samples."
    )

    st.write(
        f"**Matched observations:** {len(track_rows)} of {len(band_results)} "
        f"across {distinct_dates} calendar date(s)."
    )
    st.dataframe(track_rows, hide_index=True, use_container_width=True)

    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        east_arcsec,
        north_arcsec,
        c=matched_mjds,
        cmap="viridis",
        s=65,
    )
    axis.plot(east_arcsec, north_arcsec, color="#52c7ff", alpha=0.75)
    axis.scatter(0, 0, marker="+", s=130, color="#ff526f", label="Anchor")
    axis.set_xlabel("East offset from anchor (arcsec)")
    axis.set_ylabel("North offset from anchor (arcsec)")
    axis.set_title("Candidate path on the aligned sky grid")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="MJD")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True, clear_figure=True)
    plt.close(figure)

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(track_rows[0]))
    writer.writeheader()
    writer.writerows(track_rows)
    st.download_button(
        "Download candidate track as CSV",
        data=csv_buffer.getvalue(),
        file_name="spherex_candidate_track.csv",
        mime="text/csv",
    )

    st.warning(
        "This is a nearest-peak candidate track, not a confirmed moving "
        "object. Crowded fields, variable brightness, image artifacts, and "
        "mis-matched peaks can create false tracks. First blink the actual "
        "frames and check that the same source is selected each time. Then "
        "compare nearby stationary sources and verify the candidate against "
        "an astrometric catalog."
    )

