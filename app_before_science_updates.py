import csv
import io
import socket
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp

from spherex_helpers import flagged_pixels, load_cutout, observation_label
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


def collection_profiles(names):
    profiles = {}

    def release_number(name, marker):
        suffix = name.split(marker, 1)[1]
        digits = "".join(character for character in suffix if character.isdigit())
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
            raise ValueError("The searched coordinate is outside the cube WCS.")

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
    valid = (
        np.isfinite(image_crop)
        & np.isfinite(variance_crop)
        & (variance_crop >= 0)
        & ~flagged_pixels(flags_crop)
    )
    aperture_valid = aperture & valid
    annulus_valid = annulus & valid
    aperture_coverage = (
        np.count_nonzero(aperture_valid) / max(1, np.count_nonzero(aperture))
    )

    if aperture_coverage < 0.8:
        aperture_count = max(1, int(np.count_nonzero(aperture)))
        flagged_count = int(
            np.count_nonzero(aperture & flagged_pixels(flags_crop))
        )
        invalid_image_count = int(
            np.count_nonzero(aperture & ~np.isfinite(image_crop))
        )
        invalid_variance_count = int(
            np.count_nonzero(
                aperture
                & np.isfinite(image_crop)
                & (~np.isfinite(variance_crop) | (variance_crop < 0))
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
        raise ValueError("Too few usable pixels remain in the background annulus.")

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


def detect_bright_peaks(image, sigma_threshold, max_sources=100):
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
    is_peak = np.isfinite(image) & (
        image > background + sigma_threshold * noise
    )
    height, width = image.shape

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = np.full_like(working, -np.inf)
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = source_y0 + dy
            target_y1 = source_y1 + dy
            target_x0 = source_x0 + dx
            target_x1 = source_x1 + dx
            neighbor[target_y0:target_y1, target_x0:target_x1] = working[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
            is_peak &= working >= neighbor

    peak_pixels = np.argwhere(is_peak)
    if peak_pixels.size == 0:
        return []

    peak_pixels = sorted(
        peak_pixels,
        key=lambda position: image[position[0], position[1]],
        reverse=True,
    )
    selected = []
    min_separation_squared = 2.5**2

    for y, x in peak_pixels:
        if any(
            (x - prior_x) ** 2 + (y - prior_y) ** 2 < min_separation_squared
            for prior_y, prior_x in selected
        ):
            continue
        selected.append((int(y), int(x)))
        if len(selected) >= max_sources:
            break

    return [
        {
            "x_pixel": x,
            "y_pixel": y,
            "peak_value": float(image[y, x]),
            "approx_snr": float((image[y, x] - background) / noise),
        }
        for y, x in selected
    ]


def link_candidate_track(detections_by_frame, mjds, anchor_index, anchor_candidate, max_step_px):
    track = {anchor_index: anchor_candidate}
    median_cadence = float(np.nanmedian(np.diff(mjds))) if len(mjds) > 1 else 1.0
    if not np.isfinite(median_cadence) or median_cadence <= 0:
        median_cadence = 1.0

    for direction in (-1, 1):
        indices = (
            range(anchor_index - 1, -1, -1)
            if direction < 0
            else range(anchor_index + 1, len(mjds))
        )
        history = [anchor_index]

        for index in indices:
            prior_index = history[-1]
            prior = track[prior_index]
            predicted_x = float(prior["x_pixel"])
            predicted_y = float(prior["y_pixel"])

            if len(history) >= 2:
                earlier_index = history[-2]
                earlier = track[earlier_index]
                elapsed = float(mjds[prior_index] - mjds[earlier_index])
                target_elapsed = float(mjds[index] - mjds[prior_index])
                if elapsed != 0:
                    predicted_x += (
                        (prior["x_pixel"] - earlier["x_pixel"])
                        / elapsed
                    ) * target_elapsed
                    predicted_y += (
                        (prior["y_pixel"] - earlier["y_pixel"])
                        / elapsed
                    ) * target_elapsed

            gap_days = abs(float(mjds[index] - mjds[prior_index]))
            allowed_distance = max_step_px * max(
                1.0,
                gap_days / median_cadence,
            )
            candidates = detections_by_frame[index]
            if not candidates:
                continue

            nearest = min(
                candidates,
                key=lambda candidate: (
                    (candidate["x_pixel"] - predicted_x) ** 2
                    + (candidate["y_pixel"] - predicted_y) ** 2
                ),
            )
            distance = float(
                np.hypot(
                    nearest["x_pixel"] - predicted_x,
                    nearest["y_pixel"] - predicted_y,
                )
            )
            if distance <= allowed_distance:
                track[index] = nearest
                history.append(index)

    return dict(sorted(track.items()))


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
                with st.spinner("Finding the DR1 cube tile in IRSA…"):
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
                                "Loading the selected channel from the cube tile…"
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
                                st.info("This channel has no finite image pixels.")
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
                            st.error(f"Could not load this cube channel: {error}")

                        st.divider()
                        st.markdown("#### Quick-look spectrum at the searched coordinate")
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
                                    "Measuring the real cube channels…"
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
                                st.error(f"Spectrum extraction failed: {error}")

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
                            spectrum_axis.set_ylabel("Background-subtracted flux (mJy)")
                            spectrum_axis.set_title("SPHEREx DR1 quick-look spectrum")
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


state_defaults = {
    "target": "Custom coordinates",
    "ra_input": TARGETS["Custom coordinates"][0],
    "dec_input": TARGETS["Custom coordinates"][1],
    "collection_profile": None,
    "cube_results": None,
    "cube_results_context": None,
    "loaded_cube_url": None,
    "cube_spectrum": None,
    "cube_spectrum_key": None,
    "fov_arcmin": 6,
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
    "playback_interval": 0.75,
    "lightcurve_rows": None,
    "lightcurve_signature": None,
    "track_candidate_index": 0,
}

for key, value in state_defaults.items():
    st.session_state.setdefault(key, value)


apply_styles()
render_page_header(
    "explore",
    "Explore the infrared sky over time",
    "Search real SPHEREx observations, prepare a time-lapse, "
    "and inspect how the sky changes.",
)

st.radio(
    "Choose a workspace",
    ["Search & Results", "Animation", "Light Curve", "Track Sources"],
    horizontal=True,
    key="workspace",
    label_visibility="collapsed",
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

    refresh_release_list = st.button(
        "Refresh available releases",
        help=(
            "Ask IRSA for its current SPHEREx image collections now. "
            "The list also refreshes automatically at least hourly."
        ),
    )
    if refresh_release_list:
        available_image_collections.clear()

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
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")

            with st.spinner("Searching the IRSA archive…"):
                new_results = Irsa.query_sia(
                    pos=(coord, 1 * u.arcsec),
                    collection=selected_collections,
                    data_type="image",
                )

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
            st.rerun()

        except Exception as error:
            st.session_state["results"] = None
            st.session_state["search_context"] = None
            reset_animation()
            st.error(f"The IRSA search failed: {error}")

    results = st.session_state["results"]
    search_context = st.session_state["search_context"]

    if results is not None:
        st.divider()
        st.subheader("Observation results")
        st.metric("Observations found", len(results))

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
                "Your results are ready. Choose **Animation** above "
                "to prepare and play the observations."
            )

        render_cube_explorer(search_context)


elif st.session_state["workspace"] == "Animation":
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
            options=[0.5, 0.75, 1.0, 1.5],
            format_func=lambda seconds: {
                0.5: "Fast · 0.5 sec",
                0.75: "Medium · 0.75 sec",
                1.0: "Relaxed · 1 sec",
                1.5: "Slow · 1.5 sec",
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

        progress = st.progress(0, text="Starting preparation…")
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

        with ThreadPoolExecutor(max_workers=2) as executor:
            for index, cutout_url in enumerate(cutout_urls):
                future = executor.submit(
                    load_cutout_with_retries,
                    cutout_url,
                    active_ra,
                    active_dec,
                )
                future_indices[future] = index

            for future in as_completed(future_indices):
                index = future_indices[future]
                try:
                    downloaded[index] = future.result()
                except Exception as error:
                    failures.append(
                        (
                            index,
                            str(error).splitlines()[0][:180],
                        )
                    )

                completed_count += 1
                progress.progress(
                    completed_count / len(cutout_urls),
                    text=(
                        f"Downloading cutout {completed_count} "
                        f"of {len(cutout_urls)}…"
                    ),
                )

        if not failures:
            for index, loaded_cutout in enumerate(downloaded):
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

                progress.progress(
                    (index + 1) / len(cutout_urls),
                    text=(
                        f"Aligning frame {index + 1} "
                        f"of {len(cutout_urls)}…"
                    ),
                )

        progress.empty()
        st.session_state["preload_failures"] = failures
        st.session_state["preload_failure_signature"] = (
            preparation_signature
        )

        if not failures and all(
            frame is not None for frame in aligned_frames
        ):
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
                for frame in aligned_frames:
                    valid = np.isfinite(frame)
                    stack_sum[valid] += frame[valid]
                    stack_count[valid] += 1
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
            else:
                st.session_state["preload_failures"] = [
                    (0, "No usable unmasked image pixels were found.")
                ]

        st.rerun()

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

    play_col, pause_col, replay_col, previous_col, next_col = st.columns(5)

    if play_col.button(
        "Play / resume",
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
        "Previous",
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

    image_view_col, detection_col = st.columns(2)
    with image_view_col:
        image_appearance = st.radio(
            "Image appearance",
            ["Native pixels", "Smoothed preview"],
            horizontal=True,
            help=(
                "Smoothing changes only how the same pixels are drawn. "
                "It does not add spatial detail."
            ),
        )
    with detection_col:
        show_candidate_peaks = st.checkbox(
            "Mark bright source candidates",
            value=False,
        )

    detection_threshold = 5.0
    if show_candidate_peaks:
        detection_threshold = st.slider(
            "Candidate brightness threshold",
            min_value=3.0,
            max_value=8.0,
            value=5.0,
            step=0.5,
            help="Threshold is an approximate number of image-noise levels above the background.",
        )
        st.caption(
            "These are bright local peaks for human inspection, not confirmed "
            "astronomical sources or catalog identifications."
        )

    interpolation = (
        "bilinear" if image_appearance == "Smoothed preview" else "nearest"
    )

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

        image_col1, image_col2, image_col3 = st.columns([1, 2, 1])

        with image_col2:
            figure, axis = plt.subplots(figsize=(4.5, 4.5))
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
            axis.set_xticks([])
            axis.set_yticks([])
            figure.tight_layout(pad=1)

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
        "it, and background or PSF changes can affect the result."
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
        measurement_failures = st.session_state.get("lightcurve_failures") or []

        if measured_rows:
            flux_units = measured_rows[0]["units"]
            same_units = all(row["units"] == flux_units for row in measured_rows)

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
        "Maximum search distance per typical time step (pixels)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Increase this for faster motion, but a larger value can link unrelated sources.",
    )

    with st.spinner("Finding bright peaks in the prepared images…"):
        detections_by_frame = [
            detect_bright_peaks(frame, threshold, max_sources=80)
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
            f"pixel ({anchor_candidates[index]['x_pixel']}, "
            f"{anchor_candidates[index]['y_pixel']}) · "
            f"{anchor_candidates[index]['approx_snr']:.1f} noise levels"
        ),
        key="track_candidate_index",
    )
    anchor_candidate = anchor_candidates[candidate_index]

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
            str(band_results["date_utc"][index]).split("T")[0]
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
        date_key = str(band_results["date_utc"][observation_index]).split("T")[0]
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
        rate_col1, rate_col2, rate_col3 = st.columns(3)
        rate_col1.metric("Estimated motion", f"{total_rate:.3f} arcsec/year")
        rate_col2.metric(
            "East / north components",
            f"{east_rate:.3f} / {north_rate:.3f} arcsec/year",
        )
        rate_col3.metric(
            "Linear-fit scatter",
            f"{fit_rms:.2f} arcsec",
        )
        if rate_uncertainty is not None:
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
