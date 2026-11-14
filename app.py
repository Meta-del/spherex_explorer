import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
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
    "https://irsa.ipac.caltech.edu/spherex-archive-documentation/"
)
SPHEREX_MISSION = "https://spherex.caltech.edu/"


def reset_animation():
    st.session_state["playing"] = False
    st.session_state["prepared_signature"] = None
    st.session_state["prepared_frames"] = []
    st.session_state["prepared_metadata"] = []
    st.session_state["prepared_stretch"] = None
    st.session_state["prepared_reference_pixel"] = None
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


def make_observation_table(table):
    display_columns = [
        name
        for name in [
            "obs_id",
            "date_utc",
            "energy_bandpassname",
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


state_defaults = {
    "target": "Custom coordinates",
    "ra_input": TARGETS["Custom coordinates"][0],
    "dec_input": TARGETS["Custom coordinates"][1],
    "collection": "spherex_qr3",
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
    "preload_failures": [],
    "preload_failure_signature": None,
    "workspace": "Search & Results",
    "playback_interval": 0.75,
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
    ["Search & Results", "Animation"],
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

    search_option_col1, search_option_col2 = st.columns(2)

    with search_option_col1:
        collection = st.selectbox(
            "Data collection",
            [
                "spherex_qr3",
                "spherex_qr3_deep",
                "spherex_qr2",
                "spherex_qr2_deep",
            ],
            key="collection",
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
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")

            with st.spinner("Searching the IRSA archive…"):
                new_results = Irsa.query_sia(
                    pos=(coord, 1 * u.arcsec),
                    collection=collection,
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
            st.session_state["search_context"] = {
                "ra": ra,
                "dec": dec,
                "collection": collection,
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
                or collection != search_context["collection"]
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


else:
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
        "common-sky-grid-fixed-stretch-v1",
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
        st.session_state["preload_failures"] = []

        progress = st.progress(0, text="Starting preparation…")
        failures = []
        aligned_frames = [None] * len(cutout_urls)
        frame_metadata = [None] * len(cutout_urls)
        reference_wcs = None
        reference_shape = None
        reference_pixel = None

        for index, cutout_url in enumerate(cutout_urls):
            try:
                image, flags, metadata, header = load_cutout(
                    cutout_url,
                    active_ra,
                    active_dec,
                )

                image = np.asarray(image, dtype=float)
                bad_pixels = flagged_pixels(flags)
                usable_image = image.copy()
                usable_image[bad_pixels] = np.nan
                image_wcs = WCS(header).celestial

                if reference_wcs is None:
                    reference_wcs = image_wcs
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

            except Exception as error:
                failures.append(
                    (
                        index,
                        str(error).splitlines()[0][:180],
                    )
                )

            progress.progress(
                (index + 1) / len(cutout_urls),
                text=(
                    f"Loading and aligning frame {index + 1} "
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
