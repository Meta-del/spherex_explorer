from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp

from spherex_helpers import flagged_pixels, load_cutout, observation_label
from spherex_ui import apply_styles, render_page_header


st.set_page_config(
    page_title="Compare SPHEREx Images",
    page_icon="🌓",
    layout="wide",
)


def reset_comparison_indices():
    st.session_state["compare_a_index"] = 0
    st.session_state["compare_b_index"] = 0
    st.session_state["blink_playing"] = False


def stop_blink():
    st.session_state["blink_playing"] = False


def make_cutout_url(image_url, ra, dec, size_deg):
    separator = "&" if "?" in image_url else "?"
    return f"{image_url}{separator}center={ra},{dec}&size={size_deg}"


def make_shared_stretch(image_a, image_b):
    finite_a = image_a[np.isfinite(image_a)]
    finite_b = image_b[np.isfinite(image_b)]

    if finite_a.size == 0 or finite_b.size == 0:
        raise ValueError("These cutouts do not contain usable image values.")

    combined = np.concatenate([finite_a, finite_b])
    low, high = np.nanpercentile(combined, [5, 99])

    if high <= low:
        high = low + 1

    def stretch(data):
        data = np.nan_to_num(
            data,
            nan=low,
            posinf=high,
            neginf=low,
        )
        return np.clip((data - low) / (high - low), 0, 1)

    return stretch(image_a), stretch(image_b)


def draw_observation(
    image,
    wcs,
    sky_position,
    label,
    wavelength_um,
    figsize=(5, 5),
):
    x, y = wcs.world_to_pixel(sky_position)
    background = "#050712"

    figure, axis = plt.subplots(
        figsize=figsize,
        facecolor=background,
    )
    axis.set_facecolor(background)

    color_map = plt.get_cmap("gray").copy()
    color_map.set_bad(background)

    axis.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        cmap=color_map,
        vmin=0,
        vmax=1,
    )
    axis.scatter(
        x,
        y,
        marker="+",
        s=120,
        linewidths=2,
        color="#ff526f",
    )
    axis.set_title(
        f"{label}\n{wavelength_um:.3f} μm",
        color="#f3f7ff",
        pad=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(colors="#f3f7ff")
    figure.tight_layout(pad=1)

    st.pyplot(
        figure,
        use_container_width=False,
        clear_figure=True,
    )
    plt.close(figure)


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


st.session_state.setdefault("compare_a_index", 0)
st.session_state.setdefault("compare_b_index", 0)
st.session_state.setdefault("compare_band", None)
st.session_state.setdefault("blink_playing", False)
st.session_state.setdefault("blink_index", 0)


apply_styles()
render_page_header(
    "compare",
    "Compare SPHEREx observations",
    "Choose two dates, inspect the images and their difference, "
    "then record what you noticed.",
)


st.subheader("1 · Choose two observations")

results = st.session_state.get("results")
search_context = st.session_state.get("search_context")

if results is None or len(results) == 0 or search_context is None:
    st.info(
        "First search for a target on the Explore & Animate page. "
        "Then return here to compare its observations."
    )
    st.page_link("app.py", label="Go to Explore & Animate", icon="🔭")
    st.stop()

ra = float(search_context["ra"])
dec = float(search_context["dec"])
cutout_size_deg = float(search_context["fov_arcmin"]) / 60.0

st.caption(
    f"Target: {search_context['target']} · "
    f"Collection: {search_context['collection']} · "
    f"Field of view: {search_context['fov_arcmin']} arcminutes"
)

bands = sorted(
    {str(value) for value in results["energy_bandpassname"]}
)

if st.session_state["compare_band"] not in bands:
    st.session_state["compare_band"] = bands[0]

selected_band = st.selectbox(
    "SPHEREx band",
    bands,
    key="compare_band",
    on_change=reset_comparison_indices,
)

band_results = results[
    results["energy_bandpassname"] == selected_band
]
band_results.sort("t_min")

if len(band_results) == 0:
    st.info("There are no observations available in this band.")
    st.stop()

for key in ["compare_a_index", "compare_b_index"]:
    st.session_state[key] = min(
        st.session_state[key],
        len(band_results) - 1,
    )

choice_col_a, choice_col_b = st.columns(2)

with choice_col_a:
    index_a = st.selectbox(
        "Observation A",
        range(len(band_results)),
        format_func=lambda index: observation_label(
            band_results, index
        ),
        key="compare_a_index",
    )

with choice_col_b:
    index_b = st.selectbox(
        "Observation B",
        range(len(band_results)),
        format_func=lambda index: observation_label(
            band_results, index
        ),
        key="compare_b_index",
    )

different_observations = index_a != index_b

if not different_observations:
    st.warning(
        "Both selectors point to the same observation. "
        "Choose a different date for a useful comparison."
    )

time_gap_days = abs(
    float(band_results["t_min"][index_a])
    - float(band_results["t_min"][index_b])
)
st.metric("Time between observations", f"{time_gap_days:.2f} days")

url_a = str(band_results["access_url"][index_a])
url_b = str(band_results["access_url"][index_b])

cutout_a = make_cutout_url(url_a, ra, dec, cutout_size_deg)
cutout_b = make_cutout_url(url_b, ra, dec, cutout_size_deg)

try:
    with st.spinner("Loading the two SPHEREx cutouts…"):
        image_a, flags_a, metadata_a, header_a = load_cutout(
            cutout_a, ra, dec
        )
        image_b, flags_b, metadata_b, header_b = load_cutout(
            cutout_b, ra, dec
        )

    images_loaded = True
    display_a, display_b = make_shared_stretch(image_a, image_b)

    wcs_a = WCS(header_a).celestial
    wcs_b = WCS(header_b).celestial
    sky_position = SkyCoord(ra=ra, dec=dec, unit="deg")

except Exception as error:
    st.error(f"Could not load the selected images: {error}")
    st.stop()


st.divider()
st.subheader("2 · Inspect the images")
st.write(
    "Look for a source that changes position or brightness. "
    "Nearby stars can help you judge whether the images line up."
)

view_mode = st.radio(
    "Image view",
    ["Side by side", "Blink between images"],
    horizontal=True,
    key="compare_view_mode",
    on_change=stop_blink,
)

if view_mode == "Side by side":
    image_col_a, image_col_b = st.columns(2)

    with image_col_a:
        draw_observation(
            display_a,
            wcs_a,
            sky_position,
            observation_label(band_results, index_a),
            metadata_a["wavelength_um"],
        )

    with image_col_b:
        draw_observation(
            display_b,
            wcs_b,
            sky_position,
            observation_label(band_results, index_b),
            metadata_b["wavelength_um"],
        )

else:
    st.caption(
        "The images alternate so you can notice changes. "
        "Use Step to switch manually, or start automatic blinking."
    )

    control_col1, control_col2, control_col3 = st.columns(3)

    if control_col1.button(
        "Start blinking",
        type="primary",
        disabled=st.session_state["blink_playing"],
    ):
        st.session_state["blink_playing"] = True

    if control_col2.button(
        "Pause",
        disabled=not st.session_state["blink_playing"],
    ):
        st.session_state["blink_playing"] = False

    if control_col3.button(
        "Step to next image",
        disabled=st.session_state["blink_playing"],
    ):
        st.session_state["blink_index"] = (
            st.session_state["blink_index"] + 1
        ) % 2

    blink_interval = st.select_slider(
        "Blink speed",
        options=[0.5, 0.75, 1.0, 1.5],
        value=0.75,
        format_func=lambda seconds: f"{seconds} seconds",
        help="A longer interval makes it easier to inspect each image.",
    )

    @st.fragment(
        run_every=(
            blink_interval
            if st.session_state["blink_playing"]
            else None
        )
    )
    def show_blink_frame():
        blink_index = st.session_state["blink_index"] % 2
        image_col1, image_col2, image_col3 = st.columns([1, 1.2, 1])

        with image_col2:
            if blink_index == 0:
                draw_observation(
                    display_a,
                    wcs_a,
                    sky_position,
                    observation_label(band_results, index_a),
                    metadata_a["wavelength_um"],
                    figsize=(3.5, 3.5),
                )
            else:
                draw_observation(
                    display_b,
                    wcs_b,
                    sky_position,
                    observation_label(band_results, index_b),
                    metadata_b["wavelength_um"],
                    figsize=(3.5, 3.5),
                )

        st.caption(
            "Showing Observation A"
            if blink_index == 0
            else "Showing Observation B"
        )

        if st.session_state["blink_playing"]:
            st.session_state["blink_index"] = (blink_index + 1) % 2

    show_blink_frame()


st.divider()
st.subheader("3 · Inspect the difference")
st.write(
    "The difference image subtracts Observation B from Observation A. "
    "It can highlight changes, but it can also show artifacts."
)

wavelength_difference = abs(
    metadata_a["wavelength_um"] - metadata_b["wavelength_um"]
)
wavelength_tolerance = (
    metadata_a["bandwidth_um"] + metadata_b["bandwidth_um"]
) / 2
units_match = str(metadata_a["units"]).strip() == str(
    metadata_b["units"]
).strip()

if not different_observations:
    st.info("Choose two different observations to create a difference image.")
elif not units_match:
    st.info(
        "The observations use different image units, so a difference "
        "image is not shown."
    )
elif wavelength_difference > wavelength_tolerance:
    st.info(
        "The observations’ wavelength ranges do not overlap, so a "
        "difference image is not shown. You can still inspect and blink "
        "between the two images."
    )
else:
    aligned_b, footprint = reproject_interp(
        (image_b, wcs_b),
        wcs_a,
        shape_out=image_a.shape,
    )

    aligned_bad_flags, flag_footprint = reproject_interp(
        (flagged_pixels(flags_b).astype(float), wcs_b),
        wcs_a,
        shape_out=image_a.shape,
        order="nearest-neighbor",
    )

    invalid = (
        flagged_pixels(flags_a)
        | (aligned_bad_flags > 0.5)
        | (footprint <= 0)
        | (flag_footprint <= 0)
        | ~np.isfinite(footprint)
        | ~np.isfinite(flag_footprint)
        | ~np.isfinite(aligned_bad_flags)
        | ~np.isfinite(image_a)
        | ~np.isfinite(aligned_b)
    )

    difference = image_a - aligned_b
    difference[invalid] = np.nan
    finite_difference = difference[np.isfinite(difference)]

    if finite_difference.size == 0:
        st.info(
            "The cutouts have no overlapping valid pixels to subtract."
        )
    else:
        limit = np.nanpercentile(np.abs(finite_difference), 99)

        if limit <= 0:
            limit = 1

        difference_col1, difference_col2, difference_col3 = st.columns(
            [1, 2, 1]
        )

        with difference_col2:
            figure, axis = plt.subplots(figsize=(5, 5))
            plot = axis.imshow(
                difference,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title("Observation A − Observation B")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(
                plot,
                ax=axis,
                label=f"Pixel difference ({metadata_a['units']})",
            )
            figure.tight_layout()

            st.pyplot(
                figure,
                use_container_width=False,
                clear_figure=True,
            )
            plt.close(figure)

        with st.expander("How to read this graph"):
            st.write(
                "- **Red:** more signal in Observation A.\n"
                "- **Blue:** more signal in Observation B.\n"
                "- **Color-bar values:** pixel differences in the displayed "
                "image units.\n"
                "- **Gray or blank areas:** masked pixels or places without "
                "valid coverage in both observations."
            )
            st.caption(
                "The color scale is centered on zero and stretches to the "
                "99th percentile of the absolute pixel differences. This "
                "makes positive and negative changes easier to compare."
            )

        masked_percent = 100 * np.count_nonzero(invalid) / invalid.size
        st.caption(
            f"Excluded {masked_percent:.1f}% of pixels because of "
            "quality flags, missing coverage, or invalid values."
        )

        st.warning(
            "A red and blue pattern can be a clue that a source moved, "
            "but it is not proof. Misalignment, noise, detector artifacts, "
            "or differences in image sharpness and background can also "
            "produce patterns. These images are WCS-aligned and checked "
            "for overlapping wavelengths, but they are not matched for "
            "point-spread function or background."
        )


with st.expander("Observation details and original data links"):
    detail_col_a, detail_col_b = st.columns(2)

    with detail_col_a:
        st.markdown("**Observation A**")
        st.write(f"Date: {metadata_a['image_date']}")
        st.write(f"Wavelength: {metadata_a['wavelength_um']:.3f} μm")
        st.write(f"Detector: {metadata_a['detector']}")
        st.write(f"Units: {metadata_a['units']}")
        st.write(f"PSF FWHM: {metadata_a['psf_fwhm']}")
        st.link_button(
            "Open original Observation A FITS product",
            url_a,
            use_container_width=True,
        )

    with detail_col_b:
        st.markdown("**Observation B**")
        st.write(f"Date: {metadata_b['image_date']}")
        st.write(f"Wavelength: {metadata_b['wavelength_um']:.3f} μm")
        st.write(f"Detector: {metadata_b['detector']}")
        st.write(f"Units: {metadata_b['units']}")
        st.write(f"PSF FWHM: {metadata_b['psf_fwhm']}")
        st.link_button(
            "Open original Observation B FITS product",
            url_b,
            use_container_width=True,
        )


st.divider()
st.subheader("4 · Record what you noticed")
st.caption(
    "This records your visual assessment in this session. "
    "It is not a scientific confirmation or a report sent to scientists."
)

with st.form("save_finding_form"):
    finding_category = st.selectbox(
        "What best describes the comparison?",
        [
            "Possible motion",
            "Possible brightness change",
            "No clear change",
            "Uncertain",
            "Other",
        ],
    )

    confidence = st.select_slider(
        "How confident are you?",
        options=["Low", "Medium", "High"],
        value="Medium",
    )

    finding_note = st.text_area(
        "What did you notice?",
        placeholder=(
            "Describe the feature or change you noticed. "
            "You can also mention what made the comparison difficult."
        ),
    )

    save_clicked = st.form_submit_button(
        "Save finding",
        disabled=not images_loaded or not different_observations,
    )

if save_clicked:
    if "findings" not in st.session_state:
        st.session_state["findings"] = []

    finding = {
        "target": search_context.get("target", "Custom coordinates"),
        "ra": search_context.get("ra", ""),
        "dec": search_context.get("dec", ""),
        "band": str(selected_band),
        "date_a": observation_label(band_results, index_a),
        "date_b": observation_label(band_results, index_b),
        "obs_id_a": str(band_results["obs_id"][index_a]),
        "obs_id_b": str(band_results["obs_id"][index_b]),
        "cutout_a": cutout_a,
        "cutout_b": cutout_b,
        "category": finding_category,
        "confidence": confidence,
        "note": finding_note.strip(),
        "saved_at": datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
    }

    st.session_state["findings"].append(finding)
    st.session_state["last_saved_finding"] = finding
    st.rerun()

if "last_saved_finding" in st.session_state:
    saved = st.session_state["last_saved_finding"]
    st.success("Finding saved in this session.")
    st.write(
        f"**Review:** {saved['category']} · "
        f"**Confidence:** {saved['confidence']}"
    )
    st.write(
        f"**Compared:** {saved['date_a']} and {saved['date_b']}"
    )
    st.caption(
        f"SPHEREx products: {saved['obs_id_a']} and {saved['obs_id_b']}"
    )
