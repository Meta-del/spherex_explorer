import streamlit as st

from spherex_ui import apply_styles, render_page_header


st.set_page_config(
    page_title="SPHEREx Beginner Guide",
    page_icon="🌌",
    layout="wide",
)

IRSA_TUTORIAL_VIDEO = "https://youtu.be/KfKdILiDgi0"
NASA_SKY_SCAN_VIDEO = "https://spherex.caltech.edu/video/spherex-scans-the-sky"
FIRST_LIGHT_IMAGE = (
    "https://spherex.caltech.edu/system/avm_images/binaries/24/"
    "larger/1-PIA26280-SPHEREx-cropped.jpg?1743541621="
)
FIRST_LIGHT_PAGE = (
    "https://spherex.caltech.edu/image/"
    "spherex20250401a-spherex-first-light-images"
)
IRSA_ARCHIVE_GUIDE = (
    "https://irsa.ipac.caltech.edu/spherex-archive-documentation/"
)
IRSA_VISUALIZATION_GUIDE = (
    "https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/visualization.html"
)
IRSA_SPECTROPHOTOMETRY_GUIDE = (
    "https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html"
)

apply_styles()
render_page_header(
    "guide",
    "A beginner’s guide to SPHEREx",
    "Learn how to follow the sky over time, recognize possible changes, "
    "and understand what the images can—and cannot—tell you.",
)
st.page_link(
    "app.py",
    label="← Back to Explore and Animate",
    icon="🔭",
    use_container_width=True,
)

st.subheader("Your first time-lapse")
st.write(
    "Start with a known target, prepare its available observations, "
    "and play the images in date order. If something catches your eye, "
    "compare two dates and record what you noticed."
)

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    with st.container(border=True):
        st.markdown("### 1 · Find a target")
        st.write(
            "Choose a challenge target or enter its Right Ascension "
            "and Declination."
        )

with step_col2:
    with st.container(border=True):
        st.markdown("### 2 · Play the sky")
        st.write(
            "Choose a SPHEREx band, prepare its real archive images, "
            "then play, pause, or step through the dates."
        )

with step_col3:
    with st.container(border=True):
        st.markdown("### 3 · Investigate")
        st.write(
            "If you notice a possible change, compare two observations "
            "and save a note with your confidence."
        )

st.divider()
st.subheader("Watch: exploring SPHEREx data")

video_col, video_info_col = st.columns([1.5, 1])

with video_col:
    st.video(IRSA_TUTORIAL_VIDEO)

with video_info_col:
    st.markdown("#### Official IRSA tutorial")
    st.write(
        "This video introduces the SPHEREx Data Explorer at IRSA. "
        "The interface is different from this project, but the archive "
        "and image concepts are useful background."
    )
    st.link_button(
        "Open the IRSA tutorial",
        IRSA_TUTORIAL_VIDEO,
        use_container_width=True,
    )
    st.link_button(
        "See how SPHEREx scans the sky",
        NASA_SKY_SCAN_VIDEO,
        use_container_width=True,
    )

st.divider()
st.subheader("A real SPHEREx image")

image_col, image_text_col = st.columns([1.35, 1])

with image_col:
    st.image(
        FIRST_LIGHT_IMAGE,
        caption=(
            "SPHEREx first-light image. Credit: NASA/JPL-Caltech. "
            "This mission image is for illustration, not a frame "
            "from your searched time-lapse."
        ),
        use_container_width=True,
    )

with image_text_col:
    st.markdown("#### What are you seeing?")
    st.write(
        "Infrared light is outside the range of human vision, so the "
        "image assigns visible colors to different infrared wavelengths. "
        "The colors help show wavelength information; they are not the "
        "object’s ordinary visible-light colors."
    )
    st.warning(
        "This first-light image is uncalibrated and uses a visual "
        "brightness stretch. Use your app’s IRSA cutouts for the actual "
        "time-lapse and comparisons."
    )
    st.link_button(
        "View NASA’s image description",
        FIRST_LIGHT_PAGE,
        use_container_width=True,
    )

st.divider()
st.subheader("What are RA and Dec?")
st.write(
    "Right Ascension (RA) and Declination (Dec) are coordinates for a "
    "position on the sky. This app accepts decimal degrees. RA ranges "
    "from 0° to 360°; Dec ranges from −90° to +90°."
)

st.subheader("What do the bands and wavelengths mean?")
st.write(
    "SPHEREx measures infrared light in 102 spectral bands across "
    "approximately 0.75 to 5 microns. The app’s band choice filters the "
    "available observations. The wavelength shown for each frame is the "
    "approximate wavelength at your searched coordinate. It can vary "
    "between observations, even when you keep the same detector band."
)

with st.expander("Why might wavelength vary within a band?"):
    st.write(
        "SPHEREx uses linear variable filters. In an image, the wavelength "
        "depends on position on the detector. That means a time-lapse in "
        "one detector band can still include slightly different sampled "
        "wavelengths. Check the wavelength shown for each frame before "
        "interpreting a brightness change."
    )

st.subheader("What should I look for?")
look_col1, look_col2 = st.columns(2)

with look_col1:
    with st.container(border=True):
        st.markdown("#### Possible motion")
        st.write(
            "Compare the source with nearby stars. Stars in the background "
            "should stay in the same relative positions. A candidate that "
            "moves relative to those stars may be worth a closer look."
        )

with look_col2:
    with st.container(border=True):
        st.markdown("#### Possible brightness change")
        st.write(
            "A variable source can remain in the same place while its "
            "brightness changes. Noise, image sharpness, background, "
            "wavelength differences, and flagged pixels can also change "
            "how bright a source appears."
        )

st.info(
    "The red cross marks the coordinate you searched. It does not "
    "automatically identify a star or follow a moving object."
)

st.subheader("How to use the comparison")
st.write(
    "First watch the time-lapse. If you notice something, open Compare "
    "and select two different dates in the same band. Blink between them, "
    "then inspect the difference image if the wavelengths and units "
    "are compatible."
)

with st.expander("How to read the difference image"):
    st.write(
        "The difference image subtracts Observation B from Observation A. "
        "Red pixels have more signal in A; blue pixels have more in B. "
        "A moving source can leave a red-and-blue pattern, but alignment "
        "errors, image sharpness, background, noise, or detector artifacts "
        "can create similar patterns."
    )
    st.warning(
        "Treat a pattern as a clue to inspect, not proof of motion or "
        "variability. The app’s difference image is exploratory."
    )

st.divider()
st.subheader("Challenge targets to try")

targets = [
    {
        "Target": "Barnard’s Star",
        "RA (degrees)": 269.4463,
        "Dec (degrees)": 4.7682,
        "What to know": (
            "The preset is an approximate 2026 position. The tested D3 "
            "observations covered about two days, which is too short to "
            "expect obvious motion."
        ),
        "Archive check": "119 QR3 image records in the latest preset check",
    },
    {
        "Target": "Epsilon Indi B",
        "RA (degrees)": 331.044146,
        "Dec (degrees)": -56.782844,
        "What to know": (
            "The close brown-dwarf pair is expected to appear blended "
            "at SPHEREx image resolution."
        ),
        "Archive check": "120 QR3 image records in the latest preset check",
    },
    {
        "Target": "WISE 0855",
        "RA (degrees)": 133.795129,
        "Dec (degrees)": -7.245147,
        "What to know": (
            "This is a fast-moving brown dwarf. The app’s red cross marks "
            "a fixed searched coordinate and does not yet track its "
            "predicted position at each date."
        ),
        "Archive check": (
            "No images returned in the four tested QR2/QR3 wide "
            "and deep collections"
        ),
    },
    {
        "Target": "RW Aurigae",
        "RA (degrees)": 76.956525,
        "Dec (degrees)": 30.401438,
        "What to know": (
            "The tested D3 records covered about 3.6 days, a short "
            "window for investigating variability."
        ),
        "Archive check": "109 QR3 image records in the latest preset check",
    },
    {
        "Target": "R Draconis",
        "RA (degrees)": 248.167611,
        "Dec (degrees)": 66.754979,
        "What to know": (
            "The tested records in the selected band covered about "
            "27 days."
        ),
        "Archive check": "126 QR3 image records in the latest preset check",
    },
]

st.dataframe(
    targets,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "These counts are example results from checks in November 2026. "
    "They count archive image records, not confirmed detections of the "
    "named target, and can change as IRSA releases data."
)

st.info(
    "No images found means the selected collection returned no matching "
    "image products for the searched position. It does not mean the "
    "object is absent from the sky."
)

st.divider()
st.subheader("Explore the real archive")

resource_col1, resource_col2, resource_col3 = st.columns(3)

with resource_col1:
    st.link_button(
        "SPHEREx archive guide",
        IRSA_ARCHIVE_GUIDE,
        use_container_width=True,
    )

with resource_col2:
    st.link_button(
        "IRSA image visualization guide",
        IRSA_VISUALIZATION_GUIDE,
        use_container_width=True,
    )

with resource_col3:
    st.link_button(
        "IRSA spectrophotometry guide",
        IRSA_SPECTROPHOTOMETRY_GUIDE,
        use_container_width=True,
    )

st.caption(
    "The visualization guide explains image layers and wavelengths. "
    "The spectrophotometry guide is a deeper resource for source "
    "measurements, uncertainty, and quality flags."
)
