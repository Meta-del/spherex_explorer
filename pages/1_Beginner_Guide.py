import streamlit as st
from spherex_ui import apply_styles, render_page_header
st.set_page_config(
    page_title="SPHEREx Beginner Guide",
    page_icon="🌌",
    layout="wide",
)

apply_styles()
render_page_header(
    "guide",
    "A beginner’s guide to SPHEREx",
    "Learn what the images show, how to compare observations, "
    "and when to mark a result as uncertain.",
)

st.subheader("Start here")
st.markdown(
    """
1. Open **Explore SPHEREx Observations** in the navigation menu.
2. Choose a target or enter its RA and Dec.
3. Choose a SPHEREx data collection and click **Search observations**.
4. Select a band, then use **Play** to browse its images by date.
5. Choose a second observation to compare with the animated frame.
"""
)

st.page_link("app.py", label="Open the SPHEREx Explorer", icon="🔭")

st.divider()
st.subheader("What are RA and Dec?")
st.write(
    "Right Ascension (RA) and Declination (Dec) are coordinates that name "
    "a position on the sky. This app asks for them in decimal degrees. "
    "RA ranges from 0° to 360°. Dec ranges from −90° to +90°."
)

st.subheader("What does a SPHEREx image show?")
st.write(
    "SPHEREx images measure infrared light. The app displays real image "
    "cutouts returned by NASA/IPAC IRSA. Each result is an image product; "
    "several products can come from nearby times or different wavelength "
    "samples. A result does not by itself confirm that the named target "
    "was detected."
)

st.subheader("Understanding the bands")
st.write(
    "SPHEREx uses six detector bands, covering wavelengths from about "
    "0.75 to 5 microns. The wavelength shown under each image is the "
    "approximate wavelength at the searched coordinate for that image. "
    "Even within one band, the wavelength can vary between images."
)

st.subheader("What the red cross means")
st.write(
    "The red cross marks the coordinate you searched for. It is a position "
    "marker, not an automatic identification of a star or moving object. "
    "A bright source near the cross still needs to be identified carefully."
)

st.subheader("What the difference image means")
st.write(
    "The difference view aligns two cutouts using their sky coordinates, "
    "checks whether their wavelength ranges overlap, and masks selected "
    "quality flags. It is an exploratory comparison. The images are not "
    "matched for point-spread function or background, so a bright or dark "
    "difference does not by itself prove that an object changed."
)

st.divider()
st.subheader("Challenge targets to try")

targets = [
    {
        "Target": "Barnard’s Star",
        "RA (degrees)": 269.4463,
        "Dec (degrees)": 4.7682,
        "What to know": (
            "Preset uses an approximate 2026 position because the star "
            "moves quickly. The tested D3 observations covered only about "
            "two days, too short to see its motion."
        ),
        "Archive check": "119 QR3 image records in the latest preset check",
    },
    {
        "Target": "Epsilon Indi B",
        "RA (degrees)": 331.044146,
        "Dec (degrees)": -56.782844,
        "What to know": (
            "The close brown-dwarf pair will appear blended in SPHEREx "
            "images at this resolution."
        ),
        "Archive check": "120 QR3 image records in the latest preset check",
    },
    {
        "Target": "WISE 0855",
        "RA (degrees)": 133.795129,
        "Dec (degrees)": -7.245147,
        "What to know": (
            "No matching images were returned in the four QR2/QR3 wide "
            "and deep collections checked."
        ),
        "Archive check": "No images found in the tested collections",
    },
    {
        "Target": "RW Aurigae",
        "RA (degrees)": 76.956525,
        "Dec (degrees)": 30.401438,
        "What to know": (
            "The tested D3 records covered about 3.6 days; this is only "
            "a short time window for investigating variability."
        ),
        "Archive check": "109 QR3 image records in the latest preset check",
    },
    {
        "Target": "R Draconis",
        "RA (degrees)": 248.167611,
        "Dec (degrees)": 66.754979,
        "What to know": (
            "The selected band’s tested records covered about 27 days."
        ),
        "Archive check": "126 QR3 image records in the latest preset check",
    },
]

st.dataframe(targets, hide_index=True, use_container_width=True)

st.caption(
    "Archive counts are example results from our checks in November 2026. "
    "They count image records, not confirmed target detections, and may "
    "change as IRSA releases more data."
)

st.info(
    "No images found does not mean the object is absent from the sky. "
    "It means the selected archive collection returned no matching image "
    "products for the searched position."
)
