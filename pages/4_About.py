import streamlit as st

from spherex_ui import apply_styles, render_page_header


st.set_page_config(page_title="About SPHEREx Explorer", page_icon="🌌", layout="wide")

apply_styles()
render_page_header(
    "about",
    "Why explore SPHEREx?",
    "A public guide to the mission data, the tool, and what careful image inspection can—and cannot—tell us.",
)

st.markdown(
    """
    <section class="about-story sky-scroll-reveal">
        <span class="about-kicker">THE MISSION DATA</span>
        <h2>A changing infrared view of the whole sky</h2>
        <p>NASA’s SPHEREx mission observes the sky in infrared light across many wavelengths. This explorer searches real SPHEREx image products hosted by NASA/IPAC IRSA, then lets you inspect available observations of the same sky position over time.</p>
        <p>The archive is the source of the images. The app does not create simulated sky images or assume that every target has observations.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

about_col1, about_col2 = st.columns(2)
with about_col1:
    st.markdown(
        """
        <section class="about-card about-card-cyan sky-scroll-reveal">
            <span class="about-kicker">WHY THIS TOOL</span>
            <h3>Make a large archive easier to inspect</h3>
            <p>Observations are scientific data products, not ready-made videos. The explorer gathers matching records, sorts them by date, aligns images for viewing, and gives people simple ways to blink, compare, and record what they notice.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
with about_col2:
    st.markdown(
        """
        <section class="about-card about-card-amber sky-scroll-reveal">
            <span class="about-kicker">WHO CAN USE IT</span>
            <h3>Curious people, classrooms, and researchers</h3>
            <p>Beginners can follow the guide and compare images visually. Students and educators can use real mission data to explore time, wavelength, and image quality. Experienced users can inspect metadata and open the original archive products.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="about-story about-process sky-scroll-reveal">
        <span class="about-kicker">HOW AN EXPLORATION WORKS</span>
        <h2>Search · Watch · Compare · Record</h2>
        <div class="about-flow">
            <div><b>01</b><strong>Choose a position</strong><span>Enter RA and Dec or choose a suggested target.</span></div>
            <div><b>02</b><strong>Inspect real dates</strong><span>Search IRSA and animate available SPHEREx images in time order.</span></div>
            <div><b>03</b><strong>Check a possible change</strong><span>Compare the same band and use nearby sources as visual references.</span></div>
            <div><b>04</b><strong>Save your observation</strong><span>Record possible motion, brightness change, no clear change, or uncertainty.</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "A visual change is a candidate for further inspection, not a confirmed discovery. "
    "Image sharpness, wavelength, background, and quality flags can affect what you see. "
    "Saved findings stay in your app session and downloads; they are not sent to NASA or scientists."
)

source_col1, source_col2 = st.columns(2)
with source_col1:
    st.link_button(
        "Read the SPHEREx IRSA archive guide",
        "https://irsa.ipac.caltech.edu/spherex-archive-documentation/",
        use_container_width=True,
    )
with source_col2:
    st.link_button(
        "Visit the SPHEREx mission site",
        "https://spherex.caltech.edu/",
        use_container_width=True,
    )

st.markdown(
    """
    <section class="about-cta sky-scroll-reveal">
        <span class="about-kicker">YOUR TURN</span>
        <h2>Ready to explore a patch of sky?</h2>
        <p>Choose a target, search the real archive, and see what the observations show.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.page_link(
    "app.py",
    label="Go find something in the SPHEREx data",
    icon="🔭",
    use_container_width=True,
)
