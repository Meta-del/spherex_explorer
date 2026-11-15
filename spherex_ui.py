from html import escape
from pathlib import Path

import streamlit as st


PAGES = [
    ("explore", "01", "Explore", "app.py", "🔭"),
    ("compare", "02", "Compare", "pages/2_Compare_Images.py", "🌓"),
    ("findings", "03", "Findings", "pages/3_My_Findings.py", "📝"),
    ("guide", "04", "Guide", "pages/1_Beginner_Guide.py", "📘"),
]


def apply_styles():
    css_path = Path(__file__).with_name("styles.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="comet-field" aria-hidden="true">'
        '<span class="shooting-star shooting-star-one"></span>'
        '<span class="shooting-star shooting-star-two"></span>'
        '<span class="shooting-star shooting-star-three"></span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_page_header(active_page, title, subtitle):
    page = next((item for item in PAGES if item[0] == active_page), None)
    if page is None:
        raise ValueError(f"Unknown SPHEREx page: {active_page}")

    page_key, page_number, page_label, _, _ = page
    eyebrow = f"NASA SPHEREx · {page_number} · {page_label}"
    st.markdown(
        f"""
        <section class="sky-hero sky-hero-{escape(page_key)}">
            <div class="eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
            <div class="hero-orbit" aria-hidden="true"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="sky-top-nav"):
        columns = st.columns([1, 1, 1.1, 0.9])
        for column, item in zip(columns, PAGES):
            key, number, label, path, icon = item
            with column:
                if key == page_key:
                    st.markdown(
                        f'<div class="sky-nav-current sky-nav-current-{escape(key)}" '
                        f'aria-current="page"><span class="sky-nav-icon">'
                        f'{escape(icon)}</span><strong>{escape(label)}</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.page_link(path, label=label, icon=icon)
    render_sky_guide()


def sky_guide_answer(question):
    question = question.lower()
    if any(word in question for word in ("move", "moving", "motion", "blink")):
        return (
            "Watch one source against nearby stars. A moving object changes "
            "position while background stars stay in place. Compare the same "
            "band across dates. If the change is unclear, choose Uncertain; "
            "this app does not confirm discoveries automatically."
        )
    if any(word in question for word in ("bright", "brightness", "variable", "light curve")):
        return (
            "A variable source changes brightness while staying in about the "
            "same position. Compare similar wavelengths and check nearby "
            "stars for image-wide changes. A rejected measurement means there "
            "is not enough usable evidence, not zero brightness."
        )
    if any(word in question for word in ("flag", "quality", "error", "unusable")):
        return (
            "Flags mark pixels that may be affected by data-quality issues. "
            "If too few pixels are usable, don’t force a result; mark the "
            "comparison Uncertain."
        )
    if any(word in question for word in ("band", "wavelength", "102", "cube", "d3")):
        return (
            "The six detector arrays are labelled D1–D6. Together, SPHEREx "
            "provides 102 nominal wavelength channels in its all-sky cube; "
            "channels 51 and 52 are blank."
        )
    if any(word in question for word in ("no image", "no images", "missing")):
        return (
            "No images means the selected archive collection returned no "
            "matching products for that position. Try another collection; "
            "it does not mean the object is absent."
        )
    if any(word in question for word in ("report", "scientist", "send", "save")):
        return (
            "Saved findings stay in this app session and can be downloaded "
            "as a report. They are not automatically sent to NASA or a "
            "scientist."
        )
    return (
        "Search for a target, play its observations in date order, and look "
        "for a source that changes relative to nearby stars. Compare two "
        "dates if something catches your eye; choose Uncertain if the "
        "evidence is ambiguous."
    )


def render_sky_guide():
    st.session_state.setdefault("sky_guide_messages", [])
    with st.container(key="cass-companion-control"):
        with st.popover(
            "Cassiopeia",
            help="Touch Cassiopeia to open your sky guide.",
        ):
            st.markdown(
                '<div class="cass-chat-heading"><span class="cass-mini-avatar">✧</span>'
                '<div><strong>Cassiopeia</strong><small>Your constellation sky guide</small></div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Ask me about exploring SPHEREx. I can guide the app, but I don’t confirm discoveries.")
            messages = st.session_state["sky_guide_messages"][-4:]
            if not messages:
                st.info("I’m Cassiopeia. Ask what to look for, what the bands mean, or why an image may be blank.")
            for message in messages:
                with st.chat_message("user"):
                    st.write(message["question"])
                with st.chat_message("assistant", avatar="✦"):
                    st.write(message["answer"])

            with st.form("sky_guide_form", clear_on_submit=True):
                question = st.text_input(
                    "Ask Cassiopeia",
                    placeholder="What should I look for?",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Ask", use_container_width=True)

            if submitted and question.strip():
                st.session_state["sky_guide_messages"].append(
                    {"question": question.strip(), "answer": sky_guide_answer(question)}
                )
                st.rerun()
