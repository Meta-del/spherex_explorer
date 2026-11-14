from html import escape
from pathlib import Path

import streamlit as st


PAGES = [
    ("explore", "01", "Explore", "app.py", "🔭"),
    ("compare", "02", "Compare", "pages/2_Compare_Images.py", "🌓"),
    ("findings", "03", "My Findings", "pages/3_My_Findings.py", "📝"),
    ("guide", "04", "Beginner Guide", "pages/1_Beginner_Guide.py", "📘"),
]


def apply_styles():
    css_path = Path(__file__).with_name("styles.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def render_page_header(active_page, title, subtitle):
    page = next(
        (item for item in PAGES if item[0] == active_page),
        None,
    )

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

    columns = st.columns(len(PAGES))

    for column, item in zip(columns, PAGES):
        key, number, label, path, icon = item

        with column:
            if key == page_key:
                st.markdown(
                    f"""
                    <div class="sky-nav-current sky-nav-current-{escape(key)}"
                         aria-current="page">
                        <span class="sky-nav-icon">{escape(icon)}</span>
                        <span class="sky-nav-number">{escape(number)}</span>
                        <strong>{escape(label)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(
                    path,
                    label=f"{number} · {label}",
                    icon=icon,
                )

    st.divider()
