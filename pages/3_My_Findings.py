import csv
import io

import numpy as np
import streamlit as st

from spherex_helpers import load_cutout
from spherex_ui import apply_styles, render_page_header


st.set_page_config(page_title="My Findings", page_icon="📝", layout="wide")

if "findings" not in st.session_state:
    st.session_state["findings"] = []

apply_styles()
render_page_header(
    "findings",
    "My SPHEREx findings",
    "Review the observations you compared and download a record of your notes.",
)

st.caption(
    "Findings stay in this browser session. Download a CSV or text report "
    "if you want to keep a copy."
)


def shared_stretch(image_a, image_b):
    """Apply one brightness scale to both real FITS images."""
    finite_a = image_a[np.isfinite(image_a)]
    finite_b = image_b[np.isfinite(image_b)]

    if finite_a.size == 0 or finite_b.size == 0:
        raise ValueError("One or both cutouts have no usable image values.")

    combined = np.concatenate([finite_a, finite_b])
    low, high = np.nanpercentile(combined, [5, 99])

    if high <= low:
        high = low + 1

    def stretch(image):
        clean = np.nan_to_num(
            image,
            nan=low,
            posinf=high,
            neginf=low,
        )
        return np.clip((clean - low) / (high - low), 0, 1)

    return stretch(image_a), stretch(image_b)


findings = st.session_state["findings"]

st.divider()

if not findings:
    st.info(
        "No findings saved yet. Compare two observations, then save a finding "
        "from the Compare Images page."
    )
else:
    st.write(f"Saved findings: **{len(findings)}**")

    for index, finding in enumerate(findings):
        title = (
            f"{finding.get('target', 'Sky position')} · "
            f"{finding.get('category', 'Finding')}"
        )

        with st.expander(title, expanded=index == 0):
            st.write(
                f"**Saved:** {finding.get('saved_at', 'Previously saved')}"
            )
            st.write(
                f"**Confidence:** "
                f"{finding.get('confidence', 'Not recorded')}"
            )
            st.write(
                f"**Coordinates:** {finding.get('ra', '—')}°, "
                f"{finding.get('dec', '—')}°"
            )
            st.write(f"**Band:** {finding.get('band', '—')}")
            st.write(f"**First observation:** {finding.get('date_a', '—')}")
            st.write(f"**Second observation:** {finding.get('date_b', '—')}")
            st.write(
                f"**Note:** {finding.get('note', '') or 'No note added.'}"
            )

            st.markdown("**Image evidence**")

            cutout_a = finding.get("cutout_a")
            cutout_b = finding.get("cutout_b")

            try:
                ra = float(finding.get("ra"))
                dec = float(finding.get("dec"))
            except (TypeError, ValueError):
                ra = None
                dec = None

            if cutout_a and cutout_b and ra is not None and dec is not None:
                try:
                    with st.spinner("Loading the saved SPHEREx cutouts…"):
                        image_a, _, metadata_a, _ = load_cutout(
                            cutout_a,
                            ra,
                            dec,
                        )
                        image_b, _, metadata_b, _ = load_cutout(
                            cutout_b,
                            ra,
                            dec,
                        )

                    display_a, display_b = shared_stretch(image_a, image_b)
                    left, right = st.columns(2)

                    with left:
                        st.image(
                            display_a,
                            caption=f"First: {finding.get('date_a', '—')}",
                            clamp=True,
                        )
                        st.caption(
                            f"Product: {finding.get('obs_id_a', '—')} · "
                            f"{metadata_a.get('wavelength_um', '—')} μm · "
                            f"{metadata_a.get('units', '—')}"
                        )

                    with right:
                        st.image(
                            display_b,
                            caption=f"Second: {finding.get('date_b', '—')}",
                            clamp=True,
                        )
                        st.caption(
                            f"Product: {finding.get('obs_id_b', '—')} · "
                            f"{metadata_b.get('wavelength_um', '—')} μm · "
                            f"{metadata_b.get('units', '—')}"
                        )

                    st.caption(
                        "The images use the same brightness stretch. "
                        "This visual comparison does not by itself confirm "
                        "motion or variability."
                    )

                except Exception as error:
                    st.warning(f"Could not reload the saved cutouts: {error}")
            else:
                st.info(
                    "This finding has no saved image links. "
                    "Its dates and notes are still available."
                )

            if st.button(
                "Delete this finding",
                key=f"delete_finding_{index}",
                type="secondary",
            ):
                st.session_state["findings"].pop(index)
                st.rerun()

    st.divider()
    st.subheader("Download report")

    fieldnames = [
        "target",
        "ra",
        "dec",
        "band",
        "date_a",
        "date_b",
        "obs_id_a",
        "obs_id_b",
        "category",
        "confidence",
        "note",
        "saved_at",
        "cutout_a",
        "cutout_b",
    ]

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()

    for finding in findings:
        writer.writerow({key: finding.get(key, "") for key in fieldnames})

    st.download_button(
        "Download findings as CSV",
        data=csv_buffer.getvalue(),
        file_name="spherex_findings.csv",
        mime="text/csv",
        use_container_width=True,
    )

    report_lines = ["# My SPHEREx Findings", ""]

    for index, finding in enumerate(findings, start=1):
        report_lines.extend(
            [
                f"## Finding {index}: "
                f"{finding.get('target', 'Sky position')}",
                f"- Saved: {finding.get('saved_at', 'Previously saved')}",
                f"- Result: {finding.get('category', 'Finding')}",
                f"- Confidence: "
                f"{finding.get('confidence', 'Not recorded')}",
                f"- Coordinates: {finding.get('ra', '—')}°, "
                f"{finding.get('dec', '—')}°",
                f"- Band: {finding.get('band', '—')}",
                f"- First observation: {finding.get('date_a', '—')}",
                f"- Second observation: {finding.get('date_b', '—')}",
                f"- SPHEREx products: {finding.get('obs_id_a', '—')} and "
                f"{finding.get('obs_id_b', '—')}",
                f"- Note: {finding.get('note', '') or 'No note added.'}",
                f"- First cutout: "
                f"{finding.get('cutout_a', 'Not recorded')}",
                f"- Second cutout: "
                f"{finding.get('cutout_b', 'Not recorded')}",
                "",
            ]
        )

    st.download_button(
        "Download findings as a text report",
        data="\n".join(report_lines),
        file_name="spherex_findings.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()

    if st.button("Clear all findings", type="secondary"):
        st.session_state["findings"] = []
        st.rerun()
