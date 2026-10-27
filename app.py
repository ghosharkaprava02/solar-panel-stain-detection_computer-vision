#!/usr/bin/env python3
"""
app.py - Interactive Streamlit Application for Solar Panel Stain Detection & Severity Classification.
Allows users to upload photos, adjust coverage thresholds, view Stage A rectification,
inspect Stage B stain segmentation masks, and view sub-panel 4x4 grid heatmaps.
"""

import os
import glob
import cv2
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from infer import SolarStainInferencePipeline

st.set_page_config(
    page_title="Solar Panel Stain Inspector",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_pipeline():
    return SolarStainInferencePipeline()

pipeline = get_pipeline()

# Title and header
st.title("☀️ Solar Panel Stain Detection & Severity Classification")
st.markdown("""
**Automated Two-Stage Inspection System**:
1. **Stage A**: Solar panel localization and perspective rectification (isolating panel from roof/vegetation/sky).
2. **Stage B**: High-resolution stain segmentation (dust, bird droppings, mud) via U-Net.
3. **Severity Assessment**: Calculates **Stain Coverage %** and detects localized **Bird-Dropping Hotspots**.
""")

# Sidebar settings
st.sidebar.header("⚙️ Rule & Detection Thresholds")
st.sidebar.markdown("Configure operational severity thresholds:")

cov_thresh = st.sidebar.slider(
    "Stain Coverage Threshold (%)",
    min_value=1.0,
    max_value=30.0,
    value=float(pipeline.cov_threshold),
    step=0.5,
    help="Panels with stain coverage >= this percentage are classified as High Stain."
)
pipeline.cov_threshold = cov_thresh

blob_thresh = st.sidebar.slider(
    "Bird Dropping Blob Threshold (%)",
    min_value=0.5,
    max_value=10.0,
    value=float(pipeline.bird_blob_threshold),
    step=0.25,
    help="Any single localized bird dropping covering >= this percentage of the panel immediately triggers High Stain."
)
pipeline.bird_blob_threshold = blob_thresh

use_tta = st.sidebar.checkbox("Enable Test-Time Augmentation (TTA)", value=True)
ensemble_mode = st.sidebar.checkbox("Enable Classifier Ensemble", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Select Input Source")

# Sample selector or file upload
sample_images = sorted(glob.glob("data/raw/*/*.jpg"))
sample_names = ["Upload Custom Image..."] + [os.path.basename(p) for p in sample_images[:15]]
selected_sample = st.sidebar.selectbox("Preset Sample Panel:", sample_names)

uploaded_file = None
input_image_bgr = None

if selected_sample == "Upload Custom Image...":
    uploaded_file = st.sidebar.file_uploader("Upload Solar Panel Photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    # Match preset sample
    for p in sample_images:
        if os.path.basename(p) == selected_sample:
            input_image_bgr = cv2.imread(p)
            break

if input_image_bgr is None:
    st.info("👈 Please select a preset sample panel from the sidebar or upload a panel photo to begin inspection.")
else:
    # Run Inference
    with st.spinner("Analyzing panel with Stage A localization and Stage B segmentation..."):
        result = pipeline.process_image(input_image_bgr, use_tta=use_tta, ensemble=ensemble_mode)
        overlay_bgr = pipeline.render_overlay(input_image_bgr, result)

    # Executive Status Banner
    is_high = result["is_high_stain"]
    if is_high:
        st.error(f"### 🚨 Classification: **HIGH STAIN** (Maintenance / Cleaning Required)\n**Rationale**: {result['reasons']}")
    else:
        st.success(f"### ✅ Classification: **LOW STAIN** (Operational / Acceptable Condition)\n**Rationale**: {result['reasons']}")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Total Stain Coverage",
            value=f"{result['coverage_pct']:.2f}%",
            delta=f"{result['coverage_pct'] - pipeline.cov_threshold:+.1f}% vs Threshold",
            delta_color="inverse"
        )
    with m2:
        st.metric(
            label="Largest Bird Dropping Blob",
            value=f"{result['max_blob_pct']:.2f}%",
            delta=f"{result['max_blob_pct'] - pipeline.bird_blob_threshold:+.1f}% vs Threshold",
            delta_color="inverse"
        )
    with m3:
        st.metric(
            label="Severity Classification",
            value=result["label"]
        )
    with m4:
        st.metric(
            label="Panel Geometry Status",
            value="Rectified 512x512"
        )

    st.markdown("---")

    # Image Visualizations
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("1. Raw Input")
        st.image(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("2. Stage A Rectified")
        st.image(cv2.cvtColor(result["rectified_bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.subheader("3. Stage B Mask")
        st.image(result["binary_mask"] * 255, caption="Stain Binary Mask", use_container_width=True)

    with col4:
        st.subheader("4. Diagnostic Overlay")
        st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), caption="4x4 Grid & Hotspots", use_container_width=True)

    st.markdown("---")

    # Sub-Panel Area Analysis: 4x4 Grid
    st.subheader("📊 Sub-Panel Area Breakdown (4x4 Grid Analysis)")
    st.markdown("Quantifies localized soiling percentage across sub-regions to identify hot spots and bypass-diode risks:")

    grid_df = pd.DataFrame(
        result["grid_coverage"],
        columns=[f"Col {c+1}" for c in range(pipeline.grid_cols)],
        index=[f"Row {r+1}" for r in range(pipeline.grid_rows)]
    )

    c_table, c_heatmap = st.columns([1, 1])
    with c_table:
        st.dataframe(grid_df.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=50).format("{:.1f}%"), use_container_width=True)
    with c_heatmap:
        st.markdown(f"""
        - **Total Stained Pixels**: `{int(np.sum(result['binary_mask'])):,}` / `{result['binary_mask'].size:,}`
        - **Critical Sub-Cells (Coverage > 15%)**: `{(result['grid_coverage'] >= 15.0).sum()}` of `16`
        - **Soiling Distribution**: {'Clustered / Localized' if result['has_large_blob'] else 'Uniform / Diffuse'}
        """)

    # Download Output
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    st.download_button(
        label="📥 Download Diagnostic Overlay Image",
        data=buf.getvalue(),
        file_name="solar_panel_stain_report.jpg",
        mime="image/jpeg"
    )
