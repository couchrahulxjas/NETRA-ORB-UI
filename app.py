import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(
    page_title="NETRA-ORB Indian Satellite's Anomaly Detection and Orbit Analysis",
    layout="wide"
)

ROOT = os.getcwd()

st.sidebar.title("Analysis Mode")

mode = st.sidebar.radio(
    "Select View",
    ["Single Satellite Analysis", "Collective Analysis (All Indian Satellites)"]
)

summary_path = os.path.join(ROOT, "fleet_summary.csv")

if not os.path.exists(summary_path):
    st.error("fleet_summary.csv not found. Run build_fleet_summary.py first.")
    st.stop()

df = pd.read_csv(summary_path)



if mode == "Single Satellite Analysis":

    st.title("NETRA-ORB-Indian Satellite Anomaly Detector")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Satellite")

    satellites = sorted(df["satellite"].tolist())
    sat = st.sidebar.selectbox("Satellite", satellites)

    sat_path = os.path.join(ROOT, sat)
    img_dir = os.path.join(sat_path, "images")

    st.markdown(f"# {sat}")
    st.markdown("### Indian Satellite's Orbital Behavior, Error Analysis & Anomaly Detection")

    
    with st.popover(" What do these metrics mean?"):
        st.markdown("""
#  LSTM (Machine Learning) Orbit Prediction Metrics

These metrics evaluate how well the neural network has learned the satellite’s orbital dynamics **without explicitly using physics equations**.

---

##  Mean Error

**What it is:**  
The arithmetic average of the difference between the orbit predicted by the LSTM model and the true orbit over the entire time span.

**Why it exists:**  
It provides a **single global measure of accuracy** of the neural network.

**How to interpret:**  
- Low value → The network has successfully learned the orbital motion.  
- High value → The model is consistently making large prediction mistakes.

**What it means physically:**  
This tells you, on average, **how far your learned orbit deviates from reality**.

---

##  Median Error

**What it is:**  
The middle value of the sorted error distribution.

**Why it exists:**  
Mean can be influenced by rare extreme failures. Median shows the **typical day-to-day performance**.

**How to interpret:**  
- If Median ≈ Mean → Errors are stable and consistent.  
- If Median ≪ Mean → Occasional large failures or anomalies exist.

**What it means physically:**  
This represents the **normal operational accuracy** of the ML model.

---

##  Max Error

**What it is:**  
The **largest single error** ever produced by the neural network.

**Why it exists:**  
In safety-critical systems, worst-case behavior matters more than average behavior.

**What it means physically:**  
This corresponds to the **worst orbital misprediction** your model ever made.

---

##  P95 Error (95th Percentile)

**What it is:**  
The error value below which **95% of all prediction errors lie**.

**What it means physically:**  
This defines a **confidence envelope** around the predicted orbit.

---

##  Anomaly Count

**What it is:**  
The number of time steps where the satellite’s behavior **deviates significantly from the learned normal orbital pattern**.

---

#  SGP4 (Physics-Based Model) Error Metrics (in KM)

These metrics evaluate the **classical physics-based orbit propagation model**.

---

##  SGP4 Mean / Median / P95

They represent:
- Average accuracy  
- Typical error  
- 95% confidence bound  

> Extreme maneuver / refit discontinuities are excluded from these statistics.

---

#  High-Level Interpretation

> LSTM metrics measure learned orbital dynamics.  
> SGP4 metrics measure physics-based propagation reliability.  
> Together they form a **machine learning vs physics comparison framework** for space situational awareness.
""")

    

    row = df[df["satellite"] == sat].iloc[0]

    st.subheader(" Key Health Metrics")

    # Hide misleading SGP4 metrics
    HIDDEN_COLS = {"sgp4_max_km", "sgp4_std_km"}

    metric_cols = [c for c in df.columns if c != "satellite" and c not in HIDDEN_COLS]

    help_map = {
        "mean_error": "Average LSTM prediction error over time",
        "median_error": "Typical LSTM error (robust to spikes)",
        "max_error": "Worst LSTM prediction error",
        "p95_error": "95% of errors are below this value",
        "anomaly_count": "Number of detected abnormal behaviors",
        "sgp4_mean_km": "Average physics model error in kilometers",
        "sgp4_median_km": "Typical physics model error in kilometers",
        "sgp4_p95_km": "95% bound of physics error in kilometers",
    }

    cols = st.columns(5)
    for i, col_name in enumerate(metric_cols):
        with cols[i % 5]:
            value = row[col_name]
            label = col_name.replace("_", " ").upper()
            help_text = help_map.get(col_name, "Metric derived from orbital analysis")

            try:
                value = float(value)
                st.metric(label, f"{value:.4f}", help=help_text)
            except:
                st.metric(label, str(value), help=help_text)

    st.markdown("---")

    # ===================== IMAGES =====================

    def show_image(path, caption):
        if os.path.exists(path):
            img = Image.open(path)
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing file: {os.path.basename(path)}")

    st.subheader(" Key Results")

    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        show_image(os.path.join(img_dir, "error_plot.png"), "LSTM Prediction Error vs Time")

    with r1c2:
        show_image(os.path.join(img_dir, "anomaly_plot.png"), "Detected Anomalies")

    with r1c3:
        show_image(os.path.join(img_dir, "sgp4_error_plot.png"), "SGP4 One-step Error")

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        show_image(os.path.join(img_dir, "sgp4_error_smoothed.png"), "Smoothed SGP4 Error Trend")

    with r2c2:
        show_image(os.path.join(img_dir, "error_plot.png"), "LSTM Error Overview")

    with st.expander(" Orbital Parameter Analysis"):
        analysis_images = [
            "raan_vs_time.png",
            "inclination_vs_time.png",
            "mean_motion_vs_time.png",
            "altitude_vs_time.png",
            "eccentricity_vs_time.png",
            "bstar_vs_time.png",
            "mean_motion_dot_vs_time.png",
            "sgp4_error_vs_time.png",
            "ml_error_vs_time.png",
            "sgp4_error_vs_altitude.png",
            "ml_error_vs_altitude.png",
            "ml_error_vs_sgp4_error.png",
            "mean_motion_vs_altitude.png",
            "bstar_vs_altitude.png",
            "eccentricity_vs_altitude.png",
        ]

        cols = st.columns(3)

        for i, img in enumerate(analysis_images):
            img_path = os.path.join(img_dir, img)
            if os.path.exists(img_path):
                with cols[i % 3]:
                    show_image(img_path, img.replace(".png", "").replace("_", " ").title())

# ================= COLLECTIVE MODE =================

else:

    st.title(" Collective Analysis — All Indian Satellites")

    st.markdown("""
This view shows aggregate analysis across all Indian satellites in the dataset.
Used for understanding overall stability, reliability, and anomaly patterns.
""")

    fleet_img_dir = os.path.join(ROOT, "fleet_plots", "images")

    if not os.path.exists(fleet_img_dir):
        st.warning("fleet_plots/images folder not found. Run generate_fleet_plots.py first.")
    else:
        images = sorted(os.listdir(fleet_img_dir))

        for img in images:
            if img.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(fleet_img_dir, img)
                st.markdown(f"## {img.replace('_',' ').replace('.png','').title()}")
                st.image(Image.open(img_path), use_container_width=True)

# ================= FOOTER =================

st.markdown("---")
st.markdown("""
### About this Project

Machine learning (LSTM) based orbital behavior modeling using historical TLE data  
Next-step orbital state prediction from past orbital element sequences  
Residual-based anomaly detection in Indian satellites  
Error analysis and visualization of ML predictions over time  
Fleet-level processing and analysis of 59 Indian satellites  
Benchmarking ML prediction error against the physics-based SGP4 propagator  
Orbital parameter trend analysis (altitude, RAAN, inclination, mean motion, BSTAR, eccentricity)
""")

from streamlit_pdf_viewer import pdf_viewer

st.markdown("---")

if "show_report" not in st.session_state:
    st.session_state.show_report = False

if st.button("📄 Report"):
    st.session_state.show_report = not st.session_state.show_report

if st.session_state.show_report:
    st.subheader("📄 Project Report")

    pdf_path = "sat.pdf"

    pdf_viewer(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Download Project Report",
            data=f,
            file_name="sat.pdf",
            mime="application/pdf"
        )
