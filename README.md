
#  NETRA-ORB  


### Recognition of Anomalies in Orbits

NETRA-ORB is a deep learning based system to **learn satellite orbital behavior directly from historical TLE data**, predict future orbital states, and **automatically detect anomalies such as maneuvers, station-keeping, and mission phase changes** — without using any physics-based propagator inside the ML model.

---

## 📌 Motivation

NETRA-ORB demonstrates a **purely data-driven machine learning approach** that learns orbital evolution patterns directly from historical data.


---

## 🚀 What NETRA-ORB Does

- Uses historical **TLE / OMM data from Space-Track**
- Learns **normal orbital behavior patterns** using LSTM networks
- Predicts the **next orbital state** from past states
- Computes **prediction error (residual)**
- Detects anomalies using a **statistical thresholding approach**
- Visualizes:
  - Orbital parameters
  - Error trends
  - Anomaly events
- Scales to **fleet-level analysis (59 Indian satellites)**

---

## 🧠 Core Idea

If a satellite behaves normally:
> ML prediction error stays small and smooth.

If a satellite performs:
- Maneuvers
- Station keeping
- Orbit corrections
- Mission phase changes

> Prediction error spikes → anomaly detected.

---

## 📊 Data Source

Data is collected from:

- **Space-Track.org**
  - GP History
  - CCSDS OMM History

For each satellite:

- One CSV file
- Each row = one historical orbital state
- Fields used include:
  - Inclination
  - RAAN
  - Eccentricity
  - Argument of Perigee
  - Mean Anomaly
  - Mean Motion
  - BSTAR (drag term)

---

## 🏗️ Feature Engineering

- Each orbital state is converted into a numerical feature vector
- Features are normalized using standard scaling
- A sliding window of past states is used to predict the next state
- This converts the problem into a **sequence-to-one time series regression task**

---

## 🧬 Model Architecture

- LSTM Neural Network (PyTorch)
- Input: past sequence of orbital states
- Output: next orbital state
- The network learns a **nonlinear mapping** from past behavior to future state

---

## 🏃 Training Pipeline

1. Load and preprocess data
2. Build time-series sequences
3. Split dataset into training and testing sets
4. Train LSTM model using regression loss
5. Save trained model
6. Evaluate on unseen data
7. Compute prediction error time series

---

## 🚨 Anomaly Detection

- Normal behavior produces small prediction errors
- Abnormal behavior produces large error spikes
- A statistical threshold is used to flag anomalies automatically

This allows detection of:

- Orbit maneuvers
- Station-keeping operations
- Major orbital regime changes
- Unexpected behavior

---

## 📈 Evaluation & Analysis

For each satellite:

- Error statistics are computed
- Anomaly counts are recorded
- Long-term behavior trends are analyzed

---

## 📊 Visualization Dashboard

Built using **Streamlit**:

Shows:

- Orbital parameters vs time
- Prediction error vs time
- Error vs parameter relationships
- Visual inspection of anomaly events
- Comparison with traditional propagation trends

---

## 🛰️ Fleet-Level Processing

- Applied to **59 Indian satellites**
- Each satellite processed independently
- Results aggregated into:
  - Fleet-level comparison
  - Stability ranking
  - Behavior categorization

---

## 🔬 Observed Behaviors

1. **Stable satellites**
   - Smooth, low error curves

2. **Maneuvering satellites**
   - Frequent sharp error spikes

3. **Mission phase change satellites**
   - Very large spikes followed by unstable periods

---

## 🧰 Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- PyTorch
- Matplotlib
- Streamlit

---

## ⚠️ Limitations

- TLE data is noisy and not true ephemeris
- Time sampling is irregular
- The system detects anomalies but does not explain their physical cause
- Not intended for long-term orbit propagation
- Designed for:
  - Monitoring
  - Behavior analysis
  - Short-term prediction

---

## 🏁 Conclusion

NETRA-ORB demonstrates that:

> Deep learning models can learn orbital behavior directly from TLE data and automatically detect satellite behavior changes **without using any physics-based model inside the ML pipeline**.

---

## 🙏 Acknowledgement

Once Again Thanks to **Space-Track.org** for providing historical TLE and OMM data without which this project would not be possible.

---

## 👤 Author

**Rahul Jaswal**

