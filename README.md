# ⚡ EV Range Predictor Pro - AI-Powered Electric Vehicle Range Estimation

This engineering project combines **embedded systems modeling (CAN bus network sensor signals)** with **applied Artificial Intelligence**. The main objective is to estimate an electric vehicle's remaining range in real-time (based on a nominal 60 kWh battery pack) by analyzing driving dynamics, driver behavior, and physical constraints.

The project features an **interactive web dashboard** that simulates real-time vehicle signals and evaluates the predictions of multiple Machine Learning models.

---

## 🚀 Key Features

- **Physical Data Generation:** Realistic simulation of battery discharge cycles incorporating physical vehicle laws (speed, instantaneous power consumption, regenerative braking efficiency, and driving agility modes).
- **Temporal Feature Engineering:** Time-series processing utilizing rolling windows (extracting 30-second moving averages and standard deviations) and data normalization (`StandardScaler`).
- **Multi-Model Benchmarking:** Training and comparative analysis of two robust Machine Learning architectures: *Random Forest Regressor* and *XGBoost Regressor*.
- **Dynamic Dashboard (HMI):** A highly responsive user interface developed with **Streamlit** and **Plotly** to manipulate vehicle CAN signals and observe instantaneous impacts on the predicted discharge curve.

---

## 🕹️ Interactive Simulation Dashboard

The web application splits the workflow into two major sections:
1. **CAN Control Panel (Sidebar):** Interactive sliders to fine-tune average speed, driver aggressiveness (speed variability and throttle pedal standard deviation), reference energy consumption, remaining battery energy, and drive mode (Eco, Normal, Sport).
2. **Metrics & Visualization (Main Panel):**
   - Dynamic display of the remaining distance predicted by the selected AI model.
   - Interactive State of Charge (SoC %) gauge meter.
   - Live Plotly chart plotting the predicted linear battery energy depletion curve over the remaining distance.

---

## 📊 Model Performance & Benchmarking

The Machine Learning models were trained on simulated datasets reflecting real automotive sensor streams. Performance was validated using Mean Absolute Error (MAE) and the Coefficient of Determination ($R^2$ score):

| AI Model Architecture | Mean Absolute Error (MAE) | Global Accuracy ($R^2$ Score) |
| :--- | :---: | :---: |
| **Random Forest Regressor** | **6.30 km** | **98.27 %** |
| **XGBoost Regressor** | **14.21 km** | **91.81 %** |

> *Engineering Insight:* On this specific profile of synthetic time-series data, the Random Forest model captures the underlying physical laws of battery discharge almost perfectly, maintaining an average margin of error of just 6 kilometers.

---

## 🛠️ Installation and Setup

### 1. Configure the Virtual Environment
```powershell
# Create and activate the virtual environment (Windows)
python -m venv env_ev_project
.\env_ev_project\Scripts\Activate.ps1

# Install required dependencies
pip install streamlit pandas numpy scikit-learn xgboost joblib plotly