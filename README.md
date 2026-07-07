# ☀️ AI Solar AC Power Forecasting System

Live Weather Powered AI Prediction Dashboard

A production-ready Streamlit application that predicts a solar plant's **next 15-minute AC power output** using **live weather data** from Open-Meteo and an **already-trained Random Forest model**. The app never trains or retrains a model — it only loads and serves predictions from your existing `.pkl` file.

---

## Overview

Point the app at any location (by city name, coordinates, or your browser's current location) and it will:

1. Pull live weather data for that location from Open-Meteo (no API key required).
2. Derive the model's required features — Irradiation, Hour, Module Temperature, Ambient Temperature — from that weather data.
3. Run your trained Random Forest model to predict AC power for the next 15 minutes.
4. Estimate a confidence score from the agreement across the forest's individual trees.
5. Present everything on a clean, card-based, solar-green dashboard with live gauges, a map, and a weather summary.

---

## Features

- 🔎 **Location input** — search by city name, enter coordinates directly, or use the browser's current location (optional, via `streamlit-js-eval`).
- 🌦️ **Live weather** — temperature, cloud cover, wind speed, humidity, shortwave radiation, and weather description/icon, all from Open-Meteo.
- 🧮 **Automatic feature engineering** — irradiation and module temperature are derived automatically; the user never enters them manually.
- 🤖 **Pre-trained model only** — loads `model/best_random_forest.pkl` (or `rf_model.pkl`) via `joblib`; no training happens in the app.
- 📈 **Prediction confidence** — computed from the standard deviation of predictions across all trees in the forest.
- 📊 **Plotly gauges** for temperature, irradiation, predicted AC power, cloud cover, and wind speed.
- 🗺️ **Map view** of the selected location.
- 🧠 **Model information panel** — type, expected feature order, and feature importances (if the model exposes them).
- ⚠️ **Robust error handling** — no internet, API timeouts, unknown cities, invalid coordinates, missing model files, and failed predictions all show friendly messages instead of crashing.
- ⚡ **Caching** — the model is loaded once per server process (`st.cache_resource`); weather lookups are cached for 10 minutes per location (`st.cache_data`) to avoid hammering the free API.

---

## Screenshots

_Add screenshots of the running dashboard here once deployed._

```
docs/screenshot-dashboard.png
docs/screenshot-gauges.png
```

---

## Installation

### 1. Clone and set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your trained model

Place your already-trained Random Forest model at:

```
model/best_random_forest.pkl
```

(`model/rf_model.pkl` is also recognized as a fallback filename.)

**Don't have a trained model yet?** A placeholder generator script is included purely so you can see the app working end-to-end:

```bash
python scripts/generate_placeholder_model.py
```

This fits a small Random Forest on synthetic data — it is **not** a real forecasting model. Replace the generated file with your actual trained model before using this for anything real. The app automatically picks up whichever file exists at that path; no code changes needed.

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

---

## Folder Structure

```
solar_prediction/
├── app.py                          # Main Streamlit dashboard
├── model/
│   └── best_random_forest.pkl      # Your trained model (not included)
├── utils/
│   ├── weather.py                  # Open-Meteo geocoding + live weather
│   ├── feature_engineering.py      # Irradiation & module temp derivation
│   ├── prediction.py                # Model loading + prediction
│   ├── confidence.py                # Tree-variance based confidence score
│   └── helpers.py                   # Logging, CSS, formatting, weather codes
├── scripts/
│   └── generate_placeholder_model.py  # Demo-only model generator (not used by the app)
├── assets/
│   └── icons/                       # Optional custom icons
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml                  # Green/white solar theme
```

---

## How Prediction Works

The model expects exactly four features, in this exact order:

| # | Feature              | Source                                            |
|---|-----------------------|----------------------------------------------------|
| 1 | `IRRADIATION`         | Derived from Open-Meteo shortwave radiation         |
| 2 | `Hour`                | Current local hour (0–23)                          |
| 3 | `MODULE_TEMPERATURE`  | Estimated from ambient temperature + radiation      |
| 4 | `AMBIENT_TEMPERATURE` | Live temperature from Open-Meteo                    |

**Irradiation conversion** (`utils/feature_engineering.py`):

```python
irradiation = shortwave_radiation / 1000.0   # clipped to [0, 1.2]
```

> Adjust `RADIATION_TO_IRRADIATION_DIVISOR` in `feature_engineering.py` if your training data used a different irradiation scale.

**Module temperature estimate** (per spec):

```python
module_temperature = ambient_temperature + (shortwave_radiation / 800) * 20
```

**Confidence score** (`utils/confidence.py`):

```python
tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
confidence = 100 - (std(tree_predictions) / max_prediction_range) * 100
```

> Set `DEFAULT_MAX_PREDICTION_RANGE` in `confidence.py` to your plant's realistic maximum AC power output for a well-calibrated confidence percentage.

---

## API Details

- **Provider:** [Open-Meteo](https://open-meteo.com) — free, no API key required.
- **Geocoding endpoint:** `https://geocoding-api.open-meteo.com/v1/search`
- **Forecast endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Fields used:** `temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`, `weather_code`, `shortwave_radiation`.
- Requests time out after 8 seconds and raise specific exceptions (`CityNotFoundError`, `WeatherAPITimeoutError`, `WeatherDataUnavailableError`) that the UI translates into friendly messages.

---

## Model Information

- **Type:** Random Forest Regressor (hyperparameter tuned — set by whoever trained your model)
- **Library:** scikit-learn
- **Prediction interval:** Next 15 minutes
- **Input features (in order):** `IRRADIATION`, `Hour`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`
- **Loading:** `joblib.load("model/best_random_forest.pkl")`, cached once per server process

If your model exposes `.feature_importances_`, the dashboard's "Model Information" panel will chart it automatically.

---

## Deployment Notes

This is a single-container Streamlit app with no database, so it deploys easily to:

- **Streamlit Community Cloud** — point it at your repo, set the entry point to `app.py`.
- **Railway / Render / Fly.io** — use a simple `Dockerfile` (or their native Python buildpack) with the start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- Make sure `model/best_random_forest.pkl` is included in your deployment (either committed to the repo via Git LFS, or downloaded at startup from your own storage).

---

## Future Improvements

- Historical prediction logging and accuracy tracking against actual plant output.
- Multi-panel/multi-plant support.
- Configurable forecast horizon (e.g. next hour, next day) if a compatible model is trained.
- User accounts for saving favorite plant locations.

---

## License

This project is provided as-is for internal or educational use. Adapt the license section to match your organization's requirements before distribution.
