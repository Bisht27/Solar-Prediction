"""AI Solar AC Power Forecasting System.

A premium Streamlit dashboard that predicts the next 15-minute AC power
output of a solar plant using live Open-Meteo weather data and an
already-trained Random Forest model. No model training happens here.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.confidence import DEFAULT_MAX_PREDICTION_RANGE, calculate_confidence
from utils.feature_engineering import engineer_features_from_weather, simulate_intraday_curve
from utils.helpers import (
    APP_VERSION,
    current_local_hour,
    describe_weather_code,
    format_number,
    get_logger,
    inject_custom_css,
    timestamp_string,
)
from utils.prediction import ModelLoadError, PredictionError, load_model, predict_ac_power
from utils.weather import (
    CityNotFoundError,
    GeocodeResult,
    WeatherServiceError,
    WeatherSnapshot,
    geocode_city,
    get_current_weather,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI Solar AC Power Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model():
    """Loads the trained model once per server process."""
    return load_model()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_cached(latitude: float, longitude: float) -> WeatherSnapshot:
    """Caches weather lookups for 10 minutes per coordinate pair to avoid
    hammering the free Open-Meteo API on every rerun.
    """
    return get_current_weather(latitude, longitude)


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city_cached(city_name: str) -> list[GeocodeResult]:
    return geocode_city(city_name)


# --------------------------------------------------------------------------
# UI building blocks
# --------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ☀️ About This Project")
        st.write(
            "A live, AI-powered dashboard that forecasts a solar plant's "
            "AC power output for the next 15 minutes using real-time weather data."
        )

        st.markdown("### 🧠 Model")
        st.markdown(
            "- **Type:** Random Forest (hyperparameter tuned)\n"
            "- **Library:** scikit-learn `RandomForestRegressor`\n"
            "- **Prediction interval:** Next 15 minutes"
        )

        st.markdown("### 📥 Features Used")
        st.markdown(
            "1. Irradiation\n"
            "2. Hour\n"
            "3. Module Temperature\n"
            "4. Ambient Temperature\n"
            "5. AC Power (previous 15 min)"
        )

        st.markdown("### 🌐 Weather API")
        st.markdown("[Open-Meteo](https://open-meteo.com) — free, no API key required")

        st.markdown("### 👩‍💻 Developer Info")
        st.markdown("Built with Streamlit, scikit-learn, and Plotly.")

        st.markdown("---")
        st.caption(f"Application version {APP_VERSION}")


def render_location_picker() -> Optional[tuple[float, float, str]]:
    """Renders the location input UI and returns (lat, lon, label) or None."""
    st.markdown("#### 📍 Choose a location")
    mode = st.radio(
        "Location method",
        options=["Search by city name", "Enter coordinates"],
        horizontal=True,
        label_visibility="collapsed",
    )

    location: Optional[tuple[float, float, str]] = None

    if mode == "Search by city name":
        col1, col2 = st.columns([3, 1])
        city_name = col1.text_input("City name", placeholder="e.g. Jaipur, India", label_visibility="collapsed")
        search_clicked = col2.button("Search", use_container_width=True)

        if search_clicked or city_name:
            if not city_name.strip():
                st.info("Enter a city name to search.")
            else:
                try:
                    matches = geocode_city_cached(city_name)
                    options = [f"{m.name}, {m.country} ({m.latitude:.2f}, {m.longitude:.2f})" for m in matches]
                    selected = st.selectbox("Matching locations", options)
                    idx = options.index(selected)
                    chosen = matches[idx]
                    location = (chosen.latitude, chosen.longitude, f"{chosen.name}, {chosen.country}")
                except CityNotFoundError as exc:
                    st.warning(str(exc))
                except WeatherServiceError as exc:
                    st.error(str(exc))
    else:
        col1, col2 = st.columns(2)
        lat = col1.number_input("Latitude", min_value=-90.0, max_value=90.0, value=26.9124, format="%.4f")
        lon = col2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.7873, format="%.4f")
        location = (lat, lon, f"{lat:.4f}, {lon:.4f}")

    st.markdown("<div style='margin-top:-0.5rem'></div>", unsafe_allow_html=True)
    if st.button("📍 Use Current Location", use_container_width=False):
        browser_location = _try_browser_geolocation()
        if browser_location:
            location = browser_location

    return location


def _try_browser_geolocation() -> Optional[tuple[float, float, str]]:
    """Attempts to read the browser's geolocation via streamlit-js-eval.

    Gracefully degrades if the optional package isn't installed or the user
    denies location permission - this is a nice-to-have, not a hard dependency.
    """
    try:
        from streamlit_js_eval import get_geolocation
    except ImportError:
        st.info(
            "Browser geolocation requires the optional `streamlit-js-eval` package. "
            "Install it (see requirements.txt) or use city/coordinate search instead."
        )
        return None

    try:
        coords = get_geolocation()
        if coords and "coords" in coords:
            lat = coords["coords"]["latitude"]
            lon = coords["coords"]["longitude"]
            return (lat, lon, "Current location")
        st.info("Waiting for browser location permission… click the button again once you've allowed access.")
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Browser geolocation failed")
        st.warning("Could not retrieve your browser location. Please search by city or enter coordinates.")
        return None


def make_gauge(value: float, title: str, max_value: float, unit: str, color: str) -> go.Figure:
    """Builds a consistently styled Plotly indicator gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {unit}", "font": {"size": 26}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e6f4ea",
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "#f0fdf4"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "#dcfce7"},
                    {"range": [max_value * 0.8, max_value], "color": "#bbf7d0"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def render_metric_card(title: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="dash-card">
            <h4>{title}</h4>
            <div class="value">{value}</div>
            <div class="subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(
    predicted_power: float,
    ambient_temperature: float,
    module_temperature: float,
    irradiation: float,
    hour: int,
    confidence: float,
) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">⚡ Predicted AC Power (Next 15 Minutes)</div>
            <div class="big-value">{predicted_power:,.1f} kW</div>
            <div class="result-grid">
                <div class="item"><div class="k">Ambient Temp</div><div class="v">{format_number(ambient_temperature, ' °C')}</div></div>
                <div class="item"><div class="k">Module Temp</div><div class="v">{format_number(module_temperature, ' °C')}</div></div>
                <div class="item"><div class="k">Irradiation</div><div class="v">{format_number(irradiation, '', 3)}</div></div>
                <div class="item"><div class="k">Hour</div><div class="v">{hour}:00</div></div>
                <div class="item"><div class="k">Confidence</div><div class="v">{confidence:.0f}%</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------


def main() -> None:
    inject_custom_css(st)

    st.markdown('<p class="app-title">☀️ AI Solar AC Power Forecasting System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Live Weather Powered AI Prediction Dashboard</p>',
        unsafe_allow_html=True,
    )

    render_sidebar()

    location = render_location_picker()
    st.markdown("---")

    if not location:
        st.info("Choose a location above to generate a live prediction.")
        return

    latitude, longitude, label = location

    weather: Optional[WeatherSnapshot] = None
    model = None
    features_df: Optional[pd.DataFrame] = None
    intermediate: dict = {}
    predicted_power: Optional[float] = None
    confidence: Optional[float] = None

    status_box = st.status("Fetching weather…", expanded=False)
    try:
        status_box.update(label="Fetching weather…")
        weather = fetch_weather_cached(latitude, longitude)

        status_box.update(label="Preparing features…")
        time.sleep(0.15)
        hour = current_local_hour()
        # AC_POWER_LAG1: the AC power predicted for this same location on the
        # previous run, used as a short-term momentum feature. Defaults to
        # 0.0 the first time a location is checked in this session.
        lag_key = f"ac_power_lag1::{round(latitude, 3)}::{round(longitude, 3)}"
        ac_power_lag1 = st.session_state.get(lag_key, 0.0)

        features_df, intermediate = engineer_features_from_weather(
            ambient_temperature=weather.ambient_temperature,
            shortwave_radiation=weather.shortwave_radiation,
            hour=hour,
            ac_power_lag1=ac_power_lag1,
        )

        status_box.update(label="Running AI model…")
        model = get_model()

        status_box.update(label="Generating prediction…")
        predicted_power = predict_ac_power(model, features_df)
        confidence = calculate_confidence(model, features_df, DEFAULT_MAX_PREDICTION_RANGE)
        st.session_state[lag_key] = predicted_power

        status_box.update(label="Done", state="complete")
    except ModelLoadError as exc:
        status_box.update(label="Model unavailable", state="error")
        st.error(f"⚠️ {exc}")
        st.caption("Run `python scripts/generate_placeholder_model.py` for a demo model, or add your trained model.")
    except PredictionError as exc:
        status_box.update(label="Prediction failed", state="error")
        st.error(f"⚠️ {exc}")
    except WeatherServiceError as exc:
        status_box.update(label="Weather unavailable", state="error")
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # noqa: BLE001 - last line of defense for a friendly message
        logger.exception("Unexpected error in main flow")
        status_box.update(label="Something went wrong", state="error")
        st.error("⚠️ An unexpected error occurred. Please try again in a moment.")

    if weather is None:
        return

    description, icon = describe_weather_code(weather.weather_code)

    # ---- Top row: weather / prediction / confidence -----------------------
    top1, top2, top3 = st.columns([1, 1.4, 1])
    with top1:
        render_metric_card("Current Weather", f"{icon} {description}", label)
    with top2:
        if predicted_power is not None:
            render_metric_card("Predicted AC Power", f"{predicted_power:,.1f} kW", "Next 15 minutes")
        else:
            render_metric_card("Predicted AC Power", "—", "Unavailable")
    with top3:
        if confidence is not None:
            render_metric_card("Prediction Confidence", f"{confidence:.0f}%", "Model agreement across trees")
        else:
            render_metric_card("Prediction Confidence", "—", "Unavailable")

    st.write("")

    # ---- Second row: weather detail cards ----------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Temperature", format_number(weather.ambient_temperature, " °C"), "Ambient")
    with c2:
        render_metric_card(
            "Solar Irradiation",
            format_number(intermediate.get("irradiation"), "", 3) if intermediate else "—",
            f"{format_number(weather.shortwave_radiation, ' W/m²')} shortwave",
        )
    with c3:
        render_metric_card("Cloud Cover", format_number(weather.cloud_cover, " %"), "")
    with c4:
        render_metric_card("Wind Speed", format_number(weather.wind_speed, " km/h"), "")
    with c5:
        render_metric_card(
            "AC Power Lag-1",
            format_number(intermediate.get("ac_power_lag1"), " kW") if intermediate else "—",
            "Previous 15-min reading",
        )

    st.write("")

    # ---- Result hero card ---------------------------------------------------
    if predicted_power is not None and confidence is not None:
        render_result_card(
            predicted_power=predicted_power,
            ambient_temperature=weather.ambient_temperature,
            module_temperature=intermediate.get("module_temperature", 0.0),
            irradiation=intermediate.get("irradiation", 0.0),
            hour=intermediate.get("hour", current_local_hour()),
            confidence=confidence,
        )

    st.write("")

    # ---- Gauges ---------------------------------------------------------------
    st.markdown("#### 📊 Live Indicators")
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1:
        st.plotly_chart(
            make_gauge(weather.ambient_temperature, "Temperature (°C)", 50, "°C", "#16a34a"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            make_gauge(weather.shortwave_radiation, "Irradiation (W/m²)", 1200, "W/m²", "#eab308"),
            use_container_width=True,
        )
    with g3:
        if predicted_power is not None:
            st.plotly_chart(
                make_gauge(predicted_power, "Predicted AC Power (kW)", DEFAULT_MAX_PREDICTION_RANGE, "kW", "#15803d"),
                use_container_width=True,
            )
    with g4:
        st.plotly_chart(
            make_gauge(weather.cloud_cover or 0, "Cloud Cover (%)", 100, "%", "#64748b"),
            use_container_width=True,
        )
    with g5:
        st.plotly_chart(
            make_gauge(weather.wind_speed or 0, "Wind Speed (km/h)", 100, "km/h", "#0ea5e9"),
            use_container_width=True,
        )

    st.write("")

    # ---- Intraday trend ---------------------------------------------------
    if model is not None:
        st.markdown("#### 📈 Intraday Power Trend")
        st.caption(
            "Simulated 24-hour AC power curve for today, anchored to the current "
            "weather reading. Each hour's prediction feeds the next hour's "
            "AC Power Lag-1 feature, so this is illustrative — not a certified "
            "hour-by-hour forecast."
        )
        try:
            intraday_df = simulate_intraday_curve(
                model=model,
                ambient_temperature=weather.ambient_temperature,
                shortwave_radiation=weather.shortwave_radiation,
                current_hour=hour,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=intraday_df["hour"],
                    y=intraday_df["predicted_ac_power"],
                    mode="lines",
                    line=dict(color="#15803d", width=3, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(22, 163, 74, 0.12)",
                    name="Predicted AC Power",
                )
            )
            fig.add_vline(x=hour, line_dash="dot", line_color="#94a3b8")
            fig.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Hour of day", dtick=2, range=[0, 23]),
                yaxis=dict(title="AC Power (kW)"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render intraday trend")
            st.info("Intraday trend unavailable for this prediction.")

    st.write("")

    # ---- Third row: map, weather summary, model info -------------------------
    m1, m2, m3 = st.columns([1.2, 1, 1])
    with m1:
        st.markdown("#### 🗺️ Location")
        st.map(pd.DataFrame([{"lat": latitude, "lon": longitude}]), zoom=8, size=200)

    with m2:
        st.markdown("#### 🌤️ Weather Summary")
        with st.container(border=True):
            st.write(f"**Conditions:** {icon} {description}")
            st.write(f"**Temperature:** {format_number(weather.ambient_temperature, ' °C')}")
            st.write(f"**Feels like:** {format_number(weather.apparent_temperature, ' °C')}")
            st.write(f"**Wind:** {format_number(weather.wind_speed, ' km/h')}")
            st.write(f"**Cloud cover:** {format_number(weather.cloud_cover, ' %')}")
            st.write(f"**Radiation:** {format_number(weather.shortwave_radiation, ' W/m²')}")
            st.write(f"**Humidity:** {format_number(weather.humidity, ' %')}")
            st.caption(f"Observed at {weather.timestamp or timestamp_string()}")

    with m3:
        st.markdown("#### 🧠 Model Information")
        with st.container(border=True):
            st.write("**Model type:** Random Forest Regressor")
            st.write("**Training date:** _(set by model author)_")
            st.write("**Number of features:** 5")
            st.write("**Expected input order:**")
            st.code(
                "IRRADIATION, Hour, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, AC_POWER_LAG1",
                language="text",
            )
            st.write("**Prediction frequency:** Every 15 minutes")

            if model is not None and hasattr(model, "feature_importances_"):
                st.write("**Feature importance:**")
                importance_df = pd.DataFrame(
                    {
                        "feature": [
                            "IRRADIATION",
                            "Hour",
                            "MODULE_TEMPERATURE",
                            "AMBIENT_TEMPERATURE",
                            "AC_POWER_LAG1",
                        ],
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                st.bar_chart(importance_df.set_index("feature"))

    st.markdown("---")
    st.caption(
        "Predictions are estimates based on live weather data and a statistical model. "
        "Actual plant output may vary due to shading, soiling, equipment status, and other site factors."
    )


if __name__ == "__main__":
    main()
