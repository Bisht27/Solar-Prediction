"""Shared helper utilities used across the app: logging setup, custom CSS
injection for the premium dashboard look, WMO weather code translation,
and small formatting helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Returns a module-level logger configured with a consistent format.

    Streamlit re-imports modules across reruns, so we guard against
    attaching duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------
# WMO weather code -> human description + icon
# See: https://open-meteo.com/en/docs (WMO Weather interpretation codes)
# --------------------------------------------------------------------------

_WEATHER_CODE_MAP: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌦️"),
    56: ("Light freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snow fall", "🌨️"),
    73: ("Moderate snow fall", "🌨️"),
    75: ("Heavy snow fall", "🌨️"),
    77: ("Snow grains", "🌨️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌦️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def describe_weather_code(code: Optional[int]) -> tuple[str, str]:
    """Translates a WMO weather code into (description, emoji icon).

    Falls back to a neutral description if the code is unknown or missing,
    so the UI never breaks on unexpected API responses.
    """
    if code is None:
        return "Unknown", "❓"
    return _WEATHER_CODE_MAP.get(int(code), ("Unknown", "❓"))


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def format_number(value: Optional[float], suffix: str = "", decimals: int = 1) -> str:
    """Formats a numeric value for display, gracefully handling None."""
    if value is None:
        return "—"
    try:
        return f"{value:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def current_local_hour() -> int:
    """Returns the current local hour (0-23) used as the model's Hour feature."""
    return datetime.now().hour


def timestamp_string() -> str:
    """Human-readable current timestamp for the 'Prediction Time' display."""
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


# --------------------------------------------------------------------------
# UI styling
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #f7fdf9 0%, #ffffff 40%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Generic card */
    .dash-card {
        background: #ffffff;
        border: 1px solid #e6f4ea;
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 18px rgba(16, 24, 40, 0.05);
        height: 100%;
    }

    .dash-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .dash-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }

    .dash-card .subtext {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }

    /* Hero result card */
    .result-card {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        border-radius: 22px;
        padding: 2rem 2.2rem;
        color: white;
        box-shadow: 0 12px 30px rgba(22, 163, 74, 0.28);
    }

    .result-card .label {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    .result-card .big-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.2rem 0 1rem 0;
        line-height: 1;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }

    .result-grid .item {
        background: rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }

    .result-grid .item .k {
        font-size: 0.72rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .result-grid .item .v {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 0;
    }

    .app-subtitle {
        color: #4b5563;
        font-size: 1.02rem;
        margin-top: 0.15rem;
        margin-bottom: 1.6rem;
    }

    section[data-testid="stSidebar"] {
        background: #f7fdf9;
        border-right: 1px solid #e6f4ea;
    }
</style>
"""


def inject_custom_css(streamlit_module) -> None:
    """Injects the dashboard's custom CSS into the given streamlit module.

    Passing the module in (rather than importing streamlit here) keeps this
    helper decoupled from Streamlit for easier unit testing.
    """
    streamlit_module.markdown(CUSTOM_CSS, unsafe_allow_html=True)
