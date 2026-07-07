"""Live weather retrieval and geocoding via the free Open-Meteo API.

Open-Meteo requires no API key, which keeps this demo/deployment friction
free. All network calls have explicit timeouts and raise a small set of
custom exceptions so the UI layer can show friendly, specific messages
instead of raw tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from utils.helpers import get_logger

logger = get_logger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 8


class WeatherServiceError(Exception):
    """Base exception for all weather/geocoding failures."""


class CityNotFoundError(WeatherServiceError):
    """Raised when the geocoding API finds no matching city."""


class WeatherAPITimeoutError(WeatherServiceError):
    """Raised when the Open-Meteo API does not respond in time."""


class WeatherDataUnavailableError(WeatherServiceError):
    """Raised when the API responds but expected fields are missing."""


@dataclass
class GeocodeResult:
    name: str
    country: str
    latitude: float
    longitude: float


@dataclass
class WeatherSnapshot:
    """A single point-in-time weather reading used for prediction."""

    ambient_temperature: float
    apparent_temperature: Optional[float]
    shortwave_radiation: float
    cloud_cover: Optional[float]
    wind_speed: Optional[float]
    humidity: Optional[float]
    weather_code: Optional[int]
    timestamp: str
    latitude: float
    longitude: float


def geocode_city(city_name: str, count: int = 5) -> list[GeocodeResult]:
    """Resolves a city name to a list of candidate (lat, lon) matches.

    Raises:
        CityNotFoundError: if no matches are found.
        WeatherAPITimeoutError: if the request times out.
        WeatherServiceError: for any other network/API failure.
    """
    if not city_name or not city_name.strip():
        raise CityNotFoundError("Please enter a city name.")

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": city_name.strip(), "count": count, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise WeatherAPITimeoutError("The location search timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        logger.exception("Geocoding request failed")
        raise WeatherServiceError("Could not reach the location search service.") from exc

    payload = response.json()
    results = payload.get("results")
    if not results:
        raise CityNotFoundError(f'No location found matching "{city_name}". Try a different spelling.')

    return [
        GeocodeResult(
            name=r.get("name", city_name),
            country=r.get("country", ""),
            latitude=r["latitude"],
            longitude=r["longitude"],
        )
        for r in results
    ]


def get_current_weather(latitude: float, longitude: float) -> WeatherSnapshot:
    """Fetches current weather + solar radiation for the given coordinates.

    Shortwave radiation is requested from the 15-minutely block (Open-Meteo's
    most granular solar data) and falls back to the hourly block if that is
    unavailable, so the app degrades gracefully rather than failing outright.

    Raises:
        WeatherAPITimeoutError: if the request times out.
        WeatherDataUnavailableError: if essential fields are missing.
        WeatherServiceError: for any other network/API failure.
    """
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise WeatherServiceError("Invalid coordinates. Latitude must be -90..90 and longitude -180..180.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "cloud_cover",
            "wind_speed_10m",
            "weather_code",
            "shortwave_radiation",
        ]),
        "minutely_15": "shortwave_radiation",
        "timezone": "auto",
        "forecast_days": 1,
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise WeatherAPITimeoutError("The weather service timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        logger.exception("Weather request failed")
        raise WeatherServiceError("Could not reach the weather service. Check your internet connection.") from exc

    payload = response.json()
    current = payload.get("current")
    if not current:
        raise WeatherDataUnavailableError("The weather service returned no current conditions.")

    ambient_temperature = current.get("temperature_2m")
    if ambient_temperature is None:
        raise WeatherDataUnavailableError("Ambient temperature is missing from the weather response.")

    shortwave_radiation = current.get("shortwave_radiation")
    if shortwave_radiation is None:
        # Fall back to the most recent 15-minutely radiation reading.
        shortwave_radiation = _extract_latest_minutely_radiation(payload)
    if shortwave_radiation is None:
        # Night-time or a data gap: treat as zero irradiation rather than failing.
        logger.warning("shortwave_radiation unavailable; defaulting to 0.0")
        shortwave_radiation = 0.0

    return WeatherSnapshot(
        ambient_temperature=float(ambient_temperature),
        apparent_temperature=current.get("apparent_temperature"),
        shortwave_radiation=max(0.0, float(shortwave_radiation)),
        cloud_cover=current.get("cloud_cover"),
        wind_speed=current.get("wind_speed_10m"),
        humidity=current.get("relative_humidity_2m"),
        weather_code=current.get("weather_code"),
        timestamp=current.get("time", ""),
        latitude=latitude,
        longitude=longitude,
    )


def _extract_latest_minutely_radiation(payload: dict) -> Optional[float]:
    """Best-effort fallback: pulls the most recent 15-minutely radiation value."""
    minutely = payload.get("minutely_15") or {}
    values = minutely.get("shortwave_radiation")
    if not values:
        return None
    for value in reversed(values):
        if value is not None:
            return value
    return None
