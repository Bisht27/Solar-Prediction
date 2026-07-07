"""Converts raw live weather data into the exact feature dataframe the
trained Random Forest model expects.

IMPORTANT: the model was trained with features in this EXACT order:
    1. IRRADIATION
    2. Hour
    3. MODULE_TEMPERATURE
    4. AMBIENT_TEMPERATURE
    5. AC_POWER_LAG1
This order must never change.
"""

from __future__ import annotations

import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]

# Open-Meteo's shortwave_radiation is in W/m^2. Solar plant datasets (the kind
# these Random Forest models are typically trained on) commonly express
# IRRADIATION as a fraction of the ~1000 W/m^2 "standard test condition".
# Adjust this constant if your training data used a different scale.
RADIATION_TO_IRRADIATION_DIVISOR = 1000.0
IRRADIATION_CLIP_MIN, IRRADIATION_CLIP_MAX = 0.0, 1.2

# Realistic module temperature bounds for clipping the heuristic estimate.
MODULE_TEMP_CLIP_MIN, MODULE_TEMP_CLIP_MAX = -10.0, 85.0


def shortwave_radiation_to_irradiation(shortwave_radiation: float) -> float:
    """Converts Open-Meteo shortwave radiation (W/m^2) into the model's
    IRRADIATION feature scale, clipped to a realistic range.
    """
    irradiation = shortwave_radiation / RADIATION_TO_IRRADIATION_DIVISOR
    return float(min(max(irradiation, IRRADIATION_CLIP_MIN), IRRADIATION_CLIP_MAX))


def estimate_module_temperature(ambient_temperature: float, shortwave_radiation: float) -> float:
    """Estimates solar module temperature since Open-Meteo does not provide it.

    Formula (per project spec):
        Module Temperature = Ambient Temperature + (Shortwave Radiation / 800) * 20
    """
    estimated = ambient_temperature + (shortwave_radiation / 800.0) * 20.0
    clipped = min(max(estimated, MODULE_TEMP_CLIP_MIN), MODULE_TEMP_CLIP_MAX)
    if clipped != estimated:
        logger.info("Module temperature %.2f clipped to %.2f", estimated, clipped)
    return float(clipped)


def build_feature_dataframe(
    irradiation: float,
    hour: int,
    module_temperature: float,
    ambient_temperature: float,
    ac_power_lag1: float = 0.0,
) -> pd.DataFrame:
    """Builds the single-row dataframe in the exact column order the model expects.

    ``ac_power_lag1`` is the AC power predicted (or observed) for the
    previous 15-minute interval. Pass 0.0 when there is no prior reading yet
    (e.g. the very first prediction of a session).
    """
    return pd.DataFrame(
        [[irradiation, hour, module_temperature, ambient_temperature, ac_power_lag1]],
        columns=FEATURE_COLUMNS,
    )


def engineer_features_from_weather(
    ambient_temperature: float,
    shortwave_radiation: float,
    hour: int,
    ac_power_lag1: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: raw weather -> engineered features -> model-ready dataframe.

    Args:
        ac_power_lag1: The previous interval's AC power output (kW). This
            captures short-term momentum (e.g. a passing cloud) that the
            instantaneous weather features alone can't express. Defaults to
            0.0 when no prior reading is available yet.

    Returns the feature dataframe plus a dict of the intermediate values
    (irradiation, module_temperature) for display in the UI.
    """
    irradiation = shortwave_radiation_to_irradiation(shortwave_radiation)
    module_temperature = estimate_module_temperature(ambient_temperature, shortwave_radiation)

    features_df = build_feature_dataframe(
        irradiation=irradiation,
        hour=hour,
        module_temperature=module_temperature,
        ambient_temperature=ambient_temperature,
        ac_power_lag1=ac_power_lag1,
    )

    intermediate = {
        "irradiation": irradiation,
        "module_temperature": module_temperature,
        "hour": hour,
        "ac_power_lag1": ac_power_lag1,
    }
    return features_df, intermediate


def _daylight_factor(hour: int) -> float:
    """Simple bell-shaped daylight curve peaking at midday, zero at night."""
    import math

    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, math.sin((hour - 6) / 12 * math.pi))


def simulate_intraday_curve(
    model,
    ambient_temperature: float,
    shortwave_radiation: float,
    current_hour: int,
) -> pd.DataFrame:
    """Simulates a full 0-23h AC power curve for "today" using the current
    weather reading as an anchor point.

    Rather than requiring historical data, this scales the *current* solar
    radiation up/down across the day using a daylight bell curve normalized
    so that the simulated value at ``current_hour`` matches today's actual
    reading. AC_POWER_LAG1 is chained hour-to-hour (each hour's prediction
    feeds the next hour's lag feature), giving a smooth, self-consistent
    intraday trend for the dashboard.

    This is a visualization aid, not a certified forecast for other hours.
    """
    anchor_factor = _daylight_factor(current_hour)
    # Avoid divide-by-zero at night; fall back to a nominal peak-hour scale.
    peak_radiation = shortwave_radiation / anchor_factor if anchor_factor > 1e-3 else shortwave_radiation

    rows = []
    ac_power_lag1 = 0.0
    for hour in range(24):
        factor = _daylight_factor(hour)
        hour_radiation = max(0.0, peak_radiation * factor)
        hour_irradiation = shortwave_radiation_to_irradiation(hour_radiation)
        hour_module_temp = estimate_module_temperature(ambient_temperature, hour_radiation)

        features_df = build_feature_dataframe(
            irradiation=hour_irradiation,
            hour=hour,
            module_temperature=hour_module_temp,
            ambient_temperature=ambient_temperature,
            ac_power_lag1=ac_power_lag1,
        )
        predicted = float(max(0.0, model.predict(features_df)[0]))
        rows.append({"hour": hour, "predicted_ac_power": predicted})
        ac_power_lag1 = predicted

    return pd.DataFrame(rows)
