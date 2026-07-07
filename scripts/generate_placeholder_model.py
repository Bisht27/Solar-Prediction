"""Generates a PLACEHOLDER Random Forest model so the app can run end-to-end
before you drop in your real, already-trained model.

This script is NOT part of the application itself and is never imported or
invoked by app.py. Run it manually only if you want a working demo model:

    python scripts/generate_placeholder_model.py

It fits a small RandomForestRegressor on synthetic data that roughly mimics
the relationship between irradiation/temperature and AC power, purely so the
dashboard has something realistic to display. Replace
model/best_random_forest.pkl with your actual trained model for real use -
the app will pick it up automatically since it loads by filename.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_random_forest.pkl")
FEATURE_COLUMNS = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]


def make_synthetic_dataset(n_samples: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)

    irradiation = rng.uniform(0, 1.1, n_samples)
    hour = rng.integers(0, 24, n_samples)
    ambient_temperature = rng.uniform(15, 42, n_samples)
    module_temperature = ambient_temperature + irradiation * 25 + rng.normal(0, 1.5, n_samples)

    daylight_factor = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, 1)
    base_power = irradiation * daylight_factor * 900
    temp_derate = 1 - np.clip((module_temperature - 25) * 0.004, 0, 0.25)
    noise = rng.normal(0, 15, n_samples)

    ac_power = np.clip(base_power * temp_derate + noise, 0, None)

    # AC_POWER_LAG1 simulates the previous 15-minute interval's AC power.
    # It's strongly correlated with the current reading (short-term
    # persistence/momentum) but not identical, plus its own noise so the
    # model has to learn a genuine (not trivial) relationship.
    lag_noise = rng.normal(0, 25, n_samples)
    ac_power_lag1 = np.clip(ac_power * rng.uniform(0.85, 1.05, n_samples) + lag_noise, 0, None)

    features = pd.DataFrame(
        np.column_stack(
            [irradiation, hour, module_temperature, ambient_temperature, ac_power_lag1]
        ),
        columns=FEATURE_COLUMNS,
    )
    return features, ac_power


def main() -> None:
    features, target = make_synthetic_dataset()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, target)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Placeholder model written to {os.path.abspath(OUTPUT_PATH)}")
    print("Replace this file with your real trained model for production use.")


if __name__ == "__main__":
    main()
