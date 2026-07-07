"""Loads the pre-trained Random Forest model and runs predictions.

This module NEVER trains or retrains a model - it only loads an existing
artifact and calls .predict() on it, per the project requirements.
"""

from __future__ import annotations
from utils.download_model import download_model

import os
from typing import Any

import joblib
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATHS = [
    os.path.join("model", "best_random_forest.pkl"),
    os.path.join("model", "rf_model.pkl"),
]


class ModelLoadError(Exception):
    """Raised when the trained model file cannot be found or loaded."""


class PredictionError(Exception):
    """Raised when the model fails to produce a prediction."""


def resolve_model_path(candidates: list[str] | None = None) -> str:
    """Finds the first existing model file.
    Downloads it automatically if it is missing.
    """

    for path in candidates or DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            logger.info("Found model: %s", path)
            return path

    logger.info("Model not found locally. Downloading from Google Drive...")

    try:
        return download_model()
    except Exception as exc:
        logger.exception("Failed to download model")
        raise ModelLoadError(
            f"Unable to download trained model from Google Drive: {exc}"
        ) from exc


def load_model(model_path: str | None = None) -> Any:
    """Loads the trained Random Forest model from disk via joblib.

    Raises:
        ModelLoadError: if the file is missing or fails to deserialize.
    """
    path = model_path or resolve_model_path()
    try:
        model = joblib.load(path)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model file not found at '{path}'.") from exc
    except Exception as exc:  # noqa: BLE001 - surface any deserialization issue clearly
        logger.exception("Failed to load model from %s", path)
        raise ModelLoadError(f"Could not load the model file at '{path}': {exc}") from exc

    logger.info("Loaded model from %s (%s)", path, type(model).__name__)
    return model


def predict_ac_power(model: Any, features_df: pd.DataFrame) -> float:
    """Runs the model's prediction and returns a single float (kW).

    Raises:
        PredictionError: if the model fails to produce a prediction, e.g. due
        to a feature mismatch.
    """
    try:
        prediction = model.predict(features_df)
    except Exception as exc:  # noqa: BLE001 - convert any sklearn error into a friendly one
        logger.exception("Prediction failed")
        raise PredictionError(
            "The model could not generate a prediction from the current features. "
            "This usually means the loaded model expects different input features."
        ) from exc

    value = float(prediction[0])
    return max(0.0, value)  # AC power cannot be negative
