"""Estimates prediction confidence by measuring how much the individual
trees in the Random Forest disagree with each other.

A tight spread across trees implies the model is confident; a wide spread
implies the input sits in a less certain region of the model's learned space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

# The AC power range (in the same units as the model's target) used to
# normalize tree-disagreement into a 0-100% confidence score. Set this to
# your plant's realistic max AC power for a well-calibrated confidence value.
DEFAULT_MAX_PREDICTION_RANGE = 1000.0


def calculate_confidence(
    model: Any,
    features_df: pd.DataFrame,
    max_prediction_range: float = DEFAULT_MAX_PREDICTION_RANGE,
) -> float:
    """Returns a confidence percentage (0-100) for the current prediction.

    If the model does not expose individual trees (i.e. it isn't a bagged
    ensemble like RandomForestRegressor), a neutral default confidence of
    75% is returned rather than failing the whole prediction flow.
    """
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        logger.info("Model has no .estimators_; returning default confidence.")
        return 75.0

    try:
        tree_predictions = np.array([tree.predict(features_df)[0] for tree in estimators])
    except Exception:  # noqa: BLE001
        logger.exception("Failed to gather per-tree predictions; returning default confidence.")
        return 75.0

    std_dev = float(np.std(tree_predictions))
    safe_range = max(max_prediction_range, 1e-6)  # avoid division by zero
    confidence = 100.0 - (std_dev / safe_range) * 100.0
    return float(min(max(confidence, 0.0), 100.0))
