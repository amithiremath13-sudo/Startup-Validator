"""
inference/model_loader.py
==========================
Phase 4A — Model and Metadata Loading

Loads the trained Random Forest and associated metadata with strict
validation. Any mismatch between the model's expected feature count
and the current feature_names.json raises an error immediately —
preventing silent wrong predictions.

All paths are relative to the project root, which is inferred from this
file's location (inference/ is one level below the root).
"""

import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Resolve project root: inference/ is one level down from root
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Expected file paths (all must exist before any prediction)
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
FEAT_JSON     = os.path.join(PROJECT_ROOT, "processed_data", "feature_names.json")
META_JSON     = os.path.join(PROJECT_ROOT, "processed_data", "preprocessing_metadata.json")
MODEL_META    = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

EXPECTED_FEATURE_COUNT = 65
EXPECTED_MODEL_TYPE    = "RandomForestClassifier"


class ModelLoadError(Exception):
    """Raised when the model or metadata cannot be loaded or validated."""


def load_model_and_metadata() -> Tuple[object, List[str], Dict]:
    """
    Load the trained model, feature names, and preprocessing metadata.

    Returns
    -------
    model         : sklearn RandomForestClassifier (loaded from joblib)
    feature_names : list of str, length 65, in exact training order
    prep_meta     : dict from preprocessing_metadata.json

    Raises
    ------
    ModelLoadError : if any file is missing or validation fails.
    """
    # --- Check files exist ---
    for path in [MODEL_PATH, FEAT_JSON, META_JSON]:
        if not os.path.exists(path):
            raise ModelLoadError(f"Required file not found: {path}")

    # --- Load model ---
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {MODEL_PATH}: {e}")

    # --- Validate model type ---
    if type(model).__name__ != EXPECTED_MODEL_TYPE:
        raise ModelLoadError(
            f"Expected {EXPECTED_MODEL_TYPE}, got {type(model).__name__}"
        )

    # --- Load feature names ---
    with open(FEAT_JSON, encoding="utf-8") as f:
        feature_names = json.load(f)

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ModelLoadError(
            f"feature_names.json has {len(feature_names)} features; "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )

    # --- Validate model expects the same number of features ---
    if model.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise ModelLoadError(
            f"Model was trained on {model.n_features_in_} features; "
            f"feature_names.json has {EXPECTED_FEATURE_COUNT}. Mismatch."
        )

    # --- Load preprocessing metadata ---
    with open(META_JSON, encoding="utf-8") as f:
        prep_meta = json.load(f)

    return model, feature_names, prep_meta


def predict(model, feature_vector_list: list,
            feature_names: list = None) -> Tuple[int, float, float]:
    """
    Run inference on a single feature vector.

    Parameters
    ----------
    model              : loaded RandomForestClassifier
    feature_vector_list: ordered list of 65 numeric values
    feature_names      : optional list of feature names (from feature_names.json).
                         When supplied the vector is wrapped in a DataFrame so
                         sklearn does not emit a UserWarning about missing feature
                         names. Always supply this in normal use.

    Returns
    -------
    predicted_class   : int (0 = failure, 1 = success)
    prob_failure      : float (probability of class 0)
    prob_success      : float (probability of class 1)

    Raises
    ------
    ValueError : if the feature vector has wrong length or contains NaN.
    """
    if len(feature_vector_list) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature vector has {len(feature_vector_list)} values; "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    arr = np.array(feature_vector_list, dtype=float)

    if np.isnan(arr).any():
        nan_indices = np.where(np.isnan(arr))[0].tolist()
        raise ValueError(f"NaN values at positions {nan_indices} in feature vector.")

    # Wrap in a DataFrame with named columns so sklearn does not warn about
    # missing feature names (the model was fitted on a DataFrame with names).
    if feature_names is not None:
        X = pd.DataFrame([arr], columns=feature_names)
    else:
        X = arr.reshape(1, -1)

    predicted_class   = int(model.predict(X)[0])
    probabilities     = model.predict_proba(X)[0]

    # classes_ = [0, 1], so index 0 = P(failure), index 1 = P(success)
    prob_failure = float(probabilities[0])
    prob_success = float(probabilities[1])

    return predicted_class, prob_failure, prob_success
