"""
api/dependencies.py
====================
Startup-time resource loading for the FastAPI backend.

The ML model, feature names, and preprocessing metadata are expensive to load
(~0.3 s each time). We load them once when the application starts and share
the same objects across all requests via FastAPI's dependency-injection system.

Nothing in this file performs inference — it only handles loading and caching.
"""

import importlib.util
import os
import sys

# ── Ensure the project root is on sys.path ────────────────────────────────
_API_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_API_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.model_loader import load_model_and_metadata, ModelLoadError


# ── Load the run_full_evaluation function from 05_evaluate_idea.py ────────
# 05_evaluate_idea.py starts with a digit so it cannot be imported with a
# normal import statement — we use importlib.
def _import_run_full_evaluation():
    spec = importlib.util.spec_from_file_location(
        "evaluate_idea",
        os.path.join(PROJECT_ROOT, "05_evaluate_idea.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_full_evaluation


# ── Module-level singletons (loaded once at import time) ─────────────────
# These are populated by initialize_resources() called from the FastAPI
# lifespan handler, ensuring errors surface at startup rather than on the
# first request.

_model        = None
_feature_names = None
_prep_meta    = None
_run_full_evaluation = None


def initialize_resources() -> None:
    """
    Load all heavy resources. Called once during application startup.
    Raises ModelLoadError if the model file or metadata is missing or corrupt.
    """
    global _model, _feature_names, _prep_meta, _run_full_evaluation

    _model, _feature_names, _prep_meta = load_model_and_metadata()
    _run_full_evaluation = _import_run_full_evaluation()


def get_run_full_evaluation():
    """FastAPI dependency: return the cached run_full_evaluation callable."""
    if _run_full_evaluation is None:
        raise RuntimeError(
            "Resources not initialised. Call initialize_resources() at startup."
        )
    return _run_full_evaluation
