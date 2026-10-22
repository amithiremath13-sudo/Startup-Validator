"""
=============================================================================
04_predict_startup.py
=============================================================================
Phase 4A — Startup Idea Inference Pipeline (CLI)
Project: AI-Based Startup Viability Prediction and Validation System

PURPOSE
-------
Accept a startup idea (as free text plus optional structured inputs),
extract the 65-feature vector the trained Random Forest expects, run
inference, and print the result.

IMPORTANT LIMITATIONS
---------------------
1. This script uses ONLY the historical ML model (Component A from
   ML_METHODOLOGY.md §11). It does NOT run the idea-scoring engine
   (Component B), which is built in a later phase.

2. The model output is described as a "model-estimated success score",
   NOT a real-world probability. ROC-AUC does not prove probability
   calibration. See MODEL_EVALUATION_REPORT.md §17.

3. The model was trained on Crunchbase data from ~1990–2016. It cannot
   account for sectors or market conditions that emerged after 2016.

4. Features available at Day 0 only: sector + country + geography.
   The model cannot evaluate idea quality, team, market size, or revenue.

HOW TO RUN
----------
Interactive mode (asks for inputs):
    python 04_predict_startup.py

Single idea from command line:
    python 04_predict_startup.py --idea "AI platform for restaurants in India"

With explicit country and categories:
    python 04_predict_startup.py \
        --idea "A platform for managing invoices for small businesses" \
        --country "USA" \
        --categories "Software" "Finance"

From a JSON file:
    python 04_predict_startup.py --json-input path/to/input.json

    JSON format:
    {
        "idea_text": "...",
        "country": "India",
        "categories": ["Education"],
        "location_detail": "Bangalore, India"
    }

OUTPUTS
-------
Prints to console:
  - extracted features (structured)
  - resolved country and categories
  - warnings (if any)
  - predicted class (0 or 1)
  - model-estimated success score (probability-like output from RF)
  - caveats and limitations
=============================================================================
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inference.feature_extractor import (
    EXPECTED_FEATURE_COUNT,
    FEATURE_ORDER,
    ExtractionResult,
    StartupIdeaInput,
    extract_features,
    feature_vector_to_ordered_list,
)
from inference.model_loader import ModelLoadError, load_model_and_metadata, predict


# ---------------------------------------------------------------------------
# OUTPUT FORMATTING HELPERS
# ---------------------------------------------------------------------------

def _sep(char="=", width=65):
    return char * width


def _print_header():
    print()
    print(_sep())
    print("  AI-Based Startup Viability Prediction — Phase 4A Inference")
    print(_sep())
    print("  Prediction point : Day 0 (pre-funding idea stage)")
    print("  Model            : Random Forest (selected by CV AUC)")
    print("  Spec reference   : ML_METHODOLOGY.md Section 11")
    print(_sep())


def _print_extraction_summary(result: ExtractionResult):
    print()
    print(_sep("-"))
    print("  FEATURE EXTRACTION SUMMARY")
    print(_sep("-"))

    # Country
    if result.resolved_country:
        print(f"  Country resolved     : {result.resolved_country}")
    else:
        print(f"  Country resolved     : (none — using country_Unknown flag)")

    # Categories
    real_cats = [c for c in result.resolved_categories if c != "Unknown"]
    if real_cats:
        print(f"  Categories matched   : {', '.join(real_cats)}")
    else:
        print(f"  Categories matched   : (none — using cat_Unknown flag)")

    # Geography flags
    fv = result.feature_vector
    print(f"  is_sf_bay_area       : {fv['is_sf_bay_area']}")
    print(f"  is_california        : {fv['is_california']}")
    print(f"  n_categories         : {fv['n_categories']}")
    print(f"  category_other       : {fv['category_other']}")

    # Key flag values
    active_country_flags = [
        k for k in FEATURE_ORDER
        if k.startswith("country_") and fv.get(k, 0) == 1
    ]
    print(f"  Active country flags : {active_country_flags if active_country_flags else '(none)'}")

    active_cat_flags = [
        k for k in FEATURE_ORDER
        if k.startswith("cat_") and fv.get(k, 0) == 1
    ]
    print(f"  Active category flags: {active_cat_flags if active_cat_flags else '(none)'}")

    print(f"  Submission complete  : {result.is_complete_submission}")

    # Warnings
    if result.warnings:
        print()
        print("  WARNINGS:")
        for w in result.warnings:
            # Word-wrap at 60 chars for readability
            words = w.split()
            line = "    ! "
            for word in words:
                if len(line) + len(word) + 1 > 65:
                    print(line)
                    line = "      " + word + " "
                else:
                    line += word + " "
            print(line.rstrip())


def _print_full_feature_vector(fv: dict):
    print()
    print(_sep("-"))
    print("  FULL FEATURE VECTOR (65 features in model order)")
    print(_sep("-"))
    for i, name in enumerate(FEATURE_ORDER, 1):
        val = fv[name]
        marker = " <-- ACTIVE" if val != 0 else ""
        print(f"  {i:>3}. {name:<40} {val}{marker}")
    print(f"  Total: {len(fv)} features")


def _print_prediction(predicted_class: int, prob_failure: float,
                       prob_success: float, result: ExtractionResult):
    print()
    print(_sep())
    print("  ML PREDICTION RESULT")
    print(_sep())

    if predicted_class == 1:
        label = "SUCCESS PATTERN (1)"
        desc  = "Historical pattern matches companies that were acquired or IPO'd."
    else:
        label = "FAILURE PATTERN (0)"
        desc  = "Historical pattern matches companies that were closed."

    print(f"  Predicted class      : {label}")
    print(f"  {desc}")
    print()
    print(f"  Model-estimated success score : {prob_success:.4f}  ({prob_success*100:.1f}%)")
    print(f"  Model-estimated failure score : {prob_failure:.4f}  ({prob_failure*100:.1f}%)")
    print()
    print("  IMPORTANT CAVEATS:")
    print("  - This score is the Random Forest's output based on historical")
    print("    sector + country patterns from ~1990–2016 Crunchbase data.")
    print("  - It is NOT a real-world probability of success.")
    print("  - Probability calibration has not been applied (Phase 4B).")
    print("  - The model cannot evaluate idea quality, team, market size,")
    print("    competition, revenue potential, or technical feasibility.")

    if not result.is_complete_submission:
        print()
        print("  NOTE: This submission was incomplete (missing country or category).")
        print("  The model relied on historical missing-data artifact features")
        print("  (country_Unknown and/or cat_Unknown). These features reflect")
        print("  a data recording pattern in the Crunchbase dataset, NOT a")
        print("  real signal about your startup. Provide a country and category")
        print("  for more meaningful results.")

    print(_sep())


def _print_validation_check(fv: dict, feature_names: list):
    """Print the feature validation results."""
    count_ok    = len(fv) == EXPECTED_FEATURE_COUNT
    order_ok    = list(fv.keys()) == list(FEATURE_ORDER)
    nan_ok      = all(v is not None for v in fv.values())
    no_target   = "target" not in fv
    no_status   = "status" not in fv
    feature_match = list(fv.keys()) == feature_names

    print()
    print(_sep("-"))
    print("  FEATURE VALIDATION")
    print(_sep("-"))
    print(f"  Feature count == 65      : {'PASS' if count_ok else 'FAIL'} ({len(fv)})")
    print(f"  Feature order matches JSON: {'PASS' if order_ok else 'FAIL'}")
    print(f"  No NaN values            : {'PASS' if nan_ok else 'FAIL'}")
    print(f"  'target' absent          : {'PASS' if no_target else 'FAIL'}")
    print(f"  'status' absent          : {'PASS' if no_status else 'FAIL'}")
    print(f"  Names match feature_names: {'PASS' if feature_match else 'FAIL'}")

    if not all([count_ok, order_ok, nan_ok, no_target, no_status, feature_match]):
        print()
        print("  VALIDATION FAILED — prediction aborted.")
        sys.exit(1)
    else:
        print("  All checks passed.")


# ---------------------------------------------------------------------------
# INPUT PARSERS
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 4A: Startup Idea Inference Pipeline"
    )
    p.add_argument("--idea",       type=str, default=None,
                   help="Free-text description of the startup idea")
    p.add_argument("--country",    type=str, default=None,
                   help="Country name or ISO-3 code (e.g. 'India', 'IND', 'USA')")
    p.add_argument("--categories", type=str, nargs="*", default=None,
                   help="One or more approved category names")
    p.add_argument("--location",   type=str, default=None,
                   help="Location detail (e.g. 'San Francisco, CA')")
    p.add_argument("--json-input", type=str, default=None,
                   help="Path to JSON file with idea, country, categories, location_detail")
    p.add_argument("--verbose",    action="store_true",
                   help="Print full 65-feature vector and extraction log")
    return p.parse_args()


def _load_input(args) -> StartupIdeaInput:
    """Build a StartupIdeaInput from CLI args or JSON file."""

    if args.json_input:
        with open(args.json_input, encoding="utf-8") as f:
            data = json.load(f)
        return StartupIdeaInput(
            idea_text       = data.get("idea_text", ""),
            country         = data.get("country"),
            categories      = data.get("categories", []),
            location_detail = data.get("location_detail"),
        )

    if args.idea:
        return StartupIdeaInput(
            idea_text       = args.idea,
            country         = args.country,
            categories      = args.categories or [],
            location_detail = args.location,
        )

    # Interactive mode
    print()
    print("  No --idea provided. Running in interactive mode.")
    print("  (Press Enter to skip optional fields.)")
    print()
    idea_text = input("  Describe your startup idea: ").strip()
    if not idea_text:
        print("  Error: idea text cannot be empty.")
        sys.exit(1)
    country  = input("  Country (e.g. India, USA) [optional]: ").strip() or None
    loc      = input("  Location detail (e.g. San Francisco, CA) [optional]: ").strip() or None
    cats_raw = input(
        "  Categories (comma-separated from approved list) [optional]: "
    ).strip()
    categories = [c.strip() for c in cats_raw.split(",") if c.strip()] if cats_raw else []

    return StartupIdeaInput(
        idea_text       = idea_text,
        country         = country,
        categories      = categories,
        location_detail = loc,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_inference(input_data: StartupIdeaInput, verbose: bool = False) -> dict:
    """
    Core inference function. Usable directly from other modules (e.g. tests).

    Returns a dict with:
      predicted_class : int (0 or 1)
      prob_success    : float
      prob_failure    : float
      result          : ExtractionResult
      feature_names   : list of str
    """
    # Load model and metadata
    try:
        model, feature_names, prep_meta = load_model_and_metadata()
    except ModelLoadError as e:
        raise RuntimeError(f"Model loading failed: {e}")

    # Extract features
    result = extract_features(input_data)

    # Build ordered list for prediction
    ordered_list = feature_vector_to_ordered_list(result.feature_vector)

    # Predict
    predicted_class, prob_failure, prob_success = predict(model, ordered_list, feature_names)

    return {
        "predicted_class": predicted_class,
        "prob_success":    prob_success,
        "prob_failure":    prob_failure,
        "result":          result,
        "feature_names":   feature_names,
    }


def main():
    args  = _parse_args()
    input_data = _load_input(args)

    _print_header()
    print(f"\n  Input idea: '{input_data.idea_text[:100]}'")
    if input_data.country:
        print(f"  Input country: '{input_data.country}'")
    if input_data.categories:
        print(f"  Input categories: {input_data.categories}")

    # Load and run
    try:
        model, feature_names, prep_meta = load_model_and_metadata()
    except ModelLoadError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    result = extract_features(input_data)

    _print_extraction_summary(result)

    fv = result.feature_vector
    _print_validation_check(fv, feature_names)

    if args.verbose:
        _print_full_feature_vector(fv)
        print()
        print(_sep("-"))
        print("  EXTRACTION LOG")
        print(_sep("-"))
        for entry in result.extraction_log:
            print(f"  {entry}")

    ordered_list = feature_vector_to_ordered_list(fv)

    try:
        predicted_class, prob_failure, prob_success = predict(model, ordered_list, feature_names)
    except ValueError as e:
        print(f"\n  PREDICTION ERROR: {e}")
        sys.exit(1)

    _print_prediction(predicted_class, prob_failure, prob_success, result)


if __name__ == "__main__":
    main()
