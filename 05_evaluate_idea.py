"""
=============================================================================
05_evaluate_idea.py
=============================================================================
Phase 4B — Full Startup Idea Evaluation (Component A + Component B)

Runs both:
  Component A — Historical ML model (sector + country pattern score)
  Component B — Idea Evaluation Scoring Engine (description quality score)

and prints a combined report. The two scores are kept SEPARATE and clearly
labelled throughout.

HOW TO RUN
----------
    python 05_evaluate_idea.py --idea "Your startup description" --country "India"

    # With explicit categories (passed to Component A only):
    python 05_evaluate_idea.py --idea "..." --country "USA" --categories "Software"

    # Idea text only (Component B runs regardless; Component A gets less signal):
    python 05_evaluate_idea.py --idea "Your startup description"

    # Verbose (shows full 65-feature vector):
    python 05_evaluate_idea.py --idea "..." --country "India" --verbose

    # From JSON file:
    python 05_evaluate_idea.py --json-input path/to/input.json

IMPORTANT
---------
- The ML success score (Component A) is based on historical sector/country
  patterns from 1990–2016 Crunchbase data. It is NOT a probability.
- The Idea Evaluation Score (Component B) measures how clearly the idea
  is described. It does NOT predict business success.
- Neither score evaluates market size, competition, revenue potential,
  founder quality, or technical feasibility.
=============================================================================
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inference.feature_extractor import StartupIdeaInput, extract_features, feature_vector_to_ordered_list
from inference.idea_scorer import score_idea, MAX_OVERALL_SCORE
from inference.model_loader import ModelLoadError, load_model_and_metadata, predict


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------

def _sep(char="=", width=65):
    return char * width


def _score_bar(score: int, max_score: int = 10, width: int = 20) -> str:
    """Visual bar: ████░░░░░░  7/10"""
    filled = round(score / max_score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"{bar}  {score}/{max_score}"


def _grade(score_100: int) -> str:
    if score_100 >= 80:
        return "Strong"
    elif score_100 >= 60:
        return "Moderate"
    elif score_100 >= 40:
        return "Developing"
    elif score_100 >= 20:
        return "Weak"
    else:
        return "Very Weak"


def _print_component_b(scorer_result):
    """Print the Component B idea evaluation section."""
    print()
    print(_sep("-"))
    print("  COMPONENT B — IDEA EVALUATION SCORE")
    print("  (Rule-based description quality — designer-defined weights)")
    print(_sep("-"))
    print(f"  Overall Idea Evaluation Score : "
          f"{scorer_result.idea_evaluation_score}/{MAX_OVERALL_SCORE}  "
          f"[{_grade(scorer_result.idea_evaluation_score)}]")
    print(f"  Word count                    : {scorer_result.word_count}")
    print()
    print(f"  {'Dimension':<24} {'Score':>6}  {'Weight':>7}  {'Contribution':>12}  Bar")
    print(f"  {'-'*24} {'-'*6}  {'-'*7}  {'-'*12}  {'-'*22}")

    for d in scorer_result.dimensions:
        bar = _score_bar(d.score)
        contrib = f"{d.weighted_score:.1f}/{d.weight * MAX_OVERALL_SCORE:.0f}"
        print(f"  {d.name:<24} {d.score:>4}/10  {d.weight:>6.0%}  {contrib:>12}  {bar}")

    print()
    print("  Dimension explanations:")
    for d in scorer_result.dimensions:
        print(f"  [{d.name}]")
        print(f"    {d.explanation}")
        if d.matched_signals:
            signals_str = ", ".join(f'"{s}"' for s in d.matched_signals[:3])
            print(f"    Signals found: {signals_str}")


def _print_component_a(predicted_class, prob_success, prob_failure,
                        extraction_result):
    """Print the Component A ML model section."""
    print()
    print(_sep("-"))
    print("  COMPONENT A — ML MODEL (Historical sector/country patterns)")
    print("  (Random Forest trained on 1990–2016 Crunchbase data)")
    print(_sep("-"))

    fv = extraction_result.feature_vector
    active_cats = [k for k in fv if k.startswith("cat_") and fv[k] == 1 and k != "cat_Unknown"]
    active_ctry = [k for k in fv if k.startswith("country_") and fv[k] == 1]

    print(f"  Resolved country flags   : {active_ctry}")
    print(f"  Resolved category flags  : {active_cats}")
    print(f"  n_categories             : {fv['n_categories']}")
    print(f"  Submission complete      : {extraction_result.is_complete_submission}")

    if not extraction_result.is_complete_submission:
        print()
        print("  NOTE: Submission incomplete — artifact flags active.")
        print("  Provide a country + descriptive text for a more reliable ML score.")

    print()
    label = "SUCCESS PATTERN (1)" if predicted_class == 1 else "FAILURE PATTERN (0)"
    print(f"  Predicted pattern        : {label}")
    print(f"  Model-estimated success score : {prob_success:.4f}  ({prob_success*100:.1f}%)")
    print(f"  Model-estimated failure score : {prob_failure:.4f}  ({prob_failure*100:.1f}%)")
    print()
    print("  CAVEATS:")
    print("  - Score is based on historical sector/country patterns, not idea quality.")
    print("  - Probability calibration has not been applied.")
    print("  - Training data ends ~2016; modern sectors may not be well-represented.")
    print("  - This is NOT a guaranteed prediction of business success.")


def _print_combined_summary(scorer_result, predicted_class, prob_success):
    """Print the combined summary section."""
    print()
    print(_sep())
    print("  COMBINED SUMMARY")
    print(_sep())
    print(f"  Idea Evaluation Score  (Component B): "
          f"{scorer_result.idea_evaluation_score}/100  "
          f"[{_grade(scorer_result.idea_evaluation_score)}]")
    print(f"  ML Success Score       (Component A): "
          f"{prob_success:.4f} ({prob_success*100:.1f}%)  "
          f"[{'Success' if predicted_class == 1 else 'Failure'} pattern]")
    print()
    print("  These two scores measure different things and must not be")
    print("  averaged or combined without explicit design approval.")
    print("  Component B = description quality  |  Component A = historical sector fit")
    print(_sep())


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 4B: Full Startup Idea Evaluation"
    )
    p.add_argument("--idea",       type=str, default=None)
    p.add_argument("--country",    type=str, default=None)
    p.add_argument("--categories", type=str, nargs="*", default=None)
    p.add_argument("--location",   type=str, default=None)
    p.add_argument("--json-input", type=str, default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def _load_input(args) -> StartupIdeaInput:
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

    # Interactive
    print()
    idea_text = input("  Describe your startup idea: ").strip()
    if not idea_text:
        print("  Error: idea text cannot be empty.")
        sys.exit(1)
    country  = input("  Country (optional): ").strip() or None
    return StartupIdeaInput(idea_text=idea_text, country=country, categories=[])


# ---------------------------------------------------------------------------
# PUBLIC FUNCTION (importable by tests and future backend)
# ---------------------------------------------------------------------------

def run_full_evaluation(input_data: StartupIdeaInput) -> dict:
    """
    Run both Component A (ML) and Component B (scoring) and return
    a structured result dict.

    Returns
    -------
    {
        "scorer_result"   : IdeaScorerResult,
        "extraction_result": ExtractionResult,
        "predicted_class" : int,
        "prob_success"    : float,
        "prob_failure"    : float,
        "feature_names"   : list[str],
    }
    """
    # --- Component A: ML inference ---
    model, feature_names, _ = load_model_and_metadata()
    extraction_result = extract_features(input_data)
    ordered_list      = feature_vector_to_ordered_list(extraction_result.feature_vector)
    predicted_class, prob_failure, prob_success = predict(model, ordered_list, feature_names)

    # --- Component B: Idea scoring ---
    scorer_result = score_idea(input_data.idea_text)

    return {
        "scorer_result":     scorer_result,
        "extraction_result": extraction_result,
        "predicted_class":   predicted_class,
        "prob_success":      prob_success,
        "prob_failure":      prob_failure,
        "feature_names":     feature_names,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args       = _parse_args()
    input_data = _load_input(args)

    print()
    print(_sep())
    print("  AI-Based Startup Viability — Phase 4B Full Evaluation")
    print(_sep())
    print(f"  Idea   : '{input_data.idea_text[:90]}'")
    if input_data.country:
        print(f"  Country: '{input_data.country}'")
    if input_data.categories:
        print(f"  Categories: {input_data.categories}")

    try:
        out = run_full_evaluation(input_data)
    except ModelLoadError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n  VALIDATION ERROR: {e}")
        sys.exit(1)

    _print_component_b(out["scorer_result"])
    _print_component_a(out["predicted_class"], out["prob_success"],
                       out["prob_failure"], out["extraction_result"])
    _print_combined_summary(out["scorer_result"], out["predicted_class"],
                             out["prob_success"])


if __name__ == "__main__":
    main()
