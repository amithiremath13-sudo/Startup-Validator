"""
=============================================================================
06_generate_report.py
=============================================================================
Phase 5 — Startup Viability Validation Report (CLI)

Runs Phase 4A + Phase 4B and generates a structured validation report.

HOW TO RUN
----------
    python 06_generate_report.py --idea "Your startup description" --country "India"

    # With categories:
    python 06_generate_report.py \
        --idea "A SaaS invoicing tool for freelancers." \
        --country "USA" --categories "Software" "Finance"

    # Save report to file:
    python 06_generate_report.py --idea "..." --country "India" --output report.txt

    # Also output JSON:
    python 06_generate_report.py --idea "..." --country "India" --json-output report.json

    # From JSON input file:
    python 06_generate_report.py --json-input input.json
=============================================================================
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inference.feature_extractor import StartupIdeaInput
from inference.model_loader import ModelLoadError
from inference.report_generator import generate_report, render_report_text, report_to_dict

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evaluate_idea",
    os.path.join(SCRIPT_DIR, "05_evaluate_idea.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_full_evaluation = _mod.run_full_evaluation


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 5: Generate startup validation report")
    p.add_argument("--idea",        type=str, default=None)
    p.add_argument("--country",     type=str, default=None)
    p.add_argument("--categories",  type=str, nargs="*", default=None)
    p.add_argument("--location",    type=str, default=None)
    p.add_argument("--json-input",  type=str, default=None)
    p.add_argument("--output",      type=str, default=None,
                   help="Save text report to this file path")
    p.add_argument("--json-output", type=str, default=None,
                   help="Save JSON report to this file path")
    return p.parse_args()


def _load_input(args):
    if args.json_input:
        with open(args.json_input, encoding="utf-8") as f:
            data = json.load(f)
        return (
            StartupIdeaInput(
                idea_text       = data.get("idea_text", ""),
                country         = data.get("country"),
                categories      = data.get("categories", []),
                location_detail = data.get("location_detail"),
            ),
            data.get("country"),
            data.get("categories", []),
        )

    if args.idea:
        return (
            StartupIdeaInput(
                idea_text       = args.idea,
                country         = args.country,
                categories      = args.categories or [],
                location_detail = args.location,
            ),
            args.country,
            args.categories or [],
        )

    idea_text = input("  Describe your startup idea: ").strip()
    if not idea_text:
        print("Error: idea text cannot be empty.")
        sys.exit(1)
    country = input("  Country (optional): ").strip() or None
    return StartupIdeaInput(idea_text=idea_text, country=country, categories=[]), country, []


def main():
    args = _parse_args()
    input_data, country_raw, categories_raw = _load_input(args)

    try:
        full_eval = run_full_evaluation(input_data)
    except ModelLoadError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"VALIDATION ERROR: {e}")
        sys.exit(1)

    report      = generate_report(full_eval, country_raw, categories_raw)
    report_text = render_report_text(report)

    print(report_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n  [Saved text report to {args.output}]")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, indent=2, ensure_ascii=False)
        print(f"  [Saved JSON report to {args.json_output}]")


if __name__ == "__main__":
    main()
