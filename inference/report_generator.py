"""
inference/report_generator.py
==============================
Phase 5 — Startup Viability Validation Report Generator

PURPOSE
-------
Combines Phase 4A (ML inference) and Phase 4B (idea scoring) outputs
into one structured, human-readable validation report.

DESIGN RULES
------------
1. The ML model-estimated success score and the Idea Evaluation Score
   are kept SEPARATE in the report. They are never averaged, merged,
   or presented as a single combined number.
2. The ML score is never called a "probability of success" or a
   "guarantee." It is described as a model-estimated score based on
   historical structural patterns.
3. Model limitations (training data ending ~2016, US bias, missing
   categories, calibration gap) are always disclosed.
4. The report only reports what the data and pipeline actually support.
   It never fabricates assessments of market size, competition, revenue,
   founder quality, or technical feasibility.
5. Deterministic: same inputs always produce the same report.
6. No external APIs, no LLMs, no ML models beyond the trained RF.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from inference.idea_scorer import IdeaScorerResult, MAX_OVERALL_SCORE
from inference.feature_extractor import ExtractionResult


# ---------------------------------------------------------------------------
# REPORT DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class DimensionSummary:
    """One row in the idea-evaluation section of the report."""
    name: str
    score: int          # 0–10
    max_score: int      # always 10
    weight_pct: int     # e.g. 25 for 25%
    explanation: str
    matched_signals: List[str]


@dataclass
class ValidationReport:
    """
    Complete structured validation report for one startup idea submission.

    Fields are grouped into five logical sections:
      1. Submission   — what the user provided
      2. ML Result    — Component A output (historical pattern matching)
      3. Idea Eval    — Component B output (description quality)
      4. Narrative    — strengths, missing info, guidance
      5. Limitations  — honest disclosures about the entire pipeline

    IMPORTANT
    ---------
    ml_success_score  : float 0.0–1.0
        This is the Random Forest's output based on historical sector/country
        patterns. It is NOT a calibrated probability of real-world success.
        Label it as "model-estimated success score" in every UI or document.

    idea_evaluation_score : int 0–100
        This is a rule-based description quality score. It reflects how
        clearly the idea was described, not the quality of the business.
        Scores are based on designer-defined keyword heuristics.

    These two scores MUST NOT be combined into a single number without
    an explicit, documented design decision.
    """

    # --- Section 1: Submission ---
    idea_text: str
    country_input: Optional[str]
    categories_input: List[str]
    resolved_country: Optional[str]
    resolved_categories: List[str]
    is_complete_submission: bool
    word_count: int
    is_meaningful_submission: bool

    # --- Section 2: ML Result (Component A) ---
    ml_predicted_class: int           # 0 or 1
    ml_pattern_label: str             # "Success Pattern" or "Failure Pattern"
    ml_success_score: float           # 0.0–1.0
    ml_failure_score: float           # 0.0–1.0
    active_country_features: List[str]
    active_category_features: List[str]
    n_categories: int

    # --- Section 3: Idea Evaluation (Component B) ---
    idea_evaluation_score: int         # 0–100
    idea_evaluation_grade: str         # Strong / Moderate / Developing / Weak / Very Weak
    dimensions: List[DimensionSummary]

    # --- Section 4: Narrative ---
    strengths: List[str]
    missing_or_unclear: List[str]
    guidance: List[str]
    submission_warnings: List[str]

    # --- Section 5: Limitations ---
    limitations: List[str]


# ---------------------------------------------------------------------------
# GRADE HELPERS
# ---------------------------------------------------------------------------

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


def _ml_pattern_label(predicted_class: int) -> str:
    return "Success Pattern" if predicted_class == 1 else "Failure Pattern"


# ---------------------------------------------------------------------------
# NARRATIVE BUILDER
# ---------------------------------------------------------------------------

def _build_strengths(
    scorer_result: IdeaScorerResult,
    extraction_result: ExtractionResult,
) -> List[str]:
    """
    Identify concrete positives from the submission.
    Only reports on what is actually present.
    """
    strengths = []

    fv = extraction_result.feature_vector

    # --- From idea evaluation dimensions ---
    for d in scorer_result.dimensions:
        if d.score >= 8:
            if d.key == "problem_clarity":
                strengths.append(
                    f"Clear problem statement: the submission effectively describes "
                    f"the challenge being solved."
                )
            elif d.key == "solution_clarity":
                strengths.append(
                    f"Clear solution description: the type of product or service "
                    f"and what it does is well articulated."
                )
            elif d.key == "target_market":
                strengths.append(
                    f"Target market identified: a specific customer segment or "
                    f"market is named in the submission."
                )
            elif d.key == "specificity":
                strengths.append(
                    f"High specificity: the submission uses concrete domain "
                    f"language, technologies, or processes."
                )
            elif d.key == "completeness":
                strengths.append(
                    f"Detailed submission ({scorer_result.word_count} words): "
                    f"sufficient information is present for evaluation."
                )

    # --- From extraction results ---
    real_cats = [c for c in extraction_result.resolved_categories if c != "Unknown"]
    if len(real_cats) >= 2:
        # Fix 3: neutral factual statement — no editorial judgment
        strengths.append(
            f"Multi-sector coverage: the idea was mapped to {len(real_cats)} "
            f"historical training categories "
            f"({', '.join(real_cats[:3])}{'...' if len(real_cats) > 3 else ''})."
        )

    if extraction_result.resolved_country and extraction_result.resolved_country != "Unknown":
        strengths.append(
            f"Geographic context provided: headquarters country identified as "
            f"{extraction_result.resolved_country}."
        )

    return strengths if strengths else []


def _build_missing_or_unclear(
    scorer_result: IdeaScorerResult,
    extraction_result: ExtractionResult,
) -> List[str]:
    """
    Identify what information is absent or underspecified.
    Based solely on what the scoring engine and extractor found.
    """
    missing = []

    for d in scorer_result.dimensions:
        if d.score == 0:
            if d.key == "problem_clarity":
                missing.append(
                    "Problem statement: the submission does not describe what "
                    "challenge or pain point the startup addresses."
                )
            elif d.key == "solution_clarity":
                missing.append(
                    "Solution description: the type of product or service being "
                    "built is not mentioned."
                )
            elif d.key == "target_market":
                missing.append(
                    "Target market: no specific customer type or market segment "
                    "is identified."
                )
            elif d.key == "specificity":
                missing.append(
                    "Specific detail: the idea lacks concrete domain, technology, "
                    "or process terminology."
                )
        elif d.score <= 4:
            if d.key == "problem_clarity":
                missing.append(
                    "Problem clarity is low: a problem is hinted at but not "
                    "clearly articulated."
                )
            elif d.key == "target_market":
                missing.append(
                    "Target market is vague: consider naming a specific customer "
                    "type (e.g. SMEs, students, hospitals)."
                )

    if not extraction_result.resolved_country:
        missing.append(
            "Country not provided: geographic context helps the ML model match "
            "the idea to historical startup patterns."
        )
    elif extraction_result.feature_vector.get("country_Unknown", 0) == 1:
        missing.append(
            "Country was not recognised: provide a country name or ISO-3 code "
            "so the ML model can use geographic context."
        )

    real_cats = [c for c in extraction_result.resolved_categories if c != "Unknown"]
    if not real_cats:
        missing.append(
            "Industry category not identified: no recognised sector keywords "
            "were found. Consider selecting a category explicitly."
        )

    return missing


def _build_guidance(
    scorer_result: IdeaScorerResult,
    extraction_result: ExtractionResult,
    ml_predicted_class: int,
) -> List[str]:
    """
    Actionable suggestions based on what is missing or weak.
    Does not fabricate advice beyond what the data supports.
    """
    guidance = []

    # Suggestions driven by low-scoring dimensions
    dim_scores = {d.key: d.score for d in scorer_result.dimensions}

    if dim_scores["problem_clarity"] < 6:
        guidance.append(
            "Strengthen the problem statement: describe who faces the problem, "
            "how often it occurs, and why existing solutions are inadequate."
        )

    if dim_scores["solution_clarity"] < 6:
        guidance.append(
            "Clarify the solution: name the product type (app, platform, API, "
            "service), what it does, and how it addresses the problem."
        )

    if dim_scores["target_market"] < 6:
        guidance.append(
            "Define the target market: name a specific customer segment "
            "(e.g. 'independent restaurant owners', 'engineering students', "
            "'HR teams at mid-size companies')."
        )

    if dim_scores["specificity"] < 6:
        # Fix 2: context-aware — don't suggest adding something already present.
        # Check what the idea_text actually contains so we avoid echoing it back.
        idea_lower = (extraction_result.extraction_log[1]
                      .replace("Idea text   : ", "").strip().strip("'").lower()
                      if len(extraction_result.extraction_log) > 1 else "")

        # Build the suggestion only from genuinely missing detail types
        missing_detail = []
        if not any(kw in idea_lower for kw in
                   ("ai", "artificial intelligence", "machine learning",
                    "deep learning", "nlp", "computer vision")):
            missing_detail.append("technology used (e.g. AI, machine learning)")
        if not any(kw in idea_lower for kw in
                   ("saas", "platform", "api", "mobile app", "cloud")):
            missing_detail.append("product type (e.g. SaaS, API, mobile app)")
        if not any(kw in idea_lower for kw in
                   ("invoice", "supply chain", "diagnost", "inventory",
                    "booking", "payroll", "logistics", "ehr")):
            missing_detail.append("domain process addressed (e.g. invoicing, diagnostics)")
        if not any(kw in idea_lower for kw in
                   ("integrat", "real-time", "dashboard", "analytics")):
            missing_detail.append("integration or delivery approach")

        if missing_detail:
            guidance.append(
                "Add specific detail: the description could be strengthened by "
                "mentioning " + " and ".join(missing_detail[:2]) + "."
            )

    if scorer_result.word_count < 30:
        guidance.append(
            f"Expand the description: the current submission is {scorer_result.word_count} "
            f"words. A 40–80 word description typically provides enough information "
            f"for a more meaningful evaluation."
        )

    if not extraction_result.resolved_country:
        guidance.append(
            "Provide your target country: this allows the ML model to use "
            "geographic context from its historical training data."
        )

    return guidance


def _build_limitations() -> List[str]:
    """
    Standard set of limitations disclosed in every report.
    These are fixed, factual statements about the pipeline's constraints.
    """
    return [
        "ML model training data: the Random Forest was trained on Crunchbase "
        "startup data from approximately 1990–2016. Sector trends, market "
        "conditions, and technology categories may have changed significantly "
        "since then. The model-estimated success score reflects historical "
        "patterns, not current market dynamics.",

        "Geographic bias: 61.2% of labeled training data is from US-headquartered "
        "companies (6,533 of 10,667 training rows). Predictions for non-US startups, "
        "especially in emerging markets, are based on fewer historical examples "
        "and carry greater uncertainty.",

        "Probability calibration: the ML success score is the Random Forest's "
        "vote proportion across 100 trees. It has not been calibrated using "
        "Platt scaling or isotonic regression. It reflects relative ordering "
        "(higher score = stronger historical match) but should not be treated "
        "as an absolute probability.",

        "What the ML model cannot assess: idea quality, market size, "
        "competition, revenue potential, founder quality, and technical "
        "feasibility are absent from the training data and are not evaluated "
        "by this system.",

        "What the Idea Evaluation Score measures: description clarity only. "
        "A high Idea Evaluation Score means the idea is clearly described — "
        "not that the business will succeed. A poorly-described great idea "
        "will score lower than a clearly-described mediocre one.",

        "Vocabulary limitations: the scoring engine uses ~120 keyword patterns "
        "across 5 dimensions. Ideas described in unusual terminology, non-English "
        "phrases, or highly technical jargon may receive lower scores than their "
        "quality warrants.",
    ]


# ---------------------------------------------------------------------------
# MAIN REPORT BUILDER
# ---------------------------------------------------------------------------

def generate_report(
    full_eval_output: dict,
    country_input: Optional[str] = None,
    categories_input: Optional[List[str]] = None,
) -> ValidationReport:
    """
    Build a ValidationReport from the output of run_full_evaluation().

    Parameters
    ----------
    full_eval_output : dict
        The dict returned by 05_evaluate_idea.run_full_evaluation().
        Required keys: scorer_result, extraction_result, predicted_class,
                       prob_success, prob_failure, feature_names.

    country_input : str, optional
        The original country string the user typed (for display).

    categories_input : list of str, optional
        The original category strings the user provided (for display).

    Returns
    -------
    ValidationReport
    """
    sr: IdeaScorerResult = full_eval_output["scorer_result"]
    er: ExtractionResult = full_eval_output["extraction_result"]
    predicted_class: int = full_eval_output["predicted_class"]
    prob_success: float  = full_eval_output["prob_success"]
    prob_failure: float  = full_eval_output["prob_failure"]

    fv = er.feature_vector

    # Active feature flags
    active_country = [k for k in fv if k.startswith("country_") and fv[k] == 1]
    active_cats    = [k for k in fv if k.startswith("cat_") and fv[k] == 1
                      and k != "cat_Unknown"]

    # Dimension summaries
    dimensions = [
        DimensionSummary(
            name            = d.name,
            score           = d.score,
            max_score       = 10,
            weight_pct      = round(d.weight * 100),
            explanation     = d.explanation,
            matched_signals = d.matched_signals,
        )
        for d in sr.dimensions
    ]

    # Narrative sections
    strengths   = _build_strengths(sr, er)
    missing     = _build_missing_or_unclear(sr, er)
    guidance    = _build_guidance(sr, er, predicted_class)
    limitations = _build_limitations()

    return ValidationReport(
        # Section 1 — Submission
        idea_text               = er.extraction_log[1].replace("Idea text   : ", "").strip().strip("'"),
        country_input           = country_input,
        categories_input        = categories_input or [],
        resolved_country        = er.resolved_country,
        resolved_categories     = er.resolved_categories,
        is_complete_submission  = er.is_complete_submission,
        word_count              = sr.word_count,
        is_meaningful_submission = sr.is_meaningful_submission,

        # Section 2 — ML Result
        ml_predicted_class      = predicted_class,
        ml_pattern_label        = _ml_pattern_label(predicted_class),
        ml_success_score        = prob_success,
        ml_failure_score        = prob_failure,
        active_country_features = active_country,
        active_category_features = active_cats,
        n_categories            = fv["n_categories"],

        # Section 3 — Idea Evaluation
        idea_evaluation_score  = sr.idea_evaluation_score,
        idea_evaluation_grade  = _grade(sr.idea_evaluation_score),
        dimensions             = dimensions,

        # Section 4 — Narrative
        strengths              = strengths,
        missing_or_unclear     = missing,
        guidance               = guidance,
        submission_warnings    = er.warnings,

        # Section 5 — Limitations
        limitations            = limitations,
    )


# ---------------------------------------------------------------------------
# TEXT RENDERER
# ---------------------------------------------------------------------------

def render_report_text(report: ValidationReport) -> str:
    """
    Render a ValidationReport as a plain-text string.

    This is the authoritative human-readable representation.
    Other formats (JSON, HTML, Markdown) should derive from this structure,
    not reimplement the logic.
    """
    SEP  = "=" * 65
    DASH = "-" * 65

    lines = []
    w = lines.append  # shorthand writer

    # ── Header ──────────────────────────────────────────────────────
    w(SEP)
    w("  STARTUP VIABILITY VALIDATION REPORT")
    w("  AI-Based Startup Viability Prediction and Validation System")
    w(SEP)

    # ── Section 1: Submission ────────────────────────────────────────
    w("")
    w("  SECTION 1 — SUBMITTED IDEA")
    w(DASH)
    w(f"  Idea text  : {report.idea_text[:200]}")
    if report.country_input:
        w(f"  Country    : {report.country_input}"
          + (f" → {report.resolved_country}" if report.resolved_country else " (unresolved)"))
    else:
        w("  Country    : (not provided)")

    real_cats = [c for c in report.resolved_categories if c != "Unknown"]
    if real_cats:
        w(f"  Categories : {', '.join(real_cats)}")
    else:
        w("  Categories : (none identified — keyword matching found no match)")

    w(f"  Word count : {report.word_count}")
    w(f"  Complete submission: {'Yes' if report.is_complete_submission else 'No'}")

    if report.submission_warnings:
        w("")
        w("  Submission notices:")
        for warn in report.submission_warnings:
            _wrap_print(lines, warn, indent="    ! ", width=63)

    # ── Section 2: ML Result ─────────────────────────────────────────
    w("")
    w("  SECTION 2 — ML MODEL RESULT  (Component A)")
    w(DASH)
    w("  Source: Random Forest trained on Crunchbase data (~1990–2016)")
    w("  Features used: sector, country, geography (65 binary/integer features)")
    w("")
    w(f"  Predicted pattern         : {report.ml_pattern_label}")
    w(f"  Model-estimated success score : {report.ml_success_score:.4f}  "
      f"({report.ml_success_score * 100:.1f}%)")
    w(f"  Model-estimated failure score : {report.ml_failure_score:.4f}  "
      f"({report.ml_failure_score * 100:.1f}%)")
    w("")
    w("  Active country features : " + (", ".join(report.active_country_features)
                                         or "(none)"))
    w("  Active category features: " + (", ".join(report.active_category_features)
                                         or "(none — cat_Unknown active)"))
    w(f"  n_categories            : {report.n_categories}")
    w("")
    w("  IMPORTANT: This score reflects how structurally similar this")
    w("  submission is to companies in the historical training dataset")
    w("  that achieved an exit (acquisition or IPO) vs. those that closed.")
    w("  It is NOT a probability or a forecast of real-world success.")

    # ── Section 3: Idea Evaluation ───────────────────────────────────
    w("")
    w("  SECTION 3 — IDEA EVALUATION SCORE  (Component B)")
    w(DASH)
    w("  Source: Rule-based keyword/heuristic scorer (designer-defined weights)")
    w("  Measures: description clarity only, NOT business quality")
    w("")
    w(f"  Overall Idea Evaluation Score : {report.idea_evaluation_score} / 100"
      f"  [{report.idea_evaluation_grade}]")
    w("")
    w(f"  {'Dimension':<22}  {'Score':>6}  {'Weight':>7}  Explanation (brief)")
    w(f"  {'-'*22}  {'-'*6}  {'-'*7}  {'-'*28}")

    for d in report.dimensions:
        brief = d.explanation[:55] + ("…" if len(d.explanation) > 55 else "")
        w(f"  {d.name:<22}  {d.score:>3}/10  {d.weight_pct:>6}%  {brief}")

    w("")
    w("  Dimension detail:")
    for d in report.dimensions:
        w(f"  [{d.name}]  {d.score}/10  (weight {d.weight_pct}%)")
        _wrap_print(lines, d.explanation, indent="    ", width=63)
        if d.matched_signals:
            sigs = ", ".join(f'"{s}"' for s in d.matched_signals[:3])
            w(f"    Signals: {sigs}")

    # ── Section 4: Narrative ─────────────────────────────────────────
    w("")
    w("  SECTION 4 — NARRATIVE ASSESSMENT")
    w(DASH)

    w("")
    w("  Strengths identified in the submission:")
    if report.strengths:
        for s in report.strengths:
            _wrap_print(lines, s, indent="    + ", width=63)
    else:
        w("    (No notable strengths detected in the description)")

    w("")
    w("  Missing or unclear information:")
    if report.missing_or_unclear:
        for m in report.missing_or_unclear:
            _wrap_print(lines, m, indent="    - ", width=63)
    else:
        # Build a context-aware summary from actual dimension scores.
        # "No major clarity gaps" is only appropriate when every scored
        # dimension is 7 or above. If any dimension scored 1–6, name it.
        # Completeness is excluded — it's a meta-dimension that reflects
        # word count and coverage rather than a single missing element.
        partial_gaps = [
            f"{d.name} ({d.score}/10): {d.explanation}"
            for d in report.dimensions
            if 0 < d.score < 7 and d.name != "Completeness"
        ]
        if partial_gaps:
            w("    The rule-based evaluator identified these areas as "
              "partially addressed but improvable:")
            for gap in partial_gaps:
                _wrap_print(lines, gap, indent="    - ", width=63)
        else:
            w("    No major clarity gaps detected by the rule-based evaluator.")

    w("")
    w("  Suggestions for a stronger submission:")
    if report.guidance:
        for g in report.guidance:
            _wrap_print(lines, g, indent="    > ", width=63)
    else:
        w("    (No specific suggestions — the submission covers key areas)")

    # ── Section 5: Limitations ───────────────────────────────────────
    w("")
    w("  SECTION 5 — LIMITATIONS AND DISCLOSURES")
    w(DASH)
    for i, lim in enumerate(report.limitations, 1):
        _wrap_print(lines, lim, indent=f"  {i}. ", width=63, cont_indent="     ")

    # ── Footer ───────────────────────────────────────────────────────
    w("")
    w(SEP)
    w("  END OF REPORT")
    w("  This report is for academic and informational purposes only.")
    w("  It does not constitute investment, business, or legal advice.")
    w(SEP)

    return "\n".join(lines)


def _wrap_print(lines: list, text: str,
                indent: str = "  ", width: int = 63,
                cont_indent: str = None) -> None:
    """Word-wrap a string and append each line to `lines`."""
    if cont_indent is None:
        cont_indent = " " * len(indent)
    words = text.split()
    if not words:
        lines.append(indent.rstrip())
        return

    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = cont_indent + word + " "
        else:
            current += word + " "
    if current.strip():
        lines.append(current.rstrip())


# ---------------------------------------------------------------------------
# JSON SERIALISER (for future API use)
# ---------------------------------------------------------------------------

def report_to_dict(report: ValidationReport) -> dict:
    """
    Serialise a ValidationReport to a plain dict (JSON-compatible).
    Keeps ML score and Idea Evaluation Score as separate top-level keys.
    """
    return {
        "submission": {
            "idea_text":               report.idea_text,
            "country_input":           report.country_input,
            "categories_input":        report.categories_input,
            "resolved_country":        report.resolved_country,
            "resolved_categories":     report.resolved_categories,
            "is_complete_submission":  report.is_complete_submission,
            "word_count":              report.word_count,
            "is_meaningful_submission": report.is_meaningful_submission,
        },
        "ml_result": {
            "source":           "Random Forest trained on Crunchbase ~1990-2016",
            "predicted_class":  report.ml_predicted_class,
            "pattern_label":    report.ml_pattern_label,
            "ml_success_score": round(report.ml_success_score, 4),
            "ml_failure_score": round(report.ml_failure_score, 4),
            "note":             (
                "This is a model-estimated success score based on historical "
                "structural patterns. It is NOT a probability or guarantee of "
                "real-world startup success."
            ),
            "active_country_features":  report.active_country_features,
            "active_category_features": report.active_category_features,
            "n_categories":             report.n_categories,
        },
        "idea_evaluation": {
            "source":                 "Rule-based keyword/heuristic scorer",
            "idea_evaluation_score":  report.idea_evaluation_score,
            "max_score":              MAX_OVERALL_SCORE,
            "grade":                  report.idea_evaluation_grade,
            "note":                   (
                "This score measures description clarity only. "
                "It does not predict business success."
            ),
            "dimensions": [
                {
                    "name":            d.name,
                    "score":           d.score,
                    "max_score":       d.max_score,
                    "weight_pct":      d.weight_pct,
                    "explanation":     d.explanation,
                    "matched_signals": d.matched_signals,
                }
                for d in report.dimensions
            ],
        },
        "narrative": {
            "strengths":          report.strengths,
            "missing_or_unclear": report.missing_or_unclear,
            "guidance":           report.guidance,
            "submission_warnings": report.submission_warnings,
        },
        "limitations": report.limitations,
        "scores_are_separate": True,
        "combined_score":       None,
    }
