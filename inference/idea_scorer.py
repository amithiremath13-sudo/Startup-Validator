"""
inference/idea_scorer.py
=========================
Phase 4B — Component B: Startup Idea Evaluation / Scoring Engine

PURPOSE
-------
Evaluates the quality of a startup idea description using deterministic,
rule-based heuristics applied to the free-text submission. This is
entirely separate from the ML model (Component A).

DESIGN PRINCIPLES
-----------------
1. Evaluates ONLY what the user has actually written.
2. Never invents information not present in the submission.
3. Transparent: every score is accompanied by a plain-English reason.
4. Deterministic: same input → same output, always.
5. No external API, no LLM, no ML model.
6. Scores are designer-defined weights — not learned from data.
   This must be disclosed wherever scores are displayed.

WHAT IT EVALUATES (5 dimensions from ML_METHODOLOGY.md §11.7.2)
----------------------------------------------------------------
1. Problem Clarity  (weight 25%) — Does the idea describe a problem?
2. Solution Clarity (weight 25%) — Does the idea describe a solution?
3. Target Market    (weight 20%) — Is a customer/market mentioned?
4. Specificity      (weight 15%) — Is the idea specific, not generic?
5. Completeness     (weight 15%) — How much useful information is present?

WHAT IT DOES NOT EVALUATE
--------------------------
- Market size (no data source)
- Competition (no data source)
- Revenue potential (no data source)
- Founder quality (not in the submission)
- Technical feasibility (not assessable from text alone)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SCORING CONSTANTS
# ---------------------------------------------------------------------------

# Dimension weights — must sum to 1.0
# These are designer-defined decisions, not learned from data.
DIMENSION_WEIGHTS: Dict[str, float] = {
    "problem_clarity":  0.25,
    "solution_clarity": 0.25,
    "target_market":    0.20,
    "specificity":      0.15,
    "completeness":     0.15,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Dimension weights must sum to 1.0"

# Maximum score per dimension (raw)
MAX_DIMENSION_SCORE = 10

# Maximum overall score (after weighting)
MAX_OVERALL_SCORE = 100

# ---------------------------------------------------------------------------
# KEYWORD LISTS
# Each list drives one scoring dimension. Presence of keywords/phrases
# from these lists earns points. Lists are deliberately conservative —
# only clear, unambiguous signals are included.
# ---------------------------------------------------------------------------

# --- Problem Clarity ---
# Signals that the idea describes a problem being solved.
PROBLEM_KEYWORDS: List[str] = [
    # explicit problem words
    "problem", "challenge", "issue", "pain point", "pain",
    "struggle", "frustration", "difficulty", "barrier",
    "obstacle", "inefficiency", "inefficient", "waste",
    "costly", "expensive", "time-consuming", "slow",
    "lack of", "lack ", "missing", "gap", "underserved",
    "complicated", "complex", "broken", "outdated",
    "manual process", "manual ", "error-prone",
    # need/demand framing
    "need", "needs", "need to", "need a", "there is no",
    "no easy way", "no way to", "nobody", "hard to",
    "difficult to", "unable to", "can't easily", "cannot easily",
    # negative state framing
    "fail", "fails", "failing", "lost", "losing", "lose",
    "reduce", "reducing", "cut ", "cutting",
    "hours", "manually", "manual process", "manual ",
]

# --- Solution Clarity ---
# Signals that the idea describes a proposed solution or product.
SOLUTION_KEYWORDS: List[str] = [
    # product/service types
    "platform", "app", "application", "tool", "service",
    "software", "system", "solution", "product", "engine",
    "marketplace", "network", "portal", "dashboard", "api",
    # action verbs describing what the solution does
    "helps", "help", "enables", "enable", "allows", "allow",
    "automates", "automate", "connects", "connect",
    "provides", "provide", "simplifies", "simplify",
    "streamlines", "streamline", "optimizes", "optimize",
    "improves", "improve", "manages", "manage",
    "tracks", "track", "monitors", "monitor",
    "predicts", "predict", "analyzes", "analyze",
    "matches", "match", "recommends", "recommend",
    # we/our framing
    "we build", "we create", "we develop", "we offer",
    "we provide", "we help", "our platform", "our app",
    "our solution", "our product", "our service", "our tool",
    # using technology
    "using ai", "using machine learning", "using data",
    "powered by", "ai-powered", "data-driven",
]

# --- Target Market ---
# Signals that the idea names a specific customer or market.
TARGET_MARKET_KEYWORDS: List[str] = [
    # customer descriptors
    "customer", "customers", "user", "users", "client", "clients",
    "consumer", "consumers", "buyer", "buyers",
    # market segments
    "small business", "small businesses", "smb", "smes", "enterprise",
    "startup", "startups", "freelancer", "freelancers",
    "restaurant", "restaurants", "hospital", "hospitals",
    "farmer", "farmers", "student", "students", "teacher", "teachers",
    "doctor", "doctors", "patient", "patients",
    "developer", "developers", "designer", "designers",
    # targeting language
    "for ", "targeting", "target ", "market ",
    "segment", "audience", "demographic",
    "b2b", "b2c", "d2c",
    # geography as market indicator (country names suggest target market)
    "india", "usa", "united states", "uk", "china", "global",
    "worldwide", "international",
]

# --- Specificity --- (high-value signals that indicate a concrete, focused idea)
# These are phrases suggesting concrete domain knowledge or specific detail.
# A generic idea ("I want to build an app") will score 0 here.
SPECIFICITY_KEYWORDS: List[str] = [
    # named technologies
    "machine learning", "artificial intelligence", "nlp",
    "computer vision", "blockchain", "iot", "internet of things",
    "api", "saas", "mobile app", "cloud",
    # named verticals or processes
    "food waste", "supply chain", "invoice", "invoicing", "payroll",
    "health record", "electronic health", "ehr", "telemedicine",
    "remote patient", "crop", "loan", "lending", "insurance",
    "delivery", "logistics", "inventory", "booking", "scheduling",
    "subscription", "e-commerce", "marketplace",
    "competitive exam", "online learning", "upskilling",
    "carbon", "renewable", "solar", "recycling",
    # quantitative detail
    "%", " million", " billion", " thousand",
    "per month", "per year", "per user",
    # integration / technical specificity
    "integrate", "integration", "api", "real-time", "real time",
    "dashboard", "analytics", "reporting",
]

# --- Completeness penalties (for very short or very sparse submissions) ---
# Completeness is computed from word count + dimension coverage, not
# from its own keyword list.


# ---------------------------------------------------------------------------
# THRESHOLDS FOR SCORING
# ---------------------------------------------------------------------------

# Word-count thresholds for completeness scoring
WORD_COUNT_THRESHOLDS = {
    "empty":    0,
    "minimal":  5,    # 1–5 words  → very thin
    "brief":    15,   # 6–15 words → some info
    "moderate": 30,   # 16–30 words → decent
    "good":     50,   # 31–50 words → good
    "detailed": 80,   # 51–80 words → detailed
    # 80+ words → very detailed (max completeness score)
}

# Keyword hit thresholds for each dimension (how many distinct keywords
# must match to reach a given score band).
SCORE_BANDS = {
    # (min_hits, score)
    "absent":    (0, 0),
    "weak":      (1, 4),
    "partial":   (2, 6),
    "clear":     (3, 8),
    "strong":    (4, 10),
}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    """Score and explanation for a single evaluation dimension."""
    name: str            # human-readable dimension name
    key: str             # internal key matching DIMENSION_WEIGHTS
    score: int           # 0–10
    weight: float        # fractional weight (sums to 1.0 across all dimensions)
    weighted_score: float  # score * weight * 10  (contribution to 0–100 total)
    explanation: str     # one-sentence plain-English reason
    matched_signals: List[str]  # which keywords/phrases triggered the score


@dataclass
class IdeaScorerResult:
    """
    Complete output of the idea scoring engine.

    idea_evaluation_score : int, 0–100
        The overall weighted score across all 5 dimensions.
        This is a designer-defined heuristic score, NOT an ML prediction.
        It measures how clearly and specifically the user described their idea.

    dimensions : list of DimensionResult
        One entry per dimension, with score, weight, and explanation.

    word_count : int
        Number of words in the idea text (after basic cleaning).

    is_meaningful_submission : bool
        True if the idea has enough content to produce a useful score.
        False for empty or near-empty submissions (< 5 words).

    caveats : list of str
        Any notes about the scoring that the user should see.
    """
    idea_evaluation_score: int
    dimensions: List[DimensionResult]
    word_count: int
    is_meaningful_submission: bool
    caveats: List[str]


# ---------------------------------------------------------------------------
# HELPER: KEYWORD COUNTING
# ---------------------------------------------------------------------------

def _count_keyword_hits(text_lower: str, keyword_list: List[str]) -> Tuple[int, List[str]]:
    """
    Count how many distinct keywords from keyword_list appear in text_lower.
    Returns (count, list_of_matched_keywords).

    Longer keywords are checked first so more specific phrases take
    precedence over generic single words in the matched list.
    """
    matched = []
    sorted_kws = sorted(keyword_list, key=len, reverse=True)
    for kw in sorted_kws:
        if kw in text_lower:
            matched.append(kw)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return len(deduped), deduped


def _hits_to_score(n_hits: int) -> int:
    """Convert a keyword hit count to a 0–10 score using SCORE_BANDS."""
    if n_hits == 0:
        return 0
    elif n_hits == 1:
        return 4
    elif n_hits == 2:
        return 6
    elif n_hits == 3:
        return 8
    else:  # 4+
        return 10


# ---------------------------------------------------------------------------
# DIMENSION SCORERS
# ---------------------------------------------------------------------------

def _score_problem_clarity(text_lower: str, word_count: int) -> DimensionResult:
    """
    Score: does the idea describe a problem being solved?

    0  — No problem language at all
    4  — One weak problem signal (e.g. just "reduce")
    6  — Two distinct problem signals
    8  — Three problem signals, problem is clearly stated
    10 — Four+ problem signals, problem is specific and prominent
    """
    hits, matched = _count_keyword_hits(text_lower, PROBLEM_KEYWORDS)
    score = _hits_to_score(hits)

    if score == 0:
        expl = ("No problem statement detected. Consider describing what "
                "challenge or pain point your startup addresses.")
    elif score <= 4:
        expl = ("A problem is hinted at but not clearly articulated. "
                "Try stating the specific challenge your customers face.")
    elif score <= 6:
        expl = ("A problem is mentioned but could be more specific. "
                "Adding context about who faces this problem would strengthen clarity.")
    elif score <= 8:
        expl = ("The problem is reasonably clear. Good use of problem-oriented language.")
    else:
        expl = ("The problem is clearly and specifically described. Strong signal.")

    return DimensionResult(
        name            = "Problem Clarity",
        key             = "problem_clarity",
        score           = score,
        weight          = DIMENSION_WEIGHTS["problem_clarity"],
        weighted_score  = score * DIMENSION_WEIGHTS["problem_clarity"] * MAX_DIMENSION_SCORE,
        explanation     = expl,
        matched_signals = matched[:5],  # cap display at 5
    )


def _score_solution_clarity(text_lower: str, word_count: int) -> DimensionResult:
    """
    Score: does the idea describe a proposed solution or product?
    """
    hits, matched = _count_keyword_hits(text_lower, SOLUTION_KEYWORDS)
    score = _hits_to_score(hits)

    if score == 0:
        expl = ("No solution description detected. Describe what product, "
                "service, or platform you are building.")
    elif score <= 4:
        expl = ("A solution is implied but not clearly described. "
                "Name the type of product (app, platform, service) and what it does.")
    elif score <= 6:
        expl = ("The solution is partially described. Adding what the product "
                "specifically does for the customer would improve this score.")
    elif score <= 8:
        expl = ("The solution is reasonably clear with good action-oriented language.")
    else:
        expl = ("The solution is clearly described with strong, specific language.")

    return DimensionResult(
        name            = "Solution Clarity",
        key             = "solution_clarity",
        score           = score,
        weight          = DIMENSION_WEIGHTS["solution_clarity"],
        weighted_score  = score * DIMENSION_WEIGHTS["solution_clarity"] * MAX_DIMENSION_SCORE,
        explanation     = expl,
        matched_signals = matched[:5],
    )


def _score_target_market(text_lower: str, word_count: int) -> DimensionResult:
    """
    Score: does the idea identify a specific customer or market segment?
    """
    hits, matched = _count_keyword_hits(text_lower, TARGET_MARKET_KEYWORDS)
    score = _hits_to_score(hits)

    if score == 0:
        expl = ("No target customer or market mentioned. Who will use or buy this product?")
    elif score <= 4:
        expl = ("A market is vaguely implied. Name a specific customer type "
                "(e.g. small businesses, students, hospitals) to improve this score.")
    elif score <= 6:
        expl = ("A target market is mentioned but could be more specific. "
                "Consider naming a narrower customer segment.")
    elif score <= 8:
        expl = ("The target market is identifiable. Good customer-oriented language.")
    else:
        expl = ("The target market is clearly and specifically described.")

    return DimensionResult(
        name            = "Target Market",
        key             = "target_market",
        score           = score,
        weight          = DIMENSION_WEIGHTS["target_market"],
        weighted_score  = score * DIMENSION_WEIGHTS["target_market"] * MAX_DIMENSION_SCORE,
        explanation     = expl,
        matched_signals = matched[:5],
    )


def _score_specificity(text_lower: str, word_count: int) -> DimensionResult:
    """
    Score: how specific and concrete is the idea?

    Generic ideas ("I have an app idea") score 0.
    Ideas mentioning specific technologies, processes, or domains score higher.
    Also penalises extremely short text regardless of keywords.
    """
    hits, matched = _count_keyword_hits(text_lower, SPECIFICITY_KEYWORDS)
    score = _hits_to_score(hits)

    # Short text cap: if under 8 words, specificity can be at most 4
    # regardless of keyword hits (you can't be specific in 3 words)
    if word_count < 8:
        score = min(score, 4)

    if score == 0:
        expl = ("The idea is very generic. Include specific technologies, "
                "processes, or domain details to demonstrate depth.")
    elif score <= 4:
        expl = ("Some specificity present but the idea is still fairly broad. "
                "Naming concrete technologies or domain processes would help.")
    elif score <= 6:
        expl = ("Moderate specificity. The idea has some concrete detail but "
                "could benefit from more specific domain language.")
    elif score <= 8:
        expl = ("Good specificity. The idea demonstrates domain knowledge "
                "through specific terminology.")
    else:
        expl = ("High specificity. The idea is concrete and detailed with "
                "clear technical or domain language.")

    return DimensionResult(
        name            = "Specificity",
        key             = "specificity",
        score           = score,
        weight          = DIMENSION_WEIGHTS["specificity"],
        weighted_score  = score * DIMENSION_WEIGHTS["specificity"] * MAX_DIMENSION_SCORE,
        explanation     = expl,
        matched_signals = matched[:5],
    )


def _score_completeness(text_lower: str, word_count: int,
                         dim_scores: List[int]) -> DimensionResult:
    """
    Score: how much useful information is present overall?

    Combines word count with how many of the other 4 dimensions
    received a non-zero score. A complete submission addresses
    most dimensions with sufficient detail.

    Score breakdown:
      - Word count contributes up to 6 points
      - Dimension coverage (how many of the 4 other dims scored > 0)
        contributes up to 4 points
    """
    # Word count contribution (0–6)
    if word_count == 0:
        wc_score = 0
    elif word_count < 5:
        wc_score = 1
    elif word_count < 15:
        wc_score = 2
    elif word_count < 30:
        wc_score = 3
    elif word_count < 50:
        wc_score = 4
    elif word_count < 80:
        wc_score = 5
    else:
        wc_score = 6

    # Dimension coverage contribution (0–4)
    dims_covered = sum(1 for s in dim_scores if s > 0)
    coverage_score = dims_covered  # 0, 1, 2, 3, or 4

    score = min(wc_score + coverage_score, MAX_DIMENSION_SCORE)

    if score == 0:
        expl = "The submission is empty or near-empty. No meaningful content to evaluate."
    elif score <= 3:
        expl = (f"Very brief submission ({word_count} words). "
                "Provide more detail about the problem, solution, and target customer.")
    elif score <= 5:
        expl = (f"Brief submission ({word_count} words) covering "
                f"{dims_covered}/4 key dimensions. "
                "More detail across all dimensions would improve this score.")
    elif score <= 7:
        expl = (f"Moderate submission ({word_count} words) covering "
                f"{dims_covered}/4 key dimensions.")
    elif score <= 9:
        expl = (f"Good submission ({word_count} words) with broad coverage "
                f"across {dims_covered}/4 key dimensions.")
    else:
        expl = (f"Detailed submission ({word_count} words) with strong "
                f"coverage across all key dimensions.")

    return DimensionResult(
        name            = "Completeness",
        key             = "completeness",
        score           = score,
        weight          = DIMENSION_WEIGHTS["completeness"],
        weighted_score  = score * DIMENSION_WEIGHTS["completeness"] * MAX_DIMENSION_SCORE,
        explanation     = expl,
        matched_signals = [f"{word_count} words", f"{dims_covered}/4 dimensions covered"],
    )


# ---------------------------------------------------------------------------
# MAIN SCORING FUNCTION
# ---------------------------------------------------------------------------

def score_idea(idea_text: str) -> IdeaScorerResult:
    """
    Score a startup idea description across 5 dimensions and return
    a structured result with explanations.

    Parameters
    ----------
    idea_text : str
        The free-text description of the startup idea. This is the only
        input. Country and category are NOT used here — those go to
        the ML model (Component A).

    Returns
    -------
    IdeaScorerResult

    Notes
    -----
    - Output is deterministic: same text → same result.
    - Scores are heuristic estimates. They measure how clearly the
      idea is described, NOT the quality of the underlying business.
    - The idea_evaluation_score (0–100) is entirely separate from
      the ML model's model-estimated success score (Component A).
      Do NOT combine these two numbers without clearly labelling both.
    """
    caveats: List[str] = []

    # --- Normalise text ---
    text_clean = (idea_text or "").strip()
    # Collapse multiple spaces / newlines
    text_clean = re.sub(r"\s+", " ", text_clean)
    text_lower = text_clean.lower()

    # --- Word count ---
    word_count = len(text_clean.split()) if text_clean else 0

    # --- Empty / near-empty guard ---
    is_meaningful = word_count >= 5
    if not is_meaningful:
        caveats.append(
            "Submission is too brief to evaluate meaningfully "
            f"({word_count} word{'s' if word_count != 1 else ''}). "
            "All dimension scores are capped at low values."
        )

    # --- Score the first 4 dimensions (before completeness) ---
    d_problem  = _score_problem_clarity(text_lower, word_count)
    d_solution = _score_solution_clarity(text_lower, word_count)
    d_market   = _score_target_market(text_lower, word_count)
    d_specific = _score_specificity(text_lower, word_count)

    first_four_scores = [
        d_problem.score,
        d_solution.score,
        d_market.score,
        d_specific.score,
    ]

    # --- Score completeness (depends on the other 4) ---
    d_complete = _score_completeness(text_lower, word_count, first_four_scores)

    dimensions = [d_problem, d_solution, d_market, d_specific, d_complete]

    # --- Compute overall weighted score (0–100) ---
    raw_weighted = sum(d.weighted_score for d in dimensions)
    overall = round(raw_weighted)
    overall = max(0, min(MAX_OVERALL_SCORE, overall))

    # --- Add standard caveat ---
    caveats.append(
        "This Idea Evaluation Score (0–100) measures description clarity "
        "only. It is a rule-based heuristic score, not an ML prediction. "
        "Dimension weights are designer-defined (Problem 25%, Solution 25%, "
        "Market 20%, Specificity 15%, Completeness 15%) and do not reflect "
        "actual startup success rates."
    )
    caveats.append(
        "This score does not evaluate market size, competition, revenue "
        "potential, founder quality, or technical feasibility."
    )

    return IdeaScorerResult(
        idea_evaluation_score   = overall,
        dimensions              = dimensions,
        word_count              = word_count,
        is_meaningful_submission = is_meaningful,
        caveats                 = caveats,
    )
