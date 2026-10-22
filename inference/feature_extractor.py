"""
inference/feature_extractor.py
================================
Phase 4A — Deterministic Feature Extraction

Converts a structured startup idea submission into the 65-feature vector
that the trained Random Forest expects.

INPUTS  (via StartupIdeaInput dataclass)
  idea_text : str  — free-text description of the startup idea
  country   : str  — user-provided country name or ISO code (optional)
  categories: list[str] — user-provided category names (optional)
               If provided, these must come from the APPROVED_CATEGORIES list.
               The extractor also runs keyword matching on idea_text as a
               fallback when categories is empty.

OUTPUT  (ExtractionResult dataclass)
  feature_vector : dict[str, int | float]
                   Keys are exact feature names from feature_names.json.
                   Values are 0/1 for binary features, int for n_categories.
  resolved_country     : str | None
  resolved_categories  : list[str]   — approved categories matched
  unresolved_keywords  : list[str]   — phrases found but not mappable
  warnings             : list[str]   — non-fatal issues noted
  extraction_log       : list[str]   — step-by-step decisions

DESIGN RULES
------------
1. All feature names and values are sourced from the existing metadata.
   Nothing is invented.
2. country_Unknown = 1 ONLY when no country is provided and extraction fails.
3. cat_Unknown     = 1 ONLY when no category is provided and no keyword
   match is found.
4. Never map to a category not in APPROVED_CATEGORIES.
5. If confidence is low, leave category unresolved rather than guessing.
6. The extractor is deterministic: same input always produces same output.
7. No external APIs or LLMs are used in this module.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from inference.vocabulary import (
    APPROVED_CATEGORIES,
    APPROVED_CATEGORY_SET,
    APPROVED_COUNTRIES,
    APPROVED_COUNTRY_SET,
    CALIFORNIA_SIGNALS,
    COUNTRY_NAME_TO_CODE,
    KEYWORD_TO_CATEGORY,
    SF_BAY_AREA_SIGNALS,
)

# ---------------------------------------------------------------------------
# FEATURE ORDER — must match feature_names.json EXACTLY
# This tuple is the single source of truth for column ordering in this module.
# If preprocessing changes, regenerate this from feature_names.json.
# ---------------------------------------------------------------------------
FEATURE_ORDER = (
    "is_sf_bay_area",
    "is_california",
    "country_USA",
    "country_Unknown",
    "country_GBR",
    "country_CAN",
    "country_ISR",
    "country_DEU",
    "country_FRA",
    "country_RUS",
    "country_CHN",
    "country_IND",
    "country_other",
    "cat_Software",
    "cat_Biotechnology",
    "cat_Mobile",
    "cat_Unknown",
    "cat_Curated_Web",
    "cat_E-Commerce",
    "cat_Social_Media",
    "cat_Advertising",
    "cat_Enterprise_Software",
    "cat_Games",
    "cat_Health_Care",
    "cat_Services",
    "cat_Internet",
    "cat_Finance",
    "cat_Technology",
    "cat_Analytics",
    "cat_Hardware_plus_Software",
    "cat_Security",
    "cat_Clean_Technology",
    "cat_Apps",
    "cat_Semiconductors",
    "cat_SaaS",
    "cat_Health_and_Wellness",
    "cat_Web_Hosting",
    "cat_Networking",
    "cat_Video",
    "cat_Social_Network_Media",
    "cat_Search",
    "cat_Manufacturing",
    "cat_Music",
    "cat_Education",
    "cat_Messaging",
    "cat_Cloud_Computing",
    "cat_Medical",
    "cat_News",
    "cat_Travel",
    "cat_Information_Technology",
    "cat_Web_Development",
    "cat_Telecommunications",
    "cat_Business_Services",
    "cat_Media",
    "cat_Sales_and_Marketing",
    "cat_Entertainment",
    "cat_iPhone",
    "cat_Design",
    "cat_Fashion",
    "cat_Consulting",
    "cat_Marketplaces",
    "cat_Android",
    "cat_FinTech",
    "category_other",
    "n_categories",
)

EXPECTED_FEATURE_COUNT = 65
assert len(FEATURE_ORDER) == EXPECTED_FEATURE_COUNT, (
    f"FEATURE_ORDER has {len(FEATURE_ORDER)} entries; expected {EXPECTED_FEATURE_COUNT}"
)


def _category_to_col(category_name: str) -> str:
    """
    Convert an approved category name to its feature column name.

    This replicates exactly what 02_preprocess_data.py does:
      safe = cat.replace(' ', '_').replace('/', '_').replace('+', 'plus')
      col  = 'cat_' + safe

    Hyphens are NOT replaced (cat_E-Commerce preserves the hyphen).
    """
    safe = (
        category_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
    )
    return f"cat_{safe}"


# ---------------------------------------------------------------------------
# INPUT / OUTPUT DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class StartupIdeaInput:
    """
    Structured input for the feature extractor.

    Parameters
    ----------
    idea_text : str
        Free-text description of the startup idea. Used for keyword
        extraction when `categories` is empty or incomplete.

    country : str, optional
        Country name or ISO-3 code where the startup is based.
        Examples: "India", "IND", "United States", "USA"
        If None or empty string, country_Unknown will be set to 1.

    categories : list of str, optional
        Explicitly provided category names. Each must be a member of
        APPROVED_CATEGORIES, or it will be logged as unresolved.
        If empty, keyword matching on idea_text is used as fallback.

    location_detail : str, optional
        Additional location string used to detect SF Bay Area / California.
        E.g. "San Francisco, CA" or "Bangalore, India".
        If None, idea_text is also scanned for geography signals.
    """
    idea_text: str
    country: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    location_detail: Optional[str] = None


@dataclass
class ExtractionResult:
    """
    Full output of the feature extraction process.

    Parameters
    ----------
    feature_vector : dict
        Maps each feature name (str) to its value (0 or 1, or int for
        n_categories). Has exactly EXPECTED_FEATURE_COUNT entries.

    resolved_country : str or None
        ISO-3 code resolved from the user's country input,
        or None if unresolvable.

    resolved_categories : list of str
        Categories from APPROVED_CATEGORIES that were matched,
        either from explicit input or keyword matching.

    unresolved_text : list of str
        Phrases in the idea text that looked like categories/countries
        but could not be confidently mapped to the approved vocabulary.

    warnings : list of str
        Non-fatal notices about the extraction (e.g. ambiguous signals,
        fallback to cat_Unknown).

    extraction_log : list of str
        Step-by-step record of every extraction decision. Useful for
        debugging and for explaining the result to the user.

    is_complete_submission : bool
        True if both country and at least one real (non-Unknown) category
        were successfully resolved. A complete submission avoids the
        data-artifact features (country_Unknown, cat_Unknown).
    """
    feature_vector: Dict[str, int]
    resolved_country: Optional[str]
    resolved_categories: List[str]
    unresolved_text: List[str]
    warnings: List[str]
    extraction_log: List[str]
    is_complete_submission: bool


# ---------------------------------------------------------------------------
# COUNTRY RESOLVER
# ---------------------------------------------------------------------------

def _resolve_country(country_input: Optional[str], log: List[str]) -> Optional[str]:
    """
    Resolve a free-text country name or ISO code to an ISO-3 code.

    Returns the ISO-3 code string if matched, or None if unresolvable.
    The returned code may or may not be in APPROVED_COUNTRIES.
    If it is not in APPROVED_COUNTRIES, country_other will be set to 1.
    """
    if not country_input or not country_input.strip():
        log.append("Country: no input provided -> will set country_Unknown=1")
        return None

    raw = country_input.strip()

    # 1. Direct ISO-3 check (e.g. user typed "IND", "USA")
    upper = raw.upper()
    if upper in APPROVED_COUNTRY_SET:
        log.append(f"Country: '{raw}' matched directly as ISO-3 code '{upper}'")
        return upper

    # 2. Lookup by lowercase name/alias
    lower = raw.lower()
    if lower in COUNTRY_NAME_TO_CODE:
        code = COUNTRY_NAME_TO_CODE[lower]
        log.append(f"Country: '{raw}' resolved via alias to '{code}'")
        return code

    # 3. Partial match: check if any alias is a substring of the input
    for alias, code in COUNTRY_NAME_TO_CODE.items():
        if alias in lower:
            log.append(f"Country: '{raw}' matched via substring '{alias}' -> '{code}'")
            return code

    # 4. Unknown country — user provided something but we cannot map it
    log.append(
        f"Country: '{raw}' could not be resolved to a known code. "
        f"Will set country_other=1."
    )
    return f"__UNRESOLVED__{raw}"


# ---------------------------------------------------------------------------
# GEOGRAPHY FLAG DETECTOR
# ---------------------------------------------------------------------------

def _detect_geography_flags(
    country_code: Optional[str],
    location_detail: Optional[str],
    idea_text: str,
    log: List[str],
) -> tuple:
    """
    Determine is_sf_bay_area and is_california from available signals.

    Returns (is_sf_bay_area: int, is_california: int).

    Rules (matching the preprocessing logic):
    - is_sf_bay_area = 1 if SF Bay Area signals are found in location_detail
      or idea_text, AND country is USA (or unspecified).
    - is_california  = 1 if California signals found, AND country is USA.
    - If country is a non-US code, both flags are 0.
    """
    # Combine all text sources for signal scanning
    combined = " ".join(filter(None, [
        (location_detail or "").lower(),
        idea_text.lower(),
    ]))

    # Only meaningful for US companies
    is_usa = (country_code == "USA") or (country_code is None)
    if not is_usa:
        log.append(f"Geography flags: country={country_code} is non-US -> "
                   f"is_sf_bay_area=0, is_california=0")
        return 0, 0

    is_sf = int(any(sig in combined for sig in SF_BAY_AREA_SIGNALS))
    is_ca = int(any(sig in combined for sig in CALIFORNIA_SIGNALS))

    # Bay Area implies California
    if is_sf and not is_ca:
        is_ca = 1

    if is_sf:
        log.append(f"Geography flags: SF Bay Area signal detected -> "
                   f"is_sf_bay_area=1, is_california=1")
    elif is_ca:
        log.append(f"Geography flags: California signal detected -> "
                   f"is_sf_bay_area=0, is_california=1")
    else:
        log.append(f"Geography flags: no sub-national geography signals -> "
                   f"is_sf_bay_area=0, is_california=0")

    return is_sf, is_ca


# ---------------------------------------------------------------------------
# CATEGORY RESOLVER
# ---------------------------------------------------------------------------

def _resolve_categories(
    explicit_categories: List[str],
    idea_text: str,
    log: List[str],
    warnings: List[str],
    unresolved: List[str],
) -> List[str]:
    """
    Resolve startup categories to APPROVED_CATEGORIES members.

    Priority:
    1. Explicit categories provided by the user (validated against vocabulary).
    2. Keyword matching on idea_text (deterministic, rule-based).
    3. If nothing found: category will default to cat_Unknown.

    Returns a deduplicated list of approved category names.
    """
    resolved = []

    # --- Step 1: validate explicit categories ---
    if explicit_categories:
        log.append(f"Categories: user provided {len(explicit_categories)} explicit "
                   f"categories: {explicit_categories}")
        for cat in explicit_categories:
            cat_stripped = cat.strip()
            if cat_stripped in APPROVED_CATEGORY_SET:
                if cat_stripped not in resolved:
                    resolved.append(cat_stripped)
                    log.append(f"  Accepted explicit category: '{cat_stripped}'")
            else:
                warnings.append(
                    f"Explicit category '{cat_stripped}' is not in the approved "
                    f"vocabulary and was ignored. Use a category from APPROVED_CATEGORIES."
                )
                unresolved.append(cat_stripped)
                log.append(f"  Rejected explicit category (not in vocabulary): '{cat_stripped}'")

    # --- Step 2: keyword matching on idea_text ---
    text_lower = idea_text.lower()

    # Sort keywords by length descending so longer/more-specific phrases match first
    sorted_keywords = sorted(KEYWORD_TO_CATEGORY.keys(), key=len, reverse=True)

    keyword_matched = []
    for kw in sorted_keywords:
        if kw in text_lower:
            cat = KEYWORD_TO_CATEGORY[kw]
            if cat not in resolved and cat not in keyword_matched:
                keyword_matched.append(cat)
                log.append(f"  Keyword match: '{kw}' -> '{cat}'")

    if keyword_matched:
        log.append(f"Categories: keyword matching found {len(keyword_matched)} "
                   f"additional categories: {keyword_matched}")
        resolved.extend(keyword_matched)
    elif not resolved:
        log.append("Categories: no explicit categories and no keyword matches found.")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for cat in resolved:
        if cat not in seen:
            seen.add(cat)
            deduped.append(cat)

    return deduped


# ---------------------------------------------------------------------------
# MAIN FEATURE VECTOR BUILDER
# ---------------------------------------------------------------------------

def extract_features(input_data: StartupIdeaInput) -> ExtractionResult:
    """
    Convert a StartupIdeaInput into the exact 65-feature vector the
    trained Random Forest expects.

    This is the main entry point for the inference pipeline.

    Parameters
    ----------
    input_data : StartupIdeaInput

    Returns
    -------
    ExtractionResult containing the feature_vector dict and full audit log.
    """
    log       = []
    warnings  = []
    unresolved = []

    log.append(f"=== Feature Extraction Begin ===")
    log.append(f"Idea text   : '{input_data.idea_text[:120]}'")
    log.append(f"Country     : {input_data.country!r}")
    log.append(f"Categories  : {input_data.categories!r}")
    log.append(f"Location    : {input_data.location_detail!r}")

    # ------------------------------------------------------------------
    # STEP 1 — Resolve country
    # ------------------------------------------------------------------
    log.append("\n[Step 1] Resolve country")
    raw_code = _resolve_country(input_data.country, log)

    # Determine the specific country features
    is_unresolved_country = raw_code is not None and raw_code.startswith("__UNRESOLVED__")
    is_no_country         = raw_code is None

    if is_no_country:
        resolved_country = None
        country_unknown  = 1
        country_other    = 0
        country_flags    = {}   # all specific country flags = 0
    elif is_unresolved_country:
        resolved_country = None
        country_unknown  = 0
        country_other    = 1
        country_flags    = {}
    elif raw_code in APPROVED_COUNTRY_SET and raw_code != "Unknown":
        resolved_country = raw_code
        country_unknown  = 0
        country_other    = 0
        country_flags    = {raw_code: 1}
    else:
        # Country provided but not in our top-10 approved set
        resolved_country = raw_code
        country_unknown  = 0
        country_other    = 1
        country_flags    = {}

    log.append(f"  resolved_country={resolved_country}, "
               f"country_unknown={country_unknown}, country_other={country_other}")

    # ------------------------------------------------------------------
    # STEP 2 — Detect geography flags
    # ------------------------------------------------------------------
    log.append("\n[Step 2] Detect geography flags")
    is_sf_bay_area, is_california = _detect_geography_flags(
        resolved_country,
        input_data.location_detail,
        input_data.idea_text,
        log,
    )

    # ------------------------------------------------------------------
    # STEP 3 — Resolve categories
    # ------------------------------------------------------------------
    log.append("\n[Step 3] Resolve categories")
    resolved_cats = _resolve_categories(
        input_data.categories,
        input_data.idea_text,
        log,
        warnings,
        unresolved,
    )

    # Determine cat_Unknown flag
    # Set to 1 ONLY if no real category was found at all.
    # 'Unknown' itself is never set from user input — only as last resort.
    real_cats = [c for c in resolved_cats if c != "Unknown"]

    if not real_cats:
        # No category information at all — use the historical fallback
        cat_unknown = 1
        final_cats  = []   # no specific cat flags set
        warnings.append(
            "No category could be extracted from the idea text or explicit inputs. "
            "cat_Unknown=1 has been set. This matches how the model handled "
            "historical companies with no recorded category, but it is a weak "
            "signal. Consider providing an explicit category for better results."
        )
        log.append("  No categories resolved -> cat_Unknown=1 (fallback)")
    else:
        cat_unknown = 0
        final_cats  = real_cats
        log.append(f"  Final resolved categories: {final_cats}")

    # Determine n_categories:
    # In training: companies with cat_Unknown=1 had n_categories=1
    # (because 'Unknown' was a single tag after pipe-split).
    # For real submissions: n_categories = number of real categories matched.
    # Minimum is 1 (mirrors training distribution where minimum was 1).
    if cat_unknown == 1:
        n_categories = 1   # mirrors training: 'Unknown' counted as 1 category
    else:
        n_categories = max(1, len(final_cats))

    log.append(f"  n_categories={n_categories}, cat_unknown={cat_unknown}")

    # Determine category_other:
    # In training: 1 if the company had ANY category outside the top-50.
    # At inference: if we matched all categories within the approved vocabulary,
    # category_other = 0. If keyword matching hit a category but it mapped
    # outside the vocabulary (already filtered), or if user says "my startup
    # does something not in the list", we set it to 1.
    # Current logic: category_other = 0 for all resolved real categories,
    # since only APPROVED_CATEGORIES can be returned.
    # It is 0 by default; set to 1 only via explicit unresolved category input.
    category_other = 1 if (unresolved and real_cats) else 0
    log.append(f"  category_other={category_other} "
               f"(unresolved inputs: {unresolved if unresolved else 'none'})")

    # ------------------------------------------------------------------
    # STEP 4 — Build the full feature vector
    # ------------------------------------------------------------------
    log.append("\n[Step 4] Build feature vector")

    # Start with all features = 0
    fv: Dict[str, int] = {name: 0 for name in FEATURE_ORDER}

    # Geography flags
    fv["is_sf_bay_area"] = is_sf_bay_area
    fv["is_california"]  = is_california

    # Country flags
    # Specific known-country columns
    for ctry_code, value in country_flags.items():
        col = f"country_{ctry_code}"
        if col in fv:
            fv[col] = value
    fv["country_Unknown"] = country_unknown
    fv["country_other"]   = country_other

    # Category flags
    # FIX (D1 from PHASE_4A_AUDIT.md): raise ValueError instead of silent no-op.
    # If _category_to_col() produces a column name not in FEATURE_ORDER (which
    # would only happen if APPROVED_CATEGORIES and FEATURE_ORDER drift out of
    # sync, e.g. after a vocabulary extension that forgets to update FEATURE_ORDER),
    # we must fail loudly rather than silently leaving the feature at 0.
    for cat in final_cats:
        col = _category_to_col(cat)
        if col not in fv:
            raise ValueError(
                f"Internal error: category '{cat}' produced column name '{col}' "
                f"which is not present in FEATURE_ORDER. "
                f"APPROVED_CATEGORIES and FEATURE_ORDER have drifted out of sync. "
                f"Check that every entry in APPROVED_CATEGORIES maps to a column "
                f"that exists in FEATURE_ORDER, and that FEATURE_ORDER has exactly "
                f"{EXPECTED_FEATURE_COUNT} entries."
            )
        fv[col] = 1
    fv["cat_Unknown"]    = cat_unknown
    fv["category_other"] = category_other
    fv["n_categories"]   = n_categories

    # ------------------------------------------------------------------
    # STEP 5 — Validate the feature vector
    # ------------------------------------------------------------------
    log.append("\n[Step 5] Validate feature vector")

    # Verify count
    assert len(fv) == EXPECTED_FEATURE_COUNT, (
        f"Feature count {len(fv)} != {EXPECTED_FEATURE_COUNT}"
    )

    # Verify all values are non-negative integers (no NaN, no float)
    for name, val in fv.items():
        assert val is not None,           f"NaN in feature '{name}'"
        assert isinstance(val, int),      f"Non-int value in '{name}': {val!r}"
        assert val >= 0,                  f"Negative value in '{name}': {val}"

    # Verify no forbidden columns
    FORBIDDEN = {
        "status", "target", "funding_total_usd", "funding_rounds",
        "last_funding_at", "first_funding_at", "founding_year",
        "has_founding_date", "is_usa",
    }
    for forbidden_col in FORBIDDEN:
        assert forbidden_col not in fv, (
            f"Forbidden column '{forbidden_col}' found in feature vector"
        )

    log.append(f"  Validation passed: {EXPECTED_FEATURE_COUNT} features, "
               f"all int, no NaN, no forbidden columns.")

    # ------------------------------------------------------------------
    # STEP 6 — Determine completeness
    # ------------------------------------------------------------------
    has_real_country   = (
        resolved_country is not None
        and not resolved_country.startswith("__UNRESOLVED__")
        and country_unknown == 0
    )
    has_real_category  = cat_unknown == 0 and len(final_cats) > 0
    is_complete        = has_real_country and has_real_category

    if not is_complete:
        msg_parts = []
        if not has_real_country:
            msg_parts.append("country not resolved")
        if not has_real_category:
            msg_parts.append("no approved category found")
        warnings.append(
            f"Incomplete submission ({'; '.join(msg_parts)}). "
            f"The model will rely on missing-data artifact features "
            f"(country_Unknown and/or cat_Unknown). "
            f"Results may be less reliable than for a complete submission."
        )

    log.append(f"\n=== Extraction complete ===")
    log.append(f"  is_complete_submission = {is_complete}")
    log.append(f"  resolved_country       = {resolved_country}")
    log.append(f"  resolved_categories    = {resolved_cats}")
    log.append(f"  warnings               = {len(warnings)}")

    return ExtractionResult(
        feature_vector       = fv,
        resolved_country     = resolved_country,
        resolved_categories  = resolved_cats,
        unresolved_text      = unresolved,
        warnings             = warnings,
        extraction_log       = log,
        is_complete_submission = is_complete,
    )


def feature_vector_to_ordered_list(fv: Dict[str, int]) -> list:
    """
    Convert the feature dict to an ordered list matching FEATURE_ORDER.
    This is what gets passed to model.predict() via numpy/pandas.
    """
    return [fv[name] for name in FEATURE_ORDER]
