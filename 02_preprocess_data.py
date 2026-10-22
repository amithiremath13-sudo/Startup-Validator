"""
=============================================================================
02_preprocess_data.py
=============================================================================
Phase 2 – Data Preprocessing and Feature Engineering
Project: AI-Based Startup Viability Prediction and Validation System

AUTHORITATIVE SPECIFICATION
-----------------------------
This script follows the REVISED PREDICTION FRAMEWORK in ML_METHODOLOGY.md
Section 11. The old "first-funding" prediction point is NOT used here.

PREDICTION POINT: Day 0 — startup idea / founding moment.
No funding information is included in the ML feature matrix.

FORBIDDEN FEATURES (must never appear in output):
  - status              (it IS the target)
  - last_funding_at     (future event; critical leakage)
  - funding_total_usd   (lifetime total; post-prediction)
  - funding_rounds      (lifetime count; post-prediction)
  - first_funding_at    (future event from Day 0)
  - time_to_first_funding (cannot exist before first funding)
  - permalink, name, homepage_url (identifiers/metadata)
  - city                (cardinality too high; signal captured elsewhere)

PERMITTED FEATURES (all available at Day 0):
  - Sector / category (from category_list)
  - Country           (from country_code)
  - Geography flags   (is_sf_bay_area, is_california)
  - NOTE: founding_year EXCLUDED (temporal mismatch: training max 2016,
          inference in 2026+)
  - NOTE: has_founding_date EXCLUDED (always = 1 for any new idea)
  - NOTE: is_usa EXCLUDED (redundant with country_USA)

HOW TO RUN
----------
    python 02_preprocess_data.py

    Optionally:
    python 02_preprocess_data.py --csv "path/to/file.csv"
    python 02_preprocess_data.py --test-size 0.25 --seed 42

OUTPUT (all saved to /processed_data)
------
    train_features.csv          cleaned training feature matrix (X_train)
    train_labels.csv            training target column (y_train)
    test_features.csv           cleaned test feature matrix (X_test)
    test_labels.csv             test target column (y_test)
    feature_names.json          ordered list of final feature names
    preprocessing_metadata.json statistics computed from training set only
    preprocessing_log.txt       full human-readable run log

IMPORTANT
---------
  - Original CSV is NEVER modified.
  - All statistics (medians, category lists, country lists) are computed
    from the TRAINING set ONLY, then applied to both train and test.
  - The train/test split happens BEFORE any statistics are computed.
  - Random seed is configurable (default 42) for full reproducibility.
=============================================================================
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV   = os.path.join(SCRIPT_DIR, "big_startup_secsees_dataset.csv")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "processed_data")
DEFAULT_SEED  = 42
DEFAULT_TEST  = 0.20   # 20 % held out as test set

# Date sanity bounds (from Phase 1 EDA findings)
MIN_VALID_YEAR = 1800
MAX_VALID_YEAR = 2023

# How many top categories to multi-hot encode
TOP_N_CATEGORIES = 50

# How many top countries to one-hot encode
TOP_N_COUNTRIES  = 10

# Columns that must NEVER appear in the final feature matrix
# (checked explicitly at the end of the script)
FORBIDDEN_COLUMNS = [
    "status",
    "last_funding_at",
    "funding_total_usd",
    "funding_rounds",
    "first_funding_at",
    "permalink",
    "name",
    "homepage_url",
    "city",
    "target",            # target lives in labels file, not features file
    # Phase 2 review removals (see PREPROCESSING_REPORT.md §"Why founding_year was removed")
    "founding_year",     # temporal distribution mismatch: training max 2016, inference 2026+
    "has_founding_date", # always = 1 for any new idea submission; zero discriminatory power
    "is_usa",            # redundant: country_USA encodes the same information
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING HELPER
# ---------------------------------------------------------------------------

log_lines = []

def log(msg=""):
    """Print to console and collect for log file."""
    print(msg)
    log_lines.append(str(msg))

def section(title):
    bar = "=" * 70
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

def subsection(title):
    log(f"\n--- {title} ---")

# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2 preprocessing script – Startup Viability Project"
    )
    p.add_argument("--csv",       default=DEFAULT_CSV,  help="Path to raw CSV")
    p.add_argument("--test-size", default=DEFAULT_TEST, type=float,
                   help=f"Fraction for test set (default {DEFAULT_TEST})")
    p.add_argument("--seed",      default=DEFAULT_SEED, type=int,
                   help=f"Random seed (default {DEFAULT_SEED})")
    p.add_argument("--top-cats",  default=TOP_N_CATEGORIES, type=int,
                   help=f"Top N categories to encode (default {TOP_N_CATEGORIES})")
    p.add_argument("--top-countries", default=TOP_N_COUNTRIES, type=int,
                   help=f"Top N countries to encode (default {TOP_N_COUNTRIES})")
    return p.parse_args()

# ===========================================================================
# STEP 1 — LOAD RAW DATA
# ===========================================================================

def step1_load(csv_path):
    section("STEP 1 — LOAD RAW DATA")
    if not os.path.exists(csv_path):
        log(f"  ERROR: CSV not found at '{csv_path}'")
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    log(f"  Loaded : {csv_path}")
    log(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    log(f"  Columns: {list(df.columns)}")
    return df

# ===========================================================================
# STEP 2 — CREATE BINARY TARGET AND SPLIT INTO LABELED / UNLABELED
# ===========================================================================

def step2_create_target(df):
    section("STEP 2 — CREATE BINARY TARGET")

    # Count each status value before splitting
    subsection("Raw status distribution")
    counts = df["status"].value_counts(dropna=False)
    for s, c in counts.items():
        log(f"  {str(s):<15} {c:>7,}  ({c/len(df)*100:.2f}%)")

    # -----------------------------------------------------------------------
    # TARGET RULE (from ML_METHODOLOGY.md §2.1):
    #   acquired + ipo  →  1 (SUCCESS)
    #   closed          →  0 (FAILURE)
    #   operating       →  EXCLUDED (outcome unknown)
    # -----------------------------------------------------------------------

    labeled_mask  = df["status"].isin(["acquired", "ipo", "closed"])
    df_labeled    = df[labeled_mask].copy()
    df_operating  = df[df["status"] == "operating"].copy()
    df_other      = df[~labeled_mask & (df["status"] != "operating")].copy()

    df_labeled["target"] = df_labeled["status"].apply(
        lambda s: 1 if s in ["acquired", "ipo"] else 0
    )

    subsection("Binary target distribution (labeled rows only)")
    tc = df_labeled["target"].value_counts()
    log(f"  Label 1 (SUCCESS – acquired or ipo) : {tc.get(1, 0):>7,}")
    log(f"  Label 0 (FAILURE – closed)          : {tc.get(0, 0):>7,}")
    log(f"  Total labeled rows                  : {len(df_labeled):>7,}")
    log(f"  Operating (excluded)                : {len(df_operating):>7,}")
    log(f"  Other / NaN status (excluded)       : {len(df_other):>7,}")

    return df_labeled, df_operating

# ===========================================================================
# STEP 3 — TRAIN / TEST SPLIT
# ===========================================================================

def step3_split(df_labeled, test_size, seed):
    section("STEP 3 — TRAIN / TEST SPLIT")

    # CRITICAL: Split happens HERE, before any statistics are computed.
    # This prevents test-set distribution information from influencing
    # preprocessing decisions (preprocessing leakage).
    #
    # We use stratified split to preserve the class ratio in both halves.

    from sklearn.model_selection import train_test_split  # local import

    X = df_labeled.drop(columns=["target"])
    y = df_labeled["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y          # preserve class ratio
    )

    log(f"  Random seed  : {seed}")
    log(f"  Test fraction: {test_size:.0%}")
    log(f"  Train rows   : {len(X_train):,}  (target=1: {y_train.sum():,}  target=0: {(y_train==0).sum():,})")
    log(f"  Test rows    : {len(X_test):,}  (target=1: {y_test.sum():,}  target=0: {(y_test==0).sum():,})")

    # Reset indices so operations downstream are index-safe
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# ===========================================================================
# STEP 4 — CLEAN DATE COLUMNS
# ===========================================================================

def step4_clean_dates(X_train, X_test):
    section("STEP 4 — CLEAN DATE COLUMNS")
    log("  Parsing founded_at. Invalid years (>2023 or <1800) → NaT.")
    log("  These were identified as typos in Phase 1 EDA.")
    log("  Note: first_funding_at is parsed but will NOT become a feature.")
    log("  It is only needed to confirm data quality; it is dropped later.")

    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]

    removal_report = {}

    for split_name, df in [("train", X_train), ("test", X_test)]:
        for col in date_cols:
            if col not in df.columns:
                continue

            before_nulls = df[col].isna().sum()
            parsed = pd.to_datetime(df[col], errors="coerce")

            # Null out rows where year is outside plausible range
            invalid_future = parsed[parsed.dt.year > MAX_VALID_YEAR]
            invalid_past   = parsed[parsed.dt.year < MIN_VALID_YEAR]

            parsed.loc[parsed.dt.year > MAX_VALID_YEAR] = pd.NaT
            parsed.loc[parsed.dt.year < MIN_VALID_YEAR] = pd.NaT

            after_nulls  = parsed.isna().sum()
            newly_nulled = after_nulls - before_nulls

            df[col] = parsed  # store back as datetime

            key = f"{split_name}_{col}"
            removal_report[key] = {
                "before_nulls": int(before_nulls),
                "invalid_future_rows": int(len(invalid_future)),
                "invalid_past_rows": int(len(invalid_past)),
                "newly_nulled": int(newly_nulled),
                "after_nulls": int(after_nulls),
            }

            if newly_nulled > 0:
                log(f"  [{split_name}] {col}: {newly_nulled} rows nulled "
                    f"({len(invalid_future)} future, {len(invalid_past)} past)")

    log("  Date cleaning complete.")
    return X_train, X_test, removal_report

# ===========================================================================
# STEP 5 — ENGINEER FOUNDING YEAR FEATURE
# ===========================================================================

def step5_founding_year(X_train, X_test, seed):
    section("STEP 5 — ENGINEER founding_year AND has_founding_date")

    # Extract year from the cleaned datetime column
    for df in [X_train, X_test]:
        df["founding_year"]     = df["founded_at"].dt.year.astype("Int64")  # nullable int
        df["has_founding_date"] = df["founded_at"].notna().astype(int)

    # Impute missing founding_year with TRAINING SET MEDIAN only
    # (computed before looking at test set)
    train_year_median = int(X_train["founding_year"].median())
    log(f"  Training set founding_year median (used for imputation): {train_year_median}")

    missing_train = X_train["founding_year"].isna().sum()
    missing_test  = X_test["founding_year"].isna().sum()
    log(f"  Missing founding_year — train: {missing_train:,}  test: {missing_test:,}")

    X_train["founding_year"] = X_train["founding_year"].fillna(train_year_median).astype(int)
    X_test["founding_year"]  = X_test["founding_year"].fillna(train_year_median).astype(int)

    log(f"  founding_year range (train): "
        f"{X_train['founding_year'].min()} – {X_train['founding_year'].max()}")

    return X_train, X_test, {"founding_year_median": train_year_median,
                              "missing_train": int(missing_train),
                              "missing_test":  int(missing_test)}

# ===========================================================================
# STEP 6 — ENGINEER GEOGRAPHY FLAGS
# ===========================================================================

def step6_geography(X_train, X_test):
    section("STEP 6 — ENGINEER GEOGRAPHY FLAGS")
    log("  Creating: is_usa, is_sf_bay_area, is_california")
    log("  These replace the raw state_code and region columns.")
    log("  No statistics needed — pure rule-based derivation.")

    for df in [X_train, X_test]:
        df["is_usa"]         = (df["country_code"] == "USA").astype(int)
        df["is_sf_bay_area"] = (df["region"] == "SF Bay Area").astype(int)
        df["is_california"]  = (df["state_code"] == "CA").astype(int)

    for name, df in [("train", X_train), ("test", X_test)]:
        log(f"  [{name}] is_usa={df['is_usa'].sum():,}  "
            f"is_sf_bay_area={df['is_sf_bay_area'].sum():,}  "
            f"is_california={df['is_california'].sum():,}")

    return X_train, X_test

# ===========================================================================
# STEP 7 — ENCODE country_code
# ===========================================================================

def step7_country_encoding(X_train, X_test, top_n):
    section(f"STEP 7 — ENCODE country_code (top {top_n} countries)")

    # Fill missing with "Unknown" BEFORE computing frequencies.
    # This ensures "Unknown" appears in the frequency count if common.
    for df in [X_train, X_test]:
        df["country_code"] = df["country_code"].fillna("Unknown")

    # Compute top-N country list from TRAINING SET ONLY
    train_country_counts = X_train["country_code"].value_counts()
    top_countries = list(train_country_counts.head(top_n).index)

    log(f"  Top {top_n} countries (by training-set frequency):")
    for i, c in enumerate(top_countries, 1):
        cnt = train_country_counts[c]
        log(f"    {i:>2}. {c:<10} {cnt:>6,} rows")

    # Create binary columns for each top country
    # Then a catch-all country_other column
    for df in [X_train, X_test]:
        for country in top_countries:
            col_name = f"country_{country}"
            df[col_name] = (df["country_code"] == country).astype(int)
        # country_other = 1 if not in top_n list
        df["country_other"] = (~df["country_code"].isin(top_countries)).astype(int)

    log(f"  Created {top_n} country binary columns + country_other")

    return X_train, X_test, top_countries

# ===========================================================================
# STEP 8 — ENCODE category_list
# ===========================================================================

def step8_category_encoding(X_train, X_test, top_n):
    section(f"STEP 8 — ENCODE category_list (top {top_n} categories)")

    # Fill missing category_list before any processing
    for df in [X_train, X_test]:
        df["category_list"] = df["category_list"].fillna("Unknown")

    # Parse pipe-separated categories into lists
    def parse_cats(val):
        """Split 'A|B|C' into ['A', 'B', 'C'], stripping whitespace."""
        return [c.strip() for c in str(val).split("|") if c.strip()]

    # Compute top-N category list from TRAINING SET ONLY
    # Flatten all individual categories across every training row
    all_train_cats = []
    for val in X_train["category_list"]:
        all_train_cats.extend(parse_cats(val))

    cat_counts = Counter(all_train_cats)
    top_categories = [cat for cat, _ in cat_counts.most_common(top_n)]

    log(f"  Unique categories in training set: {len(cat_counts):,}")
    log(f"  Top {top_n} categories (training frequency):")
    for i, cat in enumerate(top_categories[:20], 1):   # print first 20 for brevity
        log(f"    {i:>2}. {cat:<35} {cat_counts[cat]:>6,}")
    if top_n > 20:
        log(f"    ... ({top_n - 20} more categories encoded, not shown)")

    # n_categories feature: count of pipe-separated items per row
    # Create this BEFORE multi-hot so it reflects the raw category count
    for df in [X_train, X_test]:
        df["n_categories"] = df["category_list"].apply(
            lambda v: len(parse_cats(v))
        )

    log(f"  n_categories — train mean: {X_train['n_categories'].mean():.2f}  "
        f"max: {X_train['n_categories'].max()}")

    # Multi-hot encode: one binary column per top category
    # category_other: 1 if company has at least one category outside top_n
    for df in [X_train, X_test]:
        row_cats = df["category_list"].apply(parse_cats)
        for cat in top_categories:
            safe_name = cat.replace(" ", "_").replace("/", "_").replace("+", "plus")
            col_name  = f"cat_{safe_name}"
            df[col_name] = row_cats.apply(lambda lst: int(cat in lst))
        # category_other flag
        df["category_other"] = row_cats.apply(
            lambda lst: int(any(c not in top_categories for c in lst))
        )

    log(f"  Created {top_n} category binary columns + category_other + n_categories")

    return X_train, X_test, top_categories, cat_counts

# ===========================================================================
# STEP 9 — ASSEMBLE FINAL FEATURE MATRIX
# ===========================================================================

def step9_assemble(X_train, X_test, top_countries, top_categories):
    section("STEP 9 — ASSEMBLE FINAL FEATURE MATRIX")

    # -----------------------------------------------------------------------
    # Define exactly which columns belong in the final feature set.
    # Every column listed here is validated against the revised methodology.
    # -----------------------------------------------------------------------

    # Geography flags
    # NOTE: is_usa EXCLUDED — redundant with country_USA (Phase 2 review decision)
    geo_flags = ["is_sf_bay_area", "is_california"]

    # Country columns (binary per top country + other)
    country_cols = [f"country_{c}" for c in top_countries] + ["country_other"]

    # Founding year features
    # NOTE: founding_year EXCLUDED — temporal distribution mismatch.
    #   Training data max year = 2016. New ideas submitted in 2026 would
    #   require 10-year extrapolation beyond the observed training range.
    #   This is not a valid use of this feature at inference time.
    # NOTE: has_founding_date EXCLUDED — at inference time (Day 0), a user
    #   submitting a new startup idea is founding it right now, so this flag
    #   would always be 1. A constant feature has zero discriminatory power.
    # Both columns are still computed internally (for potential future use
    # or analysis) but are not included in the final feature matrix.
    year_cols = []   # empty — both year features excluded

    # Category columns (binary per top category + other + count)
    safe = lambda cat: cat.replace(" ", "_").replace("/", "_").replace("+", "plus")
    category_cols = [f"cat_{safe(c)}" for c in top_categories] + \
                    ["category_other", "n_categories"]

    # Full ordered feature list
    feature_cols = geo_flags + country_cols + year_cols + category_cols

    log(f"  Total features assembled: {len(feature_cols)}")
    log(f"    Geography flags  : {len(geo_flags)}  (is_sf_bay_area, is_california)")
    log(f"    Country columns  : {len(country_cols)}")
    log(f"    Year columns     : {len(year_cols)}  (founding_year and has_founding_date excluded — see comments)")
    log(f"    Category columns : {len(category_cols)}")

    # Verify every expected column actually exists in the dataframes
    missing_in_train = [c for c in feature_cols if c not in X_train.columns]
    missing_in_test  = [c for c in feature_cols if c not in X_test.columns]
    if missing_in_train:
        log(f"  ERROR: Missing in X_train: {missing_in_train}")
        sys.exit(1)
    if missing_in_test:
        log(f"  ERROR: Missing in X_test: {missing_in_test}")
        sys.exit(1)

    # Slice to only the final feature columns
    X_train_final = X_train[feature_cols].copy()
    X_test_final  = X_test[feature_cols].copy()

    # Confirm all values are numeric (no text columns should remain)
    non_numeric_train = [c for c in feature_cols
                         if not pd.api.types.is_numeric_dtype(X_train_final[c])]
    if non_numeric_train:
        log(f"  WARNING: Non-numeric columns in train: {non_numeric_train}")

    log(f"\n  Final feature matrix shapes:")
    log(f"    X_train : {X_train_final.shape[0]:,} rows × {X_train_final.shape[1]} features")
    log(f"    X_test  : {X_test_final.shape[0]:,} rows × {X_test_final.shape[1]} features")

    return X_train_final, X_test_final, feature_cols

# ===========================================================================
# STEP 10 — LEAKAGE AUDIT
# ===========================================================================

def step10_leakage_audit(X_train_final, X_test_final, feature_cols):
    section("STEP 10 — LEAKAGE AUDIT")
    log("  Checking that no forbidden columns appear in the final feature matrix.")
    log(f"  Forbidden columns list: {FORBIDDEN_COLUMNS}")

    found_violations = []
    for col in FORBIDDEN_COLUMNS:
        # Check exact name or as a substring/prefix in feature names
        exact_match   = col in feature_cols
        partial_match = any(col in f for f in feature_cols)
        if exact_match or partial_match:
            found_violations.append(col)

    if found_violations:
        log(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log(f"  LEAKAGE VIOLATION DETECTED: {found_violations}")
        log(f"  SCRIPT WILL EXIT. Fix the issue before proceeding.")
        log(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)
    else:
        log(f"  OK — No forbidden columns found in feature matrix.")

    subsection("All final feature names")
    for i, col in enumerate(feature_cols, 1):
        log(f"  {i:>3}. {col}")

    log(f"\n  Total feature count: {len(feature_cols)}")

    # Also confirm no NaN values remain in the final matrices
    nan_train = X_train_final.isna().sum().sum()
    nan_test  = X_test_final.isna().sum().sum()
    log(f"\n  Remaining NaN values — train: {nan_train}  test: {nan_test}")
    if nan_train > 0 or nan_test > 0:
        log("  WARNING: NaN values remain. Investigate before model training.")
    else:
        log("  OK — Feature matrices are fully numeric with no NaN values.")

    return found_violations

# ===========================================================================
# STEP 11 — SAVE ALL OUTPUTS
# ===========================================================================

def step11_save(X_train_final, X_test_final, y_train, y_test,
                feature_cols, top_countries, top_categories,
                cat_counts, year_meta, removal_report, args):
    section("STEP 11 — SAVE OUTPUTS")

    # -- Training features
    train_feat_path = os.path.join(OUTPUT_DIR, "train_features.csv")
    X_train_final.to_csv(train_feat_path, index=False)
    log(f"  Saved: {train_feat_path}")

    # -- Training labels
    train_lab_path = os.path.join(OUTPUT_DIR, "train_labels.csv")
    y_train.to_frame(name="target").to_csv(train_lab_path, index=False)
    log(f"  Saved: {train_lab_path}")

    # -- Test features
    test_feat_path = os.path.join(OUTPUT_DIR, "test_features.csv")
    X_test_final.to_csv(test_feat_path, index=False)
    log(f"  Saved: {test_feat_path}")

    # -- Test labels
    test_lab_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
    y_test.to_frame(name="target").to_csv(test_lab_path, index=False)
    log(f"  Saved: {test_lab_path}")

    # -- Feature names list (JSON)
    feat_names_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    with open(feat_names_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log(f"  Saved: {feat_names_path}")

    # -- Preprocessing metadata (all statistics computed from training set)
    # This JSON must be saved alongside the model so that at inference time
    # the SAME transformations can be applied to a new startup idea.
    metadata = {
        "prediction_point": "Day 0 — startup founding / idea stage (pre-funding)",
        "authoritative_spec": "ML_METHODOLOGY.md Section 11",
        "random_seed": args.seed,
        "test_fraction": args.test_size,
        "top_n_categories": args.top_cats,
        "top_n_countries": args.top_countries,
        "min_valid_year": MIN_VALID_YEAR,
        "max_valid_year": MAX_VALID_YEAR,
        "founding_year_median_train": year_meta["founding_year_median"],
        "top_countries_list": top_countries,
        "top_categories_list": top_categories,
        "feature_count": len(feature_cols),
        "train_rows": len(X_train_final),
        "test_rows": len(X_test_final),
        "train_target_1": int(y_train.sum()),
        "train_target_0": int((y_train == 0).sum()),
        "test_target_1": int(y_test.sum()),
        "test_target_0": int((y_test == 0).sum()),
        "date_cleaning_report": removal_report,
        "forbidden_columns_verified_absent": FORBIDDEN_COLUMNS,
        "features_excluded_and_why": {
            "status": "IS the target variable",
            "last_funding_at": "Critical temporal leakage — future event",
            "funding_total_usd": "Lifetime total — post-prediction information",
            "funding_rounds": "Lifetime count — post-prediction information",
            "first_funding_at": "Future event from Day 0 prediction point",
            "time_to_first_funding": "Cannot exist before first funding occurs",
            "permalink": "Identifier — no predictive signal",
            "name": "Identifier — no predictive signal",
            "homepage_url": "Metadata — no predictive signal",
            "city": "5111 unique values — too high cardinality",
            "state_code_raw": "Replaced by is_california binary flag",
            "region_raw": "Replaced by is_sf_bay_area binary flag",
            "founded_at_raw": "Replaced by founding_year integer",
            "founding_year": "Phase 2 review removal: temporal distribution mismatch. Training max year = 2016; inference at 2026+ requires 10+ year extrapolation beyond training range.",
            "has_founding_date": "Phase 2 review removal: always = 1 for any new startup idea submission (user is founding now). Zero discriminatory power at inference time.",
            "is_usa": "Phase 2 review removal: redundant with country_USA, which encodes the same information."
        }
    }

    meta_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  Saved: {meta_path}")

    return metadata

# ===========================================================================
# STEP 12 — FINAL SUMMARY REPORT
# ===========================================================================

def step12_summary(X_train_final, X_test_final, y_train, y_test,
                   feature_cols, metadata):
    section("STEP 12 — FINAL SUMMARY")

    log(f"\n  PREDICTION POINT : Day 0 (idea / founding stage, pre-funding)")
    log(f"  SPEC REFERENCE   : ML_METHODOLOGY.md Section 11")

    subsection("Dataset sizes")
    log(f"  Training rows    : {len(X_train_final):,}")
    log(f"  Test rows        : {len(X_test_final):,}")
    log(f"  Total features   : {len(feature_cols)}")

    subsection("Target distribution")
    log(f"  Train — class 1 (success): {y_train.sum():>6,}  "
        f"({y_train.mean()*100:.1f}%)")
    log(f"  Train — class 0 (failure): {(y_train==0).sum():>6,}  "
        f"({(1-y_train.mean())*100:.1f}%)")
    log(f"  Test  — class 1 (success): {y_test.sum():>6,}  "
        f"({y_test.mean()*100:.1f}%)")
    log(f"  Test  — class 0 (failure): {(y_test==0).sum():>6,}  "
        f"({(1-y_test.mean())*100:.1f}%)")

    subsection("Feature group breakdown")
    geo   = [c for c in feature_cols if c in ("is_sf_bay_area", "is_california")]
    ctry  = [c for c in feature_cols if c.startswith("country_")]
    year  = [c for c in feature_cols if "founding" in c or "has_founding" in c]
    categ = [c for c in feature_cols if c.startswith("cat_") or
             c in ("category_other", "n_categories")]
    log(f"  Geography flags  : {len(geo)}  (is_sf_bay_area, is_california)")
    log(f"  Country flags    : {len(ctry)}")
    log(f"  Year features    : {len(year)}  (founding_year + has_founding_date excluded)")
    log(f"  Category features: {len(categ)}")

    subsection("Leakage check — forbidden columns absent")
    for col in FORBIDDEN_COLUMNS:
        present = col in feature_cols
        status  = "VIOLATION" if present else "OK"
        log(f"  {status:>9}  {col}")

    subsection("NaN check")
    nan_train = X_train_final.isna().sum().sum()
    nan_test  = X_test_final.isna().sum().sum()
    log(f"  NaN in train features: {nan_train}")
    log(f"  NaN in test features : {nan_test}")

    log("\n  Original CSV was NOT modified.")
    log("  All outputs saved to /processed_data.")
    log("\n  Phase 2 complete. DO NOT start Phase 3 / model training")
    log("  until this output has been reviewed and approved.")

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    log("=" * 70)
    log("  AI-Based Startup Viability Prediction — Phase 2 Preprocessing")
    log("=" * 70)
    log(f"  Prediction point : Day 0 (pre-funding idea stage)")
    log(f"  Spec reference   : ML_METHODOLOGY.md Section 11")
    log(f"  Random seed      : {args.seed}")
    log(f"  Test fraction    : {args.test_size:.0%}")
    log(f"  Top categories   : {args.top_cats}")
    log(f"  Top countries    : {args.top_countries}")

    # Check sklearn is available (needed for train_test_split)
    try:
        import sklearn
        log(f"  scikit-learn     : {sklearn.__version__}")
    except ImportError:
        log("  scikit-learn not found. Installing...")
        os.system("pip install scikit-learn")
        import sklearn

    # Run all steps
    df                              = step1_load(args.csv)
    df_labeled, df_operating        = step2_create_target(df)
    X_train, X_test, y_train, y_test = step3_split(df_labeled, args.test_size, args.seed)
    X_train, X_test, removal_report  = step4_clean_dates(X_train, X_test)
    X_train, X_test, year_meta       = step5_founding_year(X_train, X_test, args.seed)
    X_train, X_test                  = step6_geography(X_train, X_test)
    X_train, X_test, top_countries   = step7_country_encoding(X_train, X_test, args.top_countries)
    X_train, X_test, top_cats, cat_counts = step8_category_encoding(X_train, X_test, args.top_cats)
    X_train_f, X_test_f, feature_cols = step9_assemble(X_train, X_test, top_countries, top_cats)
    step10_leakage_audit(X_train_f, X_test_f, feature_cols)
    metadata = step11_save(X_train_f, X_test_f, y_train, y_test,
                           feature_cols, top_countries, top_cats,
                           cat_counts, year_meta, removal_report, args)
    step12_summary(X_train_f, X_test_f, y_train, y_test, feature_cols, metadata)

    # Write log file
    log_path = os.path.join(OUTPUT_DIR, "preprocessing_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n  [LOG SAVED] {log_path}")

if __name__ == "__main__":
    main()
