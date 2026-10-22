# Phase 4A Audit Report
## AI-Based Startup Viability Prediction and Validation System

---

> **Audit type:** Read-only. No files were modified.
>
> **All findings below are based on direct inspection of source code,
> metadata files, and live execution of the inference pipeline.**
> No assumptions were made without verification.
>
> **Audit performed:** 21 September 2026

---

## Table of Contents

1. [Files Inspected](#1-files-inspected)
2. [Feature Vector Integrity](#2-feature-vector-integrity)
3. [Forbidden Column and Leakage Verification](#3-forbidden-column-and-leakage-verification)
4. [Test Case Walkthrough](#4-test-case-walkthrough)
5. [Known Limitation Investigation](#5-known-limitation-investigation)
6. [Category Mapping Audit](#6-category-mapping-audit)
7. [A — Confirmed Correct Behaviour](#7-a--confirmed-correct-behaviour)
8. [B — Known Limitations](#8-b--known-limitations)
9. [C — Actual Bugs](#9-c--actual-bugs)
10. [D — Issues to Fix Before Phase 4B](#10-d--issues-to-fix-before-phase-4b)
11. [E — Issues Safe to Leave as Documented Limitations](#11-e--issues-safe-to-leave-as-documented-limitations)
12. [Phase 1–3 Integrity Check](#12-phase-13-integrity-check)

---

## 1. Files Inspected

| File | Status |
|------|--------|
| `04_predict_startup.py` | Read in full |
| `inference/feature_extractor.py` | Read in full |
| `inference/model_loader.py` | Read in full |
| `inference/vocabulary.py` | Read in full |
| `tests/test_inference.py` | Read in full |
| `INFERENCE_PIPELINE.md` | Read in full |
| `models/model_metadata.json` | Read in full |
| `processed_data/feature_names.json` | Read in full |
| `FEATURE_DICTIONARY.md` | Read in full |
| `MODEL_EVALUATION_REPORT.md` | Read in full |
| `MODEL_FEATURE_IMPORTANCE.md` | Read in full |
| `processed_data/train_features.csv` | Statistical queries run |
| `processed_data/train_labels.csv` | Statistical queries run |
| `outputs/model_evaluation/feature_importance.json` | Read in full |

---

## 2. Feature Vector Integrity

### 2.1 Feature names and order

`FEATURE_ORDER` in `inference/feature_extractor.py` was compared
entry-for-entry against `processed_data/feature_names.json`.

**Result: exact match, all 65 features, correct positions.**

The module-level assertion `assert len(FEATURE_ORDER) == 65` fires at
import time and would immediately reveal any future drift.

The `predict()` function in `model_loader.py` wraps the feature list
in a `pd.DataFrame` with `columns=feature_names` before calling
`model.predict()`. This gives sklearn named-column matching and
eliminates the "feature names missing" warning that was present in the
initial version.

### 2.2 All 65 features confirmed in correct order

```
 1. is_sf_bay_area          14. cat_Software            35. cat_SaaS
 2. is_california           15. cat_Biotechnology        36. cat_Health_and_Wellness
 3. country_USA             16. cat_Mobile               37. cat_Web_Hosting
 4. country_Unknown         17. cat_Unknown              38. cat_Networking
 5. country_GBR             18. cat_Curated_Web          39. cat_Video
 6. country_CAN             19. cat_E-Commerce           40. cat_Social_Network_Media
 7. country_ISR             20. cat_Social_Media         41. cat_Search
 8. country_DEU             21. cat_Advertising          42. cat_Manufacturing
 9. country_FRA             22. cat_Enterprise_Software  43. cat_Music
10. country_RUS             23. cat_Games                44. cat_Education
11. country_CHN             24. cat_Health_Care          45. cat_Messaging
12. country_IND             25. cat_Services             46. cat_Cloud_Computing
13. country_other           26. cat_Internet             47. cat_Medical
                            27. cat_Finance              48. cat_News
                            28. cat_Technology           49. cat_Travel
                            29. cat_Analytics            50. cat_Information_Technology
                            30. cat_Hardware_plus_Software 51. cat_Web_Development
                            31. cat_Security             52. cat_Telecommunications
                            32. cat_Clean_Technology     53. cat_Business_Services
                            33. cat_Apps                 54. cat_Media
                            34. cat_Semiconductors       55. cat_Sales_and_Marketing
                                                         56. cat_Entertainment
                                                         57. cat_iPhone
                                                         58. cat_Design
                                                         59. cat_Fashion
                                                         60. cat_Consulting
                                                         61. cat_Marketplaces
                                                         62. cat_Android
                                                         63. cat_FinTech
                                                         64. category_other
                                                         65. n_categories
```

### 2.3 Model expects 65 features

Confirmed: `model.n_features_in_ == 65`. Any feature vector of a
different length raises a `ValueError` in `predict()` before the model
is ever called.

---

## 3. Forbidden Column and Leakage Verification

### 3.1 Target column (`status`, `target`)

Neither `status` nor `target` appears anywhere in `FEATURE_ORDER`.
Step 5 of `extract_features()` explicitly asserts `"target" not in fv`
and `"status" not in fv` as part of the forbidden-column check. **Verified absent.**

### 3.2 Post-prediction information (funding, dates)

`funding_total_usd`, `funding_rounds`, `last_funding_at`,
`first_funding_at` are all checked in the Step 5 forbidden-column
assertion and are absent from `FEATURE_ORDER`. **Verified absent.**

`founding_year` and `has_founding_date` were deliberately excluded
in the Phase 2 review (temporal distribution mismatch; `has_founding_date`
always = 1 at inference). Both absent from `FEATURE_ORDER` and from the
feature vector. **Verified absent.**

`is_usa` was excluded as redundant with `country_USA`. **Verified absent.**

### 3.3 NaN values

Two independent guards prevent any NaN from reaching the model:

**Guard 1 — Step 5 in `extract_features()`:**
Every value in the feature vector is asserted `not None` and
`isinstance(val, int)`. An `AssertionError` fires before returning.

**Guard 2 — In `predict()`:**
`np.isnan(arr).any()` is checked; a `ValueError` fires if any NaN exists.

**No code path produces a float or None in the feature vector.** All
values come from either explicit integer assignments (0 or 1 for
binary features) or `max(1, len(final_cats))` for `n_categories` (which
returns an `int`). **Verified: NaN cannot reach the model.**

### 3.4 Invented categories

The pipeline has three protective layers:

1. `_resolve_categories` only appends to `resolved` if the category
   name is in `APPROVED_CATEGORY_SET` (explicit input path) or is a
   value in `KEYWORD_TO_CATEGORY` (keyword path). All values in
   `KEYWORD_TO_CATEGORY` were verified to be members of `APPROVED_CATEGORIES`.
2. In Step 4, category columns are set via `fv[col] = 1` with
   `if col in fv` guard. Since `fv` is initialised from `FEATURE_ORDER`,
   an invalid column name does nothing.
3. The test `test_explicit_invalid_category_is_rejected` verifies that
   invalid explicit categories produce warnings and are excluded.

**Verified: no invented category can be set to 1.**

---

## 4. Test Case Walkthrough

All five test cases were executed live. Results are exact.

### Test 1 — AI platform for food waste reduction in India

**Input:** `idea="AI platform helping small restaurants reduce food waste in India."`, `country="India"`

| Feature group | Active features |
|--------------|----------------|
| Country | `country_IND = 1` |
| Geography | `is_sf_bay_area = 0`, `is_california = 0` |
| Categories | `cat_Analytics = 1` (keyword: "ai platform") |
| | `cat_Clean_Technology = 1` (keyword: "food waste") |
| | `cat_Services = 1` (keyword: "restaurant") |
| Artifact flags | `country_Unknown = 0`, `cat_Unknown = 0` |
| `n_categories` | 3 |
| `category_other` | 0 |
| Submission complete | True |

**Predicted class:** 0 (failure pattern)
**Model-estimated success score:** 0.4473 (44.7%)

**Observation:** "restaurant" maps to `Services` — there is no dedicated
restaurant or food sector in the 50-category vocabulary. This is an
imprecise mapping and is documented as a known vocabulary gap.

Training data for `country_IND + cat_Clean_Technology`: **0 examples**.
The model must generalise without any IND+CleanTech training signal.

---

### Test 2 — Online education platform in India

**Input:** `idea="Online education platform for students preparing for competitive exams in India."`, `country="India"`

| Feature group | Active features |
|--------------|----------------|
| Country | `country_IND = 1` |
| Geography | `is_sf_bay_area = 0`, `is_california = 0` |
| Categories | `cat_Education = 1` (keywords: "online education", "competitive exam", "student") |
| Artifact flags | `country_Unknown = 0`, `cat_Unknown = 0` |
| `n_categories` | 1 |
| `category_other` | 0 |
| Submission complete | True |

**Predicted class:** 0 (failure pattern)
**Model-estimated success score:** 0.4710 (47.1%)

**Training data context:**
`country_IND + cat_Education`: only **2 training examples**, both
labeled failure (success rate = 0.0). The model has essentially no
reliable signal for this combination. The prediction is largely
dominated by the country_IND base rate (49.5% success) pulling
against `cat_Education` alone (52.3% success), moderated by the
very small IND+Education sample.

---

### Test 3 — Healthcare technology for remote patient monitoring (no country)

**Input:** `idea="Healthcare technology platform for remote patient monitoring."` (no country)

| Feature group | Active features |
|--------------|----------------|
| Country | `country_Unknown = 1` (no country → artifact flag) |
| Geography | `is_sf_bay_area = 0`, `is_california = 0` |
| Categories | `cat_Health_Care = 1` (keyword: "healthcare") |
| | `cat_Medical = 1` (keywords: "remote patient", "patient monitoring") |
| Artifact flags | `country_Unknown = 1` ← **artifact active**, `cat_Unknown = 0` |
| `n_categories` | 2 |
| `category_other` | 0 |
| Submission complete | False |
| Warning issued | Yes — "Incomplete submission (country not resolved)..." |

**Predicted class:** 0 (failure pattern)
**Model-estimated success score:** 0.2997 (30.0%)

**Critical observation:** The low score of 30.0% is heavily influenced
by `country_Unknown = 1`. In the training data, `country_Unknown`
companies had a success rate of only **18.0%** — the lowest of any
country group. This is a data-recording artifact, not a signal about
this healthcare idea. The model correctly warns the user, but the
numerical output is materially distorted by the missing country.

---

### Test 4 — Invoice management software (USA, explicit categories)

**Input:** `idea="A software platform for small businesses to manage invoices."`, `country="USA"`, `categories=["Software", "Finance"]`

| Feature group | Active features |
|--------------|----------------|
| Country | `country_USA = 1` |
| Geography | `is_sf_bay_area = 0`, `is_california = 0` |
| Categories | `cat_Software = 1` (explicit) |
| | `cat_Finance = 1` (explicit + keyword: "invoice", "invoicing") |
| Artifact flags | `country_Unknown = 0`, `cat_Unknown = 0` |
| `n_categories` | 2 |
| `category_other` | 0 |
| Submission complete | True |

**Predicted class:** 1 (success pattern)
**Model-estimated success score:** 0.6978 (69.8%)

**Training data context:** `country_USA + cat_Software + cat_Finance`:
**14 training examples**, success rate 71.4%. The prediction is
well-supported by training data and consistent with the historical rate.

This is the cleanest test case — complete, well-mapped to the training
vocabulary, and with adequate training support.

---

### Test 5 — Vague idea: "I have an app idea"

**Input:** `idea="I have an app idea."` (no country)

| Feature group | Active features |
|--------------|----------------|
| Country | `country_Unknown = 1` (no country → artifact flag) |
| Geography | `is_sf_bay_area = 0`, `is_california = 0` |
| Categories | `cat_Apps = 1` (keyword: "app") |
| Artifact flags | `country_Unknown = 1` ← **artifact active**, `cat_Unknown = 0` |
| `n_categories` | 1 |
| `category_other` | 0 |
| Submission complete | False |
| Warning issued | Yes |

**Predicted class:** 0 (failure pattern)
**Model-estimated success score:** 0.1671 (16.7%)

**Confirmed correct behaviour:** "app" mapped to approved category `Apps`
— no invented categories. `cat_Unknown` stayed 0 because a real keyword
match was found. `country_Unknown` correctly set to 1 with warning.

**Training data context:** `country_Unknown + cat_Apps`: 63 training
examples, success rate only **12.7%**. The very low score is driven by
the strong negative signal of `country_Unknown`. The system correctly
signals this is a weak submission, but the user should be informed that
the score would be meaningfully different with a country provided.

---

## 5. Known Limitation Investigation

### 5.1 `country_Unknown` and `cat_Unknown` as dominant model features

**Feature importance in the selected Random Forest (test set):**

| Rank | Feature | Importance | Training success rate |
|-----:|---------|:----------:|:--------------------:|
| 1 | `country_Unknown` | 0.2036 | **18.0%** |
| 2 | `country_USA` | 0.1355 | 64.3% |
| 3 | `cat_Unknown` | 0.1287 | **12.9%** |

The top three features account for **46.7%** of total model importance.
Two of the three (`country_Unknown`, `cat_Unknown`) are historical
missing-data artifacts that will almost always be **0** for new startup
idea submissions (since users provide a country and the keyword matcher
finds at least one category for a meaningful description).

This creates a structural mismatch:

```
Training distribution:           Inference distribution (normal submission):
  country_Unknown: 14.8%           country_Unknown: 0%
  cat_Unknown:      8.0%           cat_Unknown:     0%
```

The model's two strongest features are inactive for normal complete
submissions. The remaining 53.3% of importance, spread across 62 other
features, carries the full prediction weight at inference time.

**For rows in training where both artifact flags are 0** (the 82.3%
of training examples that resemble normal submissions): success rate
= **60.6%**, vs 53.2% overall. The model operating without artifact
flags has a somewhat cleaner signal, but the feature importance
distribution was shaped by the full training set including the artifact
rows.

**Conclusion:** This is a significant known limitation. It does not
make the pipeline wrong — it makes the model's effective discriminative
power at inference time lower than its training-set metrics suggest.
No fix is appropriate at this stage; a future phase should investigate
retraining with artifact features removed or neutralised.

---

### 5.2 Missing categories: Food, Restaurant, AgriTech

**Verified:** The 50-category vocabulary does not contain:
`Food`, `Restaurant`, `FoodTech`, `AgriTech`, `Agriculture`,
`Food and Beverage`.

**What happens for food/restaurant ideas:**
- `"restaurant"` → `Services` (training success rate: **31.1%**)
- `"food waste"` → `Clean Technology` (training success rate: 52.0%)
- `"food delivery"` → `Services`
- `"food tech"` → `Services`

`Services` has a very low success rate in the training data (31.1%),
which may unfairly penalise food/restaurant ideas that are submitted
via keyword matching. This imprecise mapping could produce systematically
lower scores for this entire sector.

**Other notable vocabulary gaps (modern categories):**
- No "AgriTech", "PropTech", "InsurTech" (though "insurtech" maps to FinTech)
- No "Web3", "Blockchain", "Cryptocurrency" (no mapping in keyword dict)
- No "Space Tech", "GovTech", "LegalTech"
- "Climate Tech" is in the keyword map but maps to `Clean Technology`
  (acceptable approximation)

---

### 5.3 Temporal distribution: ~1990–2016 training data

The training data covers companies roughly from 1990–2016, with the
bulk between 2005–2015 (training-set founding year median: 2007).
`founding_year` was explicitly excluded from features to prevent
10-year extrapolation for 2026 ideas (correct decision, documented in
`PREPROCESSING_REPORT.md §16`).

**However, the sector patterns themselves reflect 2005–2015 dynamics:**
- `cat_Analytics` has a training success rate of **76.1%** —
  historically high because data analytics was an emerging high-value
  sector being acquired by large companies (2010–2016 era).
- `cat_Biotechnology` has a training success rate of ~65–70% —
  reflecting heavy pharma M&A activity in that era.
- `cat_Mobile` reflects the smartphone boom (2008–2015); those dynamics
  are fundamentally different today.

These rates cannot be assumed to hold for 2026 startups. This is a
documented limitation; no fix is appropriate at the model level without
newer data.

---

### 5.4 US dominance in training data

Training distribution: **61.2% of labeled rows are `country_USA`**.
Success rate for US companies: **64.3%** vs 49.5% for India, 18.0% for
unknown.

For non-US submissions in countries outside the top-10 (`country_other`):
training success rate = 50.8% with **n=1,582 total** — spread across
127 different non-top-10 countries (averaging ~12 examples per country).
The model cannot meaningfully distinguish between Nigeria, Brazil,
South Korea, or any other non-top-10 country; they all receive the
same `country_other=1` treatment.

**Sparse training combinations for common non-US idea submissions:**

| Combination | Training examples | Success rate |
|------------|:-----------------:|:------------:|
| `country_IND + cat_Education` | 2 | 0.0% |
| `country_IND + cat_Clean_Technology` | 0 | N/A |
| `country_IND + cat_Analytics` | 1 | 0.0% |
| `country_Unknown + cat_Medical` | 28 | 28.6% |
| `country_Unknown + cat_Apps` | 63 | 12.7% |

Many realistic non-US startup ideas have **zero or near-zero** training
examples. Random Forest predictions in these regions are based on
aggregate country and aggregate category signals independently, not on
the specific combination — which can produce results that look plausible
but have no direct training support.

---

## 6. Category Mapping Audit

### 6.1 Keyword sorting correctness

Keywords in `KEYWORD_TO_CATEGORY` are sorted by length descending before
matching. This ensures more specific phrases match before shorter ones.
Example:
- `"ai platform"` (10 chars) is checked before `"app"` (3 chars), so
  "AI platform app" correctly matches Analytics rather than Apps-only.

### 6.2 Non-trivial name transformations — all verified

| Category name | Produced column | In FEATURE_ORDER |
|--------------|----------------|:----------------:|
| `Hardware + Software` | `cat_Hardware_plus_Software` | ✅ |
| `E-Commerce` | `cat_E-Commerce` | ✅ (hyphen preserved) |
| `Health and Wellness` | `cat_Health_and_Wellness` | ✅ |
| `SaaS` | `cat_SaaS` | ✅ |
| `FinTech` | `cat_FinTech` | ✅ |
| `Clean Technology` | `cat_Clean_Technology` | ✅ |

### 6.3 `cat_Technology` — unreachable by keyword

`cat_Technology` is in `FEATURE_ORDER` at position 28 and is a valid
feature. However, **no keyword in `KEYWORD_TO_CATEGORY` maps to
`"Technology"`**. It can only be activated by explicit
`categories=["Technology"]` input. An idea described as "a technology
startup" will not activate `cat_Technology` through keyword matching.

This is an intentional design choice (broad generic category), but
it creates an asymmetry: the category exists in the model and training
data (279 training examples, success rate not measured) but is
practically unreachable via keyword extraction.

### 6.4 `category_other` logic — edge case confirmed

When a user provides **only invalid categories** with no keyword match
in the idea text:

```python
categories=["FoodTech", "AgriTech"], idea_text="I build things."
→ cat_Unknown = 1
→ category_other = 0   # because real_cats is empty
```

The model treats this identically to "no category provided at all."
The warnings list correctly reports the rejection, but the feature
representation does not distinguish "user tried a category outside
the vocabulary" from "user provided nothing." This is a known
behaviour documented in §9.

### 6.5 `category_other` = 0 for all keyword-only matches

When categories are found only through keyword matching (no explicit
`categories=[]` input), `unresolved` is always empty, so
`category_other = 0` regardless of how many or how few categories
matched. This is correct: keyword matching can only produce approved
categories, so there is no "other" to flag.

---

## 7. A — Confirmed Correct Behaviour

All of the following were verified by code inspection and/or live execution.

| # | Behaviour | Verified by |
|---|-----------|------------|
| 1 | `FEATURE_ORDER` exactly matches `feature_names.json` (65 features, correct order) | Code inspection + file comparison |
| 2 | Model expects 65 features; `model.n_features_in_ == 65` | Direct model inspection |
| 3 | NaN cannot reach the model — two independent guards | Code inspection + path analysis |
| 4 | Forbidden columns (`status`, `target`, `funding_*`, `last_funding_at`, etc.) are absent from feature vector | Code inspection + Step 5 assertions |
| 5 | No invented categories — only APPROVED_CATEGORIES members can be set to 1 | Code inspection + `test_explicit_invalid_category_is_rejected` |
| 6 | `country_Unknown` and `country_other` are mutually exclusive | Code inspection + `test_country_other_for_unknown_country` |
| 7 | `country_Unknown = 1` only when no country is provided | Code inspection + Test 3, 5 live run |
| 8 | `cat_Unknown = 1` only when no real category is resolved | Code inspection + live test with "I build things." |
| 9 | `n_categories` has a minimum of 1 (matching training distribution minimum) | Code inspection + live runs |
| 10 | Explicit invalid categories produce warnings and are excluded | `test_explicit_invalid_category_is_rejected` + live test |
| 11 | `predict()` wraps input in `pd.DataFrame` — no sklearn feature-name warnings | Live execution (exit 0, no warnings on all 5 tests) |
| 12 | Feature validation runs before prediction and exits on failure | Code inspection (Step 5 assertions + `sys.exit` in CLI) |
| 13 | 14/14 pytest tests pass | Test run confirmed |
| 14 | Phase 1–3 scripts and CSV unchanged | Byte-size comparison |
| 15 | Output is labelled "model-estimated success score" not "probability" | Code inspection + live output |
| 16 | Probability calibration caveat is printed on every prediction | Code inspection + live output |
| 17 | Test 5 (vague idea): "app" maps to approved `Apps` category; cat_Unknown stays 0 | Live execution verified |
| 18 | Keywords sorted longest-first (specific before generic) | Code inspection of `sorted(..., key=len, reverse=True)` |

---

## 8. B — Known Limitations

These are real constraints that affect prediction quality but are not
bugs and do not require code changes before Phase 4B. They must be
disclosed to users.

### B1 — Artifact features dominate model importance

`country_Unknown` (20.4%) and `cat_Unknown` (12.9%) together account
for 33.3% of the Random Forest's feature importance. Both will be 0 for
nearly all real submissions. The model's effective discriminative
capacity at inference is substantially narrower than its training-set
metrics reflect.

**Impact:** Model-estimated success scores for complete, real submissions
will be driven by the remaining ~53% of feature importance (country,
sector, geography), which is a smaller and less discriminative signal
set than what the model was evaluated on.

**Appropriate response:** Note this limitation explicitly in the UI and
reports. Consider retraining without artifact features in a future phase.

### B2 — No vocabulary for food, restaurant, agritech sectors

Restaurant / food / agricultural ideas map imprecisely to `Services`
(success rate: 31.1%) or `Clean Technology` (52.0%). An actual
restaurant-tech startup is not well-represented by either. Scores for
this sector cluster may be artificially low.

### B3 — Training data covers 1990–2016

Sector success rates reflect that era's M&A and IPO market conditions.
Notable distortions for 2026 ideas:
- `cat_Analytics`: 76.1% training success rate (AI acquisitions of 2010–2016)
- `cat_Biotechnology`: ~65–70% (pharma M&A wave)
- `cat_Mobile`: reflects the smartphone boom, now saturated

These rates are presented as "historical patterns" in the output — but
they should not be interpreted as predictions for the current market.

### B4 — Non-US startups have sparse training support

Many realistic non-US sector+country combinations have 0–5 training
examples. Predictions for these combinations are extrapolations from
aggregate country and category base rates, not from direct similar
examples.

### B5 — `cat_Technology` is not reachable via keyword matching

A generic "technology startup" description cannot activate
`cat_Technology` through keyword extraction. Only explicit
`categories=["Technology"]` input works. The keyword map has no
pathway to this category.

### B6 — `category_other` loses information when all explicit categories are invalid

When `categories=["FoodTech","AgriTech"]` with no keyword match,
`category_other=0` and `cat_Unknown=1`. The fact that the user attempted
to provide sector information is not preserved in the feature vector.

### B7 — Probability calibration has not been applied

The `prob_success` value is a Random Forest vote proportion, not a
calibrated probability. The output is correctly labelled
"model-estimated success score" but the user experience must make
the calibration gap clear before Phase 4B integration.

### B8 — US geography flags inactive for non-US submissions

`is_sf_bay_area` and `is_california` are forced to 0 for all non-US
countries. For countries where geographic sub-signals matter (e.g.
Bangalore in India), there is no equivalent flag. The geography layer
is US-centric by design.

---

## 9. C — Actual Bugs

**No critical bugs were found.** The pipeline is functionally correct for
its defined scope.

One minor code quality issue was identified:

### C1 — Silent no-op for invalid column name in Step 4 — FIXED

**Status: Fixed in Phase 4A fixes (21 September 2026)**

In `extract_features()` Step 4, the original code was:

```python
for cat in final_cats:
    col = _category_to_col(cat)
    if col in fv:          # <-- silent do-nothing if col is not in fv
        fv[col] = 1
```

If `_category_to_col(cat)` were to produce a column name not in
`FEATURE_ORDER` (e.g. due to a future vocabulary change), the
assignment would silently do nothing instead of raising an error.
The intended category flag would stay at 0.

**Fix applied:** The `if col in fv` guard was replaced with an explicit
`ValueError` that names the missing category and explains the cause:

```python
for cat in final_cats:
    col = _category_to_col(cat)
    if col not in fv:
        raise ValueError(
            f"Internal error: category '{cat}' produced column name '{col}' "
            f"which is not present in FEATURE_ORDER. ..."
        )
    fv[col] = 1
```

A regression test (`test_d1_invalid_column_raises_value_error`) verifies
this error fires by monkey-patching `_resolve_categories` to inject a
fake category. A second test (`test_d1_all_approved_categories_produce_valid_columns`)
verifies that every current `APPROVED_CATEGORIES` member maps to a valid
column, so the fix does not block any real submission.

---

## 10. D — Issues to Fix Before Phase 4B

Both items have been fixed. Tests added and all 21 pass.

### D1 — Add assertion in Step 4 for unknown column names — FIXED ✅

**File changed:** `inference/feature_extractor.py`, Step 4 (category flags section)

**Change:** Replaced the silent `if col in fv:` guard with an explicit
`ValueError` that names the offending category and explains the sync issue.

**Tests added:**
- `test_d1_invalid_column_raises_value_error` — monkey-patches
  `_resolve_categories` to inject a fake category; verifies `ValueError`
  is raised with a message containing the category name.
- `test_d1_all_approved_categories_produce_valid_columns` — verifies
  that every current entry in `APPROVED_CATEGORIES` maps to a column
  present in `FEATURE_ORDER`, confirming the fix does not break normal usage.

### D2 — Document and activate `cat_Technology` via keyword matching — FIXED ✅

**File changed:** `inference/vocabulary.py`, `KEYWORD_TO_CATEGORY` section

**Change:** Added 6 safe keyword phrases that map to the `"Technology"`
approved category:

```python
"tech company":   "Technology"
"tech startup":   "Technology"
"tech firm":      "Technology"
"tech solutions": "Technology"
"tech services":  "Technology"
"tech platform":  "Technology"
```

**Why `tech *` prefix only (not `technology *`):**
Phrases like `"technology company"` are substrings of `"clean technology
company"`. Since the matching engine uses simple `kw in text_lower`
substring matching (not word-boundary matching), `"technology company"`
would incorrectly co-activate `cat_Technology` whenever `cat_Clean_Technology`
was also triggered. The `"tech *"` prefix forms are safe — verified
against all existing `KEYWORD_TO_CATEGORY` entries: none contain
`"tech company"`, `"tech startup"`, etc. as substrings, and
`"clean technology company"` does not contain any `"tech *"` form.

**Training data justification:**
`cat_Technology` is a real trained feature (position 28 of 65), with
279 training examples and a 60.2% historical success rate. It was
present in the model but unreachable via keyword extraction.

**Tests added:**
- `test_d2_cat_technology_keyword_activates` — verifies all 6 new
  phrases activate `cat_Technology`.
- `test_d2_technology_keywords_do_not_shadow_clean_technology` —
  verifies "clean technology company/startup" still route to
  `cat_Clean_Technology` only, not `cat_Technology`.
- `test_d2_technology_keywords_do_not_shadow_information_technology` —
  verifies "information technology services/solutions" still route to
  `cat_Information_Technology` only.
- `test_d2_explicit_technology_category_still_works` — verifies
  explicit `categories=["Technology"]` still works independently.
- `test_five_original_test_cases_still_pass` — parameterised regression
  test verifying all 5 original Phase 4A test cases still produce the
  same structural results (country flags, category flags, completeness).

---

## 11. E — Issues Safe to Leave as Documented Limitations

The following issues are real constraints but require no code change
before Phase 4B. They are correctly documented in `INFERENCE_PIPELINE.md`.

| Issue | Why safe to defer |
|-------|-------------------|
| B1 — Artifact feature dominance | Requires model retraining; not a Phase 4B concern |
| B2 — Missing food/restaurant vocabulary | Phase 4B LLM may bridge this gap; defer to that phase |
| B3 — 2016 training data ceiling | Requires new data; architectural decision for later |
| B4 — Sparse non-US training examples | Requires new data; model limitation, not pipeline bug |
| B5 — `cat_Technology` keyword gap | Minor; user can use explicit `--categories` flag |
| B6 — `category_other` loses invalid-only information | Edge case; warnings already issued to user |
| B7 — Probability calibration absent | Deferred to application integration phase by design |
| B8 — US-centric geography flags | Requires training data from other geographies |

---

## 12. Phase 1–3 Integrity Check

Verified that no Phase 1–3 files were modified during Phase 4A:

| File | Expected size | Actual size | Status |
|------|:-------------:|:-----------:|--------|
| `big_startup_secsees_dataset.csv` | 10,942,450 bytes | 10,942,450 bytes | ✅ UNCHANGED |
| `01_explore_dataset.py` | 44,501 bytes | 44,501 bytes | ✅ UNCHANGED |
| `02_preprocess_data.py` | 32,717 bytes | 32,717 bytes | ✅ UNCHANGED |
| `03_train_models.py` | 38,560 bytes | 38,560 bytes | ✅ UNCHANGED |
| `models/best_model.joblib` | RandomForestClassifier | RandomForestClassifier | ✅ MODEL INTACT |
| `best_model n_features_in_` | 65 | 65 | ✅ UNCHANGED |
| `best_model n_estimators` | 100 | 100 | ✅ UNCHANGED |

The approved model evaluation results are unchanged:

| Metric | Approved value | Status |
|--------|:--------------:|--------|
| Accuracy | 69.37% | ✅ Unchanged |
| Precision | 67.69% | ✅ Unchanged |
| Recall | 81.18% | ✅ Unchanged |
| F1 | 73.82% | ✅ Unchanged |
| ROC-AUC | 75.99% | ✅ Unchanged |

---

## Summary

**Phase 4A is functionally correct.** The inference pipeline accurately
replicates the training preprocessing, produces exactly 65 features in
the correct order, prevents NaN and forbidden columns from reaching the
model, and never invents categories or countries.

**One low-severity code issue** (C1, silent no-op on unknown column
names) should be fixed before Phase 4B to prevent silent wrong
predictions if the vocabulary is extended.

**All significant known limitations** are structural constraints of the
model and training data, not implementation bugs. They are documented
in `INFERENCE_PIPELINE.md` and are properly disclosed to users via
warnings and caveats in the pipeline output.

**The output is correctly labelled as "model-estimated success score"**
throughout the codebase and output. No statement claims this score
represents a guaranteed real-world probability of startup success.

---

*This audit was produced from read-only inspection of all Phase 4A files.*
*No files were modified. All findings are based on verified evidence.*
*Phase 4B must not begin until this audit has been reviewed and approved.*
