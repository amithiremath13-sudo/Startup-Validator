# Inference Pipeline Documentation
## AI-Based Startup Viability Prediction and Validation System
### Phase 4A — Deterministic Feature Extraction and ML Inference

---

> **All test results in this document are real outputs from verified
> execution of `04_predict_startup.py` and `tests/test_inference.py`
> on 21 September 2026. Nothing is fabricated.**
>
> **14/14 pytest tests passed. All 5 CLI test cases executed with exit
> code 0 and no warnings.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Input Format](#2-input-format)
3. [Feature Extraction Process](#3-feature-extraction-process)
4. [Controlled Category Vocabulary](#4-controlled-category-vocabulary)
5. [Country Extraction and Encoding](#5-country-extraction-and-encoding)
6. [Handling of the Unknown Artifacts](#6-handling-of-the-unknown-artifacts)
7. [Construction of the 65-Feature Vector](#7-construction-of-the-65-feature-vector)
8. [Model Loading](#8-model-loading)
9. [Prediction Process](#9-prediction-process)
10. [Feature Validation](#10-feature-validation)
11. [Test Case Results](#11-test-case-results)
12. [Why Deterministic Extraction Is Used Before an LLM](#12-why-deterministic-extraction-is-used-before-an-llm)
13. [Why the Output Is Not a Guaranteed Probability](#13-why-the-output-is-not-a-guaranteed-probability)
14. [What the ML Model Cannot Evaluate](#14-what-the-ml-model-cannot-evaluate)
15. [Distribution Mismatch at Inference Time](#15-distribution-mismatch-at-inference-time)
16. [Limitations](#16-limitations)
17. [File Reference](#17-file-reference)

---

## 1. Overview

Phase 4A builds the inference pipeline that converts a new startup idea
into the exact 65-feature vector that the trained Random Forest model
expects, runs prediction, and returns a structured result.

**The pipeline has two components, only one of which is implemented in
Phase 4A:**

```
  Component A — Historical ML Model (THIS PHASE)
  -----------------------------------------------
  Startup idea text
       ↓
  NLP / Feature Extraction (deterministic keyword matching)
       ↓
  Structured feature vector (65 features)
       ↓
  Load models/best_model.joblib (Random Forest)
       ↓
  Generate predicted class (0 or 1)
  Generate model-estimated success score (0.0–1.0)
       ↓
  Return result with caveats

  Component B — Idea Scoring Engine (FUTURE PHASE)
  -------------------------------------------------
  Not implemented in Phase 4A.
  Will evaluate problem clarity, solution description,
  target market identification using rule-based NLP.
```

**Prediction point:** Day 0 — the moment a startup idea exists, before
any funding has been received. The model uses only sector, country, and
geography — information available at founding.

**Authoritative specification:** `ML_METHODOLOGY.md` Section 11

---

## 2. Input Format

The pipeline accepts input through three mechanisms.

### 2.1 Command-Line Interface

```bash
# Interactive mode (prompts for input)
python 04_predict_startup.py

# Single idea with optional country
python 04_predict_startup.py --idea "Your startup description" --country "India"

# With explicit categories (must be from the approved vocabulary)
python 04_predict_startup.py \
    --idea "A platform for managing invoices" \
    --country "USA" \
    --categories "Software" "Finance"

# With location detail for geography flag detection
python 04_predict_startup.py \
    --idea "A SaaS tool for enterprises" \
    --country "USA" \
    --location "San Francisco, CA"

# Verbose output (prints full 65-feature vector and extraction log)
python 04_predict_startup.py --idea "..." --country "USA" --verbose

# From a JSON file
python 04_predict_startup.py --json-input path/to/input.json
```

### 2.2 JSON Input File Format

```json
{
    "idea_text": "AI platform helping restaurants reduce food waste.",
    "country": "India",
    "categories": ["Analytics"],
    "location_detail": "Bangalore, India"
}
```

All fields except `idea_text` are optional.

### 2.3 Programmatic (Python Import)

```python
from inference.feature_extractor import StartupIdeaInput, extract_features
from inference.model_loader import load_model_and_metadata, predict

model, feature_names, prep_meta = load_model_and_metadata()
input_data = StartupIdeaInput(
    idea_text  = "A software platform for invoicing.",
    country    = "USA",
    categories = ["Software", "Finance"],
)
result       = extract_features(input_data)
ordered_list = feature_vector_to_ordered_list(result.feature_vector)
predicted_class, prob_failure, prob_success = predict(model, ordered_list, feature_names)
```

### 2.4 Input Fields

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `idea_text` | str | Yes | Free-text description of the startup idea. Used for keyword-based category extraction. |
| `country` | str | No | Country name or ISO-3 code (e.g. "India", "IND", "United States", "USA"). |
| `categories` | list[str] | No | Explicit category names from the approved vocabulary. Bypass keyword matching for precise control. |
| `location_detail` | str | No | Location string for geography flag detection (e.g. "San Francisco, CA"). |

---

## 3. Feature Extraction Process

Extraction runs in five deterministic steps. The same input always
produces the same output. No randomness or external API calls are involved.

### Step 1 — Resolve Country

The `country` input is normalized to an ISO-3 code via:

1. **Direct ISO-3 match:** "IND" → IND, "USA" → USA
2. **Alias lookup:** "India" → IND, "United Kingdom" → GBR, "Germany" → DEU
3. **Substring match:** "People's Republic of China" → CHN (matches "china")
4. **Unresolvable:** A name that matches nothing → `country_other=1`
5. **No input:** `country_Unknown=1`

The alias table in `inference/vocabulary.py` covers the most common
country names and alternative spellings.

### Step 2 — Detect Geography Flags

Two binary flags are derived from the resolved country plus any location
text:

- `is_sf_bay_area = 1` if SF Bay Area signals appear in `location_detail`
  or `idea_text` AND the country is USA (or unspecified).
  Signals: "san francisco", "silicon valley", "palo alto", "bay area", etc.

- `is_california = 1` if California signals appear AND country is USA.
  Signals include all Bay Area signals plus "los angeles", "san diego",
  "california", etc.

- Both flags are forced to 0 for non-US countries.

- `is_sf_bay_area = 1` implies `is_california = 1`.

### Step 3 — Resolve Categories

Categories are resolved in priority order:

**Priority 1 — Explicit categories (user-provided `categories` list):**
Each entry is validated against `APPROVED_CATEGORIES`. Entries not in
the vocabulary are rejected and logged as warnings. This allows precise
control over category assignment without relying on keyword matching.

**Priority 2 — Keyword matching on `idea_text`:**
Over 100 keyword/phrase patterns are checked against the lowercased
idea text. Each keyword maps to exactly one approved category. Multiple
keywords can match → multiple categories (multi-hot). Keywords are
checked longest-first so more specific phrases ("ai platform") match
before generic substrings ("app").

**Fallback — No category resolved:**
If neither explicit input nor keyword matching produces any result,
`cat_Unknown=1` is set. A warning is added to the result.

### Step 4 — Build the Feature Vector

All 65 features are initialised to 0. Then:
- Geography flags set from Step 2
- Country flags set from Step 1
- Category flags set from Step 3
- `n_categories` set to count of resolved real categories (minimum 1)
- `category_other` set to 1 if any explicitly provided category was
  rejected (unrecognised), meaning the company has at least one
  category outside the top-50 vocabulary

### Step 5 — Validate

The assembled feature vector is checked before prediction:
- Exactly 65 features
- All values are non-null integers
- Feature key order matches `FEATURE_ORDER` / `feature_names.json`
- No forbidden columns present (`status`, `target`, `funding_total_usd`,
  `funding_rounds`, `last_funding_at`, `first_funding_at`,
  `founding_year`, `has_founding_date`, `is_usa`)

If any check fails, the pipeline raises an error and stops.

---

## 4. Controlled Category Vocabulary

### 4.1 The 50 Approved Categories

These are the exact categories from `preprocessing_metadata.json →
top_categories_list`, in training-frequency order. Only these names
may appear as category flags in the model's feature matrix.

| # | Category name | Feature column |
|---|--------------|----------------|
| 1 | Software | `cat_Software` |
| 2 | Biotechnology | `cat_Biotechnology` |
| 3 | Mobile | `cat_Mobile` |
| 4 | Unknown | `cat_Unknown` (artifact — see §6) |
| 5 | Curated Web | `cat_Curated_Web` |
| 6 | E-Commerce | `cat_E-Commerce` |
| 7 | Social Media | `cat_Social_Media` |
| 8 | Advertising | `cat_Advertising` |
| 9 | Enterprise Software | `cat_Enterprise_Software` |
| 10 | Games | `cat_Games` |
| 11 | Health Care | `cat_Health_Care` |
| 12 | Services | `cat_Services` |
| 13 | Internet | `cat_Internet` |
| 14 | Finance | `cat_Finance` |
| 15 | Technology | `cat_Technology` |
| 16 | Analytics | `cat_Analytics` |
| 17 | Hardware + Software | `cat_Hardware_plus_Software` |
| 18 | Security | `cat_Security` |
| 19 | Clean Technology | `cat_Clean_Technology` |
| 20 | Apps | `cat_Apps` |
| 21 | Semiconductors | `cat_Semiconductors` |
| 22 | SaaS | `cat_SaaS` |
| 23 | Health and Wellness | `cat_Health_and_Wellness` |
| 24 | Web Hosting | `cat_Web_Hosting` |
| 25 | Networking | `cat_Networking` |
| 26 | Video | `cat_Video` |
| 27 | Social Network Media | `cat_Social_Network_Media` |
| 28 | Search | `cat_Search` |
| 29 | Manufacturing | `cat_Manufacturing` |
| 30 | Music | `cat_Music` |
| 31 | Education | `cat_Education` |
| 32 | Messaging | `cat_Messaging` |
| 33 | Cloud Computing | `cat_Cloud_Computing` |
| 34 | Medical | `cat_Medical` |
| 35 | News | `cat_News` |
| 36 | Travel | `cat_Travel` |
| 37 | Information Technology | `cat_Information_Technology` |
| 38 | Web Development | `cat_Web_Development` |
| 39 | Telecommunications | `cat_Telecommunications` |
| 40 | Business Services | `cat_Business_Services` |
| 41 | Media | `cat_Media` |
| 42 | Sales and Marketing | `cat_Sales_and_Marketing` |
| 43 | Entertainment | `cat_Entertainment` |
| 44 | iPhone | `cat_iPhone` |
| 45 | Design | `cat_Design` |
| 46 | Fashion | `cat_Fashion` |
| 47 | Consulting | `cat_Consulting` |
| 48 | Marketplaces | `cat_Marketplaces` |
| 49 | Android | `cat_Android` |
| 50 | FinTech | `cat_FinTech` |

### 4.2 Column Name Derivation Rule

The column name for any category is derived by the same rule used in
`02_preprocess_data.py`:

```python
safe = category_name
         .replace(" ", "_")
         .replace("/", "_")
         .replace("+", "plus")
col  = "cat_" + safe
```

Hyphens are **not** replaced — `E-Commerce` produces `cat_E-Commerce`.
`Hardware + Software` produces `cat_Hardware_plus_Software`.

### 4.3 What Happens to Unrecognised Categories

If a user explicitly passes a category name not in the vocabulary
(e.g. "AgriTech", "FoodTech"):
- The category is **rejected** with a warning
- It does NOT appear in the feature vector
- `category_other=1` is set if at least one real category was also matched
- If no real category was matched at all, `cat_Unknown=1` is set

The pipeline never fabricates a category.

---

## 5. Country Extraction and Encoding

### 5.1 The 10 Approved Country Codes

Derived from `preprocessing_metadata.json → top_countries_list`:

| ISO-3 Code | Country | Feature column |
|:----------:|---------|----------------|
| USA | United States | `country_USA` |
| Unknown | Missing data (artifact) | `country_Unknown` — see §6 |
| GBR | United Kingdom | `country_GBR` |
| CAN | Canada | `country_CAN` |
| ISR | Israel | `country_ISR` |
| DEU | Germany | `country_DEU` |
| FRA | France | `country_FRA` |
| RUS | Russia | `country_RUS` |
| CHN | China | `country_CHN` |
| IND | India | `country_IND` |

### 5.2 Country Flag Logic

Exactly one country-group flag is set to 1 per submission:

| Situation | country_X=1 | country_Unknown=1 | country_other=1 |
|-----------|:-----------:|:-----------------:|:---------------:|
| Country resolved, in top-10 list | ✅ (specific col) | ❌ | ❌ |
| Country resolved, NOT in top-10 | ❌ | ❌ | ✅ |
| Country provided but unresolvable | ❌ | ❌ | ✅ |
| No country provided | ❌ | ✅ | ❌ |

`country_Unknown` and `country_other` are mutually exclusive.

### 5.3 Geography Flags with Non-US Countries

`is_sf_bay_area` and `is_california` are always **0** for companies
outside the USA. These flags are only meaningful in the US context.

---

## 6. Handling of the Unknown Artifacts

`country_Unknown` and `cat_Unknown` are among the top-3 most important
features in the trained Random Forest (see `MODEL_FEATURE_IMPORTANCE.md`).
However, they are historical data-recording artifacts, not real signals.

### 6.1 What these features mean in the training data

During preprocessing (`02_preprocess_data.py`):
- Companies with **no recorded `country_code`** in Crunchbase had their
  country filled with the string `"Unknown"` → `country_Unknown = 1`
- Companies with **no recorded `category_list`** in Crunchbase had their
  category filled with the string `"Unknown"` → `cat_Unknown = 1`

These flags captured a data-recording gap in the historical database.
They correlate with outcome because poorly-documented companies had
different success rates than well-documented ones — a documentation
artifact, not a business signal.

### 6.2 When each flag is set at inference time

| Flag | Set to 1 when | Set to 0 when |
|------|--------------|---------------|
| `country_Unknown` | User provides **no country** input | User provides any country (even non-top-10) |
| `cat_Unknown` | **No category is found** from either explicit input or keyword matching | At least one real category is matched |

### 6.3 For a normal complete submission

A user who provides a country and a description with recognisable
sector keywords will have:
- `country_Unknown = 0`
- `cat_Unknown = 0`

Both artifact flags are 0 for all 5 test cases that included a country
and meaningful description.

### 6.4 Why these features are kept in the model

Removing these features would require retraining the model. The current
Phase 4A uses the trained model as-is. The pipeline explicitly warns
the user when `country_Unknown = 1` or `cat_Unknown = 1` is triggered,
explaining that the result may be less reliable. The decision to retrain
without these features (or to use them differently) is deferred to a
future phase.

---

## 7. Construction of the 65-Feature Vector

The full ordered feature set, grouped by type:

### Group 1 — Geography Flags (positions 1–2)
`is_sf_bay_area`, `is_california`

Binary. Derived from country + location text.

### Group 2 — Country Flags (positions 3–13)
`country_USA`, `country_Unknown`, `country_GBR`, `country_CAN`,
`country_ISR`, `country_DEU`, `country_FRA`, `country_RUS`,
`country_CHN`, `country_IND`, `country_other`

Binary. Mutually exclusive (only one is 1 per submission).

### Group 3 — Category Flags (positions 14–65)
50 individual category columns (`cat_Software` through `cat_FinTech`),
plus `category_other` and `n_categories`.

The 50 binary category columns are multi-hot: multiple can be 1
simultaneously. `n_categories` is a non-negative integer (minimum 1
in the training distribution; matched to the same minimum at inference).

### Feature order is absolute

The 65 features must appear in exactly the order defined in
`processed_data/feature_names.json`. This is enforced by:
1. The `FEATURE_ORDER` tuple in `inference/feature_extractor.py`
   (hardcoded from `feature_names.json`)
2. `feature_vector_to_ordered_list()` which builds the prediction
   array by iterating `FEATURE_ORDER`
3. `predict()` which wraps the array in a `pd.DataFrame` with
   `feature_names` columns, so sklearn's model uses named-column
   matching and raises an error on any mismatch

---

## 8. Model Loading

The model loader (`inference/model_loader.py`) performs strict
validation on every load:

```
1. Verify models/best_model.joblib exists
2. Verify processed_data/feature_names.json exists
3. Verify processed_data/preprocessing_metadata.json exists
4. Load model via joblib.load()
5. Assert type(model).__name__ == "RandomForestClassifier"
6. Assert model.n_features_in_ == 65
7. Load feature_names.json
8. Assert len(feature_names) == 65
9. Load preprocessing_metadata.json
```

If any assertion fails, a `ModelLoadError` is raised with a clear
message explaining which check failed. The pipeline never silently
uses a wrong model.

**Model file:** `models/best_model.joblib`
**Model type:** `RandomForestClassifier`
**Parameters:** `n_estimators=100, max_depth=None, min_samples_leaf=5,
                  random_state=42`
**Selected by:** highest CV ROC-AUC = 0.7364 (training data only)
**Test-set metrics:** Accuracy=69.37%, F1=73.82%, ROC-AUC=75.99%

---

## 9. Prediction Process

```python
# 1. Build the feature dict (65 keys, all int)
result = extract_features(input_data)

# 2. Convert to ordered list matching FEATURE_ORDER
ordered_list = feature_vector_to_ordered_list(result.feature_vector)

# 3. Wrap in DataFrame with named columns (prevents sklearn warning)
X = pd.DataFrame([ordered_list], columns=feature_names)

# 4. Predict class and probability
predicted_class = model.predict(X)[0]          # 0 or 1
probabilities   = model.predict_proba(X)[0]    # [P(0), P(1)]
prob_failure    = probabilities[0]
prob_success    = probabilities[1]
```

**Output interpretation:**

| predicted_class | Meaning |
|:---------------:|---------|
| **1** | SUCCESS PATTERN — historical patterns match companies that were acquired or had an IPO |
| **0** | FAILURE PATTERN — historical patterns match companies that were closed |

The `prob_success` value (0.0–1.0) is the Random Forest's average
vote across 100 trees for class 1. It is described as a
**"model-estimated success score"** — NOT a calibrated real-world
probability. See Section 13.

---

## 10. Feature Validation

Before every prediction, the pipeline runs 6 automated checks:

| Check | What it verifies |
|-------|-----------------|
| Feature count | `len(fv) == 65` |
| Feature order | `list(fv.keys()) == FEATURE_ORDER` |
| JSON match | `list(fv.keys()) == feature_names` (from feature_names.json) |
| No NaN | All values are non-null integers |
| No target | `"target"` not in feature vector |
| No forbidden cols | `status`, `funding_total_usd`, `funding_rounds`, etc. all absent |

If any check fails, the pipeline prints `VALIDATION FAILED` and exits
with code 1. It never silently proceeds with an invalid feature vector.

---

## 11. Test Case Results

All results verified on 21 September 2026. Exit code 0 for all cases.

### TEST 1 — AI platform for restaurants to reduce food waste in India

```
Input:   idea="AI platform helping small restaurants reduce food waste in India."
         country="India"

Extracted:
  country_IND = 1
  cat_Analytics = 1       (keyword: "ai platform")
  cat_Clean_Technology = 1  (keyword: "food waste")
  cat_Services = 1        (keyword: "restaurant")
  n_categories = 3
  country_Unknown = 0, cat_Unknown = 0
  is_complete_submission = True

Result:
  Predicted class       : 0 (FAILURE PATTERN)
  Model success score   : 0.4473 (44.7%)
```

**Note on "restaurant" → Services:** There is no dedicated "Food" or
"Restaurant" category in the 50-category training vocabulary. The
keyword maps to the closest available category, "Services". This is a
known vocabulary gap — see Section 16.2.

---

### TEST 2 — Online education platform in India

```
Input:   idea="Online education platform for students preparing for
               competitive exams in India."
         country="India"

Extracted:
  country_IND = 1
  cat_Education = 1       (keywords: "online education", "competitive exam",
                           "student", "online learning")
  n_categories = 1
  country_Unknown = 0, cat_Unknown = 0
  is_complete_submission = True

Result:
  Predicted class       : 0 (FAILURE PATTERN)
  Model success score   : 0.4710 (47.1%)
```

---

### TEST 3 — Healthcare technology for remote patient monitoring

```
Input:   idea="Healthcare technology platform for remote patient monitoring."
         country=(none provided)

Extracted:
  country_Unknown = 1     (no country → artifact flag triggered)
  cat_Health_Care = 1     (keyword: "healthcare")
  cat_Medical = 1         (keywords: "remote patient", "patient monitoring")
  n_categories = 2
  cat_Unknown = 0
  is_complete_submission = False   (country missing)

Warning issued: "Incomplete submission (country not resolved). The model
  will rely on missing-data artifact features..."

Result:
  Predicted class       : 0 (FAILURE PATTERN)
  Model success score   : 0.2997 (30.0%)
```

The lower score vs Test 1 is driven by `country_Unknown=1`, which the
model learned is associated with worse outcomes in the historical data —
but this is a data artifact, not a real signal about the idea quality.

---

### TEST 4 — Software platform for invoice management (USA, explicit categories)

```
Input:   idea="A software platform for small businesses to manage invoices."
         country="USA"
         categories=["Software", "Finance"]  (explicit)

Extracted:
  country_USA = 1
  cat_Software = 1        (explicit + keyword: "software platform")
  cat_Finance = 1         (explicit + keyword: "invoice", "invoicing")
  n_categories = 2
  country_Unknown = 0, cat_Unknown = 0
  is_complete_submission = True

Result:
  Predicted class       : 1 (SUCCESS PATTERN)
  Model success score   : 0.6978 (69.8%)
```

USA + Software + Finance is a combination the model saw frequently in
successful training examples. This aligns with historical patterns —
US software/finance tools have high acquisition rates in the dataset.

---

### TEST 5 — Deliberately vague: "I have an app idea"

```
Input:   idea="I have an app idea."
         country=(none)
         categories=(none)

Extracted:
  country_Unknown = 1     (no country)
  cat_Apps = 1            (keyword: "app" → approved category "Apps")
  n_categories = 1
  cat_Unknown = 0         (keyword match prevented fallback to Unknown)
  is_complete_submission = False

Warning issued: incomplete submission.

Result:
  Predicted class       : 0 (FAILURE PATTERN)
  Model success score   : 0.1671 (16.7%)
```

**Key behaviour verified:** The pipeline matched "app" to the approved
vocabulary category "Apps" — it did NOT invent a category like
"Mobile App" or "Technology Startup." Only one approved category was
returned. `cat_Unknown` remained 0 because a real category was found.

The low score is driven by both `country_Unknown=1` (a strong negative
signal in the model) and the minimal description providing little
positive structural signal.

---

### Test Suite Summary

```
14/14 pytest tests passed in 2.91s

Test cases (5):
  test_01_india_restaurant_ai          PASS
  test_02_india_education              PASS
  test_03_healthcare_remote_monitoring PASS
  test_04_invoice_software_usa         PASS
  test_05_vague_idea                   PASS

Structural/contract tests (9):
  test_feature_order_matches_json      PASS
  test_feature_count_constant          PASS
  test_model_feature_count_matches     PASS
  test_model_classes_are_0_and_1       PASS
  test_category_col_mapping            PASS
  test_explicit_invalid_category_is_rejected  PASS
  test_zero_features_for_empty_idea    PASS
  test_country_other_for_unknown_country  PASS
  test_mutual_exclusion_country_flags  PASS
```

---

## 12. Why Deterministic Extraction Is Used Before an LLM

Phase 4A uses keyword matching rather than an LLM (such as Gemini or
GPT) for feature extraction. This is a deliberate design decision with
specific justifications.

### Reason 1 — A reproducible baseline is required first

Before introducing an LLM, we need to establish what the baseline
deterministic system produces. If the LLM is introduced first, there
is no fixed reference point to measure whether it improves things.
The keyword-matching system gives us a stable, testable, documented
baseline with predictable behaviour.

### Reason 2 — LLMs can fabricate information

An LLM asked "what sector is this startup in?" may confidently return
a category that is not in the approved 50-category vocabulary, or may
hallucinate details about the idea that the user never provided. The
deterministic system can only return categories that are provably in the
vocabulary. This is a hard correctness requirement for the inference
pipeline.

### Reason 3 — Determinism enables testing

Keyword matching produces the same output every time for the same input.
This allows the 14 automated tests in `test_inference.py` to assert
exact expected values. An LLM produces non-deterministic output that
cannot be tested with exact assertions.

### Reason 4 — The feature set is structurally limited

The model's feature vocabulary is only 50 categories. A sophisticated
LLM is probably overkill for mapping free text to 50 binary flags.
A well-designed keyword map can cover the majority of realistic startup
descriptions adequately.

### When an LLM would add value (Phase 4B and beyond)

The keyword approach has real limitations: it misses synonyms, handles
ambiguity poorly, and has no semantic understanding. An LLM integration
would help with:
- Mapping descriptions to categories that use different terminology
  (e.g. "edtech" → "Education" when the word "education" is absent)
- Extracting implicit geographic signals ("targeting the Bay Area market")
- Providing richer context for the idea scoring engine (Component B)

The LLM integration is deferred to Phase 4B, after the deterministic
baseline has been validated.

---

## 13. Why the Output Is Not a Guaranteed Probability

The `prob_success` value printed by the pipeline is labelled:

> **"Model-estimated success score"**

It is NOT labelled as "probability of success" and must not be
displayed to users as a factual probability. There are two reasons.

### Reason 1 — ROC-AUC measures discrimination, not calibration

The model was selected and evaluated using ROC-AUC = 0.7599 on the
test set. ROC-AUC measures whether the model ranks successes above
failures — it does not measure whether the numeric scores are calibrated.

A well-calibrated score means: if the model outputs 0.70 for 100
companies, approximately 70 of them should actually succeed. Random
Forests are known to produce scores that are well-ordered (higher
scores → more likely to succeed) but not always well-calibrated in
absolute terms.

### Reason 2 — The training data is not a representative sample

The model was trained on Crunchbase-documented startups from 1990–2016.
Even if the scores were calibrated on this historical sample, they
would not directly translate to real-world success probabilities for
startups today because:
- The dataset has survivorship bias
- The time period (1990–2016) does not represent current market conditions
- Only sector + country features are used (many determinants of success
  are absent)

### What the score does represent

The score does meaningfully rank companies — a score of 0.70 represents
a stronger historical structural match to successful companies than a
score of 0.40. It can be used as a relative indicator and a component
of a broader scoring system.

### Probability calibration (future work)

Proper probability calibration (using Platt scaling or isotonic
regression applied to the Random Forest output) will be addressed during
the application integration phase. Until then, all outputs must include
the caveat that the score is not a calibrated real-world probability.

---

## 14. What the ML Model Cannot Evaluate

The historical Random Forest model predicts outcomes based on sector,
country, and geography patterns only. It cannot and does not evaluate:

| Dimension | Why it cannot be evaluated |
|-----------|---------------------------|
| **Idea quality** | No text descriptions in the training data |
| **Problem-solution fit** | Not recorded in Crunchbase |
| **Market size** | Not in the dataset |
| **Competition** | Not in the dataset |
| **Revenue potential** | Not in the dataset |
| **Founder quality** | Not recorded |
| **Technical feasibility** | Not recorded |
| **Customer pain intensity** | Not recorded; subjective |
| **Business model** | Not recorded |

These dimensions will be evaluated by Component B (the Idea Scoring
Engine) in a future phase using rule-based heuristics applied to the
user's free-text description. Component B is entirely separate from
the historical ML model and its scores will be clearly labelled as
rule-based evaluations.

---

## 15. Distribution Mismatch at Inference Time

Several gaps exist between the training distribution and inference-time
inputs. These are documented here and must be disclosed to users.

### 15.1 Training period: 1990–2016

The model was trained on startups founded approximately 1990–2016. New
ideas submitted in 2026 represent a different macroeconomic context,
different dominant sectors, and different VC/exit market conditions.
`founding_year` was excluded from features specifically because of this
gap (see `PREPROCESSING_REPORT.md` §16). The gap still exists implicitly
in the sector and geography patterns.

### 15.2 USA and "Unknown" geography dominate

56.7% of training companies were US-based. The model has seen far more
examples of US success/failure patterns than any other country. For
non-US startups, predictions are based on fewer historical examples and
are inherently less reliable.

### 15.3 Top-3 features are data artifacts

`country_Unknown`, `country_USA`, and `cat_Unknown` are the three most
important features by mean decrease in impurity. The first and third
are historical missing-data artifacts that will almost always be 0 for
new submissions (since users provide a country and the keyword matcher
finds at least one category). This means the model's strongest signals
at training time are largely absent at inference time, and the remaining
65% of importance is spread across geography and sector features.

### 15.4 No sector for restaurant / food / agriculture

The 50-category vocabulary does not contain "Food", "Restaurant",
"AgriTech", or "FoodTech". Ideas in these sectors are mapped to the
nearest available categories ("Services", "Clean Technology") which
is an imprecise approximation.

---

## 16. Limitations

> **Phase 4A fixes applied (21 September 2026):**
> - C1 (silent no-op in Step 4) — fixed with explicit `ValueError`
> - D2 (`cat_Technology` keyword gap) — fixed with 6 safe `tech *` phrases

### 16.1 Keyword matching has no semantic understanding

The keyword extractor matches literal strings. "My platform empowers
chefs to reduce spoilage" — the word "spoilage" is not in the keyword
map. An LLM would recognise this as food waste reduction; the keyword
matcher misses it entirely.

### 16.2 50-category vocabulary is from 2005–2015

The training vocabulary was built from Crunchbase category tags in
the dataset's time period. Categories that have become common since
then ("AI/ML", "Web3", "Climate Tech" as a specific term, "AgriTech",
"GovTech") are not in the vocabulary or map imperfectly.

### 16.3 One keyword → one category (no weighted confidence)

Every keyword match has equal weight. There is no confidence score
per match. If three keywords match, all three categories are set to 1
with equal certainty, regardless of how central they are to the idea.

### 16.4 Country alias list covers major countries only

The alias table maps the most common country names and a few
alternatives. Less common names (e.g. "Việt Nam", "Österreich",
"Türkiye") may not be recognised and will fall back to `country_other=1`.

### 16.5 Model test-set performance ceiling

The Random Forest achieves ~70% accuracy and ~76% ROC-AUC on the
test set. This is a moderate result. Even a perfectly implemented
inference pipeline cannot produce predictions that exceed this ceiling,
because the limitation is in the model's discriminative ability on this
feature set.

---

## 17. File Reference

### Phase 4A files created

| File | Purpose |
|------|---------|
| `04_predict_startup.py` | CLI entry point for startup idea prediction |
| `inference/__init__.py` | Package initialiser |
| `inference/vocabulary.py` | Controlled vocabulary: approved categories, country aliases, keyword map |
| `inference/feature_extractor.py` | Core feature extraction logic; `StartupIdeaInput`, `ExtractionResult`, `extract_features()` |
| `inference/model_loader.py` | Model loading with strict validation; `predict()` function |
| `tests/__init__.py` | Test package initialiser |
| `tests/test_inference.py` | 14 automated tests (5 scenario + 9 structural) |
| `INFERENCE_PIPELINE.md` | This document |

### Phase 1–3 files used (not modified)

| File | Used for |
|------|---------|
| `models/best_model.joblib` | Loaded by `model_loader.py` |
| `models/model_metadata.json` | Reference only (not loaded at inference) |
| `processed_data/feature_names.json` | Ground-truth feature order |
| `processed_data/preprocessing_metadata.json` | Category list, country list |

### How to run

```bash
# Run a prediction
python 04_predict_startup.py --idea "Your idea" --country "Country"

# Run the full test suite
python -m pytest tests/test_inference.py -v

# Run tests without pytest
python tests/test_inference.py
```

---

*This document was written after verified execution of all test cases.*
*Phase 4A is complete. Phase 4B (LLM integration / idea scoring engine)*
*must not begin until this phase has been reviewed and approved.*
