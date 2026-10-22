# Preprocessing Report
## AI-Based Startup Viability Prediction and Validation System
### Phase 2 — Data Preprocessing and Feature Engineering

---

> **All numbers in this report are sourced from the actual execution of
> `02_preprocess_data.py` on 19 September 2026 (Phase 2 revised run).
> Nothing is estimated.**
>
> **Phase 2 review changes applied:** `founding_year`, `has_founding_date`,
> and `is_usa` removed from the feature matrix. Feature count reduced from
> 68 to **65**. See Section 16 for full explanation.
>
> **Authoritative specification:** `ML_METHODOLOGY.md` Section 11
> (Revised Prediction Framework — Day 0 prediction point)
>
> **Original CSV was not modified at any point.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prediction Point Reminder](#2-prediction-point-reminder)
3. [Raw Data Input](#3-raw-data-input)
4. [Target Variable Creation](#4-target-variable-creation)
5. [Train / Test Split](#5-train--test-split)
6. [Date Cleaning](#6-date-cleaning)
7. [Feature Engineering](#7-feature-engineering)
8. [Missing Value Handling](#8-missing-value-handling)
9. [Categorical Encoding](#9-categorical-encoding)
10. [Leakage Prevention](#10-leakage-prevention)
11. [Final Feature Matrix](#11-final-feature-matrix)
12. [Final Row and Target Counts](#12-final-row-and-target-counts)
13. [Rows Removed and Why](#13-rows-removed-and-why)
14. [Output Files](#14-output-files)
15. [Reproducibility](#15-reproducibility)
16. [Why founding_year Was Removed](#16-why-founding_year-was-removed)

---

## 1. Overview

This report documents every transformation applied to the raw Crunchbase
startup dataset to produce a clean, ML-ready feature matrix. The pipeline
is fully reproducible: re-running `02_preprocess_data.py` with the same
random seed produces byte-identical output files.

**Script:** `02_preprocess_data.py`  
**Input:** `big_startup_secsees_dataset.csv` (66,368 rows, 14 columns)  
**Output directory:** `processed_data/`  
**Random seed:** 42  
**Test fraction:** 20%

---

## 2. Prediction Point Reminder

> **Prediction point: Day 0 — the moment a startup idea exists, before
> any funding has been received.**

This means the following columns are **completely excluded** from the ML
feature matrix, no matter how strong their predictive signal might be:

| Column | Reason excluded |
|--------|----------------|
| `funding_total_usd` | Lifetime total — entirely post-prediction |
| `funding_rounds` | Lifetime count — entirely post-prediction |
| `last_funding_at` | Most recent funding date — future event |
| `first_funding_at` | First funding date — future event from Day 0 |
| `time_to_first_funding` | Cannot exist before first funding occurs |
| `founding_year` | Temporal distribution mismatch — see Section 16 |
| `has_founding_date` | Always = 1 at inference time; zero discriminatory power — see Section 16 |
| `is_usa` | Redundant with `country_USA` — see Section 16 |
| `status` | This IS the target variable |

All features in the final matrix represent only what is knowable when a
startup idea is first conceived: its sector, its geography, and its
founding year.

---

## 3. Raw Data Input

| Property | Value |
|----------|-------|
| Rows loaded | 66,368 |
| Columns | 14 |
| Encoding | UTF-8 |
| File size | 10.44 MB |

All 14 original columns were loaded without modification. The CSV was
opened in read-only mode throughout the entire pipeline.

---

## 4. Target Variable Creation

### 4.1 Rule applied

```
acquired  →  target = 1  (SUCCESS)
ipo       →  target = 1  (SUCCESS)
closed    →  target = 0  (FAILURE)
operating →  EXCLUDED (outcome unknown — cannot be labeled)
```

### 4.2 Status counts before split

| Status | Count | % of total |
|--------|------:|----------:|
| operating | 53,034 | 79.91% |
| closed | 6,238 | 9.40% |
| acquired | 5,549 | 8.36% |
| ipo | 1,547 | 2.33% |

### 4.3 Labeled dataset

| Label | Meaning | Count |
|-------|---------|------:|
| 1 | SUCCESS (acquired + ipo) | 7,096 |
| 0 | FAILURE (closed) | 6,238 |
| — | **Total labeled** | **13,334** |
| — | Operating (excluded) | 53,034 |
| — | Other / NaN (excluded) | 0 |

### 4.4 Why operating companies are excluded

A company with `status = operating` had not yet reached a final outcome
when the data was compiled. Its eventual result — acquisition, IPO, or
closure — is unknown. Assigning any label to it would fabricate training
data. It is saved separately for future inference, not used for training.

---

## 5. Train / Test Split

The split was performed **immediately after creating the target variable**
and **before any statistics were computed**. This ordering is mandatory
to prevent preprocessing leakage (where test-set distribution information
influences training-set decisions).

| Property | Value |
|----------|-------|
| Method | Stratified random split |
| Random seed | 42 |
| Test fraction | 20% |
| Train rows | **10,667** |
| Test rows | **2,667** |

### 5.1 Target distribution after split

| Set | Class 1 (success) | Class 0 (failure) | Class 1 % |
|-----|------------------:|------------------:|----------:|
| Train | 5,677 | 4,990 | 53.2% |
| Test | 1,419 | 1,248 | 53.2% |

The stratified split preserved the class ratio exactly. Both train and
test sets have 53.2% success / 46.8% failure — identical to the full
labeled dataset ratio.

### 5.2 Why stratified splitting matters

A non-stratified random split could accidentally put more failure examples
in training and more success examples in test (or vice versa), making the
model appear to perform differently from what it would on real data. The
`stratify=y` parameter in scikit-learn's `train_test_split` guarantees
that the class ratio is preserved in both halves.

---

## 6. Date Cleaning

### 6.1 Columns processed

Three date columns were parsed from string to datetime:
`founded_at`, `first_funding_at`, `last_funding_at`.

Conversion used `pd.to_datetime(..., errors='coerce')`, which turns any
unparseable string into `NaT` (Not a Time — the date equivalent of NaN).

### 6.2 Invalid year rule

From Phase 1 EDA, two categories of invalid dates were identified as
data-entry typos:
- Year > 2023: dates set in the future (e.g., year 2914, 2105)
- Year < 1800: dates set over 200 years ago (e.g., year 1014, 1015)

These rows are set to `NaT`. They are not deleted — the row is kept, but
the date value is treated as missing.

**Why NaT rather than attempting to correct the typo?**
A value like `2914-01-01` could be a transposition of 2014, 2019, 2041,
or 1914. We cannot reliably determine the correct year. Setting to NaT
and using the missing-data flag is the only defensible approach.

### 6.3 Rows affected

| Split | Column | Before nulls | Future dates nulled | Past dates nulled | After nulls |
|-------|--------|-------------:|--------------------:|------------------:|------------:|
| Train | `founded_at` | 2,994 | 2 | 0 | 2,996 |
| Train | `first_funding_at` | 1 | 0 | 0 | 1 |
| Train | `last_funding_at` | 0 | 1 | 0 | 1 |
| Test | `founded_at` | 738 | 0 | 1 | 739 |
| Test | `first_funding_at` | 1 | 0 | 0 | 1 |
| Test | `last_funding_at` | 0 | 0 | 0 | 0 |

**Total rows where an additional date was nulled: 4 rows** (3 in
`founded_at`, 1 in `last_funding_at`). This is 0.03% of the dataset.
Impact on the feature matrix is negligible. `last_funding_at` is excluded
from features regardless.

---

## 7. Feature Engineering

All features were engineered from columns that exist at Day 0 (founding
time). No derived feature requires post-founding information.

### 7.1 `founding_year` (from `founded_at`)

> **Phase 2 review decision: `founding_year` EXCLUDED from the feature
> matrix.** See Section 16 for full explanation.

The year was still computed internally during the pipeline (Step 5) for
data quality logging purposes, but it is not included in the final
`train_features.csv` or `test_features.csv`.

### 7.2 `has_founding_date` (from `founded_at`)

> **Phase 2 review decision: `has_founding_date` EXCLUDED from the feature
> matrix.**

**Reason:** When a user submits a new startup idea through the application,
they are founding a company right now — at Day 0. Therefore `has_founding_date`
would always equal 1 for every new submission. A feature that is always the
same value provides zero information to the model. Including it would waste
a feature slot without contributing any discriminatory power.

**Verification from actual data:** In the training set, 28.1% of rows have
`has_founding_date = 0` (Crunchbase had no founding date for those
historical companies). But this reflects a *data recording* gap in a
historical database — not a realistic situation for a new idea submission.
The historical missingness is a data quality artifact, not a meaningful
pattern that the application can replicate at inference time.

### 7.3 `is_usa` (from `country_code`)

> **Phase 2 review decision: `is_usa` EXCLUDED from the feature matrix.**

**Reason:** `country_USA` — produced in the country encoding step — encodes
exactly the same information: 1 if the company is in the United States,
0 otherwise. Keeping both columns would be redundant. In a linear model,
perfectly correlated features create instability. In a tree-based model,
one would be ignored entirely. `country_USA` is kept as part of the
systematic country encoding group; the standalone `is_usa` is removed.

### 7.4 `is_sf_bay_area` (from `region`)

Retained. See original Section 7.4 (unchanged).

- Binary flag: 1 if `region == "SF Bay Area"`, else 0
- Train: 1,986 SF Bay Area companies (18.6%)
- Not redundant with any country column — encodes sub-national geography.

### 7.5 `is_california` (from `state_code`)

Retained. See original Section 7.5 (unchanged).

- Binary flag: 1 if `state_code == "CA"`, else 0
- Train: 2,706 California companies (25.4%)
- Not redundant with `country_USA` — California ≠ all of USA.

---

## 8. Missing Value Handling

### 8.1 Principles

1. **Do not delete rows** — dropping rows discards all other valid
   features for that company. With only 13,334 labeled rows, each one
   matters.
2. **Pair imputation with a missing-data flag** — the model is told
   "this value was filled in, not originally present."
3. **All imputation statistics are computed from the training set only.**
   The test set's median/mode is never used to impute anything.

> **Note:** `founding_year` was imputed internally (missing filled with
> the training-set median of 2007) during the pipeline for data quality
> consistency, but the column was then excluded from the final feature
> matrix. The imputation is no longer relevant to the output files.

### 8.2 `founding_year` imputation (internal only — column excluded)

| Property | Value |
|----------|-------|
| Training-set median founding year | **2007** |
| Rows that would have been imputed in train | 2,996 (28.1%) |
| Strategy | Training-set median |
| Status | **Column excluded from feature matrix — imputation not used** |

This imputation was performed internally for pipeline consistency but
the `founding_year` column was not written to the output CSV files.

### 8.3 `category_list` missing values

- Missing `category_list` filled with the string `"Unknown"` before
  encoding.
- This creates a `cat_Unknown` binary column in the feature matrix.
  `cat_Unknown = 1` means "this company had no recorded category."
- Training set: 858 rows with `category_list = NaN` → filled "Unknown".
  `Unknown` appeared 858 times and ranked 4th in the top-50 list,
  meaning "no category" is itself a statistically meaningful group.

### 8.4 `country_code` missing values

- Missing `country_code` filled with the string `"Unknown"` before
  encoding.
- `Unknown` appeared 1,582 times in the training set and ranked 2nd
  among country codes, confirming that geographic anonymity is a
  meaningful cluster.
- Creates `country_Unknown = 1` binary column.

### 8.5 Geography flags with missing source columns

For `is_sf_bay_area` and `is_california`: when the source column
(`region` or `state_code`) is missing, the flag defaults to 0.
This is correct: if we do not know where the company is, we cannot
assert it is in SF Bay Area or California.

### 8.6 No remaining NaN values

After all imputation steps, the final feature matrices contain
**zero NaN values** in both train and test sets, verified by the
leakage audit step in the script.

---

## 9. Categorical Encoding

### 9.1 `category_list` — Multi-hot encoding

**Problem:** The raw `category_list` uses pipe-separated text:
`"Mobile|Apps|Games"`. This cannot be used directly.

**Method:** Multi-hot (binary) encoding for the top 50 categories.

**Step-by-step:**
1. Fill missing values with "Unknown"
2. Split each value on `|` to get individual category tags
3. Count all individual tags across the **training set only**
4. Select the top 50 most frequent tags
5. For each of the 50 tags, create one binary column (`cat_TagName`)
   — value 1 if the company has that tag, 0 otherwise
6. Create `category_other = 1` if the company has any tag not in the
   top 50 (ensures companies with rare categories are not entirely
   unrepresented)
7. Create `n_categories` = count of tags per company (numeric)

**Why top 50 and not all 714?**
Using all 714 training-set categories would create 714 sparse binary
columns. Most would have far fewer than 100 positive examples, making
them statistically unreliable. The top 50 each appear at least 161
times in the training set — enough for the model to learn from.

**Top 10 categories by training-set frequency:**

| Rank | Category | Training count |
|-----:|----------|---------------:|
| 1 | Software | 1,493 |
| 2 | Biotechnology | 924 |
| 3 | Mobile | 883 |
| 4 | Unknown (no category) | 858 |
| 5 | Curated Web | 776 |
| 6 | E-Commerce | 534 |
| 7 | Social Media | 499 |
| 8 | Advertising | 488 |
| 9 | Enterprise Software | 482 |
| 10 | Games | 402 |

**Critical rule:** Category frequencies were computed from the training
set only. The test set's category distribution was never examined before
encoding decisions were made.

### 9.2 `country_code` — Binary (one-hot style) encoding

**Problem:** 137+ unique country codes; cannot use as raw text or ordinal.

**Why not ordinal encoding?** Ordinal encoding (USA=1, GBR=2, etc.)
implies an ordering — that GBR is numerically "greater than" USA. For
country names, this ordering is meaningless and would mislead the model.

**Method:**
1. Fill missing with "Unknown"
2. Count country frequencies in the **training set only**
3. Select top 10 countries: USA, Unknown, GBR, CAN, ISR, DEU, FRA,
   RUS, CHN, IND
4. Create one binary column per country: `country_USA`, `country_GBR`, etc.
5. Create `country_other = 1` for any country not in the top 10

This reduces 137+ text values to 11 binary columns.

**Top 10 countries by training-set frequency:**

| Rank | Country | Training count |
|-----:|---------|---------------:|
| 1 | USA | 6,533 |
| 2 | Unknown (missing) | 1,582 |
| 3 | GBR | 440 |
| 4 | CAN | 330 |
| 5 | ISR | 164 |
| 6 | DEU | 152 |
| 7 | FRA | 147 |
| 8 | RUS | 140 |
| 9 | CHN | 136 |
| 10 | IND | 109 |

> **Note on "Unknown" as 2nd most frequent country:** This reflects the
> 14.93% missing rate in `country_code` for the training-eligible rows.
> Rather than silently imputing these as "USA" or dropping the rows, we
> preserve them as a distinct group. The model can learn whether
> "unknown geography" correlates with outcome.

---

## 10. Leakage Prevention

### 10.1 The critical ordering rule

```
Step 1  →  Create target from status
Step 2  →  Split into train and test  ← ALL statistics computed AFTER this
Step 3  →  Compute medians, category lists, country lists on TRAIN only
Step 4  →  Apply those statistics to transform BOTH train and test
```

If Step 3 happened before Step 2, the test set's distribution would
have influenced the preprocessing — this is **preprocessing leakage**
and would make the model appear better than it is on truly new data.

### 10.2 Statistics computed from training set only

| Statistic | Value | Used for |
|-----------|-------|----------|
| `founding_year` median | 2007 | Imputing missing founding years |
| Top 50 category list | See §9.1 | Deciding which categories to encode |
| Top 10 country list | See §9.2 | Deciding which countries to encode |

### 10.3 Forbidden columns — verified absent (revised list)

The leakage audit step (Step 10 of the script) checks 13 forbidden columns
and calls `sys.exit(1)` if any match is found. The list was expanded in the
Phase 2 review to include the three newly removed features.

| Forbidden column | Status | Reason |
|-----------------|--------|--------|
| `status` | ✅ ABSENT | Is the target — never a feature |
| `last_funding_at` | ✅ ABSENT | Critical temporal leakage |
| `funding_total_usd` | ✅ ABSENT | Post-prediction lifetime total |
| `funding_rounds` | ✅ ABSENT | Post-prediction lifetime count |
| `first_funding_at` | ✅ ABSENT | Future event from Day 0 |
| `permalink` | ✅ ABSENT | Identifier — no signal |
| `name` | ✅ ABSENT | Identifier — no signal |
| `homepage_url` | ✅ ABSENT | Metadata — no signal |
| `city` | ✅ ABSENT | 5,111 unique values — too sparse |
| `target` | ✅ ABSENT | In labels file, not features file |
| `founding_year` | ✅ ABSENT | **Phase 2 review: temporal mismatch** |
| `has_founding_date` | ✅ ABSENT | **Phase 2 review: always 1 at inference** |
| `is_usa` | ✅ ABSENT | **Phase 2 review: redundant with country_USA** |

---

## 11. Final Feature Matrix

### 11.1 Dimensions

| Matrix | Rows | Features |
|--------|-----:|--------:|
| `train_features.csv` | **10,667** | **65** |
| `test_features.csv` | **2,667** | **65** |

### 11.2 Feature groups

| Group | Count | Column prefix / names |
|-------|------:|----------------------|
| Geography flags | 2 | `is_sf_bay_area`, `is_california` |
| Country flags | 11 | `country_USA` … `country_IND`, `country_other` |
| Year features | 0 | *(founding_year, has_founding_date excluded — see §16)* |
| Category features | 52 | `cat_Software` … `cat_FinTech`, `category_other`, `n_categories` |
| **Total** | **65** | |

### 11.3 All 65 feature names (in order)

```
 1. is_sf_bay_area                34. cat_Semiconductors
 2. is_california                 35. cat_SaaS
 3. country_USA                   36. cat_Health_and_Wellness
 4. country_Unknown               37. cat_Web_Hosting
 5. country_GBR                   38. cat_Networking
 6. country_CAN                   39. cat_Video
 7. country_ISR                   40. cat_Social_Network_Media
 8. country_DEU                   41. cat_Search
 9. country_FRA                   42. cat_Manufacturing
10. country_RUS                   43. cat_Music
11. country_CHN                   44. cat_Education
12. country_IND                   45. cat_Messaging
13. country_other                 46. cat_Cloud_Computing
14. cat_Software                  47. cat_Medical
15. cat_Biotechnology             48. cat_News
16. cat_Mobile                    49. cat_Travel
17. cat_Unknown                   50. cat_Information_Technology
18. cat_Curated_Web               51. cat_Web_Development
19. cat_E-Commerce                52. cat_Telecommunications
20. cat_Social_Media              53. cat_Business_Services
21. cat_Advertising               54. cat_Media
22. cat_Enterprise_Software       55. cat_Sales_and_Marketing
23. cat_Games                     56. cat_Entertainment
24. cat_Health_Care               57. cat_iPhone
25. cat_Services                  58. cat_Design
26. cat_Internet                  59. cat_Fashion
27. cat_Finance                   60. cat_Consulting
28. cat_Technology                61. cat_Marketplaces
29. cat_Analytics                 62. cat_Android
30. cat_Hardware_plus_Software    63. cat_FinTech
31. cat_Security                  64. category_other
32. cat_Clean_Technology          65. n_categories
33. cat_Apps
```

### 11.4 Data types

All 65 features are numeric:
- `is_*`, `country_*`, `cat_*`, `category_other`: binary integer (0 or 1)
- `n_categories`: integer (0–17 in training set)

Zero NaN values in either matrix.

---

## 12. Final Row and Target Counts

| Metric | Value |
|--------|------:|
| Raw CSV rows | 66,368 |
| Labeled rows (known outcome) | 13,334 |
| Operating rows (excluded from training) | 53,034 |
| Other / NaN status (excluded) | 0 |
| **Training rows** | **10,667** |
| **Test rows** | **2,667** |
| Train — class 1 (success) | 5,677 (53.2%) |
| Train — class 0 (failure) | 4,990 (46.8%) |
| Test — class 1 (success) | 1,419 (53.2%) |
| Test — class 0 (failure) | 1,248 (46.8%) |
| **Total features** | **65** |

Class ratio (success : failure) = **1.14 : 1** — near-balanced.
No resampling (SMOTE, class weighting) is required, though using
`class_weight='balanced'` in the model trainer remains an option.

---

## 13. Rows Removed and Why

| Category | Rows removed | Reason |
|----------|-------------:|--------|
| `operating` status | 53,034 | Outcome unknown — cannot be labeled |
| NaN / other status | 0 | None found |
| Duplicate rows (exact) | 0 | None found in Phase 1 EDA |
| Individual date typos | 4 | Dates set to NaT (row kept) |
| **Rows deleted entirely** | **0** | No row deletions were made |

**No rows were deleted from the labeled dataset.** The only rows excluded
were the 53,034 `operating` companies, which were excluded because they
have no valid training label — not because of data quality issues.

All 13,334 labeled rows were used in training or testing.

---

## 14. Output Files

All files saved to `processed_data/`:

| File | Description | Size |
|------|-------------|-----:|
| `train_features.csv` | 10,667 × 68 feature matrix for training | 1,459 KB |
| `train_labels.csv` | 10,667 target values (0/1) | 31 KB |
| `test_features.csv` | 2,667 × 68 feature matrix for testing | 366 KB |
| `test_labels.csv` | 2,667 target values (0/1) | 8 KB |
| `feature_names.json` | Ordered list of all 68 feature names | 1 KB |
| `preprocessing_metadata.json` | All statistics computed from training set; must accompany the model at inference time | 4 KB |
| `preprocessing_log.txt` | Full human-readable execution log | 10 KB |

### 14.1 Why `preprocessing_metadata.json` is critical for inference

When a new startup idea is submitted to the application, the same
transformations that were applied during training must be applied to
the idea's structured fields. The metadata file stores:
- The training-set median founding year (2007) for imputing missing years
- The exact top-50 category list (in order) for encoding the user's sector
- The exact top-10 country list for encoding the user's country
- The exact feature column order (must match `feature_names.json`)

If these values are recomputed from scratch at inference time, the
encoding will differ from what the model was trained on, producing
wrong predictions.

---

## 15. Reproducibility

To reproduce this preprocessing exactly:

```bash
python 02_preprocess_data.py
```

All defaults match the values used in this run:
- `--seed 42`
- `--test-size 0.20`
- `--top-cats 50`
- `--top-countries 10`

To override any parameter:
```bash
python 02_preprocess_data.py --seed 42 --test-size 0.20 --top-cats 50
```

The script will always check the forbidden-column list and exit with
code 1 if any leakage violation is detected, making it safe to re-run
after any future modifications.

---

*This report was generated after verified execution of `02_preprocess_data.py`.*
*All statistics are real values from the actual dataset.*
*No model has been trained. Phase 3 / model training must not begin*
*until this report has been reviewed and approved.*

---

## 16. Why `founding_year` Was Removed

> This section was added following the Phase 2 review. It explains the
> removal of `founding_year`, `has_founding_date`, and `is_usa` in plain
> language.

### 16.1 The temporal distribution problem with `founding_year`

**What the training data looks like:**

The historical dataset contains companies whose founding years span
approximately 1990 to 2016. The maximum founding year in the training
set is **2016**. The median founding year is **2007**.

```
Training data founding year distribution:
  Earliest : ~1990s (peak activity)
  Median   : 2007
  Maximum  : 2016
  Companies founded after 2016 in training data: ZERO
```

**What happens at inference time:**

Our application evaluates startup ideas submitted in **2026** (and beyond).
A user submitting an idea today would have `founding_year = 2026`.

This means the model would receive a `founding_year` value of 2026 during
inference — a value that is **10 years beyond the maximum value it ever
saw during training** (2016).

**Why this is a problem:**

Machine learning models learn from patterns in training data. When a model
encounters an input value it has never seen before — especially one far
outside the range of its training data — it is forced to *extrapolate*.
Extrapolation means the model is making up an answer based on trends it
observed, without any actual training examples to support the prediction
at that point.

Think of it this way: if you learned to predict house prices only from
houses built between 1900 and 2016, and someone asked you to predict the
price of a house built in 2030, you would have no real evidence to work
from. You might extend the trend from 2010–2016, but that extension
carries no guarantee of being correct.

In our case, the startup ecosystem in 2026 is structurally different from
2016 in ways the training data cannot capture: different dominant sectors,
different VC dynamics, different exit markets. The model cannot know this
because none of it appeared during training.

**The decision:**

`founding_year` was excluded from the primary ML feature matrix because:
1. It would require extrapolation 10+ years beyond the training range
   for every new idea submitted in 2026 or later.
2. The model has no training examples from 2017–2026 to anchor
   predictions for modern ideas.
3. Including it would give the model a feature it cannot interpret
   reliably at inference time.

**What this does NOT mean:**

Removing `founding_year` does not guarantee the model performs better
or worse. That can only be determined by testing. What it does guarantee
is that the model is not asked to extrapolate from a feature that is
systematically out-of-distribution for every single new idea submission.

This is a data integrity decision, not a performance claim.

---

### 16.2 Why `has_founding_date` was also removed

`has_founding_date` is a binary flag that equals 1 if the company had a
recorded founding date in the historical Crunchbase dataset, and 0 if
the founding date was missing.

In the training data, 28.1% of companies had no founding date (a data
recording gap in Crunchbase for historical companies).

**At inference time:** A user submitting a new startup idea is founding
that company right now — at the moment of submission. They know their
founding date because it is today. Therefore `has_founding_date` would
always equal **1** for every single new idea submitted through the
application.

A feature that is always the same value (always 1) provides **zero
discriminatory information** to the model. The model cannot use a constant
to distinguish one startup from another. Including it would occupy a
feature slot without contributing anything.

The historical 28.1% missingness in Crunchbase data reflects incomplete
record-keeping in a database compiled from web sources — it does not
reflect a real-world pattern that will appear in our application's users.

---

### 16.3 Why `is_usa` was also removed

`is_usa` equals 1 if the company is headquartered in the United States.

`country_USA` — produced by the country encoding step — also equals 1 if
the company is headquartered in the United States.

They are **identical columns** in the feature matrix. Keeping both would
mean the model has two perfectly correlated inputs for the same information:

- In linear models, perfect correlation between features causes numerical
  instability (the model cannot determine how to split the coefficient
  between the two columns).
- In tree-based models, one column would simply be ignored, wasting a
  split opportunity.

`country_USA` is retained because it is part of the systematic,
consistently-produced country encoding group. `is_usa` is removed as the
redundant duplicate.

---

*Section 16 added following Phase 2 review on 19 September 2026.*
*No model has been trained. No performance impact is claimed.*
