# Data Quality Report
## AI-Based Startup Viability Prediction and Validation System
### Dataset: `big_startup_secsees_dataset.csv`

---

> **Important:** Every number in this report was produced by running
> `01_explore_dataset.py` against the actual CSV file on **19 September 2026**.
> No values are estimated or fabricated. The original CSV was never modified.
> Reproducible: re-run `python 01_explore_dataset.py` to regenerate all figures.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Target Variable Distribution](#2-target-variable-distribution)
3. [Missing Values](#3-missing-values)
4. [Duplicate Records](#4-duplicate-records)
5. [Funding Column Quality](#5-funding-column-quality)
6. [Date Column Quality](#6-date-column-quality)
7. [Category Column Quality](#7-category-column-quality)
8. [Geographic Column Quality](#8-geographic-column-quality)
9. [Data Leakage Assessment](#9-data-leakage-assessment)
10. [Recommended Preprocessing Steps](#10-recommended-preprocessing-steps)
11. [Phase 2 Readiness Summary](#11-phase-2-readiness-summary)

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| File | `big_startup_secsees_dataset.csv` |
| Total rows | **66,368** |
| Total columns | **14** |
| File size | ~10.44 MB |
| Encoding | UTF-8 |
| Row identifier | `permalink` (globally unique) |

### All 14 Columns

| # | Column | Pandas dtype | Role |
|---|--------|-------------|------|
| 1 | `permalink` | str | Identifier |
| 2 | `name` | str | Label |
| 3 | `homepage_url` | str | Metadata |
| 4 | `category_list` | str | Feature (needs transform) |
| 5 | `funding_total_usd` | str → float | Feature (leakage risk) |
| 6 | `status` | str | **Target** |
| 7 | `country_code` | str | Feature |
| 8 | `state_code` | str | Feature (conditional) |
| 9 | `region` | str | Feature (conditional) |
| 10 | `city` | str | Exclude / group |
| 11 | `funding_rounds` | int64 | Feature (leakage risk) |
| 12 | `founded_at` | str → datetime | Feature (engineered) |
| 13 | `first_funding_at` | str → datetime | Feature (engineered) |
| 14 | `last_funding_at` | str → datetime | Exclude (leakage) |

> **Note on dtypes:** pandas automatically inferred `funding_rounds` as
> `int64` because it contained only integers. All other columns loaded as
> `str` (string/object) because they contain text or mixed content.
> Date columns must be explicitly converted with `pd.to_datetime()`.

---

## 2. Target Variable Distribution

The `status` column is the source of our prediction target. It has **four
distinct values** in this dataset.

### 2.1 Raw Status Counts (full dataset, N = 66,368)

| Status | Count | Percentage | Role in this project |
|--------|------:|----------:|----------------------|
| `operating` | 53,034 | **79.91%** | ⛔ Excluded — unknown outcome |
| `closed` | 6,238 | 9.40% | ✅ Label = 0 (FAILURE) |
| `acquired` | 5,549 | 8.36% | ✅ Label = 1 (SUCCESS) |
| `ipo` | 1,547 | 2.33% | ✅ Label = 1 (SUCCESS) |
| Missing (NaN) | 0 | 0.00% | ⛔ Excluded |

**The `status` column has zero missing values.** Every row has a recorded
status — this is unusually clean for a real-world dataset.

### 2.2 Binary Target (training-eligible rows only)

After excluding `operating` companies, the training dataset has:

| Label | Meaning | Count | Percentage |
|-------|---------|------:|----------:|
| **0** | FAILURE (closed) | 6,238 | **46.78%** |
| **1** | SUCCESS (acquired + ipo) | 7,096 | **53.22%** |
| — | **Total training rows** | **13,334** | 100% |
| — | Operating (excluded) | 53,034 | — |

### 2.3 Class Balance Assessment

**Class ratio (failure : success) = 0.88 : 1**

This is a **near-balanced** dataset for the training subset. The two
classes are close in size, which is fortunate. Most real-world binary
classification datasets have severe imbalance (10:1 or worse).

> **Important context:** The apparent balance here is partly a consequence
> of our exclusion rule. `operating` companies (the vast majority of the
> dataset) are excluded because their outcome is unknown — not because
> they are "neutral." If `operating` companies were naively labeled as
> success, the dataset would be 87% success, which would be misleading.

**Action required:** No emergency class-balancing techniques (SMOTE etc.)
are required given the 0.88:1 ratio. However, stratified train/test splits
should still be used during model training to preserve this balance in
both the training and validation sets.

---

## 3. Missing Values

### 3.1 Full Dataset (all 66,368 rows)

| Column | Missing | % Missing | Severity | Notes |
|--------|--------:|----------:|----------|-------|
| `permalink` | 0 | 0.00% | ✅ OK | Perfect — use as row ID |
| `name` | 1 | 0.00% | ✅ OK | One nameless company |
| `homepage_url` | 5,058 | 7.62% | ⚠️ Moderate | Not used as ML feature |
| `category_list` | 3,148 | 4.74% | ⚠️ Low | Impute with "Unknown" category |
| `funding_total_usd` | 0 | 0.00% | ✅ OK | But 12,785 are non-numeric dashes — see §5 |
| `status` | 0 | 0.00% | ✅ OK | Perfect |
| `country_code` | 6,958 | 10.48% | ⚠️ Moderate | Fill with "Unknown" |
| `state_code` | 8,547 | 12.88% | ⚠️ Moderate | Expected — non-US companies |
| `region` | 8,030 | 12.10% | ⚠️ Moderate | Expected — non-US companies |
| `city` | 8,028 | 12.10% | ⚠️ Moderate | Not used as ML feature |
| `funding_rounds` | 0 | 0.00% | ✅ OK | Perfect |
| `founded_at` | 15,221 | **22.93%** | 🔴 High | Largest missing gap |
| `first_funding_at` | 24 | 0.04% | ✅ OK | Very clean |
| `last_funding_at` | 0 | 0.00% | ✅ OK | Perfect (but excluded due to leakage) |

### 3.2 Training Dataset Only (13,334 rows — known outcomes)

| Column | Missing | % Missing | Severity |
|--------|--------:|----------:|----------|
| `permalink` | 0 | 0.00% | ✅ OK |
| `name` | 1 | 0.01% | ✅ OK |
| `homepage_url` | 1,217 | 9.13% | ⚠️ Moderate |
| `category_list` | 1,086 | 8.14% | ⚠️ Moderate |
| `funding_total_usd` | 0 | 0.00% | ✅ OK |
| `status` | 0 | 0.00% | ✅ OK |
| `country_code` | 1,991 | 14.93% | ⚠️ Moderate |
| `state_code` | 2,209 | 16.57% | ⚠️ Moderate |
| `region` | 2,152 | 16.14% | ⚠️ Moderate |
| `city` | 2,152 | 16.14% | ⚠️ Moderate |
| `funding_rounds` | 0 | 0.00% | ✅ OK |
| `founded_at` | 3,732 | **27.99%** | 🔴 High |
| `first_funding_at` | 2 | 0.01% | ✅ OK |
| `last_funding_at` | 0 | 0.00% | ✅ OK |
| `target` (derived) | 0 | 0.00% | ✅ OK |

### 3.3 Key Finding: `founded_at` is the Most Problematic Column

- **22.93%** of all rows (15,221 companies) have no founding date.
- In the training set, this rises to **27.99%** (3,732 companies).
- This is the single largest missingness problem in the dataset.
- It directly affects our ability to compute `time_to_first_funding`,
  which is one of the most valuable engineered features.

**Recommended handling (Phase 2):**
1. Add a binary flag `has_founding_date` (1 = date present, 0 = missing).
2. For companies missing `founded_at`, `time_to_first_funding` cannot be
   computed — impute with the median time-to-funding across the dataset,
   but flag these rows clearly.
3. Do NOT drop these rows — they represent 28% of training data.

---

## 4. Duplicate Records

### 4.1 Exact Duplicate Rows

| Check | Count |
|-------|------:|
| Fully identical rows (all 14 columns match) | **0** |

**No exact duplicates exist.** Every row is unique when all 14 columns
are considered together. This is a clean result — no immediate deduplication
is needed.

### 4.2 Duplicate Company Names

| Check | Count |
|-------|------:|
| Company names appearing more than once | 239 |
| Total rows involved in those duplicates | 504 |
| Maximum occurrences of a single name | 4 |

**Top repeated names (sample):**

| Company Name | Occurrences |
|-------------|------------:|
| Blink | 4 |
| Karma | 4 |
| Spire | 4 |
| Roost | 4 |
| Peach | 4 |
| Bundle | 3 |
| Gather | 3 |
| Chime | 3 |

### 4.3 Permalink Uniqueness

| Check | Count |
|-------|------:|
| Duplicate permalink values | **0** |

**All 66,368 permalinks are unique.** This confirms that every row
represents a genuinely distinct company in the database.

### 4.4 Interpretation

The 239 "duplicate names" are **different companies with the same name**
— not data entry errors. Proof: all have different `permalink` values.
This is common in startup data (e.g., two startups both named "Blink"
operating in different countries or years).

**Decision:** Do NOT auto-delete duplicate names. Use `permalink` as the
definitive unique row identifier throughout all preprocessing steps.

---

## 5. Funding Column Quality

### 5.1 `funding_total_usd`

This column was loaded as a string because it contains a mix of numeric
values and literal dash characters (`-`).

| Metric | Value |
|--------|-------|
| Originally missing (blank) | 0 |
| Non-numeric entries (literal `-`) | **12,785** |
| Total NaN after numeric conversion | 12,785 |
| % of rows affected | 19.26% |
| Valid numeric values | 53,583 |

**The dash `-` is the dataset's way of encoding "funding amount unknown."**
It is not a typo — it appears exactly 12,785 times and in no other form.

**Descriptive statistics (valid numeric values only):**

| Statistic | Value |
|-----------|------:|
| Count (non-null) | 53,583 |
| Mean | $18,478,604 |
| Standard deviation | $188,013,273 |
| Minimum | $1 |
| 25th percentile | $335,624 |
| **Median** | **$2,000,000** |
| 75th percentile | $10,000,000 |
| **Maximum** | **$30,079,503,000** |

The mean ($18.5M) is nearly **9× larger** than the median ($2M). This
extreme gap means the distribution is severely right-skewed — a small
number of very large companies (Verizon, Uber, Alibaba) pull the average
far above the typical startup's funding.

### 5.2 Outlier Analysis (IQR Method)

| Metric | Value |
|--------|-------|
| Q1 (25th percentile) | $335,624 |
| Q3 (75th percentile) | $10,000,000 |
| IQR | $9,664,377 |
| Upper outlier fence | $24,496,565 |
| **Outlier count** | **7,384** |
| % of all rows | 11.13% |

**Top 5 highest funding values:**

| Company | Total Funding |
|---------|-------------:|
| Verizon Communications | $30,079,503,000 |
| Freescale Semiconductor | $17,600,000,000 |
| Cambridge Technology Group | $10,664,943,639 |
| Uber | $8,207,450,000 |
| Clearwire | $5,820,000,000 |

These values are **real** — they are not data entry errors. Multi-billion
dollar companies exist in this dataset alongside seed-stage startups. The
raw funding values span **7 orders of magnitude** ($1 to $30 billion).

**Action required:** Apply **log10 transformation** — `log10(x + 1)` —
before using funding as an ML feature. This compresses the scale so that
Uber's $8B does not overwhelm a $500K startup in the model's calculations.
Do NOT remove these high-value companies as outliers — they are genuine data.

### 5.3 `funding_rounds`

| Metric | Value |
|--------|-------|
| Missing | 0 |
| Non-numeric entries | 0 |
| Minimum | 1 |
| Maximum | 19 |
| Median | 1 |
| Mean | 1.73 |

**Distribution:** Heavily right-skewed. The vast majority of companies
(42,472 = 64%) have only 1 funding round. Only 7 companies have 17+ rounds.
This column is clean — no conversion or imputation needed. It is already
numeric with no missing values.

---

## 6. Date Column Quality

All three date columns are stored as strings in the raw CSV and require
`pd.to_datetime()` conversion.

### 6.1 Summary Table

| Column | Missing | Parse Failures | Future Dates (>2023) | Pre-1800 Dates | Date Range |
|--------|--------:|---------------:|---------------------:|---------------:|-----------|
| `founded_at` | 15,221 | 0 | **4** | 8 | 1015-01-30 → **2914-01-01** |
| `first_funding_at` | 24 | 0 | 0 | 3 | 1014-03-01 → 2015-12-05 |
| `last_funding_at` | 0 | 0 | **2** | 0 | 1971-01-01 → **2115-12-01** |

**Key observation:** No parse failures — every string that exists can be
parsed as a date. The quality problem is not format corruption; it is
**wrong year values** (typos like 2914 instead of 2014).

### 6.2 Suspicious Future Dates

These are almost certainly **data entry typos** — the year digit was
mistyped. They affect a tiny number of rows but must be handled explicitly.

**`founded_at` — 4 future dates:**

| Company | Raw Date | Likely Correct Year |
|---------|----------|---------------------|
| DugoutFC Ltd | 2105-02-17 | 2015 |
| LIVAMP | 2041-09-21 | 2014 |
| rent2cash.com | **2914-01-01** | 2014 |
| Soply | 2104-01-01 | 2014 |

**`last_funding_at` — 2 future dates:**

| Company | Raw Date | Likely Correct Year |
|---------|----------|---------------------|
| Rasyonel R&D | 2105-05-01 | 2015 |
| Wigzo Technologies | **2115-12-01** | 2015 |

**Total rows with suspicious future dates: 6** (across all date columns)

### 6.3 Suspicious Pre-1800 Dates

`founded_at`: 8 dates before year 1800. The earliest recorded date is
`1015-01-30` — over 1,000 years ago. These are certainly typos
(e.g., `1015` instead of `2015`).

**`first_funding_at`: 3 pre-1800 dates.** The column's overall earliest
date of `1014-03-01` is clearly a century-transposition typo.

**Total rows with pre-1800 dates: 11** (across all date columns)

### 6.4 Date Column Recommendations

| Situation | Recommended Action | Rationale |
|-----------|-------------------|-----------|
| Future year > 2023 | Set to NaN | Typo; cannot safely guess correct year |
| Pre-1800 year | Set to NaN | Typo; cannot safely guess correct year |
| Missing `founded_at` | Add `has_founding_date=0` flag | Do not drop; impute derived features |
| Missing `first_funding_at` | Add `has_received_funding=0` flag | Means bootstrapped or no recorded funding |
| Negative `time_to_first_funding` | Set to NaN | first_funding_at < founded_at = data entry error |

**Why set to NaN instead of correcting?**  
We cannot reliably "guess" that 2914 means 2014 vs. 1914 vs. 2024 — all
are plausible typos. Setting to NaN and flagging is the only defensible
approach for an academic project.

**Total rows affected by date quality issues: ~17 rows** (0.03% of dataset).
Impact is negligible in volume but must be handled to avoid extreme outliers
in derived date features.

---

## 7. Category Column Quality

### 7.1 Summary

| Metric | Value |
|--------|------:|
| Rows with at least one category | 63,220 |
| Rows with no category (missing) | 3,148 (4.74%) |
| Total category mentions (with repeats) | 152,658 |
| **Unique individual categories** | **858** |
| Min categories per company | 1 |
| Max categories per company | 44 |
| Median categories per company | 2 |
| Mean categories per company | 2.41 |

### 7.2 Top 10 Categories

| Category | Count | % of Categorized Rows |
|----------|------:|----------------------:|
| Software | 8,768 | 13.9% |
| Mobile | 5,557 | 8.8% |
| Biotechnology | 4,562 | 7.2% |
| E-Commerce | 4,152 | 6.6% |
| Curated Web | 3,030 | 4.8% |
| Social Media | 2,897 | 4.6% |
| Enterprise Software | 2,694 | 4.3% |
| Advertising | 2,470 | 3.9% |
| Health Care | 2,385 | 3.8% |
| Games | 2,284 | 3.6% |

### 7.3 Quality Issues

1. **858 unique categories is too many for direct encoding.** Using all
   858 as multi-hot columns would create a very wide, sparse feature matrix.
   Only the top 30–50 categories appear with enough frequency to be
   statistically reliable.

2. **Overlap between similar categories** — "Software", "Enterprise
   Software", "SaaS", and "Hardware + Software" all relate to software
   products. Some companies are tagged in all four. Grouping these into
   macro-sectors could reduce noise.

3. **Maximum 44 categories per company** is suspicious — a company tagged
   in 44 different sectors is likely over-categorised in the source data.

**Recommended action:** Multi-hot encode top 30–50 categories only. Add
`n_categories` as a numeric feature. Create an `"Other"` bucket for all
remaining categories that don't fall in the top 50.

---

## 8. Geographic Column Quality

### 8.1 Country Distribution

| Metric | Value |
|--------|-------|
| Unique country codes | 137 |
| Missing country_code | 6,958 (10.48%) |
| USA companies | 37,601 (**56.7%** of all rows) |
| Top 5 countries | USA, GBR, CAN, IND, CHN |

**The dataset is heavily USA-biased.** More than half of all companies
are US-based. This means:
- A global model will be dominated by US patterns
- Geographic features will mostly encode "is this a US company?"
- Predictions for non-US startups may be less reliable

### 8.2 State and Region Quality

| Column | Missing | Unique Values | Issue |
|--------|--------:|-------------:|-------|
| `state_code` | 8,547 (12.9%) | 311 | Contains non-US codes (e.g. `H9`, `7`, `2`) |
| `region` | 8,030 (12.1%) | 1,092 | High cardinality |
| `city` | 8,028 (12.1%) | **5,111** | Far too high for direct ML use |

**Anomaly in `state_code`:** Values like `H9`, `7`, and `2` appear among
the top 10 state codes. These are international regional codes mistakenly
stored in the same column as US state abbreviations (CA, NY, TX). This
column mixes two different coding systems.

**`city` has 5,111 unique values** — direct encoding would create 5,000+
sparse binary columns. This column should be excluded from the feature set
or reduced to a small set of hub-city binary flags (San Francisco, New York,
London, Beijing, Boston).

### 8.3 Geographic Recommendations

| Column | Action |
|--------|--------|
| `country_code` | Encode top 10 countries + `"Other"` + `"Unknown"` for missing |
| `state_code` | Derive `is_california` flag for US companies only; drop raw value |
| `region` | Derive `is_sf_bay_area` binary flag; drop raw value |
| `city` | Exclude from feature set |
| All geo columns | Fill missing with string `"Unknown"` before encoding |

---

## 9. Data Leakage Assessment

Data leakage is the single most important quality concern for this project.
It occurs when the model is trained on information that would not be
available at the time a real prediction is made.

### 9.1 Leakage Risk by Column

| Column | Leakage Risk | Reason | Recommendation |
|--------|:-----------:|--------|----------------|
| `last_funding_at` | 🔴 **CRITICAL** | Date of most recent funding. For companies about to be acquired, this date is close to the acquisition — essentially a "pre-acquisition signal." The model would learn to look at recency of funding as a proxy for imminent success/failure. | **Exclude entirely** |
| `funding_total_usd` | 🔴 **HIGH** | Reflects lifetime total funding, including rounds raised during the exit/closure process. A company acquired for $1B often had a large late-stage round just before the deal. | Use with documented caveat; apply log transform; acknowledge in report |
| `funding_rounds` | 🔴 **HIGH** | Total rounds includes all rounds ever, including those close to the outcome event. More rounds ≠ early signal; it may mean "this company was well-funded right up to its acquisition." | Use with documented caveat; acknowledge in report |
| `first_funding_at` | 🟡 **LOW-MODERATE** | Early in the company's life; available at prediction time. The derived feature `time_to_first_funding` is the most defensible use of this column. | Use as derived `time_to_first_funding` (days) |
| `founded_at` | 🟢 **LOW** | Founding predates all outcomes by definition. `founding_year` and `company_age` derived from it are safe. | Use as derived features |
| `category_list` | 🟢 **LOW** | Industry sector is fixed at founding; does not change based on outcome. | Safe to use after encoding |
| `country_code` | 🟢 **LOW** | Headquarters location is set at founding. | Safe to use after encoding |
| `status` | — | This IS the target. Using it as a feature would be trivial leakage. | Never use as a feature |

### 9.2 The Core Leakage Problem for this Dataset

This dataset records **snapshot values** — each number reflects the state
of the company at the time the data was collected, not at a specific early
point in the company's life. There is no time-series breakdown of "how much
had been raised by Year 1" vs. "how much total."

This means **perfect leakage elimination is not possible** with this dataset
in its current form. The academically honest approach is to:

1. Use the features listed above with their documented caveats
2. Acknowledge the limitation explicitly in the project report
3. Define a clear prediction point and justify which features are
   "close enough" to that point to be defensible

### 9.3 Proposed Prediction Point

> **"12–24 months after the company's first funding round."**

At this point, the following information is realistically available:
- Where the company is headquartered (`country_code`, `region`)
- What sector it operates in (`category_list`)
- When it was founded (`founded_at` → `founding_year`)
- How quickly it raised its first round (`time_to_first_funding`)
- The approximate size of that first round (proxy from `funding_total_usd`
  — with the documented caveat that this is lifetime total, not round 1 total)

---

## 10. Recommended Preprocessing Steps

These are **recommendations for Phase 2**. No preprocessing is performed
in this Phase 1 report. Each step is explained with its justification.

### Step 1 — Split dataset by known vs. unknown outcome

```
Training candidates : status in ['acquired', 'ipo', 'closed'] → 13,334 rows
Unknown outcome     : status == 'operating'                   → 53,034 rows
```

Reason: `operating` companies cannot be labeled; mixing them in would
corrupt training targets.

### Step 2 — Create binary target column

```python
target = 1  if status in ['acquired', 'ipo']
target = 0  if status == 'closed'
```

### Step 3 — Handle invalid dates

For each of `founded_at`, `first_funding_at`, `last_funding_at`:
- Parse with `pd.to_datetime(..., errors='coerce')`
- Set to NaN any date where year > 2023 (4 + 2 rows affected)
- Set to NaN any date where year < 1800 (8 + 3 rows affected)

### Step 4 — Engineer date features

```
founding_year          = founded_at.dt.year
time_to_first_funding  = (first_funding_at - founded_at).days
has_founding_date      = 1 if founded_at is not NaN, else 0
has_received_funding   = 1 if first_funding_at is not NaN, else 0
```

For negative `time_to_first_funding` values (where first_funding_at
< founded_at): set to NaN — these are data entry errors.

### Step 5 — Fix `funding_total_usd`

```python
funding_numeric = pd.to_numeric(funding_total_usd, errors='coerce')
# '-' values become NaN (12,785 rows)
# Missing NaN values: fill with 0 (no recorded funding)
# Then: log_funding = log10(funding_numeric + 1)
```

Reason for filling with 0: a missing funding value most likely means the
company has not raised external funding — not that the data is absent.
Document this assumption clearly.

### Step 6 — Process `category_list`

```
1. Fill missing category_list with "Unknown"
2. Split each value on '|'
3. Identify top 30–50 most frequent categories
4. Multi-hot encode: one binary column per top category
5. Add n_categories = count of categories per company
6. Add category_other = 1 if company has any category outside top 50
```

### Step 7 — Process geographic columns

```
country_code:
  - Fill missing with "Unknown"
  - Encode top 10 countries + group rest as "Other"
  - Add binary: is_usa = (country_code == 'USA')

state_code / region:
  - Add binary: is_sf_bay_area = (region == 'SF Bay Area')
  - Add binary: is_california  = (state_code == 'CA')
  - Drop raw columns from training features

city:
  - Exclude entirely from feature set
```

### Step 8 — Exclude leakage-risk and non-signal columns

Remove from the ML feature matrix before training:
- `permalink` — identifier
- `name` — identifier
- `homepage_url` — identifier / metadata
- `status` — target (never an input feature)
- `last_funding_at` — critical leakage risk
- `state_code` (raw) — replaced by binary flags
- `region` (raw) — replaced by binary flags
- `city` — too high cardinality

### Step 9 — Handle remaining missing values

| Feature | Strategy | Reason |
|---------|----------|--------|
| `country_code` missing | Fill "Unknown" | No imputation needed; create own bucket |
| `founding_year` missing | Fill median year | 28% missing; do not drop rows |
| `time_to_first_funding` missing | Fill median; flag with `has_founding_date=0` | Preserve rows |
| `log_funding` missing | Fill 0 (no funding) | 0 on log scale = $1 raise = effectively no funding |
| `category_list` missing | Fill "Unknown" | Preserve rows |

### Step 10 — Final feature set (proposed)

| Feature | Type | Source |
|---------|------|--------|
| `founding_year` | Integer | Derived from `founded_at` |
| `time_to_first_funding` | Integer (days) | Derived from date difference |
| `has_founding_date` | Binary | Missing indicator |
| `has_received_funding` | Binary | Missing indicator |
| `log_funding` | Float | Derived from `funding_total_usd` |
| `funding_rounds` | Integer | Direct from dataset |
| `is_usa` | Binary | Derived from `country_code` |
| `is_sf_bay_area` | Binary | Derived from `region` |
| `is_california` | Binary | Derived from `state_code` |
| `n_categories` | Integer | Derived from `category_list` |
| `category_Software` … `category_X` | Binary (×30–50) | Multi-hot from `category_list` |
| `country_[top10]` | Binary (×10) | Encoded from `country_code` |

---

## 11. Phase 2 Readiness Summary

| Area | Status | Action Needed |
|------|--------|---------------|
| Row count | ✅ 66,368 rows confirmed | None |
| Training subset | ✅ 13,334 rows identified | Split in Phase 2 |
| Target variable | ✅ Clean, zero missing | Create binary column |
| Class balance | ✅ Near-balanced (0.88:1) | Use stratified splits |
| Exact duplicates | ✅ Zero | None |
| Name duplicates | ⚠️ 239 names, 504 rows | Do not auto-delete; use permalink as ID |
| `funding_total_usd` | ⚠️ 12,785 dashes → NaN | Convert, fill 0, log-transform |
| `funding_rounds` | ✅ Clean integer | Ready to use |
| `founded_at` | 🔴 22.93% missing, 12 bad dates | Parse, null bad years, add flag |
| `first_funding_at` | ✅ 24 missing, 3 bad dates | Parse, null bad years, derive feature |
| `last_funding_at` | 🔴 Leakage risk | **Exclude from features** |
| `category_list` | ⚠️ Needs encoding | Multi-hot top 30–50 in Phase 2 |
| `country_code` | ⚠️ 10.5% missing | Fill "Unknown", encode top 10 |
| `state_code` / `region` | ⚠️ Mixed coding, partial coverage | Derive binary flags only |
| `city` | 🔴 5,111 unique values | Exclude from features |
| Data leakage | 🔴 Documented for 3 columns | Apply caveats; acknowledge in report |

### Overall Dataset Quality Assessment

**This dataset is suitable for an academic ML project with the following
conditions:**

1. The leakage risks in `funding_total_usd`, `funding_rounds`, and
   `last_funding_at` are documented and handled as described above.
2. The 13,334-row training subset (after excluding `operating`) is large
   enough for meaningful model training and evaluation.
3. The class balance in the training subset (0.88:1) is favorable — no
   emergency resampling required.
4. The largest data quality issue (`founded_at` missing for 28% of training
   rows) is manageable with imputation and a missing-data flag.
5. All findings in this report are grounded in the actual dataset.
   No performance claims about any model are made here.

---

*This report was generated as part of Phase 1 of the AI-Based Startup
Viability Prediction and Validation System academic project.*  
*All statistics were produced by `01_explore_dataset.py` running against
`big_startup_secsees_dataset.csv`.*  
*The original CSV file was not modified at any point during this analysis.*
