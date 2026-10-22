# ML Methodology Document
## AI-Based Startup Viability Prediction and Validation System

---

**Project type:** Academic minor project, 5th semester  
**Phase:** Phase 1 — Methodology Design (pre-implementation)  
**Status:** REVISED — awaiting approval before Phase 2 begins

> **Revision note (Section 11):** The original methodology defined the
> prediction point as "first funding." This was found to be misaligned
> with the actual application, which evaluates a startup *idea* before
> any funding has been received. Section 11 supersedes Sections 4 and 5
> on all matters relating to prediction point, feature selection, and
> system architecture. Sections 1–3, 6, 7, 8, and 9 remain valid and
> are not changed.

> This document records every significant methodological decision made
> before model training begins. Each decision is explained and justified
> so that it can be defended at a college project viva. No model has been
> trained yet. No performance numbers are claimed here. Nothing in this
> document has been fabricated.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Target Variable Definition](#2-target-variable-definition)
3. [Why Operating Companies Are Excluded](#3-why-operating-companies-are-excluded)
4. [Prediction Point Definition (Original — superseded by §11)](#4-prediction-point-definition)
5. [Feature Decisions (Original — superseded by §11)](#5-feature-decisions)
6. [Missing Value Strategy](#6-missing-value-strategy)
7. [Categorical Encoding Strategy](#7-categorical-encoding-strategy)
8. [Leakage Prevention Strategy](#8-leakage-prevention-strategy)
9. [Known Limitations](#9-known-limitations)
10. [Summary of Decisions (Original — superseded by §11)](#10-summary-of-all-decisions)
11. [**REVISED PREDICTION FRAMEWORK**](#11-revised-prediction-framework) ← current

---

## 1. Problem Definition

### 1.1 What We Are Building

We are building a **binary classification model** that predicts whether a
startup company will eventually achieve a successful exit (acquisition or
IPO) or shut down (close).

The model takes as input a set of features describing a company at an
early stage of its life, and outputs one of two predictions:

```
1 → Predicted SUCCESS  (likely to be acquired or go public)
0 → Predicted FAILURE  (likely to close / shut down)
```

### 1.2 What We Are NOT Building

- We are not building a model that guarantees startup success. No such
  model exists. Startup outcomes depend on macroeconomic conditions,
  timing, founder quality, and countless factors that no dataset captures.
- We are not building a model that predicts *when* a company will succeed
  or fail — only *whether* it will.
- We are not making predictions on the 53,034 currently `operating`
  companies in the dataset, because their final outcomes are unknown.

### 1.3 Why This Problem Is Suitable for Machine Learning

Machine learning is appropriate here because:

1. We have labelled historical examples — companies whose outcomes are
   already known (acquired, ipo, or closed). This makes supervised
   learning applicable.
2. There are patterns in the data that a human could not easily read
   manually (e.g., interactions between geography, sector, and funding
   speed that jointly predict outcomes).
3. The dataset is large enough (13,334 training-eligible rows) for a
   model to generalise beyond simple rules.

However, the model's predictions are **probabilistic estimates**, not
certainties. Startup success depends on factors — founder personality,
product-market fit, competitor timing — that are not captured in this
dataset. The model will make useful predictions on average; it will not
be right about every individual company.

---

## 2. Target Variable Definition

### 2.1 The Binary Label

The `status` column in the dataset contains four values. We define our
binary target as follows:

| Status value | Business meaning | Binary label |
|--------------|-----------------|:------------:|
| `acquired` | Company was purchased by another company | **1 — SUCCESS** |
| `ipo` | Company listed on a public stock exchange | **1 — SUCCESS** |
| `closed` | Company shut down / ceased operations | **0 — FAILURE** |
| `operating` | Company still running at data collection time | **Excluded** |

The derived column is named `target`.

### 2.2 Justification for the SUCCESS Definition

`acquired` and `ipo` are both treated as success because they represent
the two standard **exit events** that the startup ecosystem recognises as
positive outcomes:

- An **acquisition** means another company valued the startup enough to
  pay to own it. The founders and investors typically receive payment.
  Even if an acquisition price is not recorded, the act of being acquired
  confirms that the startup produced something of value.

- An **IPO (Initial Public Offering)** means public financial markets
  valued the company enough to let it list its shares. This is a more
  demanding standard than acquisition — IPO companies must satisfy
  regulatory requirements and demonstrate scalable financials.

Both events require **external validation by parties with financial
stakes** in being right. This makes them more reliable success indicators
than self-reported measures.

### 2.3 Justification for the FAILURE Definition

`closed` means the company stopped operating. This is the unambiguous
operational definition of failure for a startup — the company could not
sustain itself and ceased to exist as a going concern.

### 2.4 Why Not Use More Granular Labels?

An alternative design would use three or more labels (e.g., `acquired`,
`ipo`, `operating_long_term`, `closed_early`, `closed_late`). This is a
valid extension but is out of scope for a minor project for two reasons:

1. Multi-class classification adds model complexity and makes evaluation
   harder to explain.
2. The data does not provide enough information (e.g., time to closure,
   acquisition price) to construct reliable fine-grained labels.

The binary definition is academically standard for startup viability
prediction research and is the appropriate scope for this project level.

---

## 3. Why Operating Companies Are Excluded

This is the single most important data integrity decision in the project.

### 3.1 The Core Rule

In supervised machine learning, every row in the training set must have
a **known, correct label**. If you do not know the true answer for a row,
you cannot include it in training — because any label you assign would be
an invention, not a fact.

### 3.2 Why `operating` Cannot Be Labeled

A company with `status = operating` was still running when the dataset
was compiled. Its final outcome — acquisition, IPO, or closure — had not
yet occurred. We do not know what happened after the data was collected.

- If we label every `operating` company as **success (1)**: we would be
  asserting that all 53,034 of them eventually succeeded. Many of them
  will eventually close. We would be training the model on 53,034 wrong
  answers.

- If we label every `operating` company as **failure (0)**: we would be
  asserting that all of them failed. Many of them will eventually be
  acquired. We would again be training on wrong answers.

- If we label them as a **third class**: supervised binary classification
  requires exactly two classes for the target. Adding a third "unknown"
  class changes the problem entirely and still does not give us correct
  labels.

**There is no valid option other than exclusion.**

### 3.3 What Happens to Operating Companies

They are saved to `data/operating_companies.csv`. They are not discarded.

In a production version of this system, a trained model would run
inference on these 53,034 companies to predict their likely eventual
outcomes. This is the entire practical purpose of the project — making
predictions about companies whose outcomes are not yet known.

### 3.4 Impact on Training Set Size

After exclusion, the training set shrinks from 66,368 to **13,334 rows**.
This is a real cost. However, a model trained on 13,334 correctly labelled
rows is vastly more reliable than a model trained on 66,368 rows where
80% of the labels are invented.

Quality of labels always takes priority over quantity of rows.

---

## 4. Prediction Point Definition

### 4.1 Why a Prediction Point Must Be Defined

Without a defined prediction point, the question "is this feature
legitimate?" has no answer. Every feature must be evaluated against:
*"At my prediction point, would I actually have this information?"*

If the prediction point is not defined, the model can accidentally be
trained on information that only exists because of the outcome — which
is data leakage.

### 4.2 The Proposed Prediction Point

> **The prediction point is the moment a startup receives its first
> external funding round.**

At this moment, we ask: **will this company eventually succeed?**

### 4.3 Justification

**Real-world relevance:** When an investor, accelerator, or analyst
evaluates a startup for its first funding round, they are genuinely
making this prediction with their own money or reputation at stake. This
is the most natural and practically useful prediction moment.

**Earliest moment with sufficient signal:** At founding (Day 0), we
know almost nothing — a name, a country, maybe an industry. By first
funding, we additionally know how quickly the company attracted investment,
which is a meaningful signal of early traction and founder quality.

**Allows the most important derived feature:** `time_to_first_funding`
(days from founding to first funding) is only a legitimate feature if we
define the prediction point as on or after first funding. This feature
captures early momentum — a startup that attracted investment in 3 months
likely had a stronger initial pitch than one that took 3 years.

**Avoids later-stage leakage:** By anchoring to first funding rather than
a later date, we exclude all subsequent funding rounds (Series A, B, C,
etc.) and the information they carry about growing company success.

### 4.4 Information Available at the Prediction Point

The following information is legitimately known at the moment of first
funding:

| Information | Column / Feature | Status |
|-------------|-----------------|--------|
| When the company was founded | `founded_at` → `founding_year` | ✅ Safe |
| How quickly it raised first funding | `time_to_first_funding` (derived) | ✅ Safe |
| What industry it operates in | `category_list` | ✅ Safe |
| Where it is headquartered | `country_code`, `region` | ✅ Safe |
| That it has received funding | `has_received_funding` flag | ✅ Safe |
| Approximate first-round size | `funding_total_usd` — with caveat | ⚠️ Partial leakage |
| Number of rounds so far | Should be 1 at prediction point | ⚠️ Partial leakage in dataset |

The following information is **not** available at the prediction point:

| Information | Column | Reason not available |
|-------------|--------|---------------------|
| All subsequent funding rounds | `funding_rounds` total | Future rounds not happened yet |
| Date of most recent funding | `last_funding_at` | Future event |
| Whether the company succeeds | `status` | This is what we are predicting |

### 4.5 Dataset Limitation Regarding This Point

This dataset does not provide per-round funding breakdowns. `funding_total_usd`
and `funding_rounds` reflect the company's entire lifetime, not just the
state at first funding. We cannot surgically extract "round 1 amount only."

This is a **documented limitation**, not a reason to abandon the dataset.
The approach taken is to use these columns with an explicit caveat attached.
The project report must state clearly that these two features carry partial
post-prediction-point information, and that this may cause the model's
test-set performance to slightly overestimate real-world performance.

---

## 5. Feature Decisions

### 5.1 Decision Framework

Every column is evaluated on three criteria:
1. **Is it available at the prediction point?** (leakage check)
2. **Does it require transformation before use?** (format check)
3. **Does it add signal beyond noise?** (utility check)

### 5.2 Column-by-Column Decisions

---

#### `permalink`
**Decision: EXCLUDE**

This is a unique identifier — a URL slug assigned by Crunchbase.
It carries zero predictive signal. Two companies with similar slugs are
not more likely to share outcomes than two random companies. It is kept
in the dataset as a row identifier for debugging and traceability, but
it is dropped from the ML feature matrix.

---

#### `name`
**Decision: EXCLUDE as feature**

Company names have no consistent patterns that predict success. A company
called "Atlas" is not more or less likely to succeed than one called
"Zenith." The name is kept as a human-readable label but excluded from
the feature matrix.

Advanced NLP research has explored whether naming length, sentiment, or
word choice correlates with success. That is out of scope for this project.

---

#### `homepage_url`
**Decision: EXCLUDE**

Raw URLs carry no direct signal. A derived feature — `has_website` (1 if
URL is present, 0 if missing) — could weakly signal that the company has
an online presence. This is a marginal signal and is noted as an optional
extension rather than a required feature.

Additionally, some URLs may point to acquirer domains for companies that
were later acquired, creating a subtle indirect leakage risk if URL
content were ever analysed.

---

#### `category_list`
**Decision: USE — transform to multi-hot encoding**

- **Leakage risk:** None. Industry sector is defined at founding and does
  not change based on outcome.
- **Transformation required:** Yes. The raw pipe-separated string cannot
  be fed to a model. Apply multi-hot encoding for top 50 categories.
- **Additional feature:** `n_categories` (count of categories per company).
- **Missing values:** 3,148 rows (4.74%) — fill with "Unknown" before
  encoding; create `category_other = 1` for companies whose only
  categories fall outside the top 50.

Full encoding strategy is described in Section 7.

---

#### `funding_total_usd`
**Decision: USE — with transformation and documented leakage caveat**

- **Leakage risk:** PARTIAL. The value reflects lifetime total funding,
  which includes rounds raised after the prediction point. This cannot
  be fixed with the current dataset. It is used with an explicit caveat.
- **Transformation required:** Yes, two transformations:
  1. Convert string to numeric (`pd.to_numeric(..., errors='coerce')`)
  2. Apply `log10(x + 1)` to compress the 7-order-of-magnitude range
- **Missing values:** 12,785 rows show a literal dash `-` which becomes
  NaN after conversion. Strategy: fill with median of training set AND
  add `funding_known` binary flag. See Section 6.
- **Outliers:** Values up to $30 billion are real and should not be
  removed. The log transformation handles their scale.

---

#### `funding_rounds`
**Decision: USE — with documented leakage caveat**

- **Leakage risk:** PARTIAL. At the prediction point (first funding),
  every company has exactly 1 round. The lifetime count in this column
  includes all future rounds. The column therefore represents partial
  future information.
- **Transformation required:** No — it is already a clean integer with
  no missing values (confirmed by EDA).
- **Caveat:** This column should be described in the project report as
  "total rounds including post-prediction rounds" to be academically
  honest.

---

#### `country_code`
**Decision: USE — with encoding**

- **Leakage risk:** None. Country of headquarters is set at founding.
- **Transformation required:** Yes. 137 unique country codes cannot be
  used as raw strings. Two-part encoding strategy:
  1. Binary flag: `is_usa = 1` if country_code == 'USA', else 0
  2. One binary column per top-10 country by frequency, plus `country_other`
- **Missing values:** 6,958 rows (10.5%) — fill with "Unknown" category
  before encoding.

---

#### `state_code`
**Decision: DERIVE ONE FLAG, then exclude raw column**

- **Leakage risk:** None. State of headquarters is set at founding.
- **Problem:** Column mixes US state codes (CA, NY, TX) with international
  regional codes (H9, 7, 2) in the same column. Direct encoding is
  unreliable.
- **Strategy:** Derive one binary flag: `is_california = 1` if
  state_code == 'CA'. This captures the Silicon Valley geography signal
  cleanly. Drop the raw `state_code` column.
- Useful only for US companies; noise for non-US companies.

---

#### `region`
**Decision: DERIVE ONE FLAG, then exclude raw column**

- **Leakage risk:** None. Location is set at founding.
- **Problem:** 1,092 unique values — too high for direct encoding.
- **Strategy:** Derive one binary flag: `is_sf_bay_area = 1` if
  region == 'SF Bay Area'. SF Bay Area has 8,804 companies — the largest
  concentration in the dataset — and represents the world's most active
  startup ecosystem. This single binary flag captures the most
  geographically important signal.
- Drop the raw `region` column from features.

---

#### `city`
**Decision: EXCLUDE**

- **Leakage risk:** None.
- **Problem:** 5,111 unique cities. Direct encoding would create over
  5,000 sparse binary columns. Most cities would have fewer than 10
  examples — not enough to learn reliable patterns from.
- **Signal already captured:** Geographic signal is better captured by
  `country_code` (137 values → reduced to ~11 features) and
  `is_sf_bay_area` (single flag). City adds cardinality without adding
  proportional information.
- **Optional extension:** Binary flags for the top 5 hub cities
  (San Francisco, New York, London, Beijing, Boston) could be added
  in a future phase.

---

#### `founded_at`
**Decision: USE for derived features — exclude raw date**

- **Leakage risk:** None. Founding predates all outcomes by definition.
- **Transformation required:** Yes:
  1. Parse to datetime with `pd.to_datetime(..., errors='coerce')`
  2. Set to NaN any year > 2023 (4 confirmed typos, e.g., year 2914)
  3. Set to NaN any year < 1800 (8 confirmed typos)
  4. Extract `founding_year` as integer feature
  5. Create `has_founding_date` binary flag (1 if date present, 0 if NaN)
- **Missing values:** 15,221 rows (22.93% overall; 27.99% in training set)
  — the largest missing-data problem in the dataset. Do not drop these
  rows. Use `has_founding_date` flag + median imputation for `founding_year`.
- **Raw date:** Do not use as a feature. A raw date string is meaningless
  to an ML model. Only the derived year and the derived time difference
  (see `first_funding_at`) are informative.

---

#### `first_funding_at`
**Decision: USE for derived features — exclude raw date**

- **Leakage risk:** Low. This is the prediction point itself — by
  definition this information is available at prediction time.
- **Transformation required:** Yes:
  1. Parse to datetime
  2. Set to NaN any year < 1800 (3 confirmed typos, e.g., year 1014)
  3. Compute `time_to_first_funding = (first_funding_at - founded_at).days`
  4. Set `time_to_first_funding` to NaN where the result is negative
     (first_funding_at < founded_at = data entry error)
  5. Create `has_received_funding` binary flag (1 if first_funding_at
     is not NaN)
- **Missing values:** 24 rows — negligible (0.04%). Impute derived
  features with median.
- **`time_to_first_funding`** is the single most leakage-safe quantitative
  feature in the dataset. It measures how quickly a startup attracted
  its first investor — a proxy for early traction and founder credibility.

---

#### `last_funding_at`
**Decision: EXCLUDE — critical data leakage**

- **Leakage risk:** CRITICAL. This column records the date of the most
  recent funding round across the company's entire history. For companies
  that were later acquired or went public, this date is close to the
  exit event — essentially recording that "something significant is about
  to happen." Including this feature would teach the model to detect
  imminent exits rather than early-stage potential.
- **At our prediction point (first funding), future rounds have not
  happened.** `last_funding_at` therefore encodes future events by
  construction.
- **Decision is final:** No derived feature or transformation makes this
  column safe to use. It is dropped entirely before Phase 2.

---

### 5.3 Final Feature Matrix (proposed for Phase 2)

| Feature | Type | Source | Notes |
|---------|------|--------|-------|
| `founding_year` | Integer | `founded_at` | 2-step: parse date → extract year |
| `has_founding_date` | Binary | `founded_at` | Missing-data flag |
| `time_to_first_funding` | Integer (days) | `first_funding_at` − `founded_at` | Key momentum signal |
| `has_received_funding` | Binary | `first_funding_at` | Missing-data flag |
| `log_funding` | Float | `funding_total_usd` | log10(x+1); partial leakage caveat |
| `funding_known` | Binary | `funding_total_usd` | 1 if original value was numeric |
| `funding_rounds` | Integer | `funding_rounds` | Partial leakage caveat |
| `is_usa` | Binary | `country_code` | 1 if country_code == 'USA' |
| `country_GBR`, `country_CAN` … (top 10) | Binary | `country_code` | One column per top country |
| `country_other` | Binary | `country_code` | 1 if not in top 10 + not USA |
| `is_california` | Binary | `state_code` | 1 if state_code == 'CA' |
| `is_sf_bay_area` | Binary | `region` | 1 if region == 'SF Bay Area' |
| `n_categories` | Integer | `category_list` | Count of pipe-separated categories |
| `category_Software` … (top 50) | Binary | `category_list` | Multi-hot per top category |
| `category_other` | Binary | `category_list` | 1 if any category outside top 50 |

**Columns explicitly excluded from the feature matrix:**

`permalink`, `name`, `homepage_url`, `status` (it IS the target), `city`,
`state_code` (raw), `region` (raw), `last_funding_at`, `founded_at` (raw),
`first_funding_at` (raw), `funding_total_usd` (raw, pre-transform)

---

## 6. Missing Value Strategy

### 6.1 Principles

Two principles guide all missing-value decisions in this project:

**Principle 1 — Do not delete rows if it can be avoided.**  
Dropping a row because one column is missing discards all the information
from the other 13 columns for that company. With only 13,334 training rows,
we cannot afford to discard 28% of them because `founded_at` is missing.

**Principle 2 — Never hide the fact that a value was missing.**  
When you impute (fill in) a missing value, the model cannot tell that the
value was invented. To preserve this information, always pair imputation
with a binary "was this value missing?" flag column. This allows the model
to learn separately: "what does this value predict?" and "does having an
unknown value predict anything?"

### 6.2 Strategy by Column

| Column | Missing Count (training) | Strategy | Reason |
|--------|-------------------------:|----------|--------|
| `founded_at` | 3,732 (27.99%) | Parse; impute `founding_year` with median; add `has_founding_date=0` flag | Too many to drop; flag preserves the information |
| `first_funding_at` | 2 (0.01%) | Impute derived features with median; set `has_received_funding=0` | Negligible count; flagged |
| `funding_total_usd` (as dashes) | 12,785 (converted to NaN) | Impute with **median** of training set; add `funding_known=0` flag | See §6.3 for full reasoning |
| `category_list` | 1,086 (8.14%) | Fill with `"Unknown"` before encoding; creates its own `category_Unknown` column | Unknown is a valid category; preserves the row |
| `country_code` | 1,991 (14.93%) | Fill with `"Unknown"` string; creates `country_Unknown` in encoding | Unknown geography is a valid signal |
| `state_code` | 2,209 (16.57%) | Only `is_california` flag derived; if `state_code` is NaN, `is_california = 0` | No imputation needed for derived flag |
| `region` | 2,152 (16.14%) | Only `is_sf_bay_area` flag derived; if region is NaN, `is_sf_bay_area = 0` | No imputation needed for derived flag |
| `time_to_first_funding` (negative) | Small number | Set to NaN; impute with median; these are data-entry errors | Negative duration is impossible |

### 6.3 Full Reasoning for `funding_total_usd` Imputation

The 12,785 rows with a literal `-` in this column are ambiguous. The dash
could mean either:

- **"This company raised no external funding"** → correct value is $0
- **"This company's funding was not recorded in Crunchbase"** → correct
  value is unknown

A literal dash in a database field more commonly means "not recorded"
than "exactly zero." Therefore, filling with $0 is **potentially wrong**
in a direction we cannot measure. It asserts certainty where we have none.

**Chosen strategy: median imputation + `funding_known` flag**

1. Fill the NaN with the **median funding amount** from the training set
   (~$2 million). The median is used rather than the mean because the
   distribution is heavily right-skewed (mean = $18.5M, which is inflated
   by billionaire outliers and is not representative of most companies).

2. Add `funding_known = 1` if the original value was a valid number,
   `funding_known = 0` if it was a dash or blank. This lets the model
   learn whether "unknown funding" itself is a predictive signal — without
   the model confusing "unknown" with "zero."

3. Apply `log10(imputed_value + 1)` to both known and imputed values.
   The +1 ensures log(0+1) = 0, preventing log of zero errors.

**Why not mean imputation?** The mean ($18.5M) is far above the median
($2M) because a handful of massive companies pull it up. Imputing with
$18.5M would make hundreds of unknown-funding companies appear to have
been heavily funded, which is almost certainly wrong. The median is the
more representative central value for a skewed distribution.

---

## 7. Categorical Encoding Strategy

Machine learning models require numerical inputs. Columns that contain
category names, country codes, or industry tags must be converted to
numbers before training. This section defines how.

### 7.1 `category_list` — Multi-Hot Encoding of Top 50

**The problem:** 858 unique categories. Using all 858 would create 858
sparse binary columns, most containing almost entirely zeros.

**Why sparsity is harmful:**  
A column that is 99.9% zeros teaches the model almost nothing — there
are not enough examples of that category to learn a reliable pattern.
It adds noise and inflates the feature matrix without adding signal.
With 13,334 training rows and 858 category columns, the average column
would have only ~15 positive examples — statistically unreliable.

**Chosen strategy: top 50 categories + "Other" flag**

Step 1: Identify the 50 most frequently occurring individual categories
(after splitting on `|`) across the training set only. This step must
use training-set frequencies only — not the full dataset — to avoid
leakage of test-set distribution information into the feature design.

Step 2: For each of the top 50 categories, create one binary column:
`category_Software`, `category_Mobile`, `category_Biotechnology`, etc.
Value = 1 if the company has that category, 0 if not.

Step 3: Create one additional binary column `category_other`:
Value = 1 if the company has at least one category not in the top 50.

Step 4: Create `n_categories` = total number of categories the company
has (before any filtering). This numeric feature captures breadth of
focus and does not require encoding.

Step 5: For companies with `category_list = NaN`, set all category
columns to 0 and `n_categories = 0`.

**Why 50 and not 30 or 100?**  
50 is a practical compromise. The top 50 categories in this dataset each
appear in hundreds to thousands of companies — enough for reliable
statistical patterns. Categories ranked 51–858 appear far less frequently.
The exact cutoff (30, 40, 50) can be tuned in Phase 2 based on how many
training examples each category-column ends up with. The guiding rule is:
retain only categories that appear in at least 1% of training rows
(approximately 133 rows in our 13,334-row training set).

**Why not sector grouping instead?**  
Sector grouping (mapping 858 categories into ~10 macro-sectors like
"FinTech", "HealthTech") would be more elegant and produce fewer columns.
However, it requires manually reviewing and assigning 858 category names
— a significant effort involving judgment calls that would need to be
documented and defended. For this project scope, multi-hot encoding with
a reasonable cutoff is more practical and equally defensible.

Sector grouping is noted as a recommended improvement for a future phase.

### 7.2 `country_code` — Top-10 + Binary Flags

137 unique country codes. Strategy: create binary columns for the top 10
countries by frequency in the training set (USA, GBR, CAN, IND, CHN, FRA,
DEU, ISR, ESP, AUS), plus:

- `country_other = 1` for any country not in the top 10
- `country_unknown = 1` for missing country_code

This reduces 137 text values to 12 binary columns.

A simpler alternative — `is_usa` only (1/0) — would also be valid given
that USA accounts for 56.7% of the dataset. Both approaches are tried in
Phase 2 and the better-performing one is retained.

**Why not ordinal encoding?**  
Ordinal encoding assigns a number to each country (USA=1, GBR=2, etc.).
This implies an ordering — that GBR is "twice" USA — which is meaningless
for country names. Binary (one-hot style) encoding is the correct approach
for nominal (unordered) categorical variables.

### 7.3 `state_code` and `region` — Single Binary Flags Each

These columns have too many unique values for full encoding and too much
inconsistency (mixed US/international codes in `state_code`) for reliable
direct use.

The single most informative geographic signal after country is whether a
company is in the SF Bay Area (Silicon Valley + San Francisco). This is
extracted as one binary flag each from `region` and `state_code`.

---

## 8. Leakage Prevention Strategy

### 8.1 Definition

Data leakage occurs when a model is trained on information that would not
be available at prediction time in a real deployment. It causes the model
to appear more accurate than it really is during testing, but to perform
worse in practice.

There are two types of leakage relevant to this project:

**Temporal leakage:** Using information from a time point after the
prediction point. Example: `last_funding_at` records events that happen
years after the prediction point.

**Target leakage:** Using information that is caused by the outcome
rather than causing it. Example: if a company's `status` column were
accidentally left in the feature matrix, the model would learn to read
the answer directly.

### 8.2 The Four-Part Prevention Strategy

**Part 1 — Commit to a fixed prediction point.**  
The prediction point is first funding. Every feature is evaluated against:
*"Is this value knowable at the moment of first funding?"* If no, exclude
it or document it as a caveat.

**Part 2 — Exclude post-prediction columns unconditionally.**  
`last_funding_at` is excluded entirely. It records events that, by
construction, happen after the prediction point.

**Part 3 — Use derived features anchored to the prediction point.**  
Rather than using raw date columns, derive features that measure durations
and states as of the prediction point. `time_to_first_funding` is the
primary example — it is a measurement that is fully determined at first
funding and does not change afterwards.

**Part 4 — Document partial leakage with explicit caveats.**  
`funding_total_usd` and `funding_rounds` cannot be made fully clean
without per-round data that this dataset does not provide. They are used
with documented limitations in both this document and the project report.
The test-set accuracy numbers must be presented alongside this caveat —
real-world performance may be lower because these two features partially
encode post-prediction information.

### 8.3 Train/Test Split Protocol

The train/test split must be performed **before any imputation, encoding,
or feature engineering** that uses statistics from the data (e.g., medians,
category frequencies). This prevents test-set information from leaking
into the training process through the preprocessing steps themselves.

The correct order is:
```
1. Split: training rows / test rows (80/20 or 70/30, stratified on target)
2. Compute statistics on TRAINING SET ONLY (medians, category frequencies,
   country frequencies)
3. Apply those statistics to transform BOTH training set and test set
4. Train model on transformed training set
5. Evaluate on transformed test set
```

If the split happens after computing statistics, the test set's
distribution has influenced the preprocessing — this is a subtle but real
form of leakage called **preprocessing leakage**.

### 8.4 Additional Leakage Risks Identified

**`status` column:** Must be removed from the feature matrix before
training. It is the source of the target label. Leaving it in would
give the model the answer directly (trivial leakage).

**`funding_rounds = 1` at prediction point:** At the prediction point
(first funding), every company has had exactly 1 round. The lifetime
`funding_rounds` count in the dataset is always ≥1, with the surplus
representing future rounds. The column as stored therefore encodes future
activity. This is noted as a partial leakage limitation.

**Survivorship bias (not temporal leakage, but a related limitation):**
Companies appear in this dataset because they were notable enough to be
recorded in Crunchbase. Smaller companies that failed quietly without
press coverage may be under-represented in the `closed` category. The
model therefore may not generalise well to very early-stage, low-profile
startups. This must be stated as a limitation in the project report.

---

## 9. Known Limitations

These limitations are stated upfront so that the project report is
academically honest. None of them invalidates the project — they are
typical of real-world ML projects. They must be acknowledged, not hidden.

### 9.1 Partial Data Leakage in Funding Columns

`funding_total_usd` and `funding_rounds` reflect lifetime values, not
values as of the prediction point. Both columns partially encode
post-prediction-point information. This means the model's test-set
accuracy is likely to be **optimistic** — the real-world performance
when predicting a new startup at first funding would be slightly lower.

**Mitigation:** Document clearly. Do not report test accuracy as an
absolute claim. Frame results as "performance under these documented
assumptions."

### 9.2 Large Missing Rate in `founded_at`

28% of training rows have no founding date. This is handled by imputation
and a missing-data flag, but the imputed values are estimates, not facts.
Features derived from `founded_at` (including `time_to_first_funding` for
companies with no founding date) carry uncertainty.

### 9.3 No Per-Round Funding Breakdown

The dataset does not break funding into individual rounds. We cannot
know "how much was raised in the first round specifically." A future,
better-resourced version of this project could use the Crunchbase API
to retrieve per-round data.

### 9.4 Survivorship Bias

The dataset contains companies that were notable enough to be listed in
Crunchbase. The population of real-world startups includes many companies
that failed quietly and were never recorded. The model predicts outcomes
for "Crunchbase-type startups," not for all startups universally.

### 9.5 Temporal Concept Drift

The dataset spans from roughly 1990 to 2015. The startup ecosystem in 2015
was very different from 1995 — different VC volumes, different sectors,
different exit markets. A model trained on this full period may not
accurately represent patterns for startups founded after 2015. This is
called **concept drift**. Models should be retrained periodically on
fresh data when used in practice.

### 9.6 No Model Performance Claims in This Document

This methodology document is written before any model is trained. No
accuracy, precision, recall, or AUC numbers appear here because none
exist yet. Any document that claims model performance before training
has been completed contains fabricated numbers. This project makes no
such claims.

---

## 10. Summary of All Decisions

| Decision | Choice | Key Reason |
|----------|--------|-----------|
| Target definition | acquired+ipo=1, closed=0 | Standard exit-event definition; unambiguous and externally validated |
| Operating companies | Excluded from training | Outcome unknown; labeling them would fabricate training data |
| Prediction point | First funding date | Natural real-world evaluation moment; minimum viable signal available |
| `category_list` | Multi-hot top 50 + "Other" flag | 858 categories → sparsity problem; top 50 covers most frequency |
| `funding_total_usd` | Convert, median-impute, log-transform, flag | Skewed distribution; dash = unknown not zero; partial leakage documented |
| `funding_rounds` | Use as-is with caveat | Clean integer; partial leakage documented |
| `country_code` | Top 10 binary + `is_usa` + Other + Unknown | 137 values reduced to interpretable flags |
| `state_code` | Derive `is_california` flag only | Mixed US/international codes; single flag captures the key signal |
| `region` | Derive `is_sf_bay_area` flag only | 1,092 unique values; single flag captures dominant geographic signal |
| `city` | Exclude | 5,111 unique values; signal already captured by country + region flags |
| `founded_at` | Derive `founding_year` + `has_founding_date` flag | Raw date unusable; 28% missing needs imputation + flag |
| `first_funding_at` | Derive `time_to_first_funding` + `has_received_funding` | Key momentum signal; raw date unusable |
| `last_funding_at` | Exclude entirely | Critical temporal leakage — records future events |
| Missing values | Impute with median + binary missing flag | Preserves rows; does not hide the fact that value was missing |
| Class imbalance | No resampling required | 0.88:1 ratio is near-balanced; use stratified splits |
| Train/test split | Split before computing statistics | Prevents preprocessing leakage |

---

*This document was written before any model training began.*  
*All decisions are based on the actual dataset analysis performed in Phase 1.*  
*No external sources are cited that have not been directly examined.*  
*This methodology must be reviewed and approved before Phase 2 implementation begins.*

---

## 11. Revised Prediction Framework

> **This section supersedes Sections 4 and 5 in their entirety.**
> All feature and prediction-point decisions from this point forward
> come from this section. Sections 4 and 5 are retained for audit trail
> purposes only.

---

### 11.1 Why the Original Prediction Point Was Wrong

The original methodology defined the prediction point as "the moment a
startup receives its first external funding round." This was a reasonable
academic choice for a pure ML exercise, but it creates a fundamental
mismatch with the actual application being built.

**The application evaluates a startup idea submitted by a user who does
not yet have funding.** They have:
- a description of what they want to build
- a chosen industry/sector
- a country they plan to operate in
- possibly a founding year

They do not yet have:
- any funding history
- any funding amount
- a record of how long it took to raise money
- a count of funding rounds

If we train the ML model on features that include funding information, and
then ask the user "how much have you raised so far?" — the answer will
always be $0, the rounds will always be 0, and `time_to_first_funding`
cannot exist. The model would receive inputs it was never trained to
interpret at those values, and its predictions would be unreliable.

The original "first funding" prediction point is therefore misaligned with
the application's actual use case. It must be replaced.

---

### 11.2 The Architecture Problem in Plain Terms

```
What the original model was trained to do:
  Input: "Here is a startup that has ALREADY raised money.
          Tell me if it will succeed."

What the application actually needs:
  Input: "Here is a startup IDEA with no funding yet.
          Tell me if it has potential."
```

These are different problems. A model trained for the first cannot
reliably answer the second.

---

### 11.3 The Revised Prediction Point

> **The prediction point is Day 0 — the moment a startup idea exists,
> before any funding has been received.**

At this moment, we know only:
- What the startup plans to do (category / sector)
- Where it will operate (country, geography)
- When it was conceived or founded (year)
- Nothing else derived from financial or operational history

This is the only prediction point that is consistent with both:
1. What a user can truthfully provide when submitting a startup idea
2. A clean, non-leaked historical training signal from the dataset

---

### 11.4 Separation of the Two System Components

The system is redesigned as two separate, complementary components that
work together but are built and evaluated differently.

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT A — Historical ML Model                              │
│                                                                 │
│  What it does: Trained on 13,334 historical companies whose     │
│  outcomes are known. Learns the relationship between            │
│  sector + geography + founding year and eventual outcome.       │
│                                                                 │
│  What it outputs: A probability score (0.0 to 1.0) that a      │
│  startup with these structural attributes historically          │
│  resulted in a successful exit.                                 │
│                                                                 │
│  What it does NOT do: Assess idea quality, team quality,        │
│  market size, competition, or revenue potential.                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  COMPONENT B — Startup Evaluation Scoring Engine                │
│                                                                 │
│  What it does: Applies a set of rule-based and NLP-based        │
│  criteria to the free-text startup idea submitted by the user.  │
│  Each criterion is scored and weighted to produce a             │
│  structured evaluation.                                         │
│                                                                 │
│  What it outputs: Scores across dimensions like idea clarity,   │
│  problem statement, sector relevance, and geographic context.   │
│                                                                 │
│  What it does NOT do: Make a binary ML prediction. It does not  │
│  claim to know whether the idea will succeed.                   │
└─────────────────────────────────────────────────────────────────┘
```

These two components are combined into a **Final Viability Report** that
presents both outputs clearly labelled, so the user understands what each
number means and where it comes from.

---

### 11.5 Full System Architecture

```
  User submits a startup idea
           │
           ▼
  ┌─────────────────────────────────────┐
  │  NLP / Feature Extraction Layer     │
  │                                     │
  │  Extracts structured fields from    │
  │  the free-text idea description:    │
  │  • sector/category                  │
  │  • country                          │
  │  • founding year                    │
  │  • idea clarity signals             │
  │  • problem statement presence       │
  │  • solution description presence    │
  └────────────────┬────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
  ┌──────────────┐    ┌─────────────────────────┐
  │  COMPONENT A │    │  COMPONENT B            │
  │  Historical  │    │  Scoring Engine         │
  │  ML Model    │    │                         │
  │              │    │  Evaluates:             │
  │  Features:   │    │  • Problem clarity      │
  │  • category  │    │  • Solution presence    │
  │  • country   │    │  • Sector relevance     │
  │  • year      │    │  • Geographic context   │
  │              │    │  • Idea completeness    │
  │  Outputs:    │    │                         │
  │  P(success)  │    │  Outputs: scored report │
  └──────┬───────┘    └──────────┬──────────────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Final Viability    │
         │  Report             │
         │                     │
         │  • ML sector score  │
         │  • Idea eval scores │
         │  • Combined summary │
         │  • Limitations note │
         └─────────────────────┘
```

---

### 11.6 Component A — Historical ML Model: Revised Feature Set

#### 11.6.1 The Revised Prediction Point

Day 0: founding / idea conception. No funding has happened. No operational
history exists.

#### 11.6.2 Features Permitted at This Point

Only information that exists before any funding event is permitted:

| Feature | Source in dataset | Available at Day 0? | Notes |
|---------|-------------------|:-------------------:|-------|
| `category_list` (encoded) | CSV column | ✅ Yes | Sector is chosen before funding |
| `country_code` (encoded) | CSV column | ✅ Yes | Location is chosen before funding |
| `founding_year` | Derived from `founded_at` | ✅ Yes | Year is known at founding |
| `is_usa` | Derived from `country_code` | ✅ Yes | Binary flag |
| `is_sf_bay_area` | Derived from `region` | ✅ Yes | Binary geography flag |
| `is_california` | Derived from `state_code` | ✅ Yes | Binary geography flag |
| `n_categories` | Derived from `category_list` | ✅ Yes | Count of sectors listed |
| `category_[X]` (top 50) | Derived from `category_list` | ✅ Yes | Multi-hot binary columns |
| `country_[X]` (top 10) | Derived from `country_code` | ✅ Yes | Binary country columns |
| `has_founding_date` | Derived from `founded_at` | ✅ Yes | Missing-data flag |

#### 11.6.3 Features Explicitly Excluded from Component A

| Column / Feature | Reason for Exclusion |
|------------------|----------------------|
| `funding_total_usd` | Lifetime total: entirely post-prediction, or unavailable at Day 0 |
| `funding_rounds` | Lifetime count: entirely post-prediction at Day 0 |
| `time_to_first_funding` | Cannot exist at Day 0 — first funding has not happened yet |
| `has_received_funding` | At Day 0, this is always 0 — provides no discriminatory signal |
| `first_funding_at` (raw or derived) | Future event — does not exist at Day 0 |
| `last_funding_at` | Future event — critical leakage (unchanged from original) |
| `status` | This IS the target — never a feature |
| `permalink` | Identifier — no signal |
| `name` | Identifier — no signal |
| `homepage_url` | Metadata — no signal |
| `city` | Too high cardinality; signal captured by country + region |
| `state_code` (raw) | Replaced by `is_california` binary flag |
| `region` (raw) | Replaced by `is_sf_bay_area` binary flag |
| `founded_at` (raw date) | Replaced by `founding_year` integer |

#### 11.6.4 What Component A Actually Predicts

Component A predicts:

> **"Given a startup in sector X, from country Y, founded in year Z:
> what fraction of historically similar companies in this dataset
> achieved a successful exit (acquisition or IPO)?"**

This is a **base-rate structural prediction**. It answers "how have
companies like this done historically?" — not "is this specific idea
good?"

The output is a probability between 0 and 1, or equivalently a predicted
class (0 = failure-pattern, 1 = success-pattern). It reflects structural
patterns in the dataset, not idea quality.

**What Component A cannot predict, and must not claim to predict:**
- Whether this specific idea is good
- Whether the market is large enough
- Whether the founders are capable
- Whether there is competition
- Whether the product will work
- Whether customers will pay

These dimensions are absent from the Crunchbase dataset. No training
data for these factors exists. Any model that claims to score these
dimensions from our dataset would be fabricating results.

---

### 11.7 Component B — Startup Evaluation Scoring Engine

#### 11.7.1 What It Is

Component B is **not** a machine learning model. It is a structured
evaluation framework that scores the user's startup idea submission
against a set of defined criteria.

It is implemented as a rule-based + NLP text analysis system. Rules are
set by the project designers based on established startup evaluation
frameworks (e.g., problem-solution fit, clarity of value proposition).

#### 11.7.2 What It Evaluates

The scoring engine evaluates only what the user has actually written.
It does not invent information, hallucinate context, or claim to know
anything not present in the submission.

| Dimension | What is evaluated | How |
|-----------|-------------------|-----|
| **Idea clarity** | Is the idea description coherent and specific? | Text length, specificity heuristics |
| **Problem statement** | Does the submission describe a problem being solved? | Keyword/phrase presence detection |
| **Solution description** | Does the submission describe a proposed solution? | Keyword/phrase presence detection |
| **Target market** | Is a target customer or market mentioned? | Keyword/phrase presence detection |
| **Sector alignment** | Does the described idea match the chosen category? | Category vs. description consistency check |
| **Geographic relevance** | Does the chosen country affect feasibility of this sector? | Rule lookup (e.g., some sectors require specific infrastructure) |

#### 11.7.3 What It Does NOT Evaluate

The scoring engine makes no claims about:
- Market size (no data source for this)
- Competition (no data source for this)
- Revenue potential (no data source for this)
- Founder quality (not submitted by the user)
- Technical feasibility (not evaluatable from text alone)
- Customer pain intensity (subjective; not in our dataset)

If these dimensions are displayed in a report, they must be labelled
as "not assessed" rather than assigned a fabricated score.

#### 11.7.4 Scoring Design

Each evaluated dimension receives a score from 0 to 10:
- 0 = dimension completely absent from the submission
- 5 = dimension partially addressed
- 10 = dimension clearly and specifically addressed

Scores are combined into an **Idea Evaluation Score** (0–100) using
defined weights. The weights are design decisions, not ML outputs, and
must be documented as such.

**Example weight distribution (subject to revision):**

| Dimension | Weight |
|-----------|-------:|
| Problem statement clarity | 25% |
| Solution description | 25% |
| Target market identification | 20% |
| Sector/category specificity | 15% |
| Idea overall coherence | 15% |

These weights are defined by the project designers. They are not
learned from data. This must be stated clearly in the final report.

---

### 11.8 NLP / Feature Extraction Layer

#### 11.8.1 What It Must Do

When a user submits a startup idea as free text, the system must extract
structured fields that can be:
1. Passed to Component A (the ML model) as its input features
2. Evaluated by Component B (the scoring engine)

#### 11.8.2 Fields Extracted for Component A

These are the fields that map directly to the historical ML model's
feature set:

| Field needed by ML model | How it is obtained from user input |
|--------------------------|-------------------------------------|
| `category` / sector | User selects from a dropdown list matching the dataset's top 50 categories. This is NOT inferred from free text — the user explicitly chooses. |
| `country_code` | User selects their country from a list. |
| `founding_year` | User enters the year (or current year is used as default). |
| Derived binary flags | Computed from the above: `is_usa`, `is_sf_bay_area`, `is_california`, `n_categories` |

**Key design decision:** The sector and country are collected via
structured UI inputs (dropdowns), not inferred from free text. This
eliminates the risk of NLP misclassifying a description and feeding the
wrong category to the ML model.

Free text is used only for Component B scoring — where it is evaluated
for quality signals, not classified into ML-compatible codes.

#### 11.8.3 Fields Extracted for Component B

These are extracted from the free-text description:

| Signal | Extraction method |
|--------|-------------------|
| Description length (word count) | Simple word count |
| Problem keywords present | Keyword list: "problem", "challenge", "pain", "need", "gap", "struggle" etc. |
| Solution keywords present | Keyword list: "solution", "platform", "tool", "service", "app", "system" etc. |
| Customer/market keywords present | Keyword list: "customer", "user", "market", "segment", "enterprise", "consumer" etc. |
| Specificity score | Ratio of specific nouns / total words (simple heuristic) |

These are simple heuristics, not deep NLP models. They are appropriate
for a 5th-semester minor project. The limitations of keyword-based
extraction (it can miss well-written descriptions that use synonyms)
must be acknowledged in the project report.

#### 11.8.4 What the NLP Layer Does NOT Do

- It does not infer the ML category from free text. (Category is user-selected.)
- It does not predict market size. (No data source.)
- It does not assess technical feasibility. (Beyond text analysis scope.)
- It does not generate information not present in the user's submission.

---

### 11.9 Feature Source Taxonomy

Every feature in the system belongs to exactly one of three sources.
This taxonomy must be clearly communicated in the final report and UI.

#### Source 1 — Features from the historical dataset (Component A inputs)

These features are derived entirely from the Crunchbase CSV. They are
used to train the ML model and to represent the user's startup at inference
time using the same schema.

| Feature | CSV column source |
|---------|-----------------|
| `category_[X]` (top 50 binary) | `category_list` |
| `n_categories` | `category_list` |
| `category_other` | `category_list` |
| `country_[X]` (top 10 binary) | `country_code` |
| `country_other` | `country_code` |
| `is_usa` | `country_code` |
| `is_sf_bay_area` | `region` |
| `is_california` | `state_code` |
| `founding_year` | `founded_at` |
| `has_founding_date` | `founded_at` |

#### Source 2 — Features extracted from the user's free-text idea (Component B inputs)

These features do not exist in the training dataset at all. They are
produced entirely from the user's submission at inference time.

| Feature | Source |
|---------|--------|
| `word_count` | Computed from description text |
| `has_problem_statement` | Keyword detection |
| `has_solution_description` | Keyword detection |
| `has_target_market` | Keyword detection |
| `idea_specificity_score` | Heuristic ratio |

#### Source 3 — Manually defined scoring criteria (Component B weights)

These are neither learned from data nor extracted from text. They are
design decisions made by the project team.

| Element | Nature |
|---------|--------|
| Dimension weights (25%, 25%, 20%, etc.) | Designer judgment |
| Keyword lists for problem/solution/market | Designer-curated lists |
| Score thresholds (0/5/10 boundaries) | Designer judgment |

The fact that these are manually defined does not make them invalid — it
means they must be presented as design choices, not as ML predictions.
A viva examiner will ask "how did you choose these weights?" and the
correct answer is "we defined them based on standard startup evaluation
principles, and they can be tuned based on expert feedback."

---

### 11.10 How a New Startup Idea Becomes Model Inputs

This is the step-by-step process for a single user submission:

```
Step 1 — User submits:
  • Free text description: "An app that helps farmers track crop health
    using satellite imagery and AI analysis."
  • Dropdown: Category selected = "Agriculture" (or nearest match)
  • Dropdown: Country = "India"
  • Input: Founding year = 2024

Step 2 — Structured extraction:
  • country_code = "IND"
  • founding_year = 2024
  • category chosen: map "Agriculture" to the nearest matching
    historical category from the top 50 list

Step 3 — Compute Component A feature vector:
  • is_usa = 0
  • is_sf_bay_area = 0
  • is_california = 0
  • country_IND = 1, all other country flags = 0
  • founding_year = 2024
  • has_founding_date = 1
  • category_Agriculture = 1 (or nearest match), all others = 0
  • n_categories = 1
  ─────────────────────────────────────────────────────────────
  NOTE: funding features are NOT included.
  All funding columns are 0 / absent from the feature vector.
  The ML model was trained WITHOUT funding features.
  ─────────────────────────────────────────────────────────────

Step 4 — ML model produces:
  P(success | sector=Agriculture, country=India, year=2024) = e.g. 0.34
  "Historically, ~34% of similar companies in this dataset achieved exit."

Step 5 — Component B evaluates the free text:
  • word_count = 18 → moderate detail
  • has_problem_statement: "track crop health" → yes (1)
  • has_solution_description: "satellite imagery and AI" → yes (1)
  • has_target_market: "farmers" → yes (1)
  • idea_specificity_score → moderate (uses concrete nouns)
  → Idea Evaluation Score = e.g. 72/100

Step 6 — Final Viability Report:
  • Historical sector pattern score: 34% (ML model)
    "34% of Agriculture startups in similar countries historically
     achieved acquisition or IPO in our dataset."
  • Idea evaluation score: 72/100
    "Your idea description shows a clear problem, solution, and
     target market. More detail on revenue model would strengthen it."
  • Combined summary with caveats
```

---

### 11.11 What the ML Model Predicts vs. What the Scoring Engine Evaluates

These must never be conflated in the UI, the report, or the project
documentation.

| Dimension | ML Model (Component A) | Scoring Engine (Component B) |
|-----------|----------------------|------------------------------|
| Nature | Statistical / learned from data | Rule-based / designer-defined |
| Input | Sector + Country + Year | Free text description |
| Output | Probability of historical success pattern | Structured idea quality scores |
| Interpretation | "Companies like this have done X% well historically" | "Your idea submission covers these dimensions" |
| Can it assess idea quality? | ❌ No | ✅ Partially (via text heuristics) |
| Can it assess market size? | ❌ No | ❌ No |
| Can it assess founder quality? | ❌ No | ❌ No |
| Can it assess competition? | ❌ No | ❌ No |
| Is it a guarantee? | ❌ No | ❌ No |

---

### 11.12 The Mismatch Between Historical Features and Free-Text Ideas

This mismatch is the central architectural challenge of the project and
must be explicitly acknowledged.

**The historical dataset captures:**
- Structured fields: country, sector, founding year
- Funding outcomes (which we are now excluding from Component A)
- Binary exit outcomes (the target label)

**The user's startup idea submission contains:**
- Unstructured free text describing what the startup will do
- User-selected structured fields (category, country, year)

**The gap:** The historical dataset does not contain any free-text
descriptions of what each startup does. We cannot train an NLP model
to score "problem quality" or "idea clarity" from the Crunchbase data
because those dimensions were never recorded.

**How we bridge the gap:**
- Component A uses only the structured fields that exist in both the
  historical data and the user's submission (sector, country, year).
- Component B evaluates the free text using manually defined heuristics,
  completely independently of the historical training data.
- The two components produce separate outputs that are combined in the
  final report. They are never merged into a single learned model,
  because doing so would require training data that does not exist.

---

### 11.13 Honest Limitations of the Revised Architecture

**Limitation 1 — Component A has a small, clean feature set.**  
By removing all funding features, the ML model is trained on only sector,
country, and year. These three dimensions carry real signal but are
limited. The model will not be highly accurate — it will identify broad
structural patterns (e.g., some sectors and geographies have higher
historical success rates) but cannot capture individual company dynamics.
This is honest and appropriate for the available data.

**Limitation 2 — Component B scores are designer-defined, not data-driven.**  
The dimension weights (25%, 25%, 20%, etc.) and keyword lists are chosen
by the project team, not learned from labeled examples. A more rigorous
version of this project would require a dataset of human-evaluated startup
pitches with ground-truth quality scores — which does not exist in our data.

**Limitation 3 — Category mapping may not be perfect.**  
The historical dataset uses Crunchbase's category taxonomy (858 unique
categories, top 50 used). A user selecting "AgriTech" as their sector
may map to multiple historical categories or may not map cleanly to any.
The UI must use the actual category names from the top-50 list, or provide
a curated mapping layer.

**Limitation 4 — Year 2024 is outside the training distribution.**  
The historical dataset primarily covers 2005–2015. A company founded in
2024 falls outside this range. The model's `founding_year` feature will
extrapolate rather than interpolate for recent ideas. This must be
disclosed in the UI.

**Limitation 5 — Historical success rates are not predictions.**  
If Component A outputs "34% of Agriculture + India companies succeeded,"
this does not mean "your specific idea has a 34% chance of success." It
means "in the dataset, 34% of structurally similar companies reached an
exit event." The user's individual idea outcome depends on factors the
model cannot assess.

---

### 11.14 Revised Summary of All Decisions

This table supersedes the table in Section 10.

| Decision | Choice | Key Reason |
|----------|--------|-----------|
| Target definition | acquired+ipo=1, closed=0 | Standard exit-event definition; externally validated |
| Operating companies | Excluded from training | Unknown outcome; labeling fabricates training data |
| **Revised prediction point** | **Day 0 — before any funding** | Must match what user can provide: no funding info available |
| `funding_total_usd` | **EXCLUDED from Component A** | Post-prediction; not available from idea submission |
| `funding_rounds` | **EXCLUDED from Component A** | Post-prediction; not available from idea submission |
| `time_to_first_funding` | **EXCLUDED from Component A** | Cannot exist before first funding occurs |
| `first_funding_at` | **EXCLUDED from Component A** | Future event from Day 0 perspective |
| `last_funding_at` | EXCLUDED (unchanged) | Critical temporal leakage |
| `category_list` | Multi-hot top 50 + "Other" flag | Core structural signal; user selects from dropdown |
| `country_code` | Top 10 binary + `is_usa` + Other | Core structural signal; user selects from dropdown |
| `founding_year` | Integer from `founded_at` | Structural context; user provides |
| `is_sf_bay_area` | Binary flag from `region` | Geography sub-signal |
| `is_california` | Binary flag from `state_code` | Geography sub-signal |
| `city` | Excluded | Too high cardinality; signal captured elsewhere |
| `status` | Target only | Never a feature |
| `permalink`, `name`, `homepage_url` | Excluded | Identifiers / metadata |
| Missing values | Median imputation + binary flag | Preserves rows; does not hide missingness |
| Class imbalance | No resampling needed | 0.88:1 near-balanced |
| Train/test split | Before computing statistics | Prevents preprocessing leakage |
| **System architecture** | **Two-component design** | ML model + scoring engine, clearly separated |
| **Free-text evaluation** | **Rule-based heuristics (Component B)** | No labeled free-text training data exists in dataset |
| **Category input method** | **User dropdown (not NLP inference)** | Prevents NLP misclassification from entering ML model |

---

*Revised section written before Phase 2 begins.*  
*All decisions are grounded in the actual Phase 1 dataset analysis.*  
*No model has been trained. No performance numbers are claimed.*  
*This revision must be reviewed and approved before Phase 2 starts.*
