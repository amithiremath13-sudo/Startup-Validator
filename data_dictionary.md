# Data Dictionary
## AI-Based Startup Viability Prediction and Validation System
### Dataset: `big_startup_secsees_dataset.csv`

---

**Last updated:** Phase 1 – Dataset Analysis  
**Total rows:** 66,368 companies  
**Total columns:** 14  
**Source:** Crunchbase historical startup dataset  

> **How to read this document:**  
> Each column gets its own section. For every column you will find:
> - what it means in plain English
> - what data type it is
> - how many values are missing
> - whether it is useful for machine learning (ML)
> - what should be done with it before training a model

---

## Table of Contents

| # | Column | Short Description | Use as ML Feature? |
|---|--------|-------------------|--------------------|
| 1 | `permalink` | Unique URL identifier | ❌ Exclude |
| 2 | `name` | Company name | ❌ Exclude (as feature) |
| 3 | `homepage_url` | Company website URL | ❌ Exclude |
| 4 | `category_list` | Industry / sector tags | ✅ Transform |
| 5 | `funding_total_usd` | Total funding raised (USD) | ⚠️ Use with caution |
| 6 | `status` | Current company status | 🎯 TARGET ONLY |
| 7 | `country_code` | Country of headquarters | ✅ Encode |
| 8 | `state_code` | US state of headquarters | ⚠️ Conditional |
| 9 | `region` | Geographic region | ⚠️ Conditional |
| 10 | `city` | City of headquarters | ❌ Exclude or group |
| 11 | `funding_rounds` | Number of funding rounds | ⚠️ Use with caution |
| 12 | `founded_at` | Date company was founded | ✅ Engineer features |
| 13 | `first_funding_at` | Date of first funding round | ✅ Engineer features |
| 14 | `last_funding_at` | Date of most recent funding | ❌ Leakage risk |

---

## Column Details

---

### 1. `permalink`

**Plain English:**  
A unique web address slug that identifies each company in the Crunchbase
database. It looks like `/organization/company-name`. Think of it like a
username — every company gets exactly one.

**Data Type:** Text (string / `object` in pandas)

**Example values:**
```
/organization/-qounter
/organization/0-6-com
/organization/facebook
```

**Missing values:** Very few or none — this is the primary key of the dataset.

**Potential usefulness:**  
None for machine learning. This is just an ID number in disguise. The model
cannot learn anything from a URL slug. However it is useful as a **row
identifier** when debugging or merging datasets.

**What to do with it:**  
Exclude from the feature set. Keep it in the raw data for record-keeping.
When checking for true duplicate rows, use `permalink` as the ground-truth
unique identifier — if two rows share the same permalink, they are genuine
duplicates.

**ML Feature?** ❌ No — identifier column only.

---

### 2. `name`

**Plain English:**  
The public name of the company, exactly as it appears on Crunchbase.

**Data Type:** Text (string / `object`)

**Example values:**
```
:Qounter
(THE) ONE of THEM,Inc.
Facebook
Google
```

**Missing values:** Rare — almost all companies have a name.

**Potential usefulness:**  
Not useful as a raw feature. A company called "TechCorp" is not inherently
more likely to succeed than one called "BlueSky." However, advanced NLP
techniques (not covered in this phase) could theoretically extract signals
from naming patterns (e.g. length, word choice). For this academic project,
we exclude it.

**What to do with it:**  
Exclude from the feature matrix. Keep it as a human-readable label in the
dataset for debugging and presentation.

> **Note on duplicates:** Some company names appear more than once. This
> does NOT always mean they are the same company — two different startups
> can share a name. Always verify using `permalink` before deduplication.

**ML Feature?** ❌ No — too noisy, not a meaningful signal at this level.

---

### 3. `homepage_url`

**Plain English:**  
The company's official website address (URL).

**Data Type:** Text (string / `object`)

**Example values:**
```
http://www.facebook.com
http://livfame.com
http://www.qounter.com
```

**Missing values:** Moderate — some companies have no website recorded.

**Potential usefulness:**  
Not useful in raw form. In advanced NLP projects, the URL could be scraped
to analyze website content, technology stack (via Wappalyzer), or domain
registration details. That is far beyond Phase 1 scope.

A simple derived feature could be: `has_website` (1 if URL exists, 0 if
missing) — which might weakly signal professionalism.

**What to do with it:**  
Exclude from the ML feature set for this project. Optionally create a
binary `has_website` flag in Phase 2.

**ML Feature?** ❌ No (raw URL). ⚠️ Optional: derive `has_website` binary flag.

---

### 4. `category_list`

**Plain English:**  
A list of industry tags or sectors that describe what the startup does.
Multiple categories are separated by a pipe `|` character.

**Data Type:** Text (string / `object`)

**Example values:**
```
Media
Application Platforms|Real Time|Social Network Media
Apps|Games|Mobile
Curated Web
```

**Missing values:** Some companies have no category assigned — around 5–15%
of rows (exact count confirmed by running `01_explore_dataset.py`).

**Potential usefulness:**  
**High.** The industry sector a startup operates in is one of the most
important signals for viability. A SaaS company and a restaurant chain
face very different success rates. This column contains rich information
that must be extracted carefully.

**What to do with it:**  
This column CANNOT be fed directly to a machine learning model as text.
It must be **transformed** first. See the three strategies below:

**Strategy A — Multi-hot encoding (recommended):**  
Split on `|`, identify the top 30–50 most common categories, and create
one binary column per category:

| company | category_Mobile | category_Software | category_Games |
|---------|----------------|-------------------|----------------|
| Alpha   | 1              | 0                 | 1              |
| Beta    | 0              | 1                 | 0              |

This is the approach recommended for this project.

**Strategy B — Sector grouping:**  
Manually map hundreds of specific categories to a smaller set of
macro-sectors (e.g., "FinTech", "HealthTech", "Consumer Apps").

**Strategy C — Frequency / success-rate encoding:**  
Replace each category with the historical success rate of companies in
that category. Must be done with cross-validation to avoid data leakage.

**Additional derived feature:**  
`n_categories` — the number of pipe-separated values. A startup listed in
many categories might be unfocused (or just well-described in the data).

**ML Feature?** ✅ Yes — after transformation. Raw text value must be encoded first.

---

### 5. `funding_total_usd`

**Plain English:**  
The total amount of money, in US dollars, that the company raised from
investors across all funding rounds combined.

**Data Type:** Mixed — stored as text (`object`), contains numeric values
BUT also non-numeric entries like `-` (a dash), empty strings, and
possibly formatted numbers. Must be converted to numeric (`float`) before use.

**Example values:**
```
10000000        → $10 million (valid)
700000          → $700,000 (valid)
-               → missing/unknown (non-numeric, becomes NaN after conversion)
(blank)         → missing (becomes NaN)
```

**Missing values:** Two kinds:
1. Originally blank/null rows
2. Rows where the value was non-numeric (e.g. `-`) that become `NaN`
   after conversion. The EDA script counts both separately.

**Potential usefulness:**  
**High — but with a critical warning.**

Higher funding is strongly correlated with success (acquired/ipo). However
this column carries a **data leakage risk**: the value here represents the
company's TOTAL lifetime funding. For a company later acquired, this might
include funding rounds raised just before acquisition — information that
would not be available at early prediction time.

**What to do with it:**  
- Convert to numeric using `pd.to_numeric(..., errors='coerce')`
- Apply `log10(x + 1)` transformation before training — funding values
  span many orders of magnitude ($100 to $100 billion), so the raw number
  would dominate all other features if used directly
- Fill missing values with 0 (if missing likely means "no funding recorded")
  or with the median (if missing means "unknown")
- Document the leakage caveat clearly in the project report

**Outlier handling:**  
A few companies raise billions (Uber, Airbnb, etc.). These are not data
errors — they are real. Use log-scaling rather than removing them.

**ML Feature?** ⚠️ Yes, with caution. Use log-transformed. Document leakage risk.

---

### 6. `status`

**Plain English:**  
The recorded status of the company at the time the dataset was compiled.
This is the most important column in the entire dataset.

**Data Type:** Text (string / `object`)

**Possible values and their meaning:**

| Value | Meaning | Label in this project |
|-------|---------|----------------------|
| `acquired` | The company was purchased by another company | **SUCCESS = 1** |
| `ipo` | The company went public on a stock exchange | **SUCCESS = 1** |
| `closed` | The company shut down (failed) | **FAILURE = 0** |
| `operating` | The company is still running at data collection time | **EXCLUDED** |
| `(missing)` | Status was not recorded | **EXCLUDED** |

**Missing values:** Some rows have no status recorded — these are excluded
from training.

**Why `operating` is excluded:**  
A company labeled `operating` at the time the data was collected has an
UNKNOWN final outcome. It might later be acquired (success), go bankrupt
(failure), or continue operating indefinitely. Labeling it as either
success or failure would introduce incorrect, fabricated training labels.
This is not an assumption — it is a strict data integrity requirement.

**What to do with it:**  
- Create binary target: `target = 1` if `acquired` or `ipo`, `target = 0` if `closed`
- Remove all `operating` and NaN rows from the training dataset
- Save `operating` companies to a separate file for later inference

**ML Feature?** 🎯 This IS the target (label). Never use it as an input feature.

---

### 7. `country_code`

**Plain English:**  
A 3-letter ISO country code indicating where the company is headquartered.

**Data Type:** Text (string / `object`)

**Example values:**
```
USA   → United States of America
GBR   → United Kingdom
IND   → India
CHN   → China
DEU   → Germany
```

**Missing values:** Moderate — companies without a recorded location have no
country code. The EDA script reports the exact count.

**Potential usefulness:**  
**Moderate to high.** Geography affects startup success through factors like:
- access to venture capital networks (Silicon Valley effect)
- legal and regulatory environment
- market size and maturity
- exit market conditions (US IPOs, European acquisitions, etc.)

The dataset is heavily dominated by USA companies (~50–60% of all rows).

**What to do with it:**  
- Option 1 (simple): Create a binary `is_usa` flag (1 if USA, 0 otherwise)
- Option 2 (richer): Target-encode top 20 countries by frequency; group
  the rest as "Other"
- Fill missing values with "Unknown" before encoding
- Do NOT use raw string values in ML — encode to numbers first

**ML Feature?** ✅ Yes — after encoding. At minimum, a binary `is_usa` flag.

---

### 8. `state_code`

**Plain English:**  
For US companies: the two-letter abbreviation of the US state where the
company is headquartered (e.g., `CA` for California, `NY` for New York).
For non-US companies: sometimes contains a regional code, but coverage is
inconsistent and sparse.

**Data Type:** Text (string / `object`)

**Example values:**
```
CA    → California (USA)
NY    → New York (USA)
TX    → Texas (USA)
16    → (non-US numeric regional code — inconsistent)
```

**Missing values:** High — only meaningful for US companies, so all non-US
rows effectively have a non-useful or missing value here.

**Potential usefulness:**  
Limited and conditional. California (Silicon Valley) and New York have
significantly higher startup success rates than other states. This signal
is real but only applies to ~50% of the dataset (US companies).

**What to do with it:**  
- Only use this column for the US-company subset of the data
- Create a derived feature: `is_silicon_valley` (1 if state_code == 'CA'
  AND region contains 'SF Bay Area')
- For the global model, this column adds noise more than signal

**ML Feature?** ⚠️ Conditional — useful for US-only analysis; mostly noise for global model.

---

### 9. `region`

**Plain English:**  
A geographic region name, typically used for US companies to describe
metro areas (e.g., "SF Bay Area", "New York City", "Los Angeles").
For non-US companies, coverage is inconsistent.

**Data Type:** Text (string / `object`)

**Example values:**
```
SF Bay Area
New York City
Los Angeles
DE - Other
Beijing
```

**Missing values:** High for non-US companies.

**Potential usefulness:**  
Similar to `state_code` — meaningful primarily for US companies. The
"SF Bay Area" region has a disproportionately high startup success rate
due to VC ecosystem density.

**What to do with it:**  
- For a US-focused sub-analysis: encode top 10 regions
- For the global model: too sparse and inconsistent — exclude or use only
  as a derived `is_sf_bay_area` binary flag

**ML Feature?** ⚠️ Conditional — derive a binary `is_sf_bay_area` flag if useful.

---

### 10. `city`

**Plain English:**  
The city where the company is headquartered.

**Data Type:** Text (string / `object`)

**Example values:**
```
San Francisco
New York City
London
Beijing
Mumbai
```

**Missing values:** Moderate — similar to country/region, some companies
have no location recorded.

**Potential usefulness:**  
Very limited for ML in raw form. There are likely 5,000–10,000+ unique
city names in this dataset. Using city directly would create an enormous
number of columns after encoding, most of which would have very few
examples — this is called the "curse of dimensionality" and hurts models.

**What to do with it:**  
- Exclude from the ML feature set in most cases
- Exception: create binary flags for the top 5 startup hubs
  (San Francisco, New York, London, Beijing, Boston)
- Most predictive signal from city is already captured by country + region

**ML Feature?** ❌ Exclude in raw form. ⚠️ Optional: binary flags for top 5 hub cities.

---

### 11. `funding_rounds`

**Plain English:**  
The total number of distinct funding rounds the company completed. Each
time a startup raises money from investors (Seed, Series A, Series B, etc.)
that counts as one round.

**Data Type:** Numeric (integer), but stored as `object` in the raw CSV.
Requires conversion to numeric.

**Example values:**
```
1  → raised funding once (e.g., just a Seed round)
2  → raised funding twice (e.g., Seed + Series A)
7  → raised seven rounds (late-stage company)
```

**Missing values:** Some rows have no value — indicates the funding history
was not recorded.

**Potential usefulness:**  
**Moderate to high — but carries the same leakage risk as `funding_total_usd`.**

More funding rounds generally indicates investor confidence and company
longevity. However, the total rounds here includes ALL rounds ever completed,
including those raised AFTER a company's success was already determined.

**What to do with it:**  
- Convert to integer
- Fill missing values with 0 (no rounds recorded) or median
- Document that this reflects lifetime rounds, not early-stage rounds
- Use with explicit leakage caveat in the model documentation

**ML Feature?** ⚠️ Yes, with caution. Numeric. Document leakage risk.

---

### 12. `founded_at`

**Plain English:**  
The date the company was officially founded (incorporated or launched).

**Data Type:** Text (`object`) in raw CSV. Must be parsed to `datetime`.

**Example values:**
```
2014-09-04   → September 4, 2014
2007-01-01   → January 1, 2007
(blank)      → founding date not recorded
```

**Missing values:** Moderate — many companies do not have a founding date
in the dataset. This is one of the more commonly missing fields.

**Potential usefulness:**  
**High — but not as a raw date.** Raw dates are meaningless to ML models.
The useful derived features are:

1. `founding_year` — the calendar year of founding. Earlier companies in
   the dataset had more time to reach an outcome, so this matters.

2. `time_to_first_funding` — the number of days between `founded_at` and
   `first_funding_at`. This measures how quickly a startup attracted
   investment: a proxy for momentum and investor appeal.

3. `company_age_at_prediction` — if we define a prediction point (e.g.,
   2 years after founding), this is the age at that point.

**What to do with it:**  
- Parse to datetime using `pd.to_datetime(..., errors='coerce')`
- Extract `founding_year` as an integer feature
- Compute `time_to_first_funding` (see column 13)
- Flag suspicious dates (far future or before 1900)

**ML Feature?** ✅ Yes — via engineered derived features. Raw date itself: No.

---

### 13. `first_funding_at`

**Plain English:**  
The date the company received its very first external investment (e.g.,
Angel round, Pre-Seed, or Seed round).

**Data Type:** Text (`object`) in raw CSV. Must be parsed to `datetime`.

**Example values:**
```
2015-01-05   → First funding on January 5, 2015
2014-03-01   → First funding on March 1, 2014
(blank)      → No funding recorded (company may be bootstrapped)
```

**Missing values:** Moderate — companies with no recorded funding rounds
will have this blank.

**Potential usefulness:**  
**Moderate.** By itself the date is not useful. But as part of a derived
feature it is valuable:

- `time_to_first_funding = (first_funding_at - founded_at).days`
  → How long did it take investors to commit? Fast fundraising suggests
  strong early traction.

- `first_funding_year` — the year of first funding. Earlier funding
  (relative to founding) may correlate with success.

- `has_received_funding` — binary flag (1 if first_funding_at is not null)

**Leakage level:** LOW to MODERATE. The first funding date is early in the
company's life and is available at most prediction points. This is the
SAFEST of the three date columns to use.

**What to do with it:**  
- Parse to datetime
- Compute `time_to_first_funding`
- Handle negative values (where `first_funding_at < founded_at`) — these
  are data entry errors; treat as missing

**ML Feature?** ✅ Yes — via engineered derived features. Raw date itself: No.

---

### 14. `last_funding_at`

**Plain English:**  
The date of the company's most recent funding round — the last time it
raised money before the dataset was compiled.

**Data Type:** Text (`object`) in raw CSV. Must be parsed to `datetime`.

**Example values:**
```
2015-01-05   → Most recent funding on January 5, 2015
2017-09-20   → Most recent funding on September 20, 2017
(blank)      → No funding recorded
```

**Missing values:** Similar to `first_funding_at`.

**Potential usefulness:**  
**Very limited — and HIGH leakage risk.**

This column represents the MOST RECENT funding event in the company's
entire recorded history. For a company that was acquired in 2018:
- Their last funding might be from late 2017 — just before the acquisition
- The recency of funding is essentially a signal that "something important
  is about to happen"
- This means the model would learn "if last funding is recent → success"
  which is only true because of the acquisition that we are trying to predict

This is a textbook example of **temporal leakage**: information from the
future (the acquisition process) bleeds backwards into a feature (recent
funding activity).

**What to do with it:**  
**Exclude this column from the ML feature set.**

If you must use it, compute `time_since_last_funding` relative to a fixed
cut-off date (e.g., 24 months after founding) — not relative to the outcome.
But this requires knowing the timeline, which we do not have reliably.

**ML Feature?** ❌ Exclude. Very high data leakage risk.

---

## Summary Table — ML Feature Recommendations

| Column | Use? | Action Required | Leakage Risk |
|--------|------|-----------------|--------------|
| `permalink` | ❌ No | Keep as row ID only | None |
| `name` | ❌ No | Keep as label only | None |
| `homepage_url` | ❌ No | Optionally derive `has_website` | None |
| `category_list` | ✅ Yes | Multi-hot encode top 30–50 | None |
| `funding_total_usd` | ⚠️ Caution | Convert numeric, log-scale | **HIGH** |
| `status` | 🎯 Target | Create binary `target` column | N/A |
| `country_code` | ✅ Yes | Encode: `is_usa` + top countries | Low |
| `state_code` | ⚠️ Conditional | US-only; derive `is_bay_area` | Low |
| `region` | ⚠️ Conditional | US-only binary flag | Low |
| `city` | ❌ No | Too high cardinality | None |
| `funding_rounds` | ⚠️ Caution | Convert numeric | **HIGH** |
| `founded_at` | ✅ Yes | Extract year + derive age features | Low |
| `first_funding_at` | ✅ Yes | Derive `time_to_first_funding` | Moderate |
| `last_funding_at` | ❌ No | Exclude — leakage | **VERY HIGH** |

---

## Derived Features Recommended for Phase 2

These columns do not exist in the raw dataset but should be **engineered**
(calculated) from existing columns during preprocessing:

| Derived Feature | Formula | Type | Notes |
|-----------------|---------|------|-------|
| `founding_year` | year from `founded_at` | Integer | Captures era of founding |
| `time_to_first_funding` | `(first_funding_at - founded_at).days` | Integer | Early momentum signal |
| `has_received_funding` | 1 if `first_funding_at` is not null | Binary | Simple presence flag |
| `has_website` | 1 if `homepage_url` is not null | Binary | Proxy for professionalism |
| `is_usa` | 1 if `country_code == 'USA'` | Binary | Geography dominant factor |
| `is_sf_bay_area` | 1 if region == 'SF Bay Area' | Binary | VC ecosystem proximity |
| `log_funding` | `log10(funding_total_usd + 1)` | Float | Normalise skewed funding |
| `n_categories` | count of `\|` splits in `category_list` | Integer | Focus / breadth indicator |
| `category_[X]` | multi-hot per top category | Binary | One column per category |

---

*This data dictionary was prepared for academic purposes only.  
All decisions about feature inclusion must be validated by actual model testing in later phases.  
No model performance claims are made here.*
