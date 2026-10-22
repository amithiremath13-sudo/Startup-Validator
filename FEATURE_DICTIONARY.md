# Feature Dictionary
## AI-Based Startup Viability Prediction and Validation System
### Phase 2 — Final ML Feature Set

---

> **Prediction point: Day 0 — startup idea / founding stage, before any
> funding has been received.**
>
> Every feature in this dictionary is legitimately available at Day 0.
> No feature contains post-founding or post-funding information.
>
> **Total features: 65** (revised from 68 after Phase 2 review)
>
> **Phase 2 review removals:** `founding_year` (temporal distribution
> mismatch), `has_founding_date` (always 1 at inference time), `is_usa`
> (redundant with `country_USA`). See `PREPROCESSING_REPORT.md` Section 16.
>
> Source: `processed_data/feature_names.json` and `processed_data/train_features.csv`

---

## How to Read This Dictionary

Each feature entry contains:
- **Feature name** — the exact column name in the CSV files
- **Source column** — which raw CSV column it was derived from
- **Meaning** — what the feature represents in plain English
- **Type** — data type in the feature matrix
- **Values** — possible values
- **How created** — the transformation applied
- **Legitimate at Day 0?** — explicit justification for why this feature
  does not cause leakage
- **Missing value handling** — what happens if the source data was absent

---

## Group 1 — Geography Flags (2 features)

These are binary flags derived from geographic columns. They represent
where the company is headquartered — information that is fixed at founding
and does not change based on the company's outcome.

> **Phase 2 review:** `is_usa` was removed from this group (redundant
> with `country_USA`). This group now contains 2 features.

---

### `is_usa` — **REMOVED (Phase 2 review)**

> **Decision: EXCLUDED from feature matrix.**
>
> `is_usa` encoded 1 if country_code == 'USA'. This is identical to
> `country_USA` produced by the country encoding step. Keeping both
> creates a perfectly correlated pair of features with no added value.
> `country_USA` is retained; `is_usa` is removed.

---

### `is_sf_bay_area`

| Property | Value |
|----------|-------|
| **Source column** | `region` |
| **Meaning** | 1 if the company is in the San Francisco Bay Area (Silicon Valley + San Francisco), 0 otherwise |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | `(region == "SF Bay Area").astype(int)` |
| **Legitimate at Day 0?** | ✅ Yes. Office location is fixed at founding. |
| **Missing value handling** | If `region` is NaN → `is_sf_bay_area = 0` |
| **Training distribution** | 1,986 / 10,667 rows = 18.6% |
| **Why this feature matters** | The SF Bay Area has the densest concentration of venture capital and technology acquirers. It is the sub-national geography with the highest predictive relevance in this dataset. |
| **Note** | The raw `region` column (1,092 unique values) is not used directly — only this derived flag is included. |

---

### `is_california`

| Property | Value |
|----------|-------|
| **Source column** | `state_code` |
| **Meaning** | 1 if the company is in the state of California (USA), 0 otherwise |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | `(state_code == "CA").astype(int)` |
| **Legitimate at Day 0?** | ✅ Yes. State of incorporation is set at founding. |
| **Missing value handling** | If `state_code` is NaN → `is_california = 0` |
| **Training distribution** | 2,706 / 10,667 rows = 25.4% |
| **Why this feature matters** | California is broader than just the Bay Area — it includes Los Angeles, San Diego, and other technology hubs. It captures a wider California tech-corridor signal. |
| **Note** | The raw `state_code` column mixes US state codes (CA, NY) with non-US regional codes (H9, 7, 2) and is not used directly. |

---

## Group 2 — Country Flags (11 features)

These 11 features encode the company's country of headquarters as a set
of binary columns. Computed from `country_code` using training-set
frequencies only.

**Top 10 countries selected (by training-set frequency):**
USA, Unknown, GBR, CAN, ISR, DEU, FRA, RUS, CHN, IND

**Why binary (one-hot style) and not ordinal encoding?**
Ordinal encoding assigns numbers (USA=1, GBR=2, ...) which implies
USA is "less than" GBR. Country codes have no natural order. Binary
encoding gives each country its own independent dimension.

---

### `country_USA`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in the United States |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | `(country_code == "USA").astype(int)` after filling missing with "Unknown" |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 6,533 (61.2%) |
| **Note** | This overlaps with `is_usa`. Both are kept because `is_usa` is a direct derivation while `country_USA` is part of the systematic country encoding group. In Phase 3, if multicollinearity is a concern, `is_usa` may be dropped in favour of `country_USA`. |

---

### `country_Unknown`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company's country was not recorded in the dataset |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | Missing `country_code` filled with "Unknown" string; then encoded as a country |
| **Legitimate at Day 0?** | ✅ Yes. Missing geography is itself a data pattern — it is knowable at Day 0 that the location was not recorded. |
| **Training count** | 1,582 (14.8%) |
| **Why keep "Unknown" as a category?** | 1,582 training rows have no country recorded. Dropping these rows wastes 14.8% of training data. Treating "Unknown" as its own category lets the model learn whether "unrecorded geography" correlates with outcome — which it may, because well-documented companies may have systematically different success rates from poorly-documented ones. |

---

### `country_GBR`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in the United Kingdom |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 440 (4.1%) |

---

### `country_CAN`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in Canada |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 330 (3.1%) |

---

### `country_ISR`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in Israel |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 164 (1.5%) |
| **Note** | Israel has a disproportionately high number of technology acquisitions relative to its size ("Startup Nation" effect). |

---

### `country_DEU`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in Germany |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 152 (1.4%) |

---

### `country_FRA`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in France |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 147 (1.4%) |

---

### `country_RUS`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in Russia |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 140 (1.3%) |

---

### `country_CHN`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in China |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 136 (1.3%) |

---

### `country_IND`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if company is headquartered in India |
| **Type** | Binary integer |
| **Legitimate at Day 0?** | ✅ Yes |
| **Training count** | 109 (1.0%) |

---

### `country_other`

| Property | Value |
|----------|-------|
| **Source column** | `country_code` |
| **Meaning** | 1 if the company's country is not among the top 10 countries |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | `(~country_code.isin(top_10_list)).astype(int)` |
| **Legitimate at Day 0?** | ✅ Yes |
| **Why needed** | Countries outside the top 10 are too sparse for individual columns, but knowing a company is in a non-major-hub country is still a meaningful signal. |

---

## Group 3 — Year Features (0 features — both removed in Phase 2 review)

> **Both year features were excluded following the Phase 2 review.**
> See `PREPROCESSING_REPORT.md` Section 16 for full explanation.

---

### `founding_year` — **REMOVED (Phase 2 review)**

> **Decision: EXCLUDED from feature matrix.**
>
> **Source column:** `founded_at`
>
> **Why removed:** The training dataset's maximum founding year is 2016.
> New startup ideas submitted in 2026+ would have a `founding_year` value
> 10+ years beyond anything the model saw during training. Using this
> feature would require the model to extrapolate far outside its training
> distribution for every single new inference. This creates unreliable
> predictions. The feature is excluded from the primary ML matrix.
>
> This is a data integrity decision. No claim is made about whether
> removing it improves or worsens model accuracy — that requires testing.

---

### `has_founding_date` — **REMOVED (Phase 2 review)**

> **Decision: EXCLUDED from feature matrix.**
>
> **Source column:** `founded_at`
>
> **Why removed:** This flag equals 1 if the company had a recorded
> founding date in Crunchbase, 0 if missing. In the historical training
> data, 28.1% of companies had no founding date (a Crunchbase data
> recording gap).
>
> At inference time, a user submitting a startup idea is founding a
> company right now — they always have a founding date (today). Therefore
> this flag would always be 1 for every new submission, making it a
> constant with zero discriminatory power. Including a constant feature
> wastes a feature slot and contributes nothing to the model's decisions.

---

## Group 4 — Category Features (52 features)

These 52 features encode the startup's industry sector. Derived from
`category_list`, which uses pipe-separated tags (e.g. `"Mobile|Apps|Games"`).

**Top-50 categories** were selected by training-set frequency. One
binary column was created per category. Two additional columns
(`category_other`, `n_categories`) complete the group.

**Important:** Category names with spaces were converted to underscores,
`+` was converted to `plus`, and `/` to `_` to create valid column names.
The original category name is shown in the "Source category" field.

---

### `cat_Software` through `cat_FinTech` (50 features)

The following table documents all 50 individual category binary columns.
All share the same structure, so properties common to all are listed once,
followed by the per-column specifics.

**Common properties for all `cat_*` columns:**

| Property | Value |
|----------|-------|
| **Source column** | `category_list` |
| **Type** | Binary integer |
| **Values** | 0 (company does not have this category) or 1 (company has this category) |
| **How created** | Split `category_list` on `\|`; check if this category is in the resulting list |
| **Legitimate at Day 0?** | ✅ Yes. Industry sector is chosen by the founders when the company is established. It does not change based on whether the company later succeeds or fails. |
| **Missing value handling** | If `category_list` was NaN → all `cat_*` columns = 0; `cat_Unknown = 1` |
| **Encoding scope** | Categories determined from training set only; same list applied to test set |

**All 50 category columns:**

| # | Feature name | Source category | Training count |
|---|-------------|-----------------|---------------:|
| 1 | `cat_Software` | Software | 1,493 |
| 2 | `cat_Biotechnology` | Biotechnology | 924 |
| 3 | `cat_Mobile` | Mobile | 883 |
| 4 | `cat_Unknown` | (missing category) | 858 |
| 5 | `cat_Curated_Web` | Curated Web | 776 |
| 6 | `cat_E-Commerce` | E-Commerce | 534 |
| 7 | `cat_Social_Media` | Social Media | 499 |
| 8 | `cat_Advertising` | Advertising | 488 |
| 9 | `cat_Enterprise_Software` | Enterprise Software | 482 |
| 10 | `cat_Games` | Games | 402 |
| 11 | `cat_Health_Care` | Health Care | 402 |
| 12 | `cat_Services` | Services | 344 |
| 13 | `cat_Internet` | Internet | 323 |
| 14 | `cat_Finance` | Finance | 293 |
| 15 | `cat_Technology` | Technology | 279 |
| 16 | `cat_Analytics` | Analytics | 264 |
| 17 | `cat_Hardware_plus_Software` | Hardware + Software | 242 |
| 18 | `cat_Security` | Security | 234 |
| 19 | `cat_Clean_Technology` | Clean Technology | 229 |
| 20 | `cat_Apps` | Apps | 217 |
| 21 | `cat_Semiconductors` | Semiconductors | 215 |
| 22 | `cat_SaaS` | SaaS | 213 |
| 23 | `cat_Health_and_Wellness` | Health and Wellness | 209 |
| 24 | `cat_Web_Hosting` | Web Hosting | 207 |
| 25 | `cat_Networking` | Networking | 206 |
| 26 | `cat_Video` | Video | 203 |
| 27 | `cat_Social_Network_Media` | Social Network Media | 195 |
| 28 | `cat_Search` | Search | 192 |
| 29 | `cat_Manufacturing` | Manufacturing | 188 |
| 30 | `cat_Music` | Music | 183 |
| 31 | `cat_Education` | Education | 183 |
| 32 | `cat_Messaging` | Messaging | 178 |
| 33 | `cat_Cloud_Computing` | Cloud Computing | 175 |
| 34 | `cat_Medical` | Medical | 172 |
| 35 | `cat_News` | News | 171 |
| 36 | `cat_Travel` | Travel | 168 |
| 37 | `cat_Information_Technology` | Information Technology | 167 |
| 38 | `cat_Web_Development` | Web Development | 165 |
| 39 | `cat_Telecommunications` | Telecommunications | 163 |
| 40 | `cat_Business_Services` | Business Services | 163 |
| 41 | `cat_Media` | Media | 162 |
| 42 | `cat_Sales_and_Marketing` | Sales and Marketing | 162 |
| 43 | `cat_Entertainment` | Entertainment | 161 |
| 44 | `cat_iPhone` | iPhone | 161 |
| 45 | `cat_Design` | Design | 161 |
| 46 | `cat_Fashion` | Fashion | 158 |
| 47 | `cat_Consulting` | Consulting | 157 |
| 48 | `cat_Marketplaces` | Marketplaces | 155 |
| 49 | `cat_Android` | Android | 154 |
| 50 | `cat_FinTech` | FinTech | 153 |

> **Minimum training count in top 50:** 153 (FinTech)
> This is above the 1% threshold (133 rows) set in the methodology,
> confirming all 50 columns have sufficient training examples.

---

### `category_other`

| Property | Value |
|----------|-------|
| **Source column** | `category_list` |
| **Meaning** | 1 if the company has at least one category that is NOT in the top 50 |
| **Type** | Binary integer |
| **Values** | 0 or 1 |
| **How created** | Check if any of the company's parsed categories are absent from the top-50 list |
| **Legitimate at Day 0?** | ✅ Yes |
| **Why needed** | Companies operating in rare or niche sectors are not entirely unrepresented. This flag tells the model "this company has a category, but it's not a common one." |
| **Note** | A company can have `category_other = 1` AND also have `cat_Software = 1` if it is tagged in both a top-50 and a non-top-50 category. |

---

### `n_categories`

| Property | Value |
|----------|-------|
| **Source column** | `category_list` |
| **Meaning** | The number of individual category tags the company has |
| **Type** | Integer |
| **Values** | 0 (no categories) to 17 (maximum in training set) |
| **How created** | Count of pipe-separated items after splitting `category_list` on `\|` |
| **Legitimate at Day 0?** | ✅ Yes. The number of sectors a company operates in is defined at founding. |
| **Missing value handling** | If `category_list` was NaN → `n_categories = 0` (treated as zero categories). The `cat_Unknown` flag separately marks this situation. |
| **Training mean** | 2.20 categories per company |
| **Why this feature matters** | A company tagged in many categories may be unfocused (covering too many sectors) or thoroughly documented. Either interpretation could carry predictive signal. |

---

## Summary: Columns Used and Excluded

### Columns from the raw CSV and their fate

| Raw CSV Column | Used in features? | Fate |
|---------------|:-----------------:|------|
| `permalink` | ❌ | Excluded — identifier |
| `name` | ❌ | Excluded — identifier |
| `homepage_url` | ❌ | Excluded — metadata |
| `category_list` | ✅ | → 50 `cat_*` columns + `category_other` + `n_categories` |
| `funding_total_usd` | ❌ | Excluded — post-prediction lifetime total |
| `status` | ❌ | Target only — never a feature |
| `country_code` | ✅ | → 10 `country_*` columns + `country_other` |
| `state_code` | ✅ (partial) | → `is_california` only; raw column excluded |
| `region` | ✅ (partial) | → `is_sf_bay_area` only; raw column excluded |
| `city` | ❌ | Excluded — 5,111 unique values |
| `funding_rounds` | ❌ | Excluded — post-prediction lifetime count |
| `founded_at` | ❌ | `founding_year` derived but then **excluded** (Phase 2 review); raw date excluded |
| `first_funding_at` | ❌ | Excluded — future event from Day 0 |
| `last_funding_at` | ❌ | Excluded — critical temporal leakage |

### Final count: 65 features from 4 raw source columns

| Source column | Features produced |
|---------------|:-----------------:|
| `category_list` | 52 |
| `country_code` | 11 |
| `region` | 1 |
| `state_code` | 1 |
| **Total** | **65** |

---

## What These Features Can and Cannot Predict

**These 68 features can inform predictions about:**
- Which sectors have historically higher success rates
- Which geographic markets produce more exits
- Whether US / Bay Area location correlates with outcome
- Whether the founding era correlates with outcome
- Whether companies with no recorded category or geography behave differently

**These 68 features cannot inform predictions about:**
- Idea quality (no text description in the training data)
- Founder capability (not recorded)
- Market size (not in the dataset)
- Competition level (not in the dataset)
- Revenue potential (not in the dataset)
- Product-market fit (not in the dataset)

These limitations are not a flaw in the preprocessing — they reflect the
honest boundaries of what the Crunchbase dataset contains. The Startup
Evaluation Scoring Engine (Component B) addresses idea-level dimensions
separately, using rule-based evaluation of the user's free-text submission.

---

*This feature dictionary documents the exact feature set produced by*
*`02_preprocess_data.py` and stored in `processed_data/`.*
*No model has been trained. Feature selection decisions may be revisited*
*after initial model evaluation in Phase 3.*
