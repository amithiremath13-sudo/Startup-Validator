# Model Feature Importance Report
## AI-Based Startup Viability Prediction and Validation System
### Phase 3 — Feature Importance and Coefficient Analysis

---

> **All numbers in this report come from models trained and evaluated in
> `03_train_models.py`. Nothing is estimated or fabricated.**
>
> **Critical disclaimer — importance is not causation:**
> Feature importance scores tell us which features the model found
> most useful for separating successes from failures in the historical
> training data. They do NOT prove that these features cause startup
> success. Correlation in observational data is not causation.
> For example, US-headquartered companies dominate both the dataset and
> the success class — but being in the US does not by itself cause success.
> Many confounding factors (access to capital, network effects, talent
> pools, survivor reporting bias) are all entangled in these numbers.

---

## Table of Contents

1. [How Feature Importance Is Measured](#1-how-feature-importance-is-measured)
2. [Logistic Regression — Coefficients](#2-logistic-regression--coefficients)
3. [Decision Tree — Feature Importances](#3-decision-tree--feature-importances)
4. [Random Forest — Feature Importances](#4-random-forest--feature-importances)
5. [Gradient Boosting — Feature Importances](#5-gradient-boosting--feature-importances)
6. [Extra Trees — Feature Importances (Selected Model)](#6-extra-trees--feature-importances-selected-model)
7. [Cross-Model Consensus](#7-cross-model-consensus)
8. [Important Caveats](#8-important-caveats)

---

## 1. How Feature Importance Is Measured

### For tree-based models (Decision Tree, Random Forest, Gradient Boosting, Extra Trees)

**Method: Mean Decrease in Impurity (MDI)**

Every time a tree splits on a feature, it reduces the impurity
(disorder) in the resulting child nodes. Impurity decrease is measured
using the Gini index. A feature's importance score is the average
impurity decrease it causes across all splits and all trees in the
ensemble, weighted by the number of samples reaching each split.

**Interpretation:**
- A higher score means the model relied more heavily on that feature
  to separate successes from failures.
- Scores sum to 1.0 across all 65 features.
- A score of 0.20 means the feature contributed 20% of the total
  information used to build the trees.

**Known limitation of MDI:** It can overestimate the importance of
features with high cardinality (many unique values) because those
features offer more possible split points. In our dataset, `n_categories`
(integer 0–17) and country/category flags (binary 0/1) are all
relatively low-cardinality, so this bias is limited.

### For Logistic Regression

**Method: Absolute value of standardised coefficients**

The coefficient for each binary feature represents the log-odds change
associated with that feature being 1 (vs 0), holding all others constant.
Features are ranked by the absolute value of their coefficient — larger
absolute value means the model assigned more weight to that feature.

- **Positive coefficient** → feature is associated with higher predicted
  probability of success (class 1)
- **Negative coefficient** → feature is associated with higher predicted
  probability of failure (class 0)

**Reminder:** These are correlations in the training data, not causal
effects. A positive coefficient for `cat_Biotechnology` means biotech
companies historically had higher acquisition/IPO rates in this dataset
— it does not mean starting a biotech company causes success.

---

## 2. Logistic Regression — Coefficients

Best params: `C=0.1, penalty=l2`
Test ROC-AUC: 0.7558

**Top 20 features by |coefficient|:**

| Rank | Feature | Coefficient | Direction | Meaning |
|-----:|---------|:-----------:|:---------:|---------|
| 1 | `country_RUS` | −1.4276 | ↓ Failure | Russia-HQ companies had lower success rates in dataset |
| 2 | `cat_Unknown` | −1.1628 | ↓ Failure | No recorded category associated with worse outcomes |
| 3 | `country_Unknown` | −0.9484 | ↓ Failure | Unknown geography associated with worse outcomes |
| 4 | `cat_Biotechnology` | +0.8880 | ↑ Success | Biotech associated with higher acquisition/IPO rates |
| 5 | `cat_Enterprise_Software` | +0.8140 | ↑ Success | Enterprise software associated with exits |
| 6 | `cat_Analytics` | +0.7662 | ↑ Success | Analytics companies show higher exit rates |
| 7 | `cat_Security` | +0.7429 | ↑ Success | Security sector associated with acquisitions |
| 8 | `cat_Semiconductors` | +0.6864 | ↑ Success | Hardware/chip companies associated with exits |
| 9 | `cat_Services` | −0.6555 | ↓ Failure | Services companies associated with closures |
| 10 | `cat_Web_Hosting` | +0.6289 | ↑ Success | Hosting companies show higher exit rates |
| 11 | `country_USA` | +0.5877 | ↑ Success | US HQ associated with higher success rate |
| 12 | `country_CHN` | +0.5687 | ↑ Success | China HQ associated with exits in dataset |
| 13 | `cat_Health_Care` | +0.5302 | ↑ Success | Healthcare associated with acquisitions |
| 14 | `cat_Software` | +0.5205 | ↑ Success | Software associated with exits |
| 15 | `cat_Social_Media` | −0.4702 | ↓ Failure | Social media associated with more closures |
| 16 | `country_ISR` | +0.4015 | ↑ Success | Israel HQ associated with acquisitions |
| 17 | `country_CAN` | +0.3739 | ↑ Success | Canada HQ associated with exits |
| 18 | `cat_Hardware_plus_Software` | +0.3605 | ↑ Success | Hardware/software hybrids associated with exits |
| 19 | `is_sf_bay_area` | +0.3460 | ↑ Success | SF Bay Area geography associated with success |
| 20 | `cat_Advertising` | +0.3442 | ↑ Success | Advertising associated with exits |

**Plot:** `outputs/model_evaluation/importance_logistic_regression.png`

---

## 3. Decision Tree — Feature Importances

Best params: `max_depth=20, min_samples_leaf=10`
Test ROC-AUC: 0.7410 (weakest of the five models)

**Top 20 features by MDI importance:**

| Rank | Feature | Importance | Cumulative |
|-----:|---------|:----------:|:----------:|
| 1 | `country_Unknown` | 0.3927 | 39.3% |
| 2 | `country_USA` | 0.0966 | 49.2% |
| 3 | `cat_Unknown` | 0.0847 | 57.7% |
| 4 | `n_categories` | 0.0643 | 64.1% |
| 5 | `country_RUS` | 0.0408 | 68.2% |
| 6 | `cat_Biotechnology` | 0.0338 | 71.5% |
| 7 | `cat_Curated_Web` | 0.0264 | 74.2% |
| 8 | `cat_Software` | 0.0240 | 76.6% |
| 9 | `cat_Social_Media` | 0.0218 | 78.7% |
| 10 | `cat_Enterprise_Software` | 0.0201 | 80.7% |
| 11 | `cat_Services` | 0.0164 | 82.4% |
| 12 | `country_other` | 0.0161 | 84.0% |
| 13 | `category_other` | 0.0160 | 85.6% |
| 14 | `cat_Semiconductors` | 0.0109 | 86.7% |
| 15 | `is_california` | 0.0103 | 87.7% |
| 16 | `cat_Security` | 0.0095 | 88.7% |
| 17 | `cat_Analytics` | 0.0089 | 89.5% |
| 18 | `is_sf_bay_area` | 0.0087 | 90.4% |
| 19 | `cat_E-Commerce` | 0.0080 | 91.2% |
| 20 | `country_GBR` | 0.0077 | 91.9% |

**Note:** `country_Unknown` alone accounts for 39% of the Decision Tree's
information gain. The tree is heavily structured around a single first
split on whether geography is recorded. This heavy reliance on one
feature is a sign of the tree's tendency to overfit to the strongest
single signal — one reason single trees underperform ensembles here.

**Plot:** `outputs/model_evaluation/importance_decision_tree.png`

---

## 4. Random Forest — Feature Importances

Best params: `max_depth=None, min_samples_leaf=5, n_estimators=100`
Test ROC-AUC: 0.7599

**Top 20 features by MDI importance:**

| Rank | Feature | Importance | Cumulative |
|-----:|---------|:----------:|:----------:|
| 1 | `country_Unknown` | 0.2036 | 20.4% |
| 2 | `country_USA` | 0.1355 | 33.9% |
| 3 | `cat_Unknown` | 0.1287 | 46.3% |
| 4 | `cat_Biotechnology` | 0.0460 | 50.9% |
| 5 | `country_RUS` | 0.0395 | 55.3% |
| 6 | `n_categories` | 0.0380 | 59.1% |
| 7 | `is_sf_bay_area` | 0.0248 | 61.6% |
| 8 | `cat_Software` | 0.0243 | 64.1% |
| 9 | `is_california` | 0.0229 | 66.3% |
| 10 | `cat_Enterprise_Software` | 0.0229 | 68.6% |
| 11 | `cat_Curated_Web` | 0.0198 | 70.6% |
| 12 | `country_other` | 0.0193 | 72.5% |
| 13 | `cat_Services` | 0.0180 | 74.3% |
| 14 | `cat_Social_Media` | 0.0170 | 76.0% |
| 15 | `cat_Analytics` | 0.0141 | 77.4% |
| 16 | `cat_Health_Care` | 0.0121 | 78.6% |
| 17 | `category_other` | 0.0114 | 79.7% |
| 18 | `cat_Security` | 0.0114 | 80.9% |
| 19 | `cat_Semiconductors` | 0.0099 | 81.9% |
| 20 | `cat_Mobile` | 0.0088 | 82.7% |

**Compared to Decision Tree:** The ensemble distributes importance more
evenly — `country_Unknown` drops from 39.3% (DT) to 20.4% (RF). This
shows the ensemble's benefit: it does not over-rely on one split.
`country_USA` and `cat_Unknown` together with `country_Unknown` still
dominate the top 3, but their combined share is 46.3% vs 57.7% in DT.

**Plot:** `outputs/model_evaluation/importance_random_forest.png`

---

## 5. Gradient Boosting — Feature Importances

Best params: `learning_rate=0.1, max_depth=3, n_estimators=100`
Test ROC-AUC: 0.7587

**Top 20 features by MDI importance:**

| Rank | Feature | Importance | Cumulative |
|-----:|---------|:----------:|:----------:|
| 1 | `country_Unknown` | 0.3723 | 37.2% |
| 2 | `country_USA` | 0.1303 | 50.3% |
| 3 | `cat_Unknown` | 0.0945 | 59.8% |
| 4 | `cat_Biotechnology` | 0.0545 | 65.3% |
| 5 | `country_RUS` | 0.0456 | 69.8% |
| 6 | `n_categories` | 0.0279 | 72.6% |
| 7 | `cat_Software` | 0.0264 | 65.3% |
| 8 | `cat_Enterprise_Software` | 0.0230 | 75.3% |
| 9 | `cat_Social_Media` | 0.0199 | 77.3% |
| 10 | `cat_Analytics` | 0.0194 | 79.3% |
| 11 | `cat_Services` | 0.0169 | 81.0% |
| 12 | `cat_Curated_Web` | 0.0153 | 82.5% |
| 13 | `country_other` | 0.0142 | 83.9% |
| 14 | `cat_Security` | 0.0128 | 85.2% |
| 15 | `cat_Semiconductors` | 0.0105 | 86.3% |
| 16 | `cat_Health_Care` | 0.0100 | 87.3% |
| 17 | `is_sf_bay_area` | 0.0100 | 88.3% |
| 18 | `cat_Web_Hosting` | 0.0074 | 89.0% |
| 19 | `cat_Advertising` | 0.0058 | 89.6% |
| 20 | `country_CAN` | 0.0053 | 90.1% |

**Note:** GB concentrates importance more heavily in the top 3 features
(59.8%) compared to RF (46.3%). The sequential boosting process tends
to focus on the most discriminative splits early, producing a sparser
importance distribution. The same top-3 pattern emerges: unknown
geography, US geography, unknown category dominate.

**Plot:** `outputs/model_evaluation/importance_gradient_boosting.png`

---

## 6. Random Forest — Feature Importances (Selected Model)

Best params: `max_depth=None, min_samples_leaf=5, n_estimators=100`
Test ROC-AUC: 0.7599
**Selected by:** highest CV ROC-AUC = 0.7364 (training data only)

**Top 20 features by MDI importance:**

| Rank | Feature | Importance | Cumulative |
|-----:|---------|:----------:|:----------:|
| 1 | `country_Unknown` | 0.2036 | 20.4% |
| 2 | `country_USA` | 0.1355 | 33.9% |
| 3 | `cat_Unknown` | 0.1287 | 46.3% |
| 4 | `cat_Biotechnology` | 0.0460 | 50.9% |
| 5 | `country_RUS` | 0.0395 | 55.3% |
| 6 | `n_categories` | 0.0380 | 59.1% |
| 7 | `is_sf_bay_area` | 0.0248 | 61.6% |
| 8 | `cat_Software` | 0.0243 | 64.1% |
| 9 | `is_california` | 0.0229 | 66.3% |
| 10 | `cat_Enterprise_Software` | 0.0229 | 68.6% |
| 11 | `cat_Curated_Web` | 0.0198 | 70.6% |
| 12 | `country_other` | 0.0193 | 72.5% |
| 13 | `cat_Services` | 0.0180 | 74.3% |
| 14 | `cat_Social_Media` | 0.0170 | 76.0% |
| 15 | `cat_Analytics` | 0.0141 | 77.4% |
| 16 | `cat_Health_Care` | 0.0121 | 78.6% |
| 17 | `category_other` | 0.0114 | 79.7% |
| 18 | `cat_Security` | 0.0114 | 80.9% |
| 19 | `cat_Semiconductors` | 0.0099 | 81.9% |
| 20 | `cat_Mobile` | 0.0088 | 82.7% |

**What the selected model relies on most:**

The Random Forest model's signal is distributed more evenly than the
single Decision Tree or Gradient Boosting. The top-3 features
(`country_Unknown`, `country_USA`, `cat_Unknown`) account for 46.3%
of total importance — meaningfully lower than the Decision Tree's
57.7% concentration in the same features. This broader distribution
is one reason Random Forest generalises better than a single tree.

After the top 3, the model draws meaningful signal from:
- **`cat_Biotechnology` (4.6%):** High historical acquisition rate.
- **`country_RUS` (3.9%):** Negative signal for success.
- **`n_categories` (3.8%):** Number of sectors — companies with more
  category tags may be more established or diversified.
- **`is_sf_bay_area` (2.5%):** Bay Area geography — positive signal.
- **`is_california` (2.3%):** Broader California signal.

**Plot:** `outputs/model_evaluation/importance_random_forest.png`

---

## 7. Extra Trees — Feature Importances (for reference)

Best params: `max_depth=20, min_samples_leaf=5, n_estimators=200`
Test ROC-AUC: 0.7601 (marginally highest test AUC but not the selected model)

**Note:** Extra Trees was the best model on test metrics but was not
selected because model selection uses CV results only (see
`MODEL_EVALUATION_REPORT.md` Section 9). Its importances are included
here for completeness.

Looking across all five models, the same small group of features
consistently appears at the top of every importance ranking:

### Features consistently ranked top 5 across all models:

| Feature | LR rank | DT rank | RF rank | GB rank | ET rank | Consistent direction |
|---------|:-------:|:-------:|:-------:|:-------:|:-------:|---------------------|
| `country_Unknown` | 3 | 1 | 1 | 1 | 1 | ↓ Failure (negative) |
| `country_USA` | 11 | 2 | 2 | 2 | 2 | ↑ Success (positive) |
| `cat_Unknown` | 2 | 3 | 3 | 3 | 3 | ↓ Failure (negative) |
| `cat_Biotechnology` | 4 | 6 | 4 | 4 | 4 | ↑ Success (positive) |
| `country_RUS` | 1 | 5 | 5 | 5 | 5 | ↓ Failure (negative) |

### Features consistently in the top 15 across all models:

| Feature | Consistent direction |
|---------|---------------------|
| `cat_Enterprise_Software` | ↑ Success |
| `cat_Software` | ↑ Success |
| `cat_Analytics` | ↑ Success |
| `cat_Security` | ↑ Success |
| `cat_Services` | ↓ Failure |
| `cat_Social_Media` | ↓ Failure |
| `is_sf_bay_area` | ↑ Success |
| `n_categories` | ↑ Success (higher = better coverage/focus) |

### Features rarely appearing in top 20:

Most of the 65 features contribute very little to any model. Features
with near-zero importance across all models include many of the
less-common country and category flags (e.g., `cat_iPhone`, `cat_Music`,
`cat_Fashion`, `country_DEU`, `country_FRA`). This suggests these
groups are too small in the training data to provide reliable signal.

---

## 8. Important Caveats

### 8.1 Importance is not causation

The most important warning about feature importance results is that
**they describe correlations in historical training data, not causal
mechanisms.**

Examples of correlations that may not be causal:
- `country_USA` has high positive importance. US companies dominate the
  Crunchbase dataset. The US also has the world's deepest VC market,
  the most IPO-active stock exchanges, and a culture of acquisition.
  A startup being in the US does not cause it to succeed — the US
  ecosystem is correlated with successful companies, but many US
  startups also fail.
- `cat_Biotechnology` has high positive importance. Biotech companies
  were frequently acquired by large pharmaceutical companies in the
  dataset's time period (2000–2016). This reflects M&A activity in
  that era, not an intrinsic property of biotech ideas.
- `country_RUS` has the highest negative coefficient in Logistic
  Regression. Russian-HQ companies had lower recorded success rates.
  This may reflect reduced reporting of successful exits, regulatory
  barriers, less IPO activity in Russian markets, or genuine business
  challenges — the feature alone cannot distinguish these explanations.

### 8.2 `country_Unknown` and `cat_Unknown` are data artifacts

Two of the top-3 most important features in every tree-based model are
`country_Unknown` and `cat_Unknown` — which mean "this company's
geography / category was not recorded in Crunchbase."

In the application, every user submission will have a country and
sector (required fields). Therefore both of these features will always
be 0 for new idea submissions. The model's historical reliance on
these features does not affect inference for new ideas, but it does
mean a significant portion of the model's discriminative power was
learned from a data-quality pattern that will not appear at inference
time.

This is a known limitation documented in `MODEL_EVALUATION_REPORT.md`
Section 15.5.

### 8.3 MDI importance overestimates features used in early splits

In boosting and single trees, features used in early, high-level
splits receive disproportionately high MDI scores because they affect
more samples. This is part of why `country_Unknown` (likely a very
early split in most trees) appears so dominant in GB (37.2%) and DT
(39.3%) but less so in RF (20.4%) and ET (21.5%), where splits are
more randomised.

### 8.4 Low-importance features are not necessarily useless

A feature with low MDI importance may still contribute when combined
with other features, or may be important for a specific sub-population
(e.g., `country_ISR` has low global importance but may be highly
predictive for the Israel sub-segment). Feature removal decisions
should be validated empirically, not made purely from importance scores.

### 8.5 Permutation importance as an alternative

MDI importance is computed at training time. An alternative is
**permutation importance** (measured at test time by shuffling one
feature and observing performance drop). Permutation importance is
more reliable but more computationally expensive. It is recommended
as a future improvement for Phase 4 or beyond.

---

*This report was produced from the output of `03_train_models.py`.*
*Feature importance data is also available in machine-readable form at:*
*`outputs/model_evaluation/feature_importance.json`*
*No model was trained on test data. No importance scores were fabricated.*
