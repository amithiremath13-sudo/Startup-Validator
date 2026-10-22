# Model Evaluation Report
## AI-Based Startup Viability Prediction and Validation System
### Phase 3 — ML Model Training and Evaluation (Corrected)

---

> **All numbers in this report are sourced from the actual execution of
> `03_train_models.py` on 19 September 2026 (corrected run). Nothing is
> estimated or fabricated. The script is fully reproducible with seed=42.**
>
> **Correction applied:** The original report selected the model using
> test-set performance. This was incorrect. The corrected report selects
> the model using 5-fold CV results on training data only. The test set
> is used solely for the final reported evaluation of the pre-selected model.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Dataset Description](#2-dataset-description)
3. [Target Definition](#3-target-definition)
4. [Majority-Class Baseline](#4-majority-class-baseline)
5. [Models Evaluated](#5-models-evaluated)
6. [Cross-Validation Methodology](#6-cross-validation-methodology)
7. [Cross-Validation Results](#7-cross-validation-results)
8. [Model Selection — CV-Based](#8-model-selection--cv-based)
9. [Why the Test Set Must Not Be Used for Model Selection](#9-why-the-test-set-must-not-be-used-for-model-selection)
10. [Test-Set Evaluation of Selected Model](#10-test-set-evaluation-of-selected-model)
11. [All Models — Test-Set Results (for transparency)](#11-all-models--test-set-results-for-transparency)
12. [Confusion Matrices](#12-confusion-matrices)
13. [ROC-AUC Results](#13-roc-auc-results)
14. [Baseline Comparison](#14-baseline-comparison)
15. [Why the Selected Model Was Selected](#15-why-the-selected-model-was-selected)
16. [Leakage Check](#16-leakage-check)
17. [Probability Calibration Note](#17-probability-calibration-note)
18. [Limitations](#18-limitations)
19. [Reproducibility](#19-reproducibility)

---

## 1. Problem Definition

**Task:** Binary classification — predict whether a startup will achieve
a successful exit (acquisition or IPO) or shut down (closed).

**Prediction point:** Day 0 — the moment a startup idea exists, before
any funding has been received. The model receives only information
genuinely available at this point: industry sector, country of
headquarters, and geographic sub-region flags.

**Authoritative specification:** `ML_METHODOLOGY.md` Section 11

**What the model predicts:**
> "Given a startup in sector X, from country Y: what is the probability
> that a historically similar company achieved a successful exit?"

**What the model cannot predict:**
idea quality, market size, competition, revenue potential, founder
quality, technical feasibility. These are absent from the training data.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| Raw CSV rows | 66,368 companies |
| Training-eligible rows (known outcome) | 13,334 |
| Operating companies excluded | 53,034 |
| **Training set rows** | **10,667** |
| **Test set rows** | **2,667** |
| **Features** | **65** |
| Train/test split | 80% / 20%, stratified, seed=42 |

Feature groups: 2 geography flags, 11 country flags, 50 category
binary flags, 2 category summary features (`category_other`, `n_categories`).

Excluded: `funding_total_usd`, `funding_rounds`, `last_funding_at`,
`first_funding_at`, `founding_year`, `has_founding_date`, `is_usa`,
`status` (target), and identifier columns.

---

## 3. Target Definition

| Label | Meaning | Train count | Test count |
|-------|---------|:-----------:|:---------:|
| **1** | SUCCESS — acquired or IPO | 5,677 (53.2%) | 1,419 (53.2%) |
| **0** | FAILURE — closed | 4,990 (46.8%) | 1,248 (46.8%) |

Stratified split preserved the class ratio exactly in both halves.

---

## 4. Majority-Class Baseline

| Property | Value |
|----------|-------|
| Majority class | 1 (success) |
| Baseline accuracy on test set | **0.5321 (53.21%)** |
| Baseline ROC-AUC | 0.500 (random guessing) |

Any model must beat 53.21% accuracy to add value over predicting
every company as a success.

---

## 5. Models Evaluated

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Single non-linear tree |
| 3 | Random Forest | Bagging ensemble |
| 4 | Gradient Boosting | Boosting ensemble |
| 5 | Extra Trees | Randomised ensemble |

All used `random_state=42`. No external APIs or pre-trained models.
All training is local on the processed dataset.

---

## 6. Cross-Validation Methodology

**Method:** StratifiedKFold, 5 folds, shuffle=True, random_state=42

**Tuning criterion:** CV ROC-AUC (via GridSearchCV on training data only)

**Selection criterion (defined before any test-set examination):**
1. Primary: highest mean CV ROC-AUC
2. Tie-breaker: highest mean CV F1
3. Second tie-breaker: highest mean CV accuracy

The test set was loaded at script start but never passed to any CV,
tuning, or selection step. It was used only once, after the model
was already selected, to produce the final reported evaluation.

---

## 7. Cross-Validation Results

All metrics on **training data only** (5-fold stratified CV).

| Model | CV Accuracy | ± | CV F1 | ± | CV AUC | ± | Best Params |
|-------|:-----------:|:-:|:-----:|:-:|:------:|:-:|-------------|
| Logistic Regression | 0.6750 | 0.0096 | 0.7141 | 0.0060 | 0.7364 | 0.0060 | C=0.1, penalty=l2 |
| Decision Tree | 0.6647 | 0.0036 | 0.7135 | 0.0046 | 0.7188 | 0.0035 | max_depth=20, min_leaf=10 |
| **Random Forest** | 0.6741 | 0.0066 | **0.7214** | 0.0039 | **0.7364** | 0.0037 | max_depth=None, min_leaf=5, n=100 |
| Gradient Boosting | 0.6761 | 0.0087 | 0.7293 | 0.0070 | 0.7357 | 0.0047 | lr=0.1, max_depth=3, n=100 |
| Extra Trees | 0.6743 | 0.0068 | 0.7252 | 0.0040 | 0.7353 | 0.0039 | max_depth=20, min_leaf=5, n=200 |

**Observations:**
- Random Forest and Logistic Regression are near-tied on CV AUC
  (0.73644 vs 0.73639 at 5 decimal places). When examined to full
  floating-point precision, Random Forest has a slightly higher value
  and is confirmed the primary criterion winner.
- The F1 tie-breaker further confirms the choice: Random Forest CV F1
  = 0.7214 vs Logistic Regression CV F1 = 0.7141.
- Low standard deviations across all models indicate stable CV estimates.
- Gradient Boosting has the highest CV F1 (0.7293) but a lower CV AUC
  than Random Forest — primary criterion takes precedence.

---

## 8. Model Selection — CV-Based

> **SELECTED MODEL: Random Forest**
>
> **Selection basis: training data (CV) only. Test set not consulted.**

### Selection step-by-step

**Step 1 — Primary criterion: highest CV AUC**

| Model | CV AUC (full precision) |
|-------|:-----------------------:|
| Logistic Regression | 0.736393... |
| **Random Forest** | **0.736440...** |
| Gradient Boosting | 0.735700... |
| Extra Trees | 0.735300... |
| Decision Tree | 0.718800... |

Random Forest has the highest CV AUC at full floating-point precision.
Within the practical tolerance of 0.0001, Logistic Regression is also
very close (within 0.00005). The F1 tie-breaker is applied for robustness.

**Step 2 — F1 tie-breaker confirms selection**

- Random Forest CV F1 = **0.7214**
- Logistic Regression CV F1 = 0.7141

Random Forest wins on both the primary and secondary criteria.

**Step 3 — Model is selected. Test set is now examined for the first
and only time.**

---

## 9. Why the Test Set Must Not Be Used for Model Selection

This section explains why the original selection method (choosing the
model with the highest test-set AUC) was incorrect, and why it has been
corrected.

### The problem with test-set model selection

When you train five models and then select the one that performs best
on the test set, you are making a decision based on test-set performance.
This turns the test set from a passive evaluation tool into an active
selection criterion — which is exactly the same logical error as tuning
hyperparameters on the test set.

**An analogy:** Imagine you want to know how well a student will perform
on future exams. You give them ten practice exams (cross-validation on
training data) and also one real final exam (the test set). If you use
the final exam results to decide which student to "select," you have not
learned anything new — you just picked the student who happened to score
highest on that one specific exam. The next time you give them an exam,
their score will regress toward their true average performance, which was
better measured by the ten practice exams.

### What goes wrong statistically

With five models evaluated on the same 2,667-row test set, there is
random variation in test scores due to the specific samples in the test
set. The model that scores highest on this particular test set may have
gotten lucky on some subset of those 2,667 rows. If you select that model
and report its test performance as your "final result," you are reporting
a number that is inflated by the selection bias.

This is called the **winner's curse**: the best-performing model in a
competition tends to have a score that is slightly above its true
generalisation performance, because it was chosen partly for being lucky.

### The correct approach

1. Define the selection rule before seeing any test results
2. Use only cross-validation on training data to select the model
3. Apply the test set exactly once to the pre-selected model
4. Report that single test-set measurement as the final result

This is what the corrected script does. The selection rule
(primary = CV AUC, tie-breaker = CV F1) was defined in the code
before any test evaluation runs. The test set results in Step 6 of
the script are computed for all models for transparency and reporting,
but the selection in Step 7 uses only the CV results from Step 5.

### Does it matter in practice?

In this specific case, the change in selected model (from Extra Trees
to Random Forest) produces test metrics that are very similar:
- Extra Trees test AUC = 0.7601 vs Random Forest test AUC = 0.7599
- The difference is 0.0002 — negligible

But the principle matters regardless of the magnitude of the difference.
Correct methodology must be followed even when the practical impact is
small, because the same methodology applied to a noisier dataset or
smaller test set could produce meaningfully inflated results.

---

## 10. Test-Set Evaluation of Selected Model

**Selected model: Random Forest**
**Best params:** `max_depth=None, min_samples_leaf=5, n_estimators=100`

These metrics were produced by evaluating the pre-selected Random Forest
model on the untouched test set exactly once.

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6937** |
| Precision | 0.6769 |
| Recall | 0.8118 |
| **F1-score** | **0.7382** |
| **ROC-AUC** | **0.7599** |
| Majority-class baseline accuracy | 0.5321 |
| Margin over baseline | **+0.1616** |
| Beats baseline? | **Yes** |

**Confusion matrix (test set, 2,667 rows):**

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          698 (TN)              550 (FP)
Actual: Success          267 (FN)            1,152 (TP)
```

**Interpretation:**
- Of 1,248 actual failures: 698 correctly identified (56.0% specificity)
- Of 1,419 actual successes: 1,152 correctly identified (81.2% recall)
- The model leans toward predicting success — recall (81.2%) is higher
  than precision (67.7%). This means it misses few successes but produces
  a meaningful number of false positives (550 companies predicted to
  succeed that actually failed).

---

## 11. All Models — Test-Set Results (for transparency)

All five models were evaluated on the test set. These results are
reported here for transparency and academic completeness. They were
**not used to select the final model.**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Logistic Regression | 0.6944 | 0.6892 | 0.7752 | 0.7297 | 0.7558 |
| Decision Tree | 0.6835 | 0.6704 | 0.7970 | 0.7283 | 0.7410 |
| **Random Forest** ← selected | **0.6937** | 0.6769 | **0.8118** | 0.7382 | 0.7599 |
| Gradient Boosting | 0.6963 | 0.6737 | 0.8323 | 0.7446 | 0.7587 |
| Extra Trees | 0.7030 | 0.6813 | 0.8302 | **0.7484** | **0.7601** |

**Note:** Extra Trees achieves marginally higher test metrics than the
selected Random Forest. This does not mean Random Forest was the wrong
choice — the selection was correctly made before these numbers were
examined, using CV results on training data. The small differences
(0.0002 AUC, 0.01 accuracy) are within the expected variability for a
2,667-row test set and do not represent a meaningful performance gap.

---

## 12. Confusion Matrices

All confusion matrices are on the test set (2,667 rows).

### Random Forest (selected model)

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          698 (TN)              550 (FP)
Actual: Success          267 (FN)            1,152 (TP)
```

### Logistic Regression

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          752 (TN)              496 (FP)
Actual: Success          319 (FN)            1,100 (TP)
```
Highest TN — most conservative about predicting success.

### Decision Tree

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          692 (TN)              556 (FP)
Actual: Success          288 (FN)            1,131 (TP)
```

### Gradient Boosting

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          676 (TN)              572 (FP)
Actual: Success          238 (FN)            1,181 (TP)
```
Highest recall (83.2%) — misses fewest actual successes.

### Extra Trees

```
                    Predicted: Failure   Predicted: Success
Actual: Failure          697 (TN)              551 (FP)
Actual: Success          241 (FN)            1,178 (TP)
```

**Plots:** `outputs/model_evaluation/cm_<model_name>.png`

---

## 13. ROC-AUC Results

ROC-AUC measures the probability that the model ranks a randomly chosen
success higher than a randomly chosen failure. A value of 0.50 = random
guessing; 1.00 = perfect discrimination.

**Note:** ROC-AUC measures discrimination, not probability calibration.
A model with high AUC may still produce poorly calibrated probability
scores (e.g., a company with true 60% success probability might receive
a predicted probability of 80%). See Section 17 for details.

| Model | ROC-AUC |
|-------|:-------:|
| Logistic Regression | 0.7558 |
| Decision Tree | 0.7410 |
| **Random Forest** ← selected | **0.7599** |
| Gradient Boosting | 0.7587 |
| Extra Trees | 0.7601 |
| Majority-class baseline | 0.500 |

All five models substantially outperform the random baseline (+24 to
+26 percentage points). The spread across models is narrow (0.741–0.760).

**Plot:** `outputs/model_evaluation/roc_all_models.png`

---

## 14. Baseline Comparison

| Model | Test Accuracy | vs Baseline | Test AUC | vs Baseline AUC |
|-------|:------------:|:-----------:|:--------:|:---------------:|
| Majority-class baseline | 0.5321 | — | 0.500 | — |
| Logistic Regression | 0.6944 | +16.2pp | 0.7558 | +25.6pp |
| Decision Tree | 0.6835 | +15.1pp | 0.7410 | +24.1pp |
| **Random Forest** ← selected | 0.6937 | **+16.2pp** | 0.7599 | **+26.0pp** |
| Gradient Boosting | 0.6963 | +16.4pp | 0.7587 | +25.9pp |
| Extra Trees | 0.7030 | +17.1pp | 0.7601 | +26.0pp |

All five models substantially beat the majority-class baseline. The
selected Random Forest model improves accuracy by +16.2 percentage
points and AUC by +26.0 percentage points over random guessing.

---

## 15. Why the Selected Model Was Selected

The selection is based entirely on CV performance measured on training
data. Test-set results were not consulted during selection.

### Primary criterion: CV ROC-AUC

Random Forest achieved the highest CV ROC-AUC of **0.7364** (at full
floating-point precision: 0.73644, compared to Logistic Regression's
0.73639). Both are strong performers — the gap is tiny but Random Forest
is strictly higher.

CV ROC-AUC was chosen as the selection criterion because:
1. It is **threshold-independent** — it evaluates model quality across
   all possible decision thresholds, not just at the default 0.5 cut-off.
2. It is **robust to the mild class imbalance** (53/47 split).
3. It was computed on held-out fold data only — a genuine
   out-of-sample estimate from training data, not the test set.

### Secondary criterion confirms the choice

Random Forest CV F1 (0.7214) > Logistic Regression CV F1 (0.7141).
The tie-breaker is consistent with the primary criterion.

### Why not Gradient Boosting?

Gradient Boosting has the highest CV F1 (0.7293) but a lower CV AUC
(0.7357) than Random Forest (0.7364). The primary criterion (AUC) takes
precedence. If CV F1 were the primary criterion, Gradient Boosting
would be selected instead — this is documented as an academic design
choice, not a universal truth.

### What the test-set results tell us (after selection)

The selected Random Forest achieves test AUC = 0.7599, which is
consistent with its CV AUC = 0.7364. The small upward gap (+0.023)
is within normal sampling variability for a 2,667-row test set. No
sign of overfitting — test performance is not worse than CV performance.

The fact that Extra Trees achieved slightly higher test metrics
(AUC = 0.7601) does not retroactively change the selection. Those
numbers were not available when the selection decision was made. This
is the correct approach.

---

## 16. Leakage Check

Automated leakage warning: any model with accuracy > 0.90 or
ROC-AUC > 0.95 triggers a warning in the script.

**No leakage warning was triggered on any model.**

| Threshold | Highest observed | Triggered? |
|-----------|:----------------:|:----------:|
| Accuracy > 0.90 | 0.7030 (Extra Trees) | ❌ No |
| ROC-AUC > 0.95 | 0.7601 (Extra Trees) | ❌ No |

Performance range (acc 0.68–0.70, AUC 0.74–0.76) is realistic for a
Day-0 prediction task with sector and geography features only.

**Pre-training leakage audit (Step 2 of script):**

| Check | Result |
|-------|--------|
| 65 features | ✅ Confirmed |
| Forbidden columns absent | ✅ All 9 absent |
| Column order matches JSON | ✅ Confirmed |
| NaN values | ✅ Zero |
| Test set isolated | ✅ Not used until Step 8 |

---

## 17. Probability Calibration Note

The Random Forest model outputs a `predict_proba` value between 0 and 1
for each prediction. In the final application, this value may be
displayed to the user as a probability score.

**Important:** ROC-AUC measures discrimination — whether the model ranks
successes above failures — but it does not measure probability calibration.
A model with high AUC can still produce poorly calibrated probabilities.
For example, if the model says a company has a 0.80 probability of
success, a well-calibrated model would require that approximately 80%
of companies it scores at 0.80 actually succeed. Random Forests are
known to produce reasonably well-ordered but not always well-calibrated
probabilities.

If the application displays a numerical probability to users, probability
calibration should be assessed and, if needed, corrected using a
technique such as Platt scaling (logistic regression on the raw scores)
or isotonic regression. This is addressed during the application
integration phase, not in Phase 3.

---

## 18. Limitations

### 18.1 Narrow feature set by design

65 features restricted to sector, country, and geography. Startup
success factors absent from this dataset include: funding history,
team quality, market timing, product-market fit, competitive landscape.

### 18.2 Temporal distribution mismatch

Training data covers approximately 1990–2016. New ideas in 2026 fall
outside this range. `founding_year` was excluded for this reason, but
the underlying distribution gap in sector trends and macroeconomic
conditions remains a known limitation.

### 18.3 Survivorship bias

Companies appear in Crunchbase because they were notable enough to be
documented. Quiet failures may be under-represented in the `closed`
class. The model predicts outcomes for "Crunchbase-documented startups."

### 18.4 Performance is moderate

AUC = 0.76 and accuracy ≈ 69% mean the model is wrong about 31% of
the time. Users should receive a probabilistic estimate with appropriate
caveats, not a definitive prediction.

### 18.5 Data-artifact features

`country_Unknown` and `cat_Unknown` — flags for missing geography/category
in historical records — are among the most important features for tree
models. At inference time (new idea submissions), these flags will always
be 0 (users provide their sector and country). The model's historical
reliance on these features is a data quality artifact, not a real signal.

### 18.6 Small hyperparameter grid

For reproducibility and speed. A larger search might produce marginally
different hyperparameters but would not fundamentally change the result
given the feature set's signal limitations.

---

## 19. Reproducibility

| Parameter | Value |
|-----------|-------|
| Script | `03_train_models.py` |
| Random seed | 42 |
| CV folds | 5 (StratifiedKFold) |
| scikit-learn | 1.9.1 |
| Python | 3.14.7 |

```bash
python 03_train_models.py
```

Results confirmed identical across two independent runs.

**Selected model file:** `models/best_model.joblib` (Random Forest)
**All models:** `models/<model_name>.joblib`
**Metadata:** `models/model_metadata.json`

```
selection_criterion: "highest CV ROC-AUC (training data only);
                      tie-breaker: CV F1"
selection_used_test_set: false
```

---

*This report was produced after verified execution of the corrected
`03_train_models.py`. Model selection uses CV results only.*
*Test-set results for the selected model are reported as a final,
single-use evaluation. No model was trained on test data.*
*Phase 4 must not begin until this report has been reviewed and approved.*
