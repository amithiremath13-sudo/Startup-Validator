"""
=============================================================================
03_train_models.py
=============================================================================
Phase 3 – ML Model Training and Evaluation
Project: AI-Based Startup Viability Prediction and Validation System

AUTHORITATIVE SPECIFICATION
----------------------------
ML_METHODOLOGY.md Section 11 (Revised Prediction Framework)
Preprocessing: 02_preprocess_data.py output (65 features, Day-0 point)

WHAT THIS SCRIPT DOES
----------------------
1.  Loads the approved processed train/test splits (READ-ONLY)
2.  Performs pre-training leakage audit
3.  Establishes a majority-class baseline
4.  Trains 5 classifiers using training data only:
      - Logistic Regression
      - Decision Tree
      - Random Forest
      - Gradient Boosting
      - Extra Trees  (additional strong tabular ensemble)
5.  Tunes each model with a small StratifiedKFold grid search
    using TRAINING DATA ONLY
6.  Evaluates every tuned model on the UNTOUCHED test set
7.  Reports: accuracy, precision, recall, F1, ROC-AUC,
    confusion matrix, cross-validation scores
8.  Checks for suspiciously high performance (leakage indicator)
9.  Saves best model + all models + metadata to models/
10. Saves confusion matrix plots and ROC curves to
    outputs/model_evaluation/
11. Writes full results to outputs/model_evaluation/results.json
12. Writes human-readable log to outputs/model_evaluation/training_log.txt

TEST SET RULE
-------------
The test set is loaded at the top but ONLY used in the final
evaluation section. It is never passed to any tuning, fitting,
or feature-selection step.

REPRODUCIBILITY
---------------
All random states use SEED = 42.

HOW TO RUN
----------
    python 03_train_models.py

OUTPUTS
-------
    models/
        best_model.joblib           final selected model
        logistic_regression.joblib
        decision_tree.joblib
        random_forest.joblib
        gradient_boosting.joblib
        extra_trees.joblib
        model_metadata.json         feature list, threshold, selection reason

    outputs/model_evaluation/
        cm_<model>.png              confusion matrices
        roc_all_models.png          all ROC curves on one chart
        roc_<model>.png             individual ROC curves
        results.json                all numeric results
        training_log.txt            full text log
=============================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "processed_data")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
EVAL_DIR     = os.path.join(SCRIPT_DIR, "outputs", "model_evaluation")
FEAT_JSON    = os.path.join(DATA_DIR, "feature_names.json")

SEED         = 42
CV_FOLDS     = 5          # StratifiedKFold folds for cross-validation
EXPECTED_FEATURES = 65    # hard-coded guard — must match Phase 2 output

# Columns that must never appear in the feature matrix
FORBIDDEN_COLS = [
    "status", "target", "funding_total_usd", "funding_rounds",
    "last_funding_at", "first_funding_at", "founding_year",
    "has_founding_date", "is_usa",
]

# Performance thresholds that trigger a leakage warning
LEAKAGE_ACCURACY_THRESHOLD = 0.90
LEAKAGE_AUC_THRESHOLD      = 0.95

for d in [MODELS_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

log_lines = []

def log(msg=""):
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
# STEP 1 — LOAD DATA (test set loaded but not touched until §8)
# ---------------------------------------------------------------------------

def step1_load():
    section("STEP 1 — LOAD DATA")

    with open(FEAT_JSON) as f:
        feature_names = json.load(f)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv")).squeeze()

    log(f"  X_train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    log(f"  X_test  : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    log(f"  y_train : class 1 = {y_train.sum():,}  class 0 = {(y_train==0).sum():,}")
    log(f"  y_test  : class 1 = {y_test.sum():,}  class 0 = {(y_test==0).sum():,}")
    log(f"  Feature names loaded: {len(feature_names)}")

    return X_train, y_train, X_test, y_test, feature_names

# ---------------------------------------------------------------------------
# STEP 2 — PRE-TRAINING LEAKAGE AUDIT
# ---------------------------------------------------------------------------

def step2_leakage_audit(X_train, X_test, feature_names):
    section("STEP 2 — PRE-TRAINING LEAKAGE AUDIT")

    errors = []

    # Check feature count
    if X_train.shape[1] != EXPECTED_FEATURES:
        errors.append(
            f"Feature count mismatch: expected {EXPECTED_FEATURES}, "
            f"got {X_train.shape[1]}"
        )

    # Check forbidden columns
    for col in FORBIDDEN_COLS:
        if col in X_train.columns:
            errors.append(f"FORBIDDEN COLUMN IN FEATURES: {col}")

    # Check column order matches feature_names.json
    if list(X_train.columns) != feature_names:
        errors.append("Column order does not match feature_names.json")

    # Check train and test have identical columns
    if list(X_train.columns) != list(X_test.columns):
        errors.append("Train and test column order differ")

    # Check no NaN
    nan_train = X_train.isna().sum().sum()
    nan_test  = X_test.isna().sum().sum()
    if nan_train > 0:
        errors.append(f"NaN values in X_train: {nan_train}")
    if nan_test > 0:
        errors.append(f"NaN values in X_test: {nan_test}")

    if errors:
        log("  LEAKAGE AUDIT FAILED:")
        for e in errors:
            log(f"    !! {e}")
        sys.exit(1)
    else:
        log(f"  OK — {EXPECTED_FEATURES} features, no forbidden columns, "
            f"no NaN, column order matches JSON")
        log("  Test set has been loaded but will NOT be used until Step 8.")

    return feature_names

# ---------------------------------------------------------------------------
# STEP 3 — BASELINE
# ---------------------------------------------------------------------------

def step3_baseline(y_train, y_test):
    section("STEP 3 — MAJORITY-CLASS BASELINE")

    # The simplest possible model: always predict the majority class.
    # Any useful model must beat this.
    majority_class    = int(y_train.value_counts().idxmax())
    baseline_preds    = np.full(len(y_test), majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    log(f"  Majority class in training set : {majority_class}")
    log(f"  Baseline accuracy (test set)   : {baseline_accuracy:.4f}  "
        f"({baseline_accuracy*100:.2f}%)")
    log(f"  Any model must exceed {baseline_accuracy*100:.2f}% to add value.")

    return baseline_accuracy, majority_class

# ---------------------------------------------------------------------------
# STEP 4 — DEFINE MODELS AND HYPERPARAMETER GRIDS
# ---------------------------------------------------------------------------

def step4_define_models():
    section("STEP 4 — MODEL DEFINITIONS AND HYPERPARAMETER GRIDS")

    # -----------------------------------------------------------------------
    # DESIGN RATIONALE FOR EACH MODEL
    # -----------------------------------------------------------------------
    #
    # 1. Logistic Regression
    #    Linear model. Good baseline for high-dimensional binary inputs.
    #    With mostly binary features, it is interpretable via coefficients.
    #    Regularisation (C) controls overfitting.
    #    solver='lbfgs' handles multi-class and sparse features well.
    #
    # 2. Decision Tree
    #    Non-linear, interpretable. Single tree is prone to overfitting;
    #    included to compare with ensemble versions.
    #    max_depth limits overfit.
    #
    # 3. Random Forest
    #    Ensemble of trees. Reduces variance vs single tree by averaging.
    #    Strong baseline for tabular data with binary features.
    #    n_estimators=200 gives stable results without being expensive.
    #
    # 4. Gradient Boosting (sklearn GBM)
    #    Sequentially corrects errors. Often top performer on tabular data.
    #    learning_rate + n_estimators are the key levers.
    #
    # 5. Extra Trees (ExtraTreesClassifier)
    #    Similar to Random Forest but uses random splits (not best splits)
    #    — faster and sometimes better on sparse binary features like ours.
    #    Included as the "additional strong tabular classifier."
    #
    # All grids are kept SMALL (2-3 values per param) to be reproducible
    # and practical. No test data is used in GridSearchCV.
    # -----------------------------------------------------------------------

    models_and_grids = {
        "Logistic Regression": (
            LogisticRegression(
                random_state=SEED, max_iter=1000,
                solver="lbfgs", class_weight="balanced"
            ),
            {
                "C":       [0.1, 1.0, 10.0],
                "penalty": ["l2"],
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "max_depth":        [5, 10, 20, None],
                "min_samples_leaf": [1, 5, 10],
            }
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=SEED, n_jobs=-1
            ),
            {
                "n_estimators": [100, 200],
                "max_depth":    [10, 20, None],
                "min_samples_leaf": [1, 5],
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5],
            }
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                random_state=SEED, n_jobs=-1
            ),
            {
                "n_estimators":     [100, 200],
                "max_depth":        [10, 20, None],
                "min_samples_leaf": [1, 5],
            }
        ),
    }

    for name, (model, grid) in models_and_grids.items():
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        log(f"  {name:<25}  {n_combos} param combinations × {CV_FOLDS} folds "
            f"= {n_combos * CV_FOLDS} fits")

    return models_and_grids

# ---------------------------------------------------------------------------
# STEP 5 — TRAIN AND TUNE EACH MODEL (training data only)
# ---------------------------------------------------------------------------

def step5_train(X_train, y_train, models_and_grids):
    section("STEP 5 — STRATIFIEDKFOLD GRID SEARCH (training data only)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    tuned_models   = {}
    cv_results_all = {}

    for name, (base_model, param_grid) in models_and_grids.items():
        subsection(f"Tuning: {name}")
        t0 = time.time()

        search = GridSearchCV(
            estimator  = base_model,
            param_grid = param_grid,
            cv         = cv,
            scoring    = "roc_auc",   # tune on AUC — robust to class imbalance
            n_jobs     = -1,
            refit      = True,        # refit best estimator on full training set
            verbose    = 0,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best = search.best_estimator_
        tuned_models[name] = best

        log(f"  Best params   : {search.best_params_}")
        log(f"  Best CV AUC   : {search.best_score_:.4f}")
        log(f"  Time          : {elapsed:.1f}s")

        # Full 5-fold CV with best params for reporting
        cv_res = cross_validate(
            best, X_train, y_train,
            cv      = cv,
            scoring = ["accuracy", "f1", "roc_auc"],
            n_jobs  = -1,
        )
        cv_results_all[name] = {
            "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
            "cv_accuracy_std":  float(np.std(cv_res["test_accuracy"])),
            "cv_f1_mean":       float(np.mean(cv_res["test_f1"])),
            "cv_f1_std":        float(np.std(cv_res["test_f1"])),
            "cv_auc_mean":      float(np.mean(cv_res["test_roc_auc"])),
            "cv_auc_std":       float(np.std(cv_res["test_roc_auc"])),
            "best_params":      search.best_params_,
        }

        log(f"  CV Accuracy   : {cv_results_all[name]['cv_accuracy_mean']:.4f} "
            f"± {cv_results_all[name]['cv_accuracy_std']:.4f}")
        log(f"  CV F1         : {cv_results_all[name]['cv_f1_mean']:.4f} "
            f"± {cv_results_all[name]['cv_f1_std']:.4f}")
        log(f"  CV AUC        : {cv_results_all[name]['cv_auc_mean']:.4f} "
            f"± {cv_results_all[name]['cv_auc_std']:.4f}")

    return tuned_models, cv_results_all

# ---------------------------------------------------------------------------
# STEP 6 — EVALUATE ON TEST SET (called once, after all tuning is done)
# ---------------------------------------------------------------------------

def step6_test_evaluation(tuned_models, X_test, y_test, baseline_accuracy):
    section("STEP 6 — TEST SET EVALUATION (untouched set, final evaluation only)")

    test_results = {}

    for name, model in tuned_models.items():
        subsection(f"Test evaluation: {name}")

        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]   # prob of class 1

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred)

        log(f"  Accuracy  : {acc:.4f}  (baseline: {baseline_accuracy:.4f}  "
            f"delta: {acc - baseline_accuracy:+.4f})")
        log(f"  Precision : {prec:.4f}")
        log(f"  Recall    : {rec:.4f}")
        log(f"  F1-score  : {f1:.4f}")
        log(f"  ROC-AUC   : {auc:.4f}")
        log(f"  Confusion matrix (rows=actual, cols=predicted):")
        log(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
        log(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

        # Leakage warning: suspiciously high scores
        if acc > LEAKAGE_ACCURACY_THRESHOLD or auc > LEAKAGE_AUC_THRESHOLD:
            log(f"  *** LEAKAGE WARNING: {name} has very high performance.")
            log(f"  *** Accuracy {acc:.4f} > {LEAKAGE_ACCURACY_THRESHOLD} OR "
                f"AUC {auc:.4f} > {LEAKAGE_AUC_THRESHOLD}.")
            log(f"  *** Investigate for data leakage before trusting these results.")

        test_results[name] = {
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "roc_auc":   float(auc),
            "confusion_matrix": cm.tolist(),
            "y_pred":    y_pred.tolist(),
            "y_prob":    y_prob.tolist(),
        }

    return test_results

# ---------------------------------------------------------------------------
# STEP 7 — SELECT BEST MODEL (using CV results only — test set NOT used)
# ---------------------------------------------------------------------------

def step7_select_model(test_results, cv_results_all, baseline_accuracy):
    section("STEP 7 — MODEL SELECTION (CV-based, training data only)")

    log("  SELECTION RULE (defined before any test-set results are examined):")
    log("  Primary   criterion = highest mean CV ROC-AUC (5-fold, training only)")
    log("  Secondary criterion = highest mean CV F1 (tie-breaker)")
    log("  Tertiary  criterion = highest mean CV accuracy (second tie-breaker)")
    log()
    log("  WHY TEST-SET RESULTS ARE NOT USED FOR SELECTION:")
    log("  Using the test set to choose between models is a form of test-set")
    log("  contamination. If we evaluate all models on the test set and pick")
    log("  the one with the highest test metric, we are effectively optimising")
    log("  for the test set — the same error we prevent during hyperparameter")
    log("  tuning. The selected model would then appear better on the test set")
    log("  than it really is, because we chose it precisely because it scored")
    log("  highest on that data. The test set must produce a single, unbiased")
    log("  estimate of performance for the pre-selected model only.")
    log()

    log(f"  {'Model':<25} {'CV AUC':>10} {'CV F1':>10} {'CV Acc':>10}")
    log(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in cv_results_all.items():
        log(f"  {name:<25} {r['cv_auc_mean']:>10.4f} {r['cv_f1_mean']:>10.4f} "
            f"{r['cv_accuracy_mean']:>10.4f}")

    # Apply selection rule on CV results only
    best_cv_auc  = max(r["cv_auc_mean"] for r in cv_results_all.values())
    # Use tolerance for floating-point near-ties (within 0.0001)
    auc_tolerance = 0.0001
    candidates   = {n: r for n, r in cv_results_all.items()
                    if best_cv_auc - r["cv_auc_mean"] <= auc_tolerance}

    if len(candidates) == 1:
        best_name = list(candidates.keys())[0]
        log(f"\n  Primary criterion resolved: {best_name} has highest CV AUC "
            f"= {best_cv_auc:.4f}")
    else:
        # Tie-breaker 1: CV F1
        best_cv_f1  = max(r["cv_f1_mean"] for r in candidates.values())
        f1_candidates = {n: r for n, r in candidates.items()
                         if best_cv_f1 - r["cv_f1_mean"] <= auc_tolerance}
        if len(f1_candidates) == 1:
            best_name = list(f1_candidates.keys())[0]
            log(f"\n  CV AUC near-tie among: {list(candidates.keys())}")
            log(f"  F1 tie-breaker resolved: {best_name} CV F1 = {best_cv_f1:.4f}")
        else:
            # Tie-breaker 2: CV accuracy
            best_name = max(f1_candidates,
                            key=lambda n: cv_results_all[n]["cv_accuracy_mean"])
            log(f"\n  Accuracy tie-breaker resolved: {best_name}")

    log(f"\n  SELECTED MODEL : {best_name}")
    log(f"  Selection basis: CV AUC = {cv_results_all[best_name]['cv_auc_mean']:.4f}  "
        f"CV F1 = {cv_results_all[best_name]['cv_f1_mean']:.4f}")
    log(f"  Selection used TRAINING DATA ONLY. Test set not consulted.")

    # Now report test-set metrics for the selected model (informational only)
    log(f"\n  Test-set metrics for selected model (reported AFTER selection,")
    log(f"  NOT used to make the selection decision):")
    best_r = test_results[best_name]
    log(f"  Test Accuracy  : {best_r['accuracy']:.4f}  "
        f"(baseline: {baseline_accuracy:.4f}, delta: {best_r['accuracy']-baseline_accuracy:+.4f})")
    log(f"  Test Precision : {best_r['precision']:.4f}")
    log(f"  Test Recall    : {best_r['recall']:.4f}")
    log(f"  Test F1        : {best_r['f1']:.4f}")
    log(f"  Test ROC-AUC   : {best_r['roc_auc']:.4f}")
    log(f"\n  NOTE on probability calibration:")
    log(f"  ROC-AUC measures discrimination (ranking), not calibration.")
    log(f"  If the application displays a numerical probability score,")
    log(f"  probability calibration (e.g. Platt scaling or isotonic")
    log(f"  regression) should be addressed during application integration.")

    if best_r["accuracy"] <= baseline_accuracy:
        log(f"\n  WARNING: Selected model does NOT beat the majority-class baseline.")
        log(f"  This suggests the features have limited predictive power.")

    return best_name

# ---------------------------------------------------------------------------
# STEP 8 — SAVE MODELS AND METADATA
# ---------------------------------------------------------------------------

def step8_save(tuned_models, best_name, test_results, cv_results_all,
               feature_names, baseline_accuracy):
    section("STEP 8 — SAVE MODELS AND METADATA")

    # Map model names to safe filenames
    name_to_file = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree":       "decision_tree.joblib",
        "Random Forest":       "random_forest.joblib",
        "Gradient Boosting":   "gradient_boosting.joblib",
        "Extra Trees":         "extra_trees.joblib",
    }

    # Save all models individually
    for name, model in tuned_models.items():
        path = os.path.join(MODELS_DIR, name_to_file[name])
        joblib.dump(model, path)
        log(f"  Saved: {path}")

    # Save best model separately for easy loading
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(tuned_models[best_name], best_path)
    log(f"  Saved best model: {best_path}")

    # Save model metadata (everything needed to run inference correctly)
    metadata = {
        "phase": "Phase 3 — Model Training",
        "spec_reference": "ML_METHODOLOGY.md Section 11",
        "random_seed": SEED,
        "cv_folds": CV_FOLDS,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "expected_feature_count": EXPECTED_FEATURES,
        "prediction_point": "Day 0 — pre-funding idea stage",
        "target_definition": {
            "1": "SUCCESS (acquired or ipo)",
            "0": "FAILURE (closed)"
        },
        "selected_model_name": best_name,
        "selected_model_file": "best_model.joblib",
        "selection_criterion": "highest CV ROC-AUC (training data only); tie-breaker: CV F1",
        "selection_used_test_set": False,
        "baseline_accuracy": float(baseline_accuracy),
        "test_results": test_results,
        "cv_results": cv_results_all,
        "leakage_warning_thresholds": {
            "accuracy": LEAKAGE_ACCURACY_THRESHOLD,
            "roc_auc":  LEAKAGE_AUC_THRESHOLD,
        },
        "forbidden_columns_verified_absent": FORBIDDEN_COLS,
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    # Strip y_pred and y_prob (too large for metadata) before saving
    meta_save = json.loads(json.dumps(metadata))
    for name in meta_save["test_results"]:
        meta_save["test_results"][name].pop("y_pred", None)
        meta_save["test_results"][name].pop("y_prob", None)

    with open(meta_path, "w") as f:
        json.dump(meta_save, f, indent=2)
    log(f"  Saved metadata: {meta_path}")

    # Also save full results (with predictions) for report generation
    results_path = os.path.join(EVAL_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  Saved full results: {results_path}")

    return metadata

# ---------------------------------------------------------------------------
# STEP 9 — CONFUSION MATRIX PLOTS
# ---------------------------------------------------------------------------

def step9_confusion_matrices(tuned_models, X_test, y_test):
    section("STEP 9 — CONFUSION MATRIX VISUALISATIONS")

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Failure (0)", "Success (1)"]
        )
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(EVAL_DIR, f"cm_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# STEP 10 — ROC CURVES
# ---------------------------------------------------------------------------

def step10_roc_curves(tuned_models, X_test, y_test, test_results):
    section("STEP 10 — ROC CURVE VISUALISATIONS")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Combined ROC plot (all models on one axes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for (name, model), color in zip(tuned_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = test_results[name]["roc_auc"]
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()

    combined_path = os.path.join(EVAL_DIR, "roc_all_models.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {combined_path}")

    # Individual ROC plots
    for (name, model), color in zip(tuned_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = test_results[name]["roc_auc"]
        fpr_i, tpr_i, _ = roc_curve(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_i, tpr_i, lw=2, color=color,
                label=f"{name} (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {name}", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(EVAL_DIR, f"roc_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# STEP 11 — FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def step11_feature_importance(tuned_models, feature_names):
    section("STEP 11 — FEATURE IMPORTANCE")

    importance_data = {}

    for name, model in tuned_models.items():
        if hasattr(model, "feature_importances_"):
            # Tree-based models (DT, RF, GB, ET)
            importances = model.feature_importances_
            indices     = np.argsort(importances)[::-1]

            log(f"\n  {name} — Top 20 features by impurity importance:")
            log(f"  {'Rank':<5} {'Feature':<40} {'Importance':>12}")
            log(f"  {'-'*5} {'-'*40} {'-'*12}")

            top20 = []
            for rank, idx in enumerate(indices[:20], 1):
                feat  = feature_names[idx]
                score = float(importances[idx])
                log(f"  {rank:<5} {feat:<40} {score:>12.6f}")
                top20.append({"feature": feat, "importance": score})

            importance_data[name] = {
                "type": "feature_importances",
                "top_20": top20,
                "all": [
                    {"feature": feature_names[i], "importance": float(importances[i])}
                    for i in range(len(feature_names))
                ]
            }

            # Bar chart for top 20
            fig, ax = plt.subplots(figsize=(10, 7))
            top20_feat  = [d["feature"] for d in top20]
            top20_score = [d["importance"] for d in top20]
            colors_bar  = ["#e74c3c" if s == max(top20_score) else "#3498db"
                           for s in top20_score]
            bars = ax.barh(top20_feat[::-1], top20_score[::-1],
                           color=colors_bar[::-1], edgecolor="white")
            ax.set_xlabel("Feature Importance (mean impurity decrease)")
            ax.set_title(f"Top 20 Feature Importances — {name}",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()

            safe_name = name.lower().replace(" ", "_")
            path = os.path.join(EVAL_DIR, f"importance_{safe_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log(f"  Plot saved: {path}")

        elif hasattr(model, "coef_"):
            # Logistic Regression — use absolute coefficients
            coefs    = model.coef_[0]
            abs_coef = np.abs(coefs)
            indices  = np.argsort(abs_coef)[::-1]

            log(f"\n  Logistic Regression — Top 20 features by |coefficient|:")
            log(f"  {'Rank':<5} {'Feature':<40} {'Coefficient':>12} {'|Coef|':>8}")
            log(f"  {'-'*5} {'-'*40} {'-'*12} {'-'*8}")

            top20 = []
            for rank, idx in enumerate(indices[:20], 1):
                feat  = feature_names[idx]
                coef  = float(coefs[idx])
                acoef = float(abs_coef[idx])
                log(f"  {rank:<5} {feat:<40} {coef:>12.4f} {acoef:>8.4f}")
                top20.append({"feature": feat, "coefficient": coef,
                               "abs_coefficient": acoef})

            importance_data[name] = {
                "type": "coefficients",
                "note": "Absolute coefficient magnitude used for ranking. "
                        "Positive = associated with success (class 1). "
                        "Negative = associated with failure (class 0). "
                        "This is correlation, NOT causation.",
                "top_20": top20,
            }

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 7))
            top20_feat = [d["feature"] for d in top20]
            top20_coef = [d["coefficient"] for d in top20]
            bar_colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top20_coef]
            ax.barh(top20_feat[::-1], top20_coef[::-1],
                    color=bar_colors[::-1], edgecolor="white")
            ax.axvline(x=0, color="black", lw=0.8, linestyle="--")
            ax.set_xlabel("Coefficient (green = success, red = failure)")
            ax.set_title("Top 20 Feature Coefficients — Logistic Regression",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()

            path = os.path.join(EVAL_DIR, "importance_logistic_regression.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log(f"  Plot saved: {path}")

    # Save importance data to JSON
    imp_path = os.path.join(EVAL_DIR, "feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    log(f"\n  Feature importance data saved: {imp_path}")

    return importance_data

# ---------------------------------------------------------------------------
# STEP 12 — FINAL VALIDATION
# ---------------------------------------------------------------------------

def step12_final_validation(tuned_models, best_name, feature_names, X_test, y_test):
    section("STEP 12 — FINAL VALIDATION")

    errors   = []
    best_mdl = tuned_models[best_name]

    # 1. Model loads successfully (already in memory, but confirm predict works)
    try:
        preds = best_mdl.predict(X_test)
        probs = best_mdl.predict_proba(X_test)[:, 1]
        log(f"  Predictions generated: {len(preds)} rows, "
            f"prob range [{probs.min():.4f}, {probs.max():.4f}]")
    except Exception as e:
        errors.append(f"Prediction failed: {e}")

    # 2. Reload from disk and predict again — confirms serialisation
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    loaded    = joblib.load(best_path)
    try:
        preds2 = loaded.predict(X_test)
        match  = np.array_equal(preds, preds2)
        log(f"  Reloaded model predictions match original: {match}")
        if not match:
            errors.append("Loaded model predictions differ from in-memory model")
    except Exception as e:
        errors.append(f"Loaded model prediction failed: {e}")

    # 3. Feature count
    if X_test.shape[1] != EXPECTED_FEATURES:
        errors.append(f"Feature count {X_test.shape[1]} != {EXPECTED_FEATURES}")
    else:
        log(f"  Feature count : {X_test.shape[1]} == {EXPECTED_FEATURES}  OK")

    # 4. Feature order matches feature_names.json
    if list(X_test.columns) == feature_names:
        log(f"  Feature order matches feature_names.json  OK")
    else:
        errors.append("Feature order does not match feature_names.json")

    # 5. No NaN in test features
    nans = X_test.isna().sum().sum()
    log(f"  NaN in test features: {nans}  {'OK' if nans == 0 else 'WARNING'}")
    if nans > 0:
        errors.append(f"NaN in X_test: {nans}")

    # 6. Target not in features
    if "target" in X_test.columns:
        errors.append("'target' column present in test features")
    else:
        log("  'target' absent from feature matrix  OK")

    # 7. Forbidden columns absent
    for col in FORBIDDEN_COLS:
        if col in X_test.columns:
            errors.append(f"Forbidden column in test: {col}")
    log(f"  All {len(FORBIDDEN_COLS)} forbidden columns absent  OK")

    if errors:
        log("\n  VALIDATION ERRORS:")
        for e in errors:
            log(f"    !! {e}")
    else:
        log("\n  All validation checks passed.")

    return len(errors) == 0

# ---------------------------------------------------------------------------
# STEP 13 — FINAL SUMMARY TO CONSOLE
# ---------------------------------------------------------------------------

def step13_summary(test_results, cv_results_all, best_name, baseline_accuracy):
    section("STEP 13 — FINAL SUMMARY")

    log(f"\n  Prediction point : Day 0 (pre-funding idea stage)")
    log(f"  Baseline accuracy: {baseline_accuracy:.4f}  ({baseline_accuracy*100:.2f}%)")
    log(f"  Model selection  : CV ROC-AUC (training data only)")
    log(f"\n  {'Model':<25} {'CV AUC':>9} {'CV F1':>9} {'Test Acc':>9} "
        f"{'Test F1':>9} {'Test AUC':>9} {'Selected':>10}")
    log(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    for name in test_results:
        tr  = test_results[name]
        cv  = cv_results_all[name]
        sel = "<-- SELECTED" if name == best_name else ""
        log(f"  {name:<25} {cv['cv_auc_mean']:>9.4f} {cv['cv_f1_mean']:>9.4f} "
            f"{tr['accuracy']:>9.4f} {tr['f1']:>9.4f} {tr['roc_auc']:>9.4f} {sel:>10}")

    best_cv = cv_results_all[best_name]
    best_ts = test_results[best_name]
    log(f"\n  Selected model      : {best_name}")
    log(f"  Selection basis     : highest CV AUC = {best_cv['cv_auc_mean']:.4f} "
        f"(training data only)")
    log(f"  Test Accuracy       : {best_ts['accuracy']:.4f}")
    log(f"  Test Precision      : {best_ts['precision']:.4f}")
    log(f"  Test Recall         : {best_ts['recall']:.4f}")
    log(f"  Test F1             : {best_ts['f1']:.4f}")
    log(f"  Test ROC-AUC        : {best_ts['roc_auc']:.4f}")
    log(f"  Beats baseline      : {best_ts['accuracy'] > baseline_accuracy}")
    log(f"\n  Phase 3 complete. DO NOT start Phase 4 / application integration")
    log(f"  until results have been reviewed and approved.")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("  AI-Based Startup Viability Prediction — Phase 3 Model Training")
    log("=" * 70)
    log(f"  Random seed      : {SEED}")
    log(f"  CV folds         : {CV_FOLDS}")
    log(f"  Expected features: {EXPECTED_FEATURES}")

    X_train, y_train, X_test, y_test, feature_names = step1_load()
    step2_leakage_audit(X_train, X_test, feature_names)
    baseline_accuracy, majority_class = step3_baseline(y_train, y_test)
    models_and_grids = step4_define_models()
    tuned_models, cv_results_all = step5_train(X_train, y_train, models_and_grids)
    test_results = step6_test_evaluation(tuned_models, X_test, y_test, baseline_accuracy)
    best_name = step7_select_model(test_results, cv_results_all, baseline_accuracy)
    metadata  = step8_save(tuned_models, best_name, test_results,
                            cv_results_all, feature_names, baseline_accuracy)
    step9_confusion_matrices(tuned_models, X_test, y_test)
    step10_roc_curves(tuned_models, X_test, y_test, test_results)
    importance_data = step11_feature_importance(tuned_models, feature_names)
    validation_ok = step12_final_validation(
        tuned_models, best_name, feature_names, X_test, y_test
    )
    step13_summary(test_results, cv_results_all, best_name, baseline_accuracy)

    # Write log
    log_path = os.path.join(EVAL_DIR, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n  [LOG SAVED] {log_path}")

    if not validation_ok:
        print("\n  WARNING: Final validation had errors. Review training_log.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
