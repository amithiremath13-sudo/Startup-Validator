# AI-Based Startup Viability Prediction and Validation System

An academic project (5th-semester minor project) that combines a trained
machine learning model with a rule-based idea-evaluation engine to produce
a structured viability report for a startup idea.

---

## Table of Contents

1. [Problem and Objectives](#1-problem-and-objectives)
2. [System Architecture](#2-system-architecture)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Prerequisites](#4-prerequisites)
5. [Running Locally](#5-running-locally)
6. [API Reference](#6-api-reference)
7. [Score Definitions and Limitations](#7-score-definitions-and-limitations)
8. [Model Evaluation (Historical Test-Set Results)](#8-model-evaluation-historical-test-set-results)
9. [Running the Backend Tests](#9-running-the-backend-tests)
10. [Troubleshooting](#10-troubleshooting)
11. [Key Documentation Files](#11-key-documentation-files)

---

## 1. Problem and Objectives

Evaluating a startup idea early — before significant time and money is
invested — is difficult without access to expert investors or large
datasets. This project explores whether historical startup-outcome patterns
from Crunchbase can be used to provide a structured, data-grounded
assessment of a new idea at the concept stage (Day 0, before any funding
is sought).

**Intended users:** Students and researchers studying startup ecosystems;
anyone who wants a structured, multi-dimensional evaluation of an early-stage
idea as a learning or decision-support tool.

**Objectives:**
- Train a supervised ML classifier on historical startup outcomes
  (acquisition/IPO = success; closure = failure) using only features
  available at founding: sector, country, and geography.
- Build a deterministic rule-based evaluator that scores how clearly a
  startup idea is described across five dimensions.
- Combine both components into a single structured validation report
  delivered through a REST API and a React web interface.
- Be explicit about what the system can and cannot predict.

**What this system does NOT do:**
- It does not evaluate market size, competition, revenue potential,
  founder quality, or technical feasibility.
- It does not guarantee or predict real-world startup success.
- The ML score is not a calibrated probability.

---

## 2. System Architecture

```
User (browser)
    │
    │  POST /api/validate { "idea": "...", "country": "..." }
    ▼
React SPA — frontend/
    │  (VITE_API_BASE_URL from .env.development or .env.production)
    ▼
FastAPI backend — backend/
    │
    ├── Component A: ML inference
    │   ├── Deterministic keyword-based feature extraction (65 features)
    │   ├── RandomForestClassifier (trained on 13,334 historical startups)
    │   └── Outputs: predicted_class (0 or 1) + ml_success_score (0–1)
    │
    └── Component B: Idea clarity scoring
        ├── Rule-based keyword/heuristic scoring (5 dimensions)
        └── Outputs: idea_evaluation_score (0–100) + per-dimension explanations
                     │
                     ▼
              Structured JSON report
              (two scores always kept separate; combined_score is always null)
```

### ML Model (Component A)

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Features | 65 binary/integer features — sector, country, geography |
| Prediction point | Day 0: founding / idea stage, before any funding |
| Selected by | Highest 5-fold CV ROC-AUC on training data only |
| Training dataset | 13,334 labeled startups (acquired/IPO = 1; closed = 0) |
| Source data | Crunchbase-derived dataset, ~1990–2016 |

### Idea Clarity Evaluator (Component B)

| Dimension | Weight |
|-----------|-------:|
| Problem Clarity | 25% |
| Solution Clarity | 25% |
| Target Market | 20% |
| Specificity | 15% |
| Completeness | 15% |

Each dimension scores 0–10. The weighted sum produces the
Idea Evaluation Score (0–100). Scores are designer-defined heuristics,
not learned from data.

---

## 3. Project Folder Structure

```
startup validator/
│
├── backend/                     ← FastAPI application (authoritative runtime)
│   ├── api/
│   │   ├── app.py               ← Routes, validation, CORS, error handling
│   │   └── dependencies.py      ← Startup-time model loading
│   ├── inference/               ← ML pipeline
│   │   ├── feature_extractor.py ← 65-feature extraction from idea + country
│   │   ├── idea_scorer.py       ← Rule-based 5-dimension scoring engine
│   │   ├── model_loader.py      ← Model loading and predict()
│   │   ├── report_generator.py  ← Report assembly, narrative, limitations
│   │   └── vocabulary.py        ← Category keywords, country aliases
│   ├── models/
│   │   ├── best_model.joblib    ← Trained Random Forest (do not modify)
│   │   └── model_metadata.json
│   ├── processed_data/
│   │   ├── feature_names.json   ← 65 feature names in exact training order
│   │   └── preprocessing_metadata.json
│   ├── tests/                   ← 130 pytest tests
│   ├── 05_evaluate_idea.py      ← Pipeline orchestrator
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                    ← React + Vite single-page app
│   ├── src/
│   │   ├── api.js               ← fetch wrapper for POST /api/validate
│   │   ├── App.jsx              ← Root component, API state
│   │   ├── index.css            ← Global styles
│   │   └── components/
│   │       ├── IdeaForm.jsx
│   │       ├── ScoreCards.jsx
│   │       ├── DimensionBreakdown.jsx
│   │       ├── NarrativeSection.jsx
│   │       ├── MLDetails.jsx
│   │       └── Limitations.jsx
│   ├── .env.development         ← VITE_API_BASE_URL=http://localhost:8000
│   ├── .env.production          ← Set VITE_API_BASE_URL before building
│   ├── vite.config.js
│   ├── package.json
│   └── README.md
│
├── ── Root-level training and CLI scripts (not part of the running app) ──
│
├── 01_explore_dataset.py        ← Phase 1: EDA
├── 02_preprocess_data.py        ← Phase 2: Feature engineering, train/test split
├── 03_train_models.py           ← Phase 3: Model training and evaluation
├── 04_predict_startup.py        ← Phase 4A: CLI inference tool
├── 05_evaluate_idea.py          ← Phase 4B/5: CLI evaluation + report
├── 06_generate_report.py        ← Phase 5: CLI report generator
├── big_startup_secsees_dataset.csv  ← Raw Crunchbase-derived dataset (66,368 rows)
│
├── models/                      ← Output directory for 03_train_models.py
├── processed_data/              ← Output directory for 02_preprocess_data.py
│                                   (also contains train/test CSV splits)
├── inference/                   ← Copy used by root CLI scripts (04–06)
├── data/                        ← Intermediate data from preprocessing
├── outputs/                     ← EDA plots, model evaluation charts
│
└── ── Documentation ──
    ├── README.md                    ← This file
    ├── ML_METHODOLOGY.md            ← Prediction framework, feature decisions
    ├── MODEL_EVALUATION_REPORT.md   ← Full Phase 3 evaluation results
    ├── PREPROCESSING_REPORT.md      ← Phase 2 preprocessing decisions
    ├── FEATURE_DICTIONARY.md        ← All 65 features documented
    ├── INFERENCE_PIPELINE.md        ← Phase 4A pipeline documentation
    ├── DATA_QUALITY_REPORT.md       ← Dataset quality findings
    ├── MODEL_FEATURE_IMPORTANCE.md  ← Feature importance analysis
    ├── PHASE_4A_AUDIT.md            ← Inference pipeline audit
    └── PROJECT_AUDIT.md             ← Full project audit (post-migration)
```

---

## 4. Prerequisites

| Requirement | Version used | Notes |
|-------------|:------------:|-------|
| Python | 3.14.7 | 3.10+ should work |
| Node.js | 24.14.0 | 18+ should work |
| npm | 11.9.0 | Comes with Node.js |

All Python dependencies are pinned in `backend/requirements.txt`.
The frontend uses only React 19 and Vite 8 (no additional UI library).

---

## 5. Running Locally

### 5.1 Backend setup

Open a PowerShell terminal and run the following from the project root:

```powershell
# Navigate to the backend folder
Set-Location "backend"

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn api.app:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.

- Swagger UI (interactive docs): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

> **Important:** Run `uvicorn` from inside the `backend/` directory.
> The backend resolves all paths (model files, feature metadata) relative
> to its own location. Running it from the project root will cause
> `ModelLoadError` at startup.

### 5.2 Frontend setup

Open a **second** PowerShell terminal and run:

```powershell
# Navigate to the frontend folder (from the project root)
Set-Location "frontend"

# Install dependencies (only needed once, or after package.json changes)
npm install

# Start the Vite development server
npm run dev
```

The frontend is now available at `http://localhost:5173`.

The dev server reads `frontend/.env.development` which sets:
```
VITE_API_BASE_URL=http://localhost:8000
```

All API requests from the browser go to `http://localhost:8000/api/validate`.

### 5.3 Verifying the connection

With both servers running:

1. Open `http://localhost:5173` in a browser.
2. Enter a startup idea description (aim for 40–80 words).
3. Select a country from the dropdown.
4. Click **Validate idea**.
5. The report appears below the form with two separate scores and
   per-dimension explanations.

### 5.4 Frontend production build

The production build is used for static hosting (e.g. S3/CloudFront).
Before building, set the real backend URL in `frontend/.env.production`:

```powershell
# Edit .env.production and replace the placeholder
# VITE_API_BASE_URL=https://YOUR_ACTUAL_BACKEND_URL

# Then build
Set-Location "frontend"
npm run build
# Output goes to frontend/dist/
```

The URL is baked into the bundle at build time. If the backend URL
changes after building, you must rebuild and redeploy.

---

## 6. API Reference

### `POST /api/validate`

Validates a startup idea and returns a structured report.

**Request body (JSON):**

```json
{
  "idea":    "An AI-powered SaaS platform helping small restaurants reduce food waste.",
  "country": "India"
}
```

| Field | Type | Required | Constraints |
|-------|------|:--------:|-------------|
| `idea` | string | ✅ | Non-empty, non-whitespace |
| `country` | string | ✅ | Non-empty; any country name or ISO-3 code |

**Example request (PowerShell):**

```powershell
$body = '{"idea": "An AI-powered SaaS platform helping small restaurants reduce food waste.", "country": "India"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/validate" -Method POST -ContentType "application/json" -Body $body
```

**Example response (200 OK) — actual values from live run:**

```json
{
  "submission": {
    "idea_text": "An AI-powered SaaS platform helping small restaurants reduce food waste.",
    "country_input": "India",
    "resolved_country": "IND",
    "resolved_categories": ["SaaS", "Analytics", "Clean Technology", "Services"],
    "is_complete_submission": true,
    "word_count": 11
  },
  "ml_result": {
    "predicted_class": 0,
    "pattern_label": "Failure Pattern",
    "ml_success_score": 0.4593,
    "ml_failure_score": 0.5407,
    "note": "This is a model-estimated success score based on historical structural patterns. It is NOT a probability or guarantee of real-world startup success.",
    "active_country_features": ["country_IND"],
    "active_category_features": ["cat_SaaS", "cat_Analytics", "cat_Clean_Technology", "cat_Services"],
    "n_categories": 4,
    "source": "Random Forest trained on Crunchbase ~1990-2016"
  },
  "idea_evaluation": {
    "idea_evaluation_score": 65,
    "max_score": 100,
    "grade": "Moderate",
    "note": "This score measures description clarity only. It does not predict business success.",
    "dimensions": [
      {
        "name": "Problem Clarity",
        "score": 6,
        "max_score": 10,
        "weight_pct": 25,
        "explanation": "A problem is mentioned but could be more specific...",
        "matched_signals": ["reduce", "waste"]
      }
    ]
  },
  "narrative": {
    "strengths": ["..."],
    "missing_or_unclear": ["..."],
    "guidance": ["..."],
    "submission_warnings": []
  },
  "limitations": ["..."],
  "scores_are_separate": true,
  "combined_score": null
}
```

**Key response fields:**

| Field | What it means |
|-------|---------------|
| `ml_result.ml_success_score` | The Random Forest's vote proportion for class 1 (0.0–1.0). Reflects how similar the sector+country combination is to historical startups that achieved an exit. Not a calibrated probability. |
| `ml_result.pattern_label` | `"Success Pattern"` (class 1) or `"Failure Pattern"` (class 0) based on the 0.5 threshold. |
| `idea_evaluation.idea_evaluation_score` | Weighted keyword-based description quality score (0–100). Measures clarity, not business viability. |
| `idea_evaluation.grade` | `Strong` / `Moderate` / `Developing` / `Weak` / `Very Weak` |
| `scores_are_separate` | Always `true`. The two scores must never be averaged or combined. |
| `combined_score` | Always `null`. No combined score is produced. |

**Error responses:**

| Status | Cause |
|--------|-------|
| 400 | Empty or whitespace-only `idea` or `country` |
| 422 | Required field (`idea` or `country`) missing from body |
| 503 | ML model failed to load at startup |
| 500 | Unexpected internal error (no stack trace exposed to client) |

### `GET /health`

Returns `{"status": "ok"}` — use as a liveness probe.

---

## 7. Score Definitions and Limitations

### ML Model Score (`ml_success_score`)

**What it is:** The Random Forest's average vote across 100 trees for
class 1 (success), expressed as a value between 0.0 and 1.0.

**What it measures:** How structurally similar the submitted sector +
country combination is to historical startups in the training dataset
that achieved an acquisition or IPO outcome.

**What it is NOT:**
- It is **not** a calibrated probability of real-world startup success.
  The model was trained on historical Crunchbase data from approximately
  1990–2016. Current-market accuracy has not been established.
- It does **not** evaluate idea quality, team capability, market size,
  competition, revenue potential, or technical feasibility.
- A high score reflects a favorable historical structural pattern, not
  a recommendation to invest or proceed.

**Dataset note:** The raw dataset contains 66,368 company records.
After excluding companies with unknown outcomes (`operating` status:
53,034 rows), the labeled training population is 13,334 companies
(success = acquired or IPO; failure = closed). This dataset is heavily
US-biased (61.2% of labeled training rows are US-headquartered companies).

### Idea Clarity Score (`idea_evaluation_score`)

**What it is:** A weighted sum of five rule-based dimension scores
(0–10 each), each determined by keyword and phrase matching against
the submitted text.

**What it measures:** How clearly and specifically the startup idea was
described — not the quality or viability of the underlying business.
A well-described poor idea can outscore a poorly-described great one.

**Dimension weights (designer-defined, not learned from data):**

| Dimension | Weight | What it checks |
|-----------|-------:|----------------|
| Problem Clarity | 25% | Does the text describe a problem or pain point? |
| Solution Clarity | 25% | Does the text describe a product or solution? |
| Target Market | 20% | Is a customer or market segment named? |
| Specificity | 15% | Are concrete domain terms, technologies, or processes mentioned? |
| Completeness | 15% | Word count and coverage across the other four dimensions |

---

## 8. Model Evaluation (Historical Test-Set Results)

These metrics were measured once on the held-out test set (20% of labeled
data, 2,667 rows) after model selection was completed using
cross-validation on training data only. They reflect performance on
historical data, not current-market performance.

| Metric | Value |
|--------|------:|
| **Accuracy** | **69.37%** |
| Precision | 67.69% |
| Recall | 81.18% |
| F1-score | 73.82% |
| **ROC-AUC** | **75.99%** |
| Majority-class baseline accuracy | 53.21% |

**Model selection:** Random Forest was selected by the highest 5-fold
stratified CV ROC-AUC (0.7364 ± 0.0037) measured on training data only.
The test set was used exactly once, for final evaluation.

**Best hyperparameters:** `n_estimators=100, max_depth=None, min_samples_leaf=5`

**Confusion matrix (test set, 2,667 rows):**

```
                    Predicted Failure   Predicted Success
Actual Failure           698 (TN)            550 (FP)
Actual Success           267 (FN)          1,152 (TP)
```

**Interpretation:** The model is more likely to predict success than
failure (recall 81.2%, precision 67.7%). This recall bias is acceptable
for the use case — missing a genuinely viable idea is worse than
over-optimistically scoring a borderline one. The idea-clarity score
(Component B) provides complementary assessment.

---

## 9. Running the Backend Tests

```powershell
# From the backend/ directory
Set-Location "backend"
python -m pytest tests/ -q
```

**Expected result (verified):**
```
130 passed, 5 subtests passed in ~8–16s
```

Test coverage includes:
- 34 API endpoint tests (valid request, empty input, missing fields,
  unresolvable country, simulated pipeline failure)
- 27 idea scorer unit tests
- 21 inference pipeline tests
- 48 report generator tests

> Run `python -m pytest tests/ -v` for full per-test output.

---

## 10. Troubleshooting

### Backend does not start / `ModelLoadError`

**Cause:** The backend is being run from the wrong directory.

**Fix:** Always run `uvicorn` from inside `backend/`:
```powershell
Set-Location "backend"
uvicorn api.app:app --reload --port 8000
```

### Frontend shows "Could not reach the server"

**Cause:** The backend is not running, or it is running on a different port.

**Fix:**
1. Confirm the backend is running: `Invoke-RestMethod http://localhost:8000/health`
2. Confirm `frontend/.env.development` contains `VITE_API_BASE_URL=http://localhost:8000`
3. Restart the Vite dev server after any `.env` change.

### Frontend shows a CORS error in the browser console

**Cause:** The browser is blocking the cross-origin request.

**Fix:** The backend's default CORS configuration allows
`http://localhost:5173` and `http://localhost:4173`. Check that
the Vite dev server is running on one of these ports.
If you changed the Vite port, update the `ALLOWED_ORIGINS`
environment variable before starting the backend:
```powershell
$env:ALLOWED_ORIGINS = "http://localhost:YOUR_PORT"
uvicorn api.app:app --reload --port 8000
```

### `npm install` fails

**Cause:** Node.js is not installed or is an incompatible version.

**Fix:** Verify with `node --version` (18+ required). Download from
[nodejs.org](https://nodejs.org) if missing.

### `pip install -r requirements.txt` fails

**Cause:** Python is not installed, is the wrong version, or pip
cannot find the packages.

**Fix:** Verify with `python --version` (3.10+ required). Ensure pip
is up to date: `python -m pip install --upgrade pip`, then retry.

---

## 11. Key Documentation Files

| File | Contents |
|------|----------|
| `ML_METHODOLOGY.md` | Prediction framework, feature decisions, leakage prevention |
| `MODEL_EVALUATION_REPORT.md` | Full Phase 3 model evaluation with selection methodology |
| `PREPROCESSING_REPORT.md` | Phase 2 data cleaning, feature engineering, train/test split |
| `FEATURE_DICTIONARY.md` | All 65 model features with descriptions and justifications |
| `INFERENCE_PIPELINE.md` | Phase 4A feature extraction and inference pipeline |
| `DATA_QUALITY_REPORT.md` | EDA findings, missing values, data quality assessment |
| `MODEL_FEATURE_IMPORTANCE.md` | Feature importance analysis for all five trained models |
| `PHASE_4A_AUDIT.md` | Read-only audit of the inference pipeline |
| `PROJECT_AUDIT.md` | Full project audit after backend/frontend migration |
| `backend/README.md` | Backend-specific setup, CORS, endpoint reference |
| `frontend/README.md` | Frontend-specific setup, env variables, build instructions |

---

*This project was developed for academic purposes only.*
*It does not constitute investment, business, or legal advice.*
*The ML model reflects historical Crunchbase data from approximately 1990–2016.*
*Real-world startup success depends on many factors not captured in this system.*
