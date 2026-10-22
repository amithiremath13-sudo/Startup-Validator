# Project Audit Report
## AI-Based Startup Viability Prediction and Validation System

---

> **Audit type:** Read-only. No files were modified, moved, renamed, or deleted.
> No model was retrained. No dependencies were installed.
>
> **All findings below are based on direct inspection of source code,
> metadata files, and live test/build execution.** Verified facts are
> distinguished from assumptions.

---

## Table of Contents

- [A. Current Architecture and Folder Structure](#a-current-architecture-and-folder-structure)
- [B. Verified Working Components and Test/Build Results](#b-verified-working-components-and-testbuild-results)
- [C. Bugs and Inconsistencies](#c-bugs-and-inconsistencies)
- [D. Missing Functionality for a Complete Academic Demonstration](#d-missing-functionality-for-a-complete-academic-demonstration)
- [E. Proposed Repair Sequence](#e-proposed-repair-sequence)

---

## A. Current Architecture and Folder Structure

### A.1 Top-level layout

```
startup validator/
├── backend/                    ← Phase 6A: FastAPI backend (the live backend)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py               ← FastAPI app with CORS
│   │   └── dependencies.py      ← Startup model loading
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py ← Phase 4A: 65-feature extraction
│   │   ├── idea_scorer.py       ← Phase 4B: rule-based scoring (5 dimensions)
│   │   ├── model_loader.py      ← Model loading + predict()
│   │   ├── report_generator.py ← Phase 5: report assembly + narrative
│   │   └── vocabulary.py        ← Category keywords + country aliases
│   ├── models/
│   │   ├── best_model.joblib    ← Trained Random Forest (65 features)
│   │   ├── model_metadata.json
│   │   └── (4 other .joblib files)
│   ├── processed_data/
│   │   ├── feature_names.json   ← 65 feature names in exact order
│   │   └── preprocessing_metadata.json
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_idea_scorer.py
│   │   ├── test_inference.py
│   │   └── test_report_generator.py
│   ├── 05_evaluate_idea.py       ← Pipeline orchestrator (importlib-loaded)
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                    ← Phase 6A: React + Vite
│   ├── src/
│   │   ├── api.js               ← fetch wrapper for POST /api/validate
│   │   ├── App.jsx              ← Root component, owns API state
│   │   ├── main.jsx
│   │   ├── index.css            ← Global styles
│   │   ├── App.css              ← Empty (placeholder)
│   │   └── components/
│   │       ├── IdeaForm.jsx
│   │       ├── ScoreCards.jsx
│   │       ├── DimensionBreakdown.jsx
│   │       ├── NarrativeSection.jsx
│   │       ├── MLDetails.jsx
│   │       └── Limitations.jsx
│   ├── public/
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src/assets/              ← Stale Vite defaults (react.svg, vite.svg, hero.png)
│   ├── .env.development         ← VITE_API_BASE_URL=http://localhost:8000
│   ├── .env.production          ← VITE_API_BASE_URL=https://REPLACE_WITH_YOUR_EC2_URL
│   ├── vite.config.js
│   ├── package.json
│   └── README.md
│
├── api/                         ← ⚠ STALE DUPLICATE of backend/api/ (no CORS)
├── inference/                   ← ⚠ STALE DUPLICATE of backend/inference/
├── models/                      ← ⚠ STALE DUPLICATE of backend/models/
├── processed_data/              ← ⚠ STALE DUPLICATE (also has train/test CSVs)
├── tests/                       ← ⚠ STALE DUPLICATE of backend/tests/
├── 05_evaluate_idea.py          ← ⚠ STALE DUPLICATE of backend/05_evaluate_idea.py
├── requirements-api.txt         ← ⚠ STALE DUPLICATE of backend/requirements.txt
│
├── 01_explore_dataset.py        ← Training/EDA scripts (stay at root)
├── 02_preprocess_data.py
├── 03_train_models.py
├── 04_predict_startup.py
├── 06_generate_report.py
├── big_startup_secsees_dataset.csv  ← Raw training dataset
├── data/                        ← Preprocessing output (labeled CSVs)
├── outputs/                     ← EDA plots, model evaluation
├── *.md                         ← 12 documentation files
└── API_README.md               ← Phase 6A API docs (stale — predates backend/ move)
```

### A.2 System architecture (data flow)

```
User (browser)
    │
    ▼
React SPA (frontend/)
    │  POST /api/validate  { idea, country }
    │  VITE_API_BASE_URL → backend URL
    ▼
FastAPI (backend/api/app.py)
    │  CORS: ALLOWED_ORIGINS env var or localhost defaults
    │  Pydantic validation → 400/422
    ▼
05_evaluate_idea.py::run_full_evaluation()
    │
    ├── Component A: inference/feature_extractor.py + model_loader.py
    │   └── 65-feature vector → RandomForestClassifier → predicted_class + scores
    │
    ├── Component B: inference/idea_scorer.py
    │   └── 5-dimension keyword scoring → idea_evaluation_score (0–100)
    │
    └── inference/report_generator.py
        └── generate_report() + report_to_dict() → structured JSON
            │
            ▼
        ValidateResponse (Pydantic model) → JSON to frontend
```

### A.3 Key design properties (verified)

| Property | Status | How verified |
|----------|--------|--------------|
| ML score and Idea Evaluation Score are separate | ✅ Verified | `combined_score: null`, `scores_are_separate: true` in API response; frontend renders them as distinct cards |
| ML score never called a probability | ✅ Verified | `ml_result.note` says "NOT a probability"; frontend renders this note in MLDetails; tests assert no "probability of success" phrase |
| 65 features in exact order | ✅ Verified | `model.n_features_in_ == 65`; `feature_names.json` has 65 entries; `FEATURE_ORDER` in `feature_extractor.py` matches |
| Model is RandomForestClassifier | ✅ Verified | `type(model).__name__ == "RandomForestClassifier"`; `n_estimators=100`, `max_depth=None` |
| No funding/future features | ✅ Verified | Forbidden columns list checked in tests; none appear in feature vector |
| CORS is environment-based | ✅ Verified | `ALLOWED_ORIGINS` env var with localhost fallback; `allow_credentials=False` |
| Frontend renders all data dynamically | ✅ Verified | All components read from `report.*` props; no hardcoded scores |

---

## B. Verified Working Components and Test/Build Results

### B.1 Backend tests (executed from `backend/`)

```
Command:  python -m pytest tests/ -q
Result:   130 passed, 5 subtests passed, 0 failures
Time:     16.35s
```

All 130 tests pass. This includes:
- 34 API tests (`test_api.py`)
- 27 idea scorer tests (`test_idea_scorer.py`)
- 21 inference tests (`test_inference.py`)
- 48 report generator tests (`test_report_generator.py`)

### B.2 Root-level tests (executed from project root)

```
Command:  python -m pytest tests/ -q
Result:   130 passed, 5 subtests passed, 0 failures
Time:     7.88s
```

The root-level tests also pass because they use the same relative-path logic and the duplicate files at root are identical to the backend copies (except `api/app.py` which lacks CORS — but tests don't check CORS).

### B.3 Frontend production build

```
Command:  npm run build  (from frontend/)
Result:   ✓ built in 540ms
Output:   dist/index.html (0.45 KB)
          dist/assets/index-n_VUd3ow.css (4.69 KB)
          dist/assets/index-CbMz6Q63.js (231.75 KB, 72.53 KB gzip)
```

Build succeeds with no errors or warnings. The `VITE_API_BASE_URL` placeholder (`https://REPLACE_WITH_YOUR_EC2_URL`) is correctly baked into the bundle — this is expected for a pre-deployment build.

### B.4 Model integrity (verified from `backend/`)

| Property | Value |
|----------|-------|
| Model type | `RandomForestClassifier` |
| `n_features_in_` | 65 |
| `n_estimators` | 100 |
| `max_depth` | None |
| `min_samples_leaf` | 5 |
| `classes_` | [0, 1] |
| `feature_names.json` count | 65 |
| Feature order matches model | ✅ Yes |
| `top_categories_list` count | 50 |
| `top_countries_list` count | 10 |

### B.5 Idea scorer weights (verified)

| Dimension | Weight |
|-----------|-------:|
| Problem Clarity | 25% |
| Solution Clarity | 25% |
| Target Market | 20% |
| Specificity | 15% |
| Completeness | 15% |
| **Sum** | **100%** |

Assertion `abs(sum(weights) - 1.0) < 1e-9` is present in the source and would fail at import if weights drift.

### B.6 Frontend-backend schema consistency

The frontend accesses these API response fields (verified by grep of all `.jsx` files):

| Frontend access | API response key | Match? |
|----------------|------------------|:------:|
| `report.ml_result` | `ml_result` | ✅ |
| `report.ml_result.ml_success_score` | `ml_result.ml_success_score` | ✅ |
| `report.ml_result.predicted_class` | `ml_result.predicted_class` | ✅ |
| `report.ml_result.pattern_label` | `ml_result.pattern_label` | ✅ |
| `report.ml_result.source` | `ml_result.source` | ✅ |
| `report.ml_result.note` | `ml_result.note` | ✅ |
| `report.ml_result.active_country_features` | `ml_result.active_country_features` | ✅ |
| `report.ml_result.active_category_features` | `ml_result.active_category_features` | ✅ |
| `report.ml_result.n_categories` | `ml_result.n_categories` | ✅ |
| `report.idea_evaluation.idea_evaluation_score` | `idea_evaluation.idea_evaluation_score` | ✅ |
| `report.idea_evaluation.grade` | `idea_evaluation.grade` | ✅ |
| `report.idea_evaluation.note` | `idea_evaluation.note` | ✅ |
| `report.idea_evaluation.dimensions` | `idea_evaluation.dimensions` | ✅ |
| `dim.name`, `dim.score`, `dim.max_score`, `dim.weight_pct`, `dim.explanation` | All present in dimension schema | ✅ |
| `report.narrative.strengths` | `narrative.strengths` | ✅ |
| `report.narrative.missing_or_unclear` | `narrative.missing_or_unclear` | ✅ |
| `report.narrative.guidance` | `narrative.guidance` | ✅ |
| `report.narrative.submission_warnings` | `narrative.submission_warnings` | ✅ |
| `report.submission.resolved_country` | `submission.resolved_country` | ✅ |
| `report.submission.is_complete_submission` | `submission.is_complete_submission` | ✅ |
| `report.limitations` | `limitations` | ✅ |

**No schema mismatches found.** Every field the frontend reads exists in the API response with the correct type.

### B.7 CORS configuration (verified)

```python
# backend/api/app.py
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _DEFAULT_ORIGINS
)
```

- Local development: defaults allow Vite dev/preview servers.
- Production: `ALLOWED_ORIGINS` env var must be set on EC2.
- `allow_credentials=False` — correct for a public API that doesn't use cookies.
- `allow_methods=["GET", "POST", "OPTIONS"]` — minimal set.
- Never uses `["*"]` in production — verified.
- The old root-level `api/app.py` does NOT have CORS middleware — it is a stale copy.

---

## C. Bugs and Inconsistencies

### CRITICAL

**None found.** The backend tests all pass, the frontend builds cleanly, the model is intact, and the API response schema matches the frontend's expectations.

### HIGH

#### H1 — Duplicate root-level files from incomplete backend migration

**Severity:** High (causes confusion, potential import ambiguity, and maintenance hazard)

During the `backend/` migration, files were **copied** into `backend/` but the originals were **not removed** from the project root. The following duplicate copies exist and are stale:

| Root-level file | Backend equivalent | Identical? |
|----------------|-------------------|:----------:|
| `api/app.py` | `backend/api/app.py` | ❌ No — root version lacks CORS middleware |
| `api/dependencies.py` | `backend/api/dependencies.py` | ✅ Yes |
| `api/__init__.py` | `backend/api/__init__.py` | ✅ Yes |
| `inference/*.py` (5 files) | `backend/inference/*.py` | ✅ Yes |
| `models/*.joblib` (7 files) | `backend/models/*.joblib` | ✅ Yes (binary) |
| `processed_data/*.json` (2 files) | `backend/processed_data/*.json` | ✅ Yes |
| `tests/*.py` (4 files) | `backend/tests/*.py` | ✅ Yes |
| `05_evaluate_idea.py` | `backend/05_evaluate_idea.py` | ✅ Yes |
| `requirements-api.txt` | `backend/requirements.txt` | ✅ Yes (renamed) |

**Risk:** If someone runs `uvicorn api.app:app` from the project root, they get the old `api/app.py` without CORS — the frontend will fail with CORS errors. If someone runs `python -m pytest tests/` from the root, they test the old copies, not the backend.

**Recommendation:** Remove the stale root-level duplicates after confirming the backend copies are the canonical ones. Keep the root-level `processed_data/` only if the training scripts (`01_`–`03_`) still need the train/test CSVs — but the two JSON files the API needs are already in `backend/processed_data/`.

#### H2 — No root-level `README.md` or project-level documentation

**Severity:** High (academic project deliverable gap)

There is no `README.md` at the project root. An examiner or reviewer opening the project has no entry point explaining:
- What the project does
- How to run the backend
- How to run the frontend
- Where the documentation lives
- What the test results are

The `backend/README.md` and `frontend/README.md` exist but are not discoverable without knowing to look inside those subdirectories. `API_README.md` exists at root but documents the pre-migration API.

### MEDIUM

#### M1 — Frontend HTML title is still Vite default "frontend"

**Severity:** Medium (presentation quality)

`frontend/index.html` line 7:
```html
<title>frontend</title>
```

Should be something like "AI Startup Validator" — the current title appears in the browser tab and search results.

#### M2 — Stale Vite default assets in `frontend/src/assets/`

**Severity:** Low (unused files, no functional impact)

`frontend/src/assets/` contains `react.svg`, `vite.svg`, and `hero.png` from the Vite scaffold. None of these are imported by any component. They are dead files that add noise to the project.

#### M3 — `frontend/src/App.css` is an empty placeholder

**Severity:** Low (dead file)

`App.css` contains only a comment: `/* App-specific overrides — global styles live in index.css */`. It is not imported by any component. It could be removed, but it causes no harm.

#### M4 — `API_README.md` at root is stale

**Severity:** Medium (documentation accuracy)

`API_README.md` documents the API as it existed before the `backend/` migration. It references running from the project root (`uvicorn api.app:app`) and does not mention the `backend/` folder. It should either be updated to point to `backend/` or removed in favour of `backend/README.md`.

#### M5 — No `.gitignore` at project root

**Severity:** Medium (version control hygiene)

If the project is committed to Git, Python `__pycache__/` directories, `.pytest_cache/`, `node_modules/`, and `dist/` at various levels would be included. The `frontend/.gitignore` covers `frontend/` but not the Python artifacts at root or in `backend/`.

### LOW

#### L1 — `frontend/src/assets/hero.png` is an unverified file

`hero.png` exists in `frontend/src/assets/` but is not imported anywhere. Its provenance is unknown — it may be a Vite scaffold artifact or an old placeholder. No functional impact.

#### L2 — Deprecation warnings in test output

The backend test run produces 3 deprecation warnings:
- `anyio.abc.BlockingPortal` alias deprecated (from Starlette TestClient)
- `asyncio.iscoroutinefunction` deprecated (from FastAPI routing)

These are library-level warnings, not project bugs. They will resolve when `starlette` and `fastapi` update. No action required.

---

## D. Missing Functionality for a Complete Academic Demonstration

### D1 — No root-level `README.md`

**What's missing:** A single entry-point document at the project root that explains the project, how to run it, and where to find documentation.

**Why it matters for a viva:** The examiner's first question is typically "show me the project." A root `README.md` is the expected starting point.

### D2 — No end-to-end integration test (frontend → backend)

**What's missing:** There is no test that starts the backend server, sends a request from the frontend, and verifies the full round-trip. The backend tests use `TestClient` (in-process) and the frontend has no tests at all.

**Why it matters:** For an academic demo, proving the frontend and backend work together is important. However, this is a common gap in projects at this scale and can be demonstrated manually.

### D3 — No frontend tests

**What's missing:** There are no React component tests (e.g. with Vitest or React Testing Library). The frontend is only verified by the production build succeeding.

**Why it matters:** For an academic project, this is acceptable — the frontend is a thin rendering layer over the API. The backend tests cover the core logic.

### D4 — No deployment configuration files

**What's missing:** No `docker-compose.yml`, `Procfile`, `nginx.conf`, or AWS deployment scripts. The `backend/README.md` and `frontend/README.md` contain instructions but no automation.

**Why it matters:** This is intentionally deferred per the project constraints. Not a gap for the local academic demonstration.

---

## E. Proposed Repair Sequence

### Priority 1 (highest) — Remove stale root-level duplicates

**Issue:** H1

**Why first:** This is the only issue that could cause incorrect behavior (running the wrong `api/app.py` without CORS). It also creates confusion about which files are canonical and clutters the project structure.

**Proposed fix:**
1. Delete the root-level `api/` directory (stale, lacks CORS)
2. Delete the root-level `inference/` directory (identical copy in `backend/`)
3. Delete the root-level `tests/` directory (identical copy in `backend/`)
4. Delete the root-level `05_evaluate_idea.py` (identical copy in `backend/`)
5. Delete the root-level `requirements-api.txt` (replaced by `backend/requirements.txt`)
6. Keep `models/` at root if the training scripts (`01_`–`03_`) reference it; otherwise delete
7. Keep `processed_data/` at root if training scripts need the train/test CSVs; the two JSON files are already in `backend/processed_data/`
8. Update `API_README.md` to point to `backend/README.md`, or delete it

**Verification after fix:**
- Run `python -m pytest tests/ -q` from `backend/` — must still pass 130 tests
- Run `npm run build` from `frontend/` — must still build cleanly

### Priority 2 — Create root-level `README.md`

**Issue:** H2

**Proposed fix:** Write a project-root `README.md` that:
- Names the project and its purpose
- Lists the folder structure (`backend/`, `frontend/`, training scripts, docs)
- Gives quick-start instructions for both backend and frontend
- Links to key documentation files (`ML_METHODOLOGY.md`, `MODEL_EVALUATION_REPORT.md`, etc.)
- States the test results

### Priority 3 — Fix frontend HTML title

**Issue:** M1

**Proposed fix:** Change `<title>frontend</title>` to `<title>AI Startup Validator</title>` in `frontend/index.html`.

### Priority 4 — Clean up stale frontend assets

**Issue:** M2, M3, L1

**Proposed fix:** Delete `frontend/src/assets/react.svg`, `frontend/src/assets/vite.svg`, `frontend/src/assets/hero.png`, and `frontend/src/App.css`. None are imported.

### Priority 5 — Add root-level `.gitignore`

**Issue:** M5

**Proposed fix:** Create a `.gitignore` at the project root covering `__pycache__/`, `.pytest_cache/`, `*.pyc`, `node_modules/`, `dist/`, and `.env.local`.

### Priority 6 — Update or remove `API_README.md`

**Issue:** M4

**Proposed fix:** Either delete `API_README.md` (since `backend/README.md` supersedes it) or update it to redirect to `backend/README.md`.

---

*This audit was performed as a read-only inspection. No files were modified.
All test and build results were verified by live execution.
The proposed repair sequence is a recommendation — no fixes have been implemented.*
