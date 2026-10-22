"""
api/app.py
===========
Phase 6A — FastAPI backend for the AI Startup Validator.

SINGLE ENDPOINT
---------------
POST /api/validate
  Input  : { "idea": str, "country": str }
  Output : structured validation report (JSON)

DESIGN RULES
------------------------------------------
- ML score and Idea Evaluation Score are kept as SEPARATE keys.
- The ML score is never labelled as a calibrated probability.
- Stack traces are never exposed to API clients.
- All ML / scoring logic lives in the existing inference package.
  This file contains only routing, validation, and serialisation.

CORS
----
Allowed origins are read from the ALLOWED_ORIGINS environment variable
(comma-separated list).  If the variable is not set the default allows
only localhost development origins.

  Local development (default):
    http://localhost:5173
    http://localhost:4173
    http://127.0.0.1:5173

  Production: set ALLOWED_ORIGINS in the EC2 environment before starting
  the server, e.g.:
    export ALLOWED_ORIGINS="https://your-cloudfront-url.cloudfront.net"
    uvicorn api.app:app --host 0.0.0.0 --port 8000

HOW TO RUN
----------
  # From the backend/ directory:
  uvicorn api.app:app --reload --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ── Project root on path ─────────────────────────────────────────────────
# This file lives at  backend/api/app.py
# PROJECT_ROOT resolves to  backend/
_API_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_API_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.feature_extractor import StartupIdeaInput
from inference.model_loader import ModelLoadError
from inference.report_generator import generate_report, report_to_dict

from api.dependencies import initialize_resources, get_run_full_evaluation

logger = logging.getLogger(__name__)


# ── CORS origins ─────────────────────────────────────────────────────────
# Read from ALLOWED_ORIGINS env var (comma-separated).
# Falls back to localhost-only defaults when the variable is not set.
# NEVER use ["*"] in production: the browser would allow any site to call
# the API on behalf of a user.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",    # Vite dev server default
    "http://localhost:4173",    # Vite preview
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: List[str] = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _DEFAULT_ORIGINS
)


# ── Lifespan: load model once at startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model and metadata before accepting any requests."""
    try:
        initialize_resources()
        logger.info("AI Startup Validator: model loaded from %s", PROJECT_ROOT)
    except ModelLoadError as exc:
        logger.error("Failed to load model at startup: %s", exc)
    yield


# ── Application ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Startup Validator API",
    description=(
        "Validates a startup idea using a trained Random Forest model "
        "(Component A, historical sector/country patterns) and a "
        "rule-based idea-evaluation scorer (Component B, description quality). "
        "The two scores are kept separate and are never combined."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — must be added before any routes or other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# ── Validation error handler ─────────────────────────────────────────────
# "missing" field errors stay as 422 Unprocessable Entity.
# Empty/whitespace value errors from field_validator become 400 Bad Request.

@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    has_missing = any(e.get("type") == "missing" for e in errors)
    if has_missing:
        return JSONResponse(
            status_code=422,
            content={"detail": [
                {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                for e in errors
            ]},
        )
    msg = errors[0].get("msg", "Invalid input") if errors else "Invalid input"
    msg = msg.removeprefix("Value error, ")
    return JSONResponse(status_code=400, content={"detail": msg})


# ── Request / Response models ─────────────────────────────────────────────

class ValidateRequest(BaseModel):
    idea:    str
    country: str

    @field_validator("idea")
    @classmethod
    def idea_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("idea must not be empty or whitespace only")
        return stripped

    @field_validator("country")
    @classmethod
    def country_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("country must not be empty or whitespace only")
        return stripped


class DimensionScore(BaseModel):
    name:            str
    score:           int
    max_score:       int
    weight_pct:      int
    explanation:     str
    matched_signals: List[str]


class MLResult(BaseModel):
    source:                   str
    predicted_class:          int
    pattern_label:            str
    ml_success_score:         float
    ml_failure_score:         float
    note:                     str
    active_country_features:  List[str]
    active_category_features: List[str]
    n_categories:             int


class IdeaEvaluation(BaseModel):
    source:                str
    idea_evaluation_score: int
    max_score:             int
    grade:                 str
    note:                  str
    dimensions:            List[DimensionScore]


class Narrative(BaseModel):
    strengths:           List[str]
    missing_or_unclear:  List[str]
    guidance:            List[str]
    submission_warnings: List[str]


class Submission(BaseModel):
    idea_text:                str
    country_input:            Optional[str]
    categories_input:         List[str]
    resolved_country:         Optional[str]
    resolved_categories:      List[str]
    is_complete_submission:   bool
    word_count:               int
    is_meaningful_submission: bool


class ValidateResponse(BaseModel):
    """
    Structured validation report.

    ml_result.ml_success_score  — model-estimated success score (0–1).
        NOT a calibrated probability. See ml_result.note.

    idea_evaluation.idea_evaluation_score — rule-based clarity score (0–100).
        Measures description quality, not business viability.

    scores_are_separate : always True.
    combined_score      : always null.
    """
    submission:          Submission
    ml_result:           MLResult
    idea_evaluation:     IdeaEvaluation
    narrative:           Narrative
    limitations:         List[str]
    scores_are_separate: bool
    combined_score:      None


# ── Endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/api/validate",
    response_model=ValidateResponse,
    summary="Validate a startup idea",
)
def validate_idea(
    body: ValidateRequest,
    run_full_evaluation=Depends(get_run_full_evaluation),
) -> ValidateResponse:
    input_data = StartupIdeaInput(
        idea_text       = body.idea,
        country         = body.country,
        categories      = [],
        location_detail = None,
    )

    try:
        full_eval = run_full_evaluation(input_data)
    except ModelLoadError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(status_code=503,
            detail="The ML model is currently unavailable.")
    except ValueError as exc:
        logger.warning("Pipeline validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected pipeline error: %s", exc)
        raise HTTPException(status_code=500,
            detail="An internal error occurred. Please try again.")

    try:
        report      = generate_report(full_eval, country_input=body.country)
        report_dict = report_to_dict(report)
    except Exception as exc:
        logger.exception("Report generation failed: %s", exc)
        raise HTTPException(status_code=500,
            detail="An internal error occurred while generating the report.")

    return ValidateResponse(
        submission      = Submission(**report_dict["submission"]),
        ml_result       = MLResult(**report_dict["ml_result"]),
        idea_evaluation = IdeaEvaluation(
            **{k: v for k, v in report_dict["idea_evaluation"].items()
               if k != "dimensions"},
            dimensions=[DimensionScore(**d)
                        for d in report_dict["idea_evaluation"]["dimensions"]],
        ),
        narrative           = Narrative(**report_dict["narrative"]),
        limitations         = report_dict["limitations"],
        scores_are_separate = True,
        combined_score      = None,
    )


# ── Global exception handler ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )


# ── Health check ─────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
def health() -> Dict[str, str]:
    return {"status": "ok"}
