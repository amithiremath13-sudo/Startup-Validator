# AI Startup Validator — Backend

FastAPI backend exposing `POST /api/validate`.

---

## Directory layout

```
backend/
├── api/
│   ├── app.py           ← FastAPI application, CORS, request/response models
│   └── dependencies.py  ← Startup-time model loading
├── inference/           ← ML pipeline (feature extraction, scoring, reporting)
├── models/              ← Trained Random Forest + metadata
├── processed_data/      ← feature_names.json + preprocessing_metadata.json
├── tests/               ← pytest test suite (130 tests)
├── 05_evaluate_idea.py  ← Pipeline orchestrator (loaded by api/dependencies.py)
└── requirements.txt
```

---

## Local development

```bash
# 1. Install dependencies (from backend/)
pip install -r requirements.txt

# 2. Start the dev server
uvicorn api.app:app --reload --port 8000

# API is available at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Run tests

```bash
# From backend/
python -m pytest tests/ -q
```

---

## CORS configuration

Allowed origins are controlled by the `ALLOWED_ORIGINS` environment variable
(comma-separated list of allowed frontend URLs).

**Local development** — no variable needed. Defaults allow:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:4173` (Vite preview)
- `http://127.0.0.1:5173`

**Production on EC2** — set `ALLOWED_ORIGINS` before starting the server:

```bash
# Single origin (CloudFront):
export ALLOWED_ORIGINS="https://d1234abcd.cloudfront.net"

# Multiple origins (CloudFront + custom domain):
export ALLOWED_ORIGINS="https://d1234abcd.cloudfront.net,https://www.yourdomain.com"

uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Never use `ALLOWED_ORIGINS=*` in production. Always list the specific
frontend origin.

---

## AWS EC2 deployment (when ready)

These are preparation notes only. No AWS resources are created here.

1. **Instance**: Launch an EC2 instance (e.g. `t3.medium`, Amazon Linux 2023).
2. **Install Python 3.11+** and `pip install -r requirements.txt`.
3. **Transfer files**: copy the entire `backend/` folder to the instance.
4. **Set environment variables**:
   ```bash
   export ALLOWED_ORIGINS="https://your-cloudfront-domain.cloudfront.net"
   ```
5. **Run with a process manager** (e.g. `systemd` or `gunicorn`):
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```
6. **Security group**: open port 8000 (or 443 if you add HTTPS via nginx/ALB).
7. Note the EC2 public DNS or Elastic IP — the frontend's `VITE_API_BASE_URL`
   must point to this URL before the frontend is built for production.

> **HTTPS on EC2**: For production, put nginx or an Application Load Balancer
> in front of uvicorn and terminate TLS there. The S3/CloudFront frontend
> requires HTTPS on the API origin (mixed-content blocking).

---

## Endpoint reference

### `POST /api/validate`

Request body:
```json
{ "idea": "string", "country": "string" }
```

Response (200 OK):
```json
{
  "submission":        { ... },
  "ml_result":         { "ml_success_score": 0.46, "pattern_label": "...", ... },
  "idea_evaluation":   { "idea_evaluation_score": 72, "dimensions": [...], ... },
  "narrative":         { "strengths": [...], "missing_or_unclear": [...], ... },
  "limitations":       [...],
  "scores_are_separate": true,
  "combined_score":    null
}
```

`ml_result.ml_success_score` is a model-estimated score (0–1) based on
historical sector/country patterns from 1990–2016 Crunchbase data.
It is **not** a calibrated probability of startup success.

`idea_evaluation.idea_evaluation_score` (0–100) measures how clearly the
idea was described, not whether the business will succeed.

The two scores are always separate. `combined_score` is always `null`.

### `GET /health`

Returns `{"status": "ok"}` — use as a liveness probe.
