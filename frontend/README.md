# AI Startup Validator — Frontend

React + Vite single-page application. Calls the FastAPI backend and
renders the full validation report.

---

## Local development

```bash
# 1. Install dependencies (from frontend/)
npm install

# 2. Start the dev server
npm run dev
# → http://localhost:5173
```

The dev server reads `.env.development`:
```
VITE_API_BASE_URL=http://localhost:8000
```

Make sure the backend is running at `localhost:8000` first:
```bash
# In a separate terminal, from backend/
uvicorn api.app:app --reload --port 8000
```

---

## Production build (for S3 / CloudFront)

### Step 1 — Set the real backend URL

Open `.env.production` and replace the placeholder with your actual
deployed backend URL:

```
VITE_API_BASE_URL=https://your-ec2-url-or-alb-url
```

This value is **baked into the static bundle at build time**.
If the backend URL changes after building, you must rebuild and redeploy.

> **HTTPS requirement**: CloudFront serves HTTPS. Your EC2 backend must
> also be reachable over HTTPS, otherwise browsers will block the request
> as mixed content. Use nginx + a certificate or an Application Load
> Balancer on EC2 to terminate TLS.

### Step 2 — Build

```bash
npm run build
# Output: frontend/dist/
```

### Step 3 — Preview locally (optional)

```bash
npm run preview
# → http://localhost:4173
```

### Step 4 — Deploy to S3

These are preparation notes only. No AWS resources are created here.

1. Create an S3 bucket with **static website hosting** enabled.
2. Upload everything inside `dist/` to the bucket root.
3. Set the bucket policy to allow public read, **or** (recommended) create
   a CloudFront distribution pointing at the bucket and keep the bucket private.
4. In CloudFront:
   - Set the **Origin** to your S3 bucket.
   - Set the **Default root object** to `index.html`.
   - Add a **Custom error response** for 403 and 404 → `index.html` / 200
     (required for React single-page routing).
5. Note the CloudFront domain (e.g. `https://d1234abcd.cloudfront.net`).
6. Add that domain to the backend's `ALLOWED_ORIGINS` environment variable
   before (re)starting the backend server.

---

## Environment variables

| Variable | File | Purpose |
|----------|------|---------|
| `VITE_API_BASE_URL` | `.env.development` | Backend URL for local dev |
| `VITE_API_BASE_URL` | `.env.production` | Backend URL baked into production build |

Only variables prefixed with `VITE_` are embedded in the bundle.

---

## Project structure

```
frontend/
├── public/          ← static assets (copied as-is to dist/)
├── src/
│   ├── api.js              ← fetch wrapper for POST /api/validate
│   ├── App.jsx             ← root component, owns API state
│   ├── index.css           ← global styles
│   └── components/
│       ├── IdeaForm.jsx        ← input form
│       ├── ScoreCards.jsx      ← ML score + Idea Clarity score
│       ├── DimensionBreakdown.jsx  ← five evaluation dimensions
│       ├── NarrativeSection.jsx    ← strengths / gaps / suggestions
│       ├── MLDetails.jsx       ← extracted features, country, categories
│       └── Limitations.jsx     ← collapsible disclosures
├── .env.development
├── .env.production    ← !! edit VITE_API_BASE_URL before building !!
├── vite.config.js
└── package.json
```
