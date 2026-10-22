import { useState } from 'react'
import { validateIdea } from './api'
import IdeaForm          from './components/IdeaForm'
import ScoreCards        from './components/ScoreCards'
import DimensionBreakdown from './components/DimensionBreakdown'
import NarrativeSection  from './components/NarrativeSection'
import MLDetails         from './components/MLDetails'
import Limitations       from './components/Limitations'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [report,  setReport]  = useState(null)

  async function handleSubmit(idea, country) {
    setLoading(true)
    setError(null)
    setReport(null)
    try {
      const data = await validateIdea(idea, country)
      setReport(data)
      setTimeout(() => {
        document.getElementById('results-top')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 120)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-wrapper">
      {/* ── Navbar ─────────────────────────────────────── */}
      <nav className="navbar" aria-label="Site navigation">
        <div className="navbar__inner">
          <a href="/" className="navbar__brand">
            <div className="navbar__icon" aria-hidden="true">🚀</div>
            <span className="navbar__name">StartupValidator</span>
          </a>
          <span className="navbar__badge">AI-Powered · Academic</span>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────── */}
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero__eyebrow">
          <span aria-hidden="true">⚡</span> ML + Rule-Based Analysis
        </div>
        <h1 id="hero-title" className="hero__title">
          Validate Your<br /><span>Startup Idea</span><br />in Seconds
        </h1>
        <p className="hero__sub">
          Combines a Random Forest model trained on 13,334 historical startups
          with a structured idea-clarity evaluator to give you a data-grounded
          assessment — not a guarantee.
        </p>
      </section>

      {/* ── Main ───────────────────────────────────────── */}
      <main className="container" style={{ paddingBottom: '3rem', flex: 1 }}>
        <IdeaForm onSubmit={handleSubmit} loading={loading} />

        {error && (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">⚠</span>
            <span><strong>Error: </strong>{error}</span>
          </div>
        )}

        {report && (
          <div id="results-top" className="results-appear">

            {/* Scores */}
            <div className="card card--highlight" style={{ marginTop: '1.75rem' }}>
              <div className="card__title">Validation Result</div>
              <ScoreCards
                mlResult={report.ml_result}
                ideaEvaluation={report.idea_evaluation}
              />
            </div>

            {/* Dimensions */}
            <div className="card">
              <div className="card__title">Idea Clarity — Dimension Breakdown</div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1.1rem' }}>
                {report.idea_evaluation.note}
              </p>
              <DimensionBreakdown dimensions={report.idea_evaluation.dimensions} />
            </div>

            {/* Narrative */}
            <div className="card">
              <div className="card__title">Narrative Assessment</div>
              <NarrativeSection narrative={report.narrative} />
            </div>

            {/* ML Details */}
            <div className="card">
              <div className="card__title">ML Model Details</div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.9rem' }}>
                {report.ml_result.source}
              </p>
              <MLDetails submission={report.submission} mlResult={report.ml_result} />
            </div>

            {/* Limitations */}
            <div className="card">
              <Limitations limitations={report.limitations} />
            </div>

          </div>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────── */}
      <footer className="site-footer">
        <p>
          Academic project only · Not investment advice ·
          ML model trained on Crunchbase data ~1990–2016 ·
          <strong style={{ color: 'var(--text-secondary)' }}> Not a probability of success</strong>
        </p>
      </footer>
    </div>
  )
}
