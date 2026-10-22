/**
 * App.jsx
 * --------
 * Root component. Owns API state and renders the full dashboard.
 *
 * Layout:
 *   Header
 *   IdeaForm            ← user input
 *   (error banner)
 *   ── results (only shown after a successful API response) ──
 *   ScoreCards          ← ML score + Idea Clarity score (separate)
 *   DimensionBreakdown  ← five idea-evaluation dimensions
 *   NarrativeSection    ← strengths / gaps / suggestions
 *   MLDetails           ← extracted features, country, categories
 *   Limitations         ← collapsible disclosures
 *   Footer
 */
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
      // Scroll results into view on mobile
      setTimeout(() => {
        document.getElementById('results-top')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="site-header">
        <div className="container">
          <span style={{ fontSize: '1.5rem' }} aria-hidden="true">🚀</span>
          <div>
            <h1>AI Startup Validator</h1>
            <p className="tagline">
              Historical pattern analysis · Not a guarantee of success
            </p>
          </div>
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────── */}
      <main className="container" style={{ paddingBottom: '2rem' }}>

        <IdeaForm onSubmit={handleSubmit} loading={loading} />

        {/* Error state */}
        {error && (
          <div className="error-banner" role="alert">
            <strong>Error: </strong>{error}
          </div>
        )}

        {/* Results */}
        {report && (
          <div id="results-top">

            {/* ── Score overview ─── */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
              <h2>Validation result</h2>
              <ScoreCards
                mlResult={report.ml_result}
                ideaEvaluation={report.idea_evaluation}
              />
            </div>

            {/* ── Idea evaluation dimensions ─── */}
            <div className="card">
              <h2>Idea evaluation — dimension breakdown</h2>
              <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '1rem' }}>
                {report.idea_evaluation.note}
              </p>
              <DimensionBreakdown dimensions={report.idea_evaluation.dimensions} />
            </div>

            {/* ── Strengths / gaps / suggestions ─── */}
            <div className="card">
              <h2>Narrative assessment</h2>
              <NarrativeSection narrative={report.narrative} />
            </div>

            {/* ── ML model details ─── */}
            <div className="card">
              <h2>ML model details</h2>
              <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '1rem' }}>
                Source: {report.ml_result.source}
              </p>
              <MLDetails
                submission={report.submission}
                mlResult={report.ml_result}
              />
            </div>

            {/* ── Limitations ─── */}
            <div className="card">
              <Limitations limitations={report.limitations} />
            </div>

          </div>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="site-footer">
        <p>
          For academic purposes only · Does not constitute investment or business advice
        </p>
        <p style={{ marginTop: '0.25rem' }}>
          ML model trained on Crunchbase data ~1990–2016 ·{' '}
          <strong>Not a probability of success</strong>
        </p>
      </footer>
    </>
  )
}
