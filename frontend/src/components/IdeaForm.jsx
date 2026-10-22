import { useState } from 'react'

const COUNTRIES = [
  'India', 'USA', 'United Kingdom', 'Canada', 'Germany',
  'France', 'Israel', 'China', 'Russia', 'Australia',
  'Brazil', 'Singapore', 'Netherlands', 'Sweden', 'Japan',
  'South Korea', 'Nigeria', 'Kenya', 'UAE', 'Other',
]

export default function IdeaForm({ onSubmit, loading }) {
  const [idea,    setIdea]    = useState('')
  const [country, setCountry] = useState('')
  const [custom,  setCustom]  = useState('')

  const effectiveCountry = country === 'Other' ? custom : country
  const wordCount = idea.trim().split(/\s+/).filter(Boolean).length
  const canSubmit = idea.trim().length > 0 && effectiveCountry.trim().length > 0

  function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit || loading) return
    onSubmit(idea.trim(), effectiveCountry.trim())
  }

  const wordColor = wordCount === 0
    ? 'var(--text-muted)'
    : wordCount < 20
    ? 'var(--amber-400)'
    : 'var(--green-400)'

  return (
    <form className="card" onSubmit={handleSubmit} noValidate>
      <div className="card__title">Describe Your Startup Idea</div>

      <div className="field" style={{ marginBottom: '1.1rem' }}>
        <label htmlFor="idea">
          Startup idea description <span style={{ color: 'var(--red-400)' }}>*</span>
        </label>
        <textarea
          id="idea"
          placeholder="e.g. An AI-powered platform that helps small restaurant owners predict daily demand to reduce food waste. Target customers are independent restaurant owners in India who struggle with over-ordering and spoilage costs."
          value={idea}
          onChange={e => setIdea(e.target.value)}
          disabled={loading}
          aria-required="true"
        />
        <span className="field-hint" style={{ color: wordColor }}>
          {wordCount} words — aim for 40–80 for best results
        </span>
      </div>

      <div className="field-row" style={{ marginBottom: '1.4rem' }}>
        <div className="field">
          <label htmlFor="country">
            Country <span style={{ color: 'var(--red-400)' }}>*</span>
          </label>
          <select
            id="country"
            value={country}
            onChange={e => setCountry(e.target.value)}
            disabled={loading}
            aria-required="true"
          >
            <option value="">Select a country…</option>
            {COUNTRIES.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {country === 'Other' && (
          <div className="field">
            <label htmlFor="country-custom">Country name</label>
            <input
              id="country-custom"
              type="text"
              placeholder="Type your country"
              value={custom}
              onChange={e => setCustom(e.target.value)}
              disabled={loading}
            />
          </div>
        )}
      </div>

      <button
        type="submit"
        className="btn btn-primary btn-full"
        disabled={!canSubmit || loading}
        aria-busy={loading}
      >
        {loading ? (
          <><span className="spinner" aria-hidden="true" /> Analysing idea…</>
        ) : (
          <><span aria-hidden="true">✦</span> Analyse Startup Idea</>
        )}
      </button>
    </form>
  )
}
