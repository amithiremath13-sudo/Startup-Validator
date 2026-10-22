/**
 * IdeaForm.jsx
 * The input form: idea textarea + country input + submit button.
 */
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
  const canSubmit = idea.trim().length > 0 && effectiveCountry.trim().length > 0

  function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit || loading) return
    onSubmit(idea.trim(), effectiveCountry.trim())
  }

  return (
    <form className="card" onSubmit={handleSubmit} noValidate>
      <h2>Describe your startup idea</h2>

      <div className="field" style={{ marginBottom: '1rem' }}>
        <label htmlFor="idea">
          Startup idea <span style={{ color: '#dc2626' }}>*</span>
        </label>
        <textarea
          id="idea"
          placeholder="e.g. An AI-powered platform that helps small restaurants predict daily demand to reduce food waste. Target customers are independent restaurant owners in India."
          value={idea}
          onChange={e => setIdea(e.target.value)}
          disabled={loading}
          aria-required="true"
        />
        <span style={{ fontSize: '0.78rem', color: '#64748b' }}>
          {idea.trim().split(/\s+/).filter(Boolean).length} words — aim for 40–80 for best results
        </span>
      </div>

      <div className="field-row" style={{ marginBottom: '1.25rem' }}>
        <div className="field">
          <label htmlFor="country">
            Country <span style={{ color: '#dc2626' }}>*</span>
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
          <><span className="spinner" aria-hidden="true" /> Analysing…</>
        ) : (
          'Validate idea'
        )}
      </button>
    </form>
  )
}
