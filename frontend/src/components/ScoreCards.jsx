/**
 * ScoreCards.jsx
 * Shows the ML success/failure score and the Idea Evaluation score
 * as two clearly labelled, visually distinct cards.
 *
 * The two scores are always kept separate — never averaged or combined.
 */

function PatternBadge({ patternLabel }) {
  const isSuccess = patternLabel?.toLowerCase().includes('success')
  return (
    <span style={{
      display: 'inline-block',
      padding: '0.2rem 0.6rem',
      borderRadius: '9999px',
      fontSize: '0.72rem',
      fontWeight: 700,
      letterSpacing: '0.05em',
      background: isSuccess ? '#dcfce7' : '#fee2e2',
      color:      isSuccess ? '#15803d' : '#b91c1c',
    }}>
      {patternLabel}
    </span>
  )
}

function GradeBadge({ grade }) {
  const colors = {
    'Strong':     { bg: '#dbeafe', text: '#1d4ed8' },
    'Moderate':   { bg: '#fef9c3', text: '#854d0e' },
    'Developing': { bg: '#fef3c7', text: '#92400e' },
    'Weak':       { bg: '#fee2e2', text: '#b91c1c' },
    'Very Weak':  { bg: '#fce7f3', text: '#9d174d' },
  }
  const { bg, text } = colors[grade] ?? { bg: '#f1f5f9', text: '#475569' }
  return (
    <span style={{
      display: 'inline-block',
      padding: '0.2rem 0.6rem',
      borderRadius: '9999px',
      fontSize: '0.72rem',
      fontWeight: 700,
      letterSpacing: '0.05em',
      background: bg,
      color: text,
    }}>
      {grade}
    </span>
  )
}

export default function ScoreCards({ mlResult, ideaEvaluation }) {
  const pct = (mlResult.ml_success_score * 100).toFixed(1)
  const isSuccess = mlResult.predicted_class === 1

  return (
    <>
      <div className="score-grid">
        {/* ── Component A: ML model score ─────────────────── */}
        <div className={`score-card ${isSuccess ? 'success' : 'failure'}`}>
          <div className="label">ML Model Score</div>
          <div className="value" style={{ color: isSuccess ? 'var(--success)' : 'var(--failure)' }}>
            {pct}%
          </div>
          <div className="sub" style={{ marginTop: '0.4rem' }}>
            <PatternBadge patternLabel={mlResult.pattern_label} />
          </div>
          <div className="sub" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
            Historical sector / country fit
          </div>
        </div>

        {/* ── Component B: Idea evaluation score ──────────── */}
        <div className="score-card idea">
          <div className="label">Idea Clarity Score</div>
          <div className="value" style={{ color: 'var(--brand)' }}>
            {ideaEvaluation.idea_evaluation_score}
            <span style={{ fontSize: '1rem', fontWeight: 400, color: '#94a3b8' }}>/100</span>
          </div>
          <div className="sub" style={{ marginTop: '0.4rem' }}>
            <GradeBadge grade={ideaEvaluation.grade} />
          </div>
          <div className="sub" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
            Description quality
          </div>
        </div>
      </div>

      {/* Separation notice — always shown */}
      <p className="disclaimer">
        <strong>These two scores measure different things</strong> and are never
        combined. The ML score reflects how similar this submission is to
        historical startups that achieved an exit — it is{' '}
        <strong>not</strong> a calibrated probability of real-world success.
        The Idea Clarity Score measures how well the idea was described.
      </p>
    </>
  )
}
