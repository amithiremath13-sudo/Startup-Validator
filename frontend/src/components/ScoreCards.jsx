function PatternBadge({ patternLabel }) {
  const isSuccess = patternLabel?.toLowerCase().includes('success')
  return (
    <span className={isSuccess ? 'badge badge-success' : 'badge badge-failure'}>
      {isSuccess ? '▲' : '▼'} {patternLabel}
    </span>
  )
}

function GradeBadge({ grade }) {
  const map = {
    'Strong':     'badge-blue',
    'Moderate':   'badge-purple',
    'Developing': 'badge-amber',
    'Weak':       'badge-failure',
    'Very Weak':  'badge-failure',
  }
  return <span className={`badge ${map[grade] ?? 'badge-purple'}`}>{grade}</span>
}

export default function ScoreCards({ mlResult, ideaEvaluation }) {
  const pct       = (mlResult.ml_success_score * 100).toFixed(1)
  const isSuccess = mlResult.predicted_class === 1

  return (
    <>
      <div className="score-grid">
        {/* Component A — ML model */}
        <div className={`score-card ${isSuccess ? 'success' : 'failure'}`}>
          <div className="label">ML Viability Score</div>
          <div
            className="value"
            style={{ color: isSuccess ? 'var(--green-400)' : 'var(--red-400)' }}
          >
            {pct}%
          </div>
          <div className="sub" style={{ marginTop: '0.5rem' }}>
            <PatternBadge patternLabel={mlResult.pattern_label} />
          </div>
          <div className="sub" style={{ marginTop: '0.4rem' }}>
            Historical sector &amp; country fit
          </div>
        </div>

        {/* Component B — Idea clarity */}
        <div className="score-card idea">
          <div className="label">Idea Clarity Score</div>
          <div className="value" style={{ color: 'var(--blue-400)' }}>
            {ideaEvaluation.idea_evaluation_score}
            <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-muted)' }}>
              /100
            </span>
          </div>
          <div className="sub" style={{ marginTop: '0.5rem' }}>
            <GradeBadge grade={ideaEvaluation.grade} />
          </div>
          <div className="sub" style={{ marginTop: '0.4rem' }}>
            Description quality
          </div>
        </div>
      </div>

      <p className="disclaimer">
        <strong style={{ color: 'var(--text-secondary)' }}>These two scores measure different things</strong> and are never
        combined. The ML Viability Score reflects how similar this submission is
        to historical startups that achieved an exit — it is{' '}
        <strong style={{ color: 'var(--text-secondary)' }}>not</strong> a calibrated probability of real-world success.
        The Idea Clarity Score measures how well the idea was described.
      </p>
    </>
  )
}
