export default function DimensionBreakdown({ dimensions }) {
  return (
    <div>
      {dimensions.map(dim => (
        <DimensionRow key={dim.name} dim={dim} />
      ))}
    </div>
  )
}

function barGradient(score) {
  if (score >= 8) return 'linear-gradient(90deg,#22c55e,#4ade80)'
  if (score >= 6) return 'linear-gradient(90deg,#3b82f6,#60a5fa)'
  if (score >= 4) return 'linear-gradient(90deg,#d97706,#fbbf24)'
  return 'linear-gradient(90deg,#dc2626,#f87171)'
}

function DimensionRow({ dim }) {
  const pct = (dim.score / dim.max_score) * 100
  return (
    <div className="progress-wrap">
      <div className="progress-header">
        <span className="dim-name">
          {dim.name}
          <span style={{
            marginLeft: '0.4rem', fontSize: '0.68rem',
            color: 'var(--text-muted)', fontWeight: 400,
          }}>
            {dim.weight_pct}%
          </span>
        </span>
        <span className="dim-score">{dim.score}/{dim.max_score}</span>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: barGradient(dim.score) }}
          role="progressbar"
          aria-valuenow={dim.score}
          aria-valuemin={0}
          aria-valuemax={dim.max_score}
          aria-label={dim.name}
        />
      </div>
      <p className="progress-explanation">{dim.explanation}</p>
    </div>
  )
}
