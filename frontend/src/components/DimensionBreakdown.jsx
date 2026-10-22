/**
 * DimensionBreakdown.jsx
 * Shows all five idea-evaluation dimension scores as labelled progress bars
 * with brief explanations.
 */
export default function DimensionBreakdown({ dimensions }) {
  return (
    <div>
      {dimensions.map(dim => (
        <DimensionRow key={dim.name} dim={dim} />
      ))}
    </div>
  )
}

function barColor(score) {
  if (score >= 8) return '#16a34a'
  if (score >= 6) return '#2563eb'
  if (score >= 4) return '#d97706'
  return '#dc2626'
}

function DimensionRow({ dim }) {
  const pct = (dim.score / dim.max_score) * 100
  return (
    <div className="progress-wrap">
      <div className="progress-header">
        <span className="dim-name">
          {dim.name}
          <span style={{
            marginLeft: '0.4rem',
            fontSize: '0.72rem',
            color: '#94a3b8',
            fontWeight: 400,
          }}>
            {dim.weight_pct}% weight
          </span>
        </span>
        <span className="dim-score">{dim.score}/{dim.max_score}</span>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: barColor(dim.score) }}
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
