export default function MLDetails({ submission, mlResult }) {
  const realCats = mlResult.active_category_features
    .map(f => f.replace(/^cat_/, '').replace(/_/g, ' '))

  const countries = mlResult.active_country_features
    .map(f => f.replace(/^country_/, ''))

  return (
    <div>
      <div className="detail-row">
        <span className="detail-label">Resolved country</span>
        <span className="detail-value">
          {submission.resolved_country
            ? <span className="tag tag-green">{submission.resolved_country}</span>
            : <span style={{ color: 'var(--red-400)' }}>Not recognised — country_Unknown active</span>}
        </span>
      </div>

      <div className="detail-row">
        <span className="detail-label">Detected categories</span>
        <span className="detail-value">
          {realCats.length > 0
            ? realCats.map(c => <span key={c} className="tag tag-blue">{c}</span>)
            : <span style={{ color: 'var(--red-400)' }}>None — cat_Unknown active</span>}
        </span>
      </div>

      <div className="detail-row">
        <span className="detail-label">Country flags</span>
        <span className="detail-value">
          {countries.map(c => <span key={c} className="tag tag-purple">{c}</span>)}
        </span>
      </div>

      <div className="detail-row">
        <span className="detail-label">n_categories</span>
        <span className="detail-value">{mlResult.n_categories}</span>
      </div>

      <div className="detail-row">
        <span className="detail-label">Complete submission</span>
        <span className="detail-value">
          {submission.is_complete_submission
            ? <span className="badge badge-success">Yes</span>
            : <span className="badge badge-failure">No — results less reliable</span>}
        </span>
      </div>

      <p className="disclaimer" style={{ marginTop: '1rem' }}>{mlResult.note}</p>
    </div>
  )
}
