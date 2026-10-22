/**
 * MLDetails.jsx
 * Shows what the ML model extracted: resolved country, detected categories,
 * feature counts, and the ML source note.
 */
export default function MLDetails({ submission, mlResult }) {
  const realCats = mlResult.active_category_features
    .map(f => f.replace(/^cat_/, '').replace(/_/g, ' '))

  const countries = mlResult.active_country_features
    .map(f => f.replace(/^country_/, ''))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>

      <Row label="Resolved country">
        {submission.resolved_country
          ? submission.resolved_country
          : <span style={{ color: '#dc2626' }}>Not recognised — country_Unknown flag active</span>}
      </Row>

      <Row label="Detected categories">
        {realCats.length > 0
          ? <TagList tags={realCats} color="#dbeafe" textColor="#1e40af" />
          : <span style={{ color: '#dc2626' }}>None matched — cat_Unknown flag active</span>}
      </Row>

      <Row label="Active country flags">
        <TagList tags={countries} color="#f0fdf4" textColor="#15803d" />
      </Row>

      <Row label="n_categories">{mlResult.n_categories}</Row>

      <Row label="Complete submission">
        {submission.is_complete_submission
          ? <span style={{ color: '#15803d', fontWeight: 600 }}>Yes</span>
          : <span style={{ color: '#dc2626', fontWeight: 600 }}>No — results may be less reliable</span>}
      </Row>

      <p className="disclaimer" style={{ marginTop: 0 }}>
        {mlResult.note}
      </p>
    </div>
  )
}

function Row({ label, children }) {
  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', fontSize: '0.875rem' }}>
      <span style={{ minWidth: '160px', color: '#64748b', flexShrink: 0 }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{children}</span>
    </div>
  )
}

function TagList({ tags, color, textColor }) {
  return (
    <span style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
      {tags.map(t => (
        <span key={t} style={{
          background: color,
          color: textColor,
          borderRadius: '9999px',
          padding: '0.1rem 0.6rem',
          fontSize: '0.78rem',
          fontWeight: 600,
        }}>
          {t}
        </span>
      ))}
    </span>
  )
}
