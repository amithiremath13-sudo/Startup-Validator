/**
 * NarrativeSection.jsx
 * Strengths, missing/unclear information, and improvement suggestions
 * sourced directly from the API response narrative object.
 */

function Icon({ type }) {
  if (type === 'strength')   return <span className="icon" aria-hidden="true">✅</span>
  if (type === 'gap')        return <span className="icon" aria-hidden="true">⚠️</span>
  if (type === 'suggestion') return <span className="icon" aria-hidden="true">💡</span>
  return null
}

function NarrativeList({ items, type, emptyMsg }) {
  if (!items?.length) {
    return <p style={{ fontSize: '0.85rem', color: '#64748b' }}>{emptyMsg}</p>
  }
  return (
    <ul className="narrative-list">
      {items.map((item, i) => (
        <li key={i}>
          <Icon type={type} />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  )
}

export default function NarrativeSection({ narrative }) {
  const { strengths, missing_or_unclear, guidance, submission_warnings } = narrative

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

      {/* Submission warnings (e.g. incomplete submission) */}
      {submission_warnings?.length > 0 && (
        <div className="error-banner" style={{ background: '#fffbeb', borderColor: '#fde68a', color: '#92400e' }}>
          {submission_warnings.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}

      {/* Strengths */}
      <div>
        <p className="section-title">Strengths identified</p>
        <NarrativeList
          items={strengths}
          type="strength"
          emptyMsg="No notable strengths detected in the description. Try adding more detail."
        />
      </div>

      {/* Gaps */}
      <div>
        <p className="section-title">Missing or unclear information</p>
        <NarrativeList
          items={missing_or_unclear}
          type="gap"
          emptyMsg="No major clarity gaps detected by the rule-based evaluator."
        />
      </div>

      {/* Suggestions */}
      <div>
        <p className="section-title">Suggestions for a stronger submission</p>
        <NarrativeList
          items={guidance}
          type="suggestion"
          emptyMsg="No specific suggestions — the submission covers the key areas."
        />
      </div>
    </div>
  )
}
