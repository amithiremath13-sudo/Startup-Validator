function Icon({ type }) {
  if (type === 'strength')   return <span className="icon" aria-hidden="true">✦</span>
  if (type === 'gap')        return <span className="icon" aria-hidden="true">◈</span>
  if (type === 'suggestion') return <span className="icon" aria-hidden="true">◆</span>
  return null
}

function NarrativeList({ items, type, emptyMsg }) {
  if (!items?.length) {
    return <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{emptyMsg}</p>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>

      {submission_warnings?.length > 0 && (
        <div className="error-banner" style={{
          background: 'rgba(251,191,36,.08)',
          borderColor: 'rgba(251,191,36,.3)',
          color: '#fde68a',
        }}>
          <span aria-hidden="true">⚠</span>
          <div>{submission_warnings.map((w, i) => <p key={i}>{w}</p>)}</div>
        </div>
      )}

      <div>
        <p className="section-title">Strengths identified</p>
        <NarrativeList
          items={strengths}
          type="strength"
          emptyMsg="No notable strengths detected. Try adding more detail about your idea."
        />
      </div>

      <div>
        <p className="section-title">Missing or unclear information</p>
        <NarrativeList
          items={missing_or_unclear}
          type="gap"
          emptyMsg="No major clarity gaps detected by the rule-based evaluator."
        />
      </div>

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
