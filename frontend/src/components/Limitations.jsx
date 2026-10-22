/**
 * Limitations.jsx
 * Collapsible section showing the six model and pipeline limitations
 * from the API response. Always disclosed — never hidden by default
 * in a way that prevents users from seeing them.
 */
import { useState } from 'react'

export default function Limitations({ limitations }) {
  const [open, setOpen] = useState(false)

  return (
    <div>
      <button
        className="limitations-toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span>{open ? '▲' : '▼'}</span>
        {open ? 'Hide' : 'Show'} model limitations & disclosures ({limitations.length})
      </button>

      {open && (
        <ul className="limitations-list">
          {limitations.map((lim, i) => (
            <li key={i}>{lim}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
