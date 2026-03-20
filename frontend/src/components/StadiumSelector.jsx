import React, { useEffect, useState } from 'react'
import { fetchStadiums } from '../utils/api'

export default function StadiumSelector({ value, onChange }) {
  const [stadiums, setStadiums] = useState([])

  useEffect(() => {
    fetchStadiums()
      .then((list) => setStadiums(list.sort((a, b) => a.name.localeCompare(b.name))))
      .catch(() => {})
  }, [])

  return (
    <div className="stadium-section">
      <div className="stadium-select-wrap">
        <span className="stadium-label">Stadium</span>
        <select
          className="stadium-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">Select a stadium…</option>
          {stadiums.map((s) => (
            <option key={s.token} value={s.token}>
              {s.name} ({s.team})
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
