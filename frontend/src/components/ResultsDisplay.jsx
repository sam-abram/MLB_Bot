import React, { useState } from 'react'

const ANALYTICS_OPTIONS = [
  { key: 'leagueAvg',  label: 'League Average',  field: 'league_avg',  color: '#D4AF37' },
  { key: 'batterAvg',  label: 'Batter Average',   field: 'batter_avg',  color: '#4A90D9' },
  { key: 'pitcherAvg', label: 'Pitcher Average',  field: 'pitcher_avg', color: '#9B59B6' },
]

function Bar({ prob, comparisons }) {
  const modelW = prob * 100
  return (
    <div className="result-bars">
      <div className="bar-row">
        <span className="bar-tag" style={{ color: '#00e676' }}>MODEL</span>
        <div className="bar-track">
          <div className="bar-fill model" style={{ width: `${modelW}%` }} />
        </div>
        <span className="bar-label">{modelW.toFixed(1)}%</span>
      </div>
      {comparisons.map(({ label, value, color }) => {
        const w = value * 100
        return (
          <div key={label} className="bar-row">
            <span className="bar-tag" style={{ color, fontSize: '0.62rem' }}>{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${w}%`, background: color, opacity: 0.7 }} />
            </div>
            <span className="bar-label" style={{ color }}>{w.toFixed(1)}%</span>
          </div>
        )
      })}
    </div>
  )
}

const ORDER = ['K', 'BIPO', 'BB', '1B', 'XBH', 'HR']

export default function ResultsDisplay({ result }) {
  const [analytics, setAnalytics] = useState({
    leagueAvg: false,
    batterAvg: false,
    pitcherAvg: false,
  })

  if (!result) return null

  const { outcomes, batter, pitcher, stadium } = result
  const byCode = Object.fromEntries(outcomes.map((oc) => [oc.code, oc]))

  const toggle = (key) => setAnalytics((prev) => ({ ...prev, [key]: !prev[key] }))

  const activeComparisons = ANALYTICS_OPTIONS.filter((o) => analytics[o.key])

  return (
    <div className="results-section">
      <div className="results-header">
        <div className="analytics-controls">
          <div className="analytics-label-row">
            <span className="analytics-label">Analytics</span>
            <div className="info-icon-wrap">
              <span className="info-icon">i</span>
              <div className="info-tooltip">
                <strong>League</strong> — how often each outcome occurs across all MLB plate appearances.<br />
                <strong>Batter</strong> — how often each outcome occurs for the selected batter.<br />
                <strong>Pitcher</strong> — how often each outcome occurs for the selected pitcher.
              </div>
            </div>
          </div>
          <div className="analytics-checkboxes">
            {ANALYTICS_OPTIONS.map((opt) => (
              <label key={opt.key} className="analytics-checkbox-label">
                <input
                  type="checkbox"
                  checked={analytics[opt.key]}
                  onChange={() => toggle(opt.key)}
                  className="analytics-checkbox"
                />
                <span style={{ color: opt.color }}>{opt.label}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="results-title-group">
          <span className="results-title">Prediction</span>
          <span className="results-matchup">
            {batter.name} vs {pitcher.name} @ {stadium}
          </span>
        </div>
      </div>

      <div className="results-grid">
        {ORDER.map((code) => {
          const oc = byCode[code]
          if (!oc) return null

          const comparisons = activeComparisons.map((opt) => ({
            label: opt.label.split(' ')[0].toUpperCase(),
            value: oc[opt.field],
            color: opt.color,
          }))

          return (
            <div key={oc.code} className="result-card">
              <div className="result-label">
                <span className="result-label-name">{oc.name}</span>
                <span className="result-label-code">{oc.code === 'XBH' ? '2B/3B' : oc.code === 'BB' ? 'BB / HBP' : oc.code}</span>
              </div>
              <Bar prob={oc.prob} comparisons={comparisons} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
