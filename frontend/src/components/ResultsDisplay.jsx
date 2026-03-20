import React from 'react'

const MAX_PROB = 0.65  // scale bars relative to this cap

function Bar({ prob, league, label }) {
  const modelW  = Math.min((prob / MAX_PROB) * 100, 100)
  const leagueW = Math.min((league / MAX_PROB) * 100, 100)
  return (
    <div className="result-bars">
      <div className="bar-row">
        <span className="bar-tag" style={{ color: '#D4AF37', fontSize: '0.65rem' }}>MODEL</span>
        <div className="bar-track">
          <div className="bar-fill model" style={{ width: `${modelW}%` }} />
        </div>
        <span className="bar-label">{(prob * 100).toFixed(1)}%</span>
      </div>
      <div className="bar-row">
        <span className="bar-tag" style={{ color: '#5a5a6a', fontSize: '0.65rem' }}>AVG</span>
        <div className="bar-track">
          <div className="bar-fill league" style={{ width: `${leagueW}%` }} />
        </div>
        <span className="bar-label" style={{ color: 'var(--text-muted)' }}>
          {(league * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

export default function ResultsDisplay({ result }) {
  if (!result) return null

  const { outcomes, batter, pitcher, stadium } = result

  return (
    <div className="results-section">
      <div className="results-header">
        <span className="results-title">Prediction</span>
        <span className="results-matchup">
          {batter.name} vs {pitcher.name} @ {stadium}
        </span>
      </div>

      <div className="results-table">
        {outcomes.map((oc) => {
          const sign = oc.vs_pct >= 0 ? '+' : ''
          const deltaClass =
            Math.abs(oc.vs_pct) < 2
              ? 'neutral'
              : oc.vs_pct > 0
              ? 'positive'
              : 'negative'

          return (
            <div key={oc.code} className="result-row">
              <div className="result-label">
                <span className="result-label-name">{oc.name}</span>
                <span className="result-label-code">{oc.code}</span>
              </div>
              <Bar prob={oc.prob} league={oc.league_avg} />
              <div className={`result-delta ${deltaClass}`}>
                {sign}{oc.vs_pct.toFixed(1)}%
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
