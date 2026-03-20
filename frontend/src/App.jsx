import React, { useCallback, useState } from 'react'
import StadiumSelector from './components/StadiumSelector'
import PlayerCard from './components/PlayerCard'
import PredictButton from './components/PredictButton'
import ResultsDisplay from './components/ResultsDisplay'
import { searchBatters, searchPitchers, predict } from './utils/api'
import { STADIUM_COLORS } from './utils/stadiums'

function getBgStyle(stadiumToken) {
  if (!stadiumToken || !STADIUM_COLORS[stadiumToken]) {
    return { background: 'var(--bg-default)' }
  }
  const { primary, secondary } = STADIUM_COLORS[stadiumToken]
  return {
    background: `radial-gradient(ellipse at 30% 0%, ${secondary}40 0%, ${primary}cc 45%, #0a0a0f 100%)`,
  }
}

export default function App() {
  const [stadium, setStadium]   = useState('')
  const [batter, setBatter]     = useState(null)
  const [pitcher, setPitcher]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const canPredict = batter && pitcher && stadium

  const handlePredict = useCallback(async () => {
    if (!canPredict) return
    setLoading(true)
    setError(null)
    try {
      const data = await predict(batter.mlbam_id, pitcher.mlbam_id, stadium)
      // Merge handedness back into local state for display
      if (data.batter)  setBatter((b) => ({ ...b, stand:    data.batter.stand }))
      if (data.pitcher) setPitcher((p) => ({ ...p, p_throws: data.pitcher.p_throws }))
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [canPredict, batter, pitcher, stadium])

  const handleSelectBatter  = useCallback((p) => { setBatter(p);  setResult(null) }, [])
  const handleSelectPitcher = useCallback((p) => { setPitcher(p); setResult(null) }, [])
  const handleStadiumChange = useCallback((t) => { setStadium(t); setResult(null) }, [])

  return (
    <>
      {/* Full-page background layer */}
      <div className="bg-layer" style={getBgStyle(stadium)} />

      <div className="app-shell">
        <header className="app-header">
          <h1>MLB At-Bat Predictor</h1>
          <p>AI-powered plate appearance outcome probabilities</p>
        </header>

        <main className="app-main">
          <StadiumSelector value={stadium} onChange={handleStadiumChange} />

          <div className="players-row">
            <PlayerCard
              role="batter"
              player={batter}
              onSelect={handleSelectBatter}
              searchFn={searchBatters}
            />
            <div className="vs-divider">
              <span className="vs-text">VS</span>
            </div>
            <PlayerCard
              role="pitcher"
              player={pitcher}
              onSelect={handleSelectPitcher}
              searchFn={searchPitchers}
            />
          </div>

          <PredictButton
            disabled={!canPredict}
            loading={loading}
            onClick={handlePredict}
          />

          {error && (
            <div style={{ color: 'var(--red)', fontSize: '0.9rem', textAlign: 'center' }}>
              {error}
            </div>
          )}

          <ResultsDisplay result={result} />
        </main>
      </div>
    </>
  )
}
