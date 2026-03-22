import React, { useCallback, useEffect, useState } from 'react'
import StadiumSelector from './components/StadiumSelector'
import PlayerCard from './components/PlayerCard'
import PredictButton from './components/PredictButton'
import ResultsDisplay from './components/ResultsDisplay'
import { searchBatters, searchPitchers, predict } from './utils/api'
import { STADIUM_COLORS } from './utils/stadiums'

function getBgGradient(stadiumToken) {
  if (!stadiumToken || !STADIUM_COLORS[stadiumToken]) {
    return { background: 'var(--bg-default)' }
  }
  const { primary, secondary } = STADIUM_COLORS[stadiumToken]
  return {
    background: `radial-gradient(ellipse at 30% 0%, ${secondary}40 0%, ${primary}cc 45%, #0a0a0f 100%)`,
  }
}

export default function App() {
  const [activeTab, setActiveTab]   = useState('predictor')
  const [stadium, setStadium]       = useState('')
  const [batter, setBatter]         = useState(null)
  const [pitcher, setPitcher]       = useState(null)
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [photoLoaded, setPhotoLoaded] = useState(false)

  const canPredict = batter && pitcher && stadium

  // Reset photo state when stadium changes so new image crossfades in
  useEffect(() => { setPhotoLoaded(false) }, [stadium])

  const handlePredict = useCallback(async () => {
    if (!canPredict) return
    setLoading(true)
    setError(null)
    try {
      const data = await predict(batter.mlbam_id, pitcher.mlbam_id, stadium)
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

  const stadiumData = stadium ? STADIUM_COLORS[stadium] : null
  const imageUrl    = stadiumData?.imageUrl ?? null

  return (
    <>
      {/* Gradient background layer (always visible) */}
      <div className="bg-gradient" style={getBgGradient(stadium)} />

      {/* Stadium photo background (fades in after image loads) */}
      <div
        className={`bg-photo${photoLoaded ? ' loaded' : ''}`}
        style={photoLoaded ? { backgroundImage: `url(${imageUrl})` } : {}}
      />

      {/* Dark overlay for readability when photo is shown */}
      <div className={`bg-overlay${photoLoaded ? ' active' : ''}`} />

      {/* Hidden preloader — remounts on stadium change to trigger fresh load */}
      {imageUrl && (
        <img
          key={imageUrl}
          src={imageUrl}
          alt=""
          style={{ display: 'none' }}
          onLoad={() => setPhotoLoaded(true)}
          onError={() => setPhotoLoaded(false)}
        />
      )}

      <div className="app-shell">
        <header className="app-header">
          <h1>MLB At-Bat Predictor</h1>
          <p>AI-powered plate appearance outcome probabilities</p>
          <nav className="header-tabs">
            <button
              className={`tab-btn${activeTab === 'predictor' ? ' active' : ''}`}
              onClick={() => setActiveTab('predictor')}
            >
              Predictor
            </button>
            <button
              className={`tab-btn${activeTab === 'about' ? ' active' : ''}`}
              onClick={() => setActiveTab('about')}
            >
              About
            </button>
          </nav>
        </header>

        {activeTab === 'predictor' ? (
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
        ) : (
          <main className="about-page">
            <h2>How It Works</h2>
          </main>
        )}
      </div>
    </>
  )
}
