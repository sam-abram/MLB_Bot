import React, { useCallback, useState } from 'react'
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

  const canPredict = batter && pitcher && stadium

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

  return (
    <>
      {/* Gradient background layer (always visible) */}
      <div
        className="bg-gradient"
        style={activeTab === 'about'
          ? { background: 'radial-gradient(ellipse at 20% 0%, #1a0a3a 0%, #0a0a1f 50%, #0a0a0f 100%)' }
          : getBgGradient(stadium)
        }
      />

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
            <div className="about-container">
              <div className="about-intro-space" />
              <div className="about-columns">
                <div />
                <section className="about-section">
                  <h2 className="about-heading">How It Works</h2>
                  <div className="about-body">
                    <p>The model is trained on every pitch thrown in Major League Baseball from 2023 through 2025 – 3 seasons of Statcast data covering velocity, spin, location, and outcomes for every plate appearance.</p>
                    <p>The model first learns embeddings for every batter and pitcher in the league. These embeddings are numerical representations of players learned directly from outcomes, allowing the model to discover patterns on its own: which hitters are similar, which pitchers share tendencies, and how specific matchups play out. This is the part of the model that can pick up on subtleties that traditional statistics might miss.</p>
                    <p>Alongside those embeddings, the model is fed a set of carefully constructed statistical features. For each matchup, the system makes its prediction using six different factors — the batter's overall rates, the pitcher's overall rates, the stadium's effect, each player's platoon split against the other's handedness, and how the pitcher's specific arsenal interacts with the batter's strengths and weaknesses. Each factor produces a full outcome distribution, and all six are expressed as deviations from the league average. A technique called Bayesian shrinkage ensures that a rookie with 50 plate appearances isn't treated with the same confidence as a ten-year veteran — small samples are pulled toward the league average, while established players' rates are trusted more.</p>
                    <p>A learned gate sits between these two approaches, monitoring how much they agree or disagree. The gate forces the model to lean on statistical features where the data is clear, and gives the learned representations more influence when the numbers alone don't tell the full story.</p>
                  </div>
                </section>

                <aside className="about-bio">
                  <img
                    src="/Adobe Express - file.png"
                    alt="Sam Abramowicz"
                    className="about-photo"
                  />
                  <div className="about-bio-text">
                    <p>The MLB At-Bat Predictor was developed by <strong>Sam Abramowicz</strong>, a Mathematics and Computer Science student at the University of Virginia.</p>
                  </div>
                </aside>
              </div>
            </div>
          </main>
        )}
      </div>
    </>
  )
}
