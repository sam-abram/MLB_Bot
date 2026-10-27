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
                    <p>Pick a batter, a pitcher, and a ballpark, and the model estimates how likely each ending to that plate appearance is:</p>
                    <ul className="about-outcomes">
                      <li><strong>K</strong><span>Strikeout</span></li>
                      <li><strong>BB</strong><span>Walk or hit by pitch</span></li>
                      <li><strong>1B</strong><span>Single</span></li>
                      <li><strong>XBH</strong><span>Double or triple</span></li>
                      <li><strong>HR</strong><span>Home run</span></li>
                      <li><strong>BIPO</strong><span>Ball in play for an out: groundouts, flyouts, double plays, sacrifices, fielder's choices (reaching on an error counts here too)</span></li>
                    </ul>
                    <p>It learns from every pitch tracked by MLB's Statcast system since Opening Day 2023, and it is retrained every night, so each day's games feed into the next day's predictions.</p>
                    <p>Each prediction blends two views of the matchup. The first is a traditional scouting-report view: how the batter, the pitcher, and the ballpark compare to league average, including platoon splits and how the batter has done against the pitch types this pitcher throws. The second is a neural network that builds its own profile of each player from results and from Statcast measurements of every pitch type: velocity, spin, movement, location, and how hard the batter hits each one. For each of the six outcomes, the model learns how much to trust each view based on how much they disagree.</p>

                    <h3 className="about-subheading">How accurate is it?</h3>
                    <p>Very, when you measure the right thing. Most of a single plate appearance is chance, and even a star is only a few percentage points better than an average player at any one outcome. The model's job is to get those few points right for every matchup, and it does. In the final two weeks of the 2026 regular season, it was tested on <strong>13,140 plate appearances it never trained on</strong>:</p>
                    <ul className="about-list">
                      <li>In the matchups it rated most strikeout-prone (its top 10%), it predicted 34.4% and batters struck out <strong>34.6%</strong> of the time. In its 10% least strikeout-prone matchups, it predicted 11.5% and the actual rate was <strong>11.0%</strong>.</li>
                      <li>Across all six outcomes, its probabilities were off by an average of just <strong>0.7 percentage points</strong>.</li>
                    </ul>
                    <p>In log loss, the standard score for probability forecasts, it beat league-average rates by <strong>1.4%</strong> (95% confidence interval: 1.1%–1.7%). That number looks small only because league average is already a close guess for most players. The model wins by consistently catching the matchups where it isn't.</p>

                    <h3 className="about-subheading">What it doesn't know</h3>
                    <ul className="about-list">
                      <li>The game situation: count, inning, score, or runners on base</li>
                      <li>Weather, wind, or day-to-day changes like injuries, rest, or how many pitches the pitcher has thrown</li>
                      <li>Much about players with little MLB history. Their numbers lean heavily on league average, and a player with no MLB history yet has no profile at all.</li>
                    </ul>

                    <details className="about-details">
                      <summary>Technical details</summary>
                      <div className="about-details-body">
                        <p><strong>Statistical side.</strong> Six rate tables, each turned into log-odds differences from league average across all six outcomes: batter overall, pitcher overall, ballpark, batter vs. the pitcher's hand, pitcher vs. the batter's side, and pitch mix (the pitcher's usage of each pitch type, weighted by the batter's results against that type). Each rate is pulled toward league average by a prior worth 150 plate appearances. A hitter with 150 PA is weighted half on his own numbers and half on league average; at 600 PA it's about 80/20. The model learns one weight per table and adds the weighted differences to league-average log-odds.</p>
                        <p><strong>Neural side.</strong> Learned embeddings for the batter, pitcher, ballpark, and both players' handedness, plus 210 Statcast features. For each of 10 pitch types, these cover the pitcher's usage, velocity, spin rate and axis, movement, release point and extension, location spread, and zone/chase rates, and the batter's exit velocity, launch angle, and contact rate against that pitch. A multilayer network turns these into its own six-outcome log-odds.</p>
                        <p><strong>The gate.</strong> A small network looks at how far apart the two sides are on each outcome, how confident each side is (entropy), and their overall disagreement, and outputs a weight between 0 and 1 for each outcome. Final log-odds = statistical + weight × (neural − statistical), converted to probabilities with a softmax. Everything is trained jointly.</p>
                        <p><strong>Training and updates.</strong> Cross-entropy loss, the AdamW optimizer, and early stopping on held-out games. Every night a GitHub Actions job downloads Statcast data through the previous day, rebuilds features, retrains, and publishes the model to S3. The live server picks up the new model automatically.</p>
                        <p><strong>Evaluation.</strong> Trained through Aug 31, 2026, with early stopping on Sep 1–14, and scored on Sep 15–27 (13,140 PAs). Log loss is 1.4197 vs. 1.4394 for league-average base rates, a gain of 0.0198 nats per PA (1.37%). The 95% CI is 1.07%–1.68%, from standard errors clustered by batter (z = 8.8). The geometric-mean probability ratio is e<sup>0.0198</sup> ≈ 1.020: on average the model gives the outcome that actually happened about 2% more probability than league averages do.</p>
                        <p><strong>Calibration.</strong> Test PAs are split into deciles of predicted probability for each outcome. Expected calibration error by outcome: K 1.05, BIPO 1.09, BB 0.38, 1B 0.89, XBH 0.56, HR 0.37 percentage points. With about 1,300 PAs per decile, that is roughly what sampling noise alone would produce for a perfectly calibrated model. Predictions span 14.2%–30.7% for K and 1.8%–4.5% for HR (10th–90th percentile), against observed rates of 23.0% and 3.0%. That narrow spread is why log-loss gains over league average are small for any plate-appearance model.</p>
                      </div>
                    </details>
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
