const BASE = '/api'

export async function fetchStadiums() {
  const res = await fetch(`${BASE}/stadiums`)
  if (!res.ok) throw new Error('Failed to fetch stadiums')
  return res.json()
}

export async function searchBatters(q) {
  const url = q ? `${BASE}/search/batters?q=${encodeURIComponent(q)}` : `${BASE}/search/batters`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Search failed')
  return res.json()
}

export async function searchPitchers(q) {
  const url = q ? `${BASE}/search/pitchers?q=${encodeURIComponent(q)}` : `${BASE}/search/pitchers`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Search failed')
  return res.json()
}

export async function predict(batterId, pitcherId, stadium) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ batter_id: batterId, pitcher_id: pitcherId, stadium }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}
