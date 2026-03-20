const BASE = '/api'

export async function fetchStadiums() {
  const res = await fetch(`${BASE}/stadiums`)
  if (!res.ok) throw new Error('Failed to fetch stadiums')
  return res.json()
}

export async function searchBatters(q) {
  if (!q || q.length < 2) return []
  const res = await fetch(`${BASE}/search/batters?q=${encodeURIComponent(q)}`)
  if (!res.ok) throw new Error('Search failed')
  return res.json()
}

export async function searchPitchers(q) {
  if (!q || q.length < 2) return []
  const res = await fetch(`${BASE}/search/pitchers?q=${encodeURIComponent(q)}`)
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
