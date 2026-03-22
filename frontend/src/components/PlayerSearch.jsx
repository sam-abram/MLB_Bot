import React, { useCallback, useEffect, useRef, useState } from 'react'
import { getHeadshotThumbUrl } from '../utils/stadiums'

// Re-sort backend results to put prefix matches above substring matches,
// keeping pa_count as the tiebreaker within each tier.
function rankResults(items, query) {
  if (!query) return items
  const q = query.toLowerCase()
  return [...items].sort((a, b) => {
    const aName = a.name.toLowerCase()
    const bName = b.name.toLowerCase()
    const aPrefix = aName.startsWith(q) || aName.split(' ').some((w) => w.startsWith(q))
    const bPrefix = bName.startsWith(q) || bName.split(' ').some((w) => w.startsWith(q))
    if (aPrefix && !bPrefix) return -1
    if (!aPrefix && bPrefix) return 1
    return (b.pa_count || 0) - (a.pa_count || 0)
  })
}

export default function PlayerSearch({ role, onSelect, searchFn }) {
  const [query, setQuery]           = useState('')
  const [results, setResults]       = useState([])
  const [defaultResults, setDefaultResults] = useState([])
  const [open, setOpen]             = useState(false)
  const [loading, setLoading]       = useState(false)
  const timerRef = useRef(null)
  const wrapRef  = useRef(null)

  // Fetch featured/default players on mount
  useEffect(() => {
    searchFn('').then((data) => setDefaultResults(data)).catch(() => {})
  }, [searchFn])

  const handleChange = useCallback(
    (e) => {
      const val = e.target.value
      setQuery(val)
      clearTimeout(timerRef.current)

      if (val.length === 0) {
        setResults(defaultResults)
        setOpen(defaultResults.length > 0)
        return
      }

      timerRef.current = setTimeout(async () => {
        setLoading(true)
        try {
          const data = await searchFn(val)
          const ranked = rankResults(data, val)
          setResults(ranked)
          setOpen(ranked.length > 0)
        } catch {
          setResults([])
        } finally {
          setLoading(false)
        }
      }, 200)
    },
    [searchFn, defaultResults]
  )

  const handleFocus = useCallback(() => {
    if (query.length === 0) {
      setResults(defaultResults)
      setOpen(defaultResults.length > 0)
    } else if (results.length > 0) {
      setOpen(true)
    }
  }, [query, results, defaultResults])

  // Click outside to close
  useEffect(() => {
    const handler = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSelect = (player) => {
    setQuery(player.name)
    setOpen(false)
    onSelect(player)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') setOpen(false)
  }

  const isDefault = query.length === 0
  const displayResults = isDefault ? defaultResults : results

  return (
    <div className="player-search-wrap" ref={wrapRef}>
      <input
        className="player-search-input"
        type="text"
        value={query}
        onChange={handleChange}
        onFocus={handleFocus}
        onKeyDown={handleKeyDown}
        placeholder={`Search ${role}…`}
        autoComplete="off"
        spellCheck={false}
      />
      {open && displayResults.length > 0 && (
        <div className="search-dropdown">
          {isDefault && (
            <div className="search-dropdown-label">Featured Players</div>
          )}
          {displayResults.map((player) => (
            <div
              key={player.mlbam_id}
              className="search-result"
              onMouseDown={() => handleSelect(player)}
            >
              <img
                className="search-result-photo"
                src={getHeadshotThumbUrl(player.mlbam_id)}
                alt=""
                onError={(e) => { e.target.style.display = 'none' }}
              />
              <span className="search-result-name">{player.name}</span>
              {!player.in_vocab && (
                <span className="search-result-unk" title="Limited training data">
                  limited data
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
