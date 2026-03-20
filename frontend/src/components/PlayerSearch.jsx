import React, { useCallback, useEffect, useRef, useState } from 'react'

export default function PlayerSearch({ role, onSelect, searchFn }) {
  const [query, setQuery]     = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen]       = useState(false)
  const [loading, setLoading] = useState(false)
  const timerRef = useRef(null)
  const wrapRef  = useRef(null)

  // Debounced search
  const handleChange = useCallback(
    (e) => {
      const val = e.target.value
      setQuery(val)
      clearTimeout(timerRef.current)
      if (val.length < 2) {
        setResults([])
        setOpen(false)
        return
      }
      timerRef.current = setTimeout(async () => {
        setLoading(true)
        try {
          const data = await searchFn(val)
          setResults(data)
          setOpen(data.length > 0)
        } catch {
          setResults([])
        } finally {
          setLoading(false)
        }
      }, 300)
    },
    [searchFn]
  )

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

  return (
    <div className="player-search-wrap" ref={wrapRef}>
      <input
        className="player-search-input"
        type="text"
        value={query}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onFocus={() => results.length > 0 && setOpen(true)}
        placeholder={`Search ${role}…`}
        autoComplete="off"
        spellCheck={false}
      />
      {open && (
        <div className="search-dropdown">
          {results.map((player) => (
            <div
              key={player.mlbam_id}
              className="search-result"
              onMouseDown={() => handleSelect(player)}
            >
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
