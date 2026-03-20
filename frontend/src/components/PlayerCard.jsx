import React from 'react'
import PlayerSearch from './PlayerSearch'
import { getHeadshotUrl } from '../utils/stadiums'

function Silhouette() {
  return (
    <div className="player-photo-placeholder">
      <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="30" cy="20" r="12" fill="white" />
        <ellipse cx="30" cy="48" rx="18" ry="14" fill="white" />
      </svg>
    </div>
  )
}

export default function PlayerCard({ role, player, onSelect, searchFn }) {
  const label = role === 'batter' ? 'BATTER' : 'PITCHER'
  const handLabel = role === 'batter'
    ? (player ? `Bats ${player.stand ?? '?'}` : null)
    : (player ? `Throws ${player.p_throws ?? '?'}` : null)

  return (
    <div className="player-card">
      <div className="player-card-label">{label}</div>
      <PlayerSearch role={label.toLowerCase()} onSelect={onSelect} searchFn={searchFn} />
      <div className="player-photo-wrap">
        {player ? (
          <>
            <img
              className="player-photo"
              src={getHeadshotUrl(player.mlbam_id)}
              alt={player.name}
              onError={(e) => {
                e.target.style.display = 'none'
                e.target.nextSibling && (e.target.nextSibling.style.display = 'flex')
              }}
            />
            <div style={{ display: 'none' }}><Silhouette /></div>
            <div className="player-name">{player.name}</div>
            <div className="player-meta">
              {handLabel && <span className="player-meta-badge">{handLabel}</span>}
              <span className="player-meta-badge">ID {player.mlbam_id}</span>
            </div>
            {!player.in_vocab && (
              <div className="unk-warning">Limited training data — predictions less reliable</div>
            )}
          </>
        ) : (
          <Silhouette />
        )}
      </div>
    </div>
  )
}
